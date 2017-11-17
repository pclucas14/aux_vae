import pdb
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tf_utils.adamax import AdamaxOptimizer
from tf_utils.hparams import HParams
from tf_utils.common import img_stretch, img_tile
from tf_utils.common import assign_to_gpu, split, CheckpointLoader, average_grads, NotBuggySupervisor
from tf_utils.layers import conv2d, deconv2d, ar_multiconv2d, resize_nearest_neighbor
from tf_utils.distributions import DiagonalGaussian, discretized_logistic, compute_lowerbound, repeat
from tf_utils.data_utils import get_inputs, get_images
import tqdm


class IAFLayer(object):
    def __init__(self, hps, mode, downsample):
        self.hps = hps
        self.mode = mode
        self.downsample = downsample

    def up(self, input, **_):
        hps = self.hps
        h_size = hps.nr_filters
        z_size = hps.z_size
        stride = [2, 2] if self.downsample else [1, 1]

        with arg_scope([conv2d]):
            x = tf.nn.elu(input)
            x = conv2d("up_conv1", x, 2 * z_size + 2 * h_size, stride=stride)
            self.qz_mean, self.qz_logsd, self.up_context, h = split(x, 1, [z_size, z_size, h_size, h_size])

            h = tf.nn.elu(h)
            h = conv2d("up_conv3", h, h_size)
            if self.downsample:
                input = resize_nearest_neighbor(input, 0.5)
            return input + 0.1 * h

    def down(self, input):
        hps = self.hps
        h_size = hps.nr_filters
        z_size = hps.z_size

        with arg_scope([conv2d, ar_multiconv2d]):
            x = tf.nn.elu(input)
            x = conv2d("down_conv1", x, 4 * z_size + h_size * 2)
            pz_mean, pz_logsd, rz_mean, rz_logsd, down_context, h_det = split(x, 1, [z_size] * 4 + [h_size] * 2)

            prior = DiagonalGaussian(pz_mean, 2 * pz_logsd)
            posterior = DiagonalGaussian(rz_mean + self.qz_mean, 2 * (rz_logsd + self.qz_logsd))
            context = self.up_context + down_context

            if self.mode in ["init", "sample"]:
                z = prior.sample
            else:
                z = posterior.sample

            if self.mode == "sample":
                kl_cost = kl_obj = tf.zeros([hps.batch_size * hps.k])
            else:
                logqs = posterior.logps(z)
                x = ar_multiconv2d("ar_multiconv2d", z, context, [h_size, h_size], [z_size, z_size])
                arw_mean, arw_logsd = x[0] * 0.1, x[1] * 0.1
                z = (z - arw_mean) / tf.exp(arw_logsd)
                logqs += arw_logsd
                logps = prior.logps(z)

                kl_cost = logqs - logps

                if hps.kl_min > 0:
                    # [0, 1, 2, 3] -> [0, 1] -> [1] / (b * k)
                    kl_ave = tf.reduce_mean(tf.reduce_sum(kl_cost, [2, 3]), [0], keep_dims=True)
                    kl_ave = tf.maximum(kl_ave, hps.kl_min)
                    kl_ave = tf.tile(kl_ave, [hps.batch_size * hps.k, 1])
                    kl_obj = tf.reduce_sum(kl_ave, [1])
                else:
                    kl_obj = tf.reduce_sum(kl_cost, [1, 2, 3])
                kl_cost = tf.reduce_sum(kl_cost, [1, 2, 3])
            
            h = tf.concat([z, h_det], 1)
            h = tf.nn.elu(h)
            if self.downsample:
                input = resize_nearest_neighbor(input, 2)
                h = deconv2d("down_deconv2", h, h_size)
            else:
                h = conv2d("down_conv2", h, h_size)
            output = input + 0.1 * h
            return output, kl_obj, kl_cost


def get_default_hparams():
    return HParams(
        batch_size=16,        # Batch size on one GPU.
        eval_batch_size=100,  # Batch size for evaluation.
        num_gpus=8,           # Number of GPUs (effectively increases batch size).
        learning_rate=2e-3,   # Learning rate.
        z_size=32,            # Size of z variables.
        h_size=160,           # Size of resnet block.
        kl_min=0.25,          # Number of "free bits/nats".
        depth=2,              # Number of downsampling blocks.
        num_blocks=2,         # Number of resnet blocks for each downsampling layer.
        k=1,                  # Number of samples for IS objective.
        dataset="cifar10",    # Dataset name.
        image_size=32,        # Image size.
    )

'''
class CVAE1(object):
    def __init__(self, hps=None, mode='train', x=None):
        self.hps = hps is hps else get_default_hparams()
        self.mode = mode
        input_shape = [hps.batch_size * hps.num_gpus, 3, hps.image_size, hps.image_size]
        self.x = tf.placeholder(tf.uint8, shape=input_shape) if x is None else x
        self.m_trunc = []
        self.dec_log_stdv = tf.get_variable("dec_log_stdv", initializer=tf.constant(0.0))
         
        losses = []
        grads = []
        # xs = tf.split(0, hps.num_gpus, self.x)
        xs = tf.split(self.x, hps.num_gpus, 0)
        opt = AdamaxOptimizer(hps.learning_rate)

        num_pixels = 3 * hps.image_size * hps.image_size
        for i in range(hps.num_gpus):
            with tf.device(assign_to_gpu(i)):
                m, obj, loss = self._forward(xs[i], i)
                losses += [loss]
                self.m_trunc += [m]

                # obj /= (np.log(2.) * num_pixels * hps.batch_size)
                if mode == "train":
                    grads += [opt.compute_gradients(obj)]

        self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer(),
                                            trainable=False)
        self.bits_per_dim = tf.add_n(losses) / (np.log(2.) * num_pixels * hps.batch_size * hps.num_gpus) 
         
        if mode == "train":
            # add gradients together and get training updates
            grad = average_grads(grads)
            self.train_op = opt.apply_gradients(grad, global_step=self.global_step)
            tf.summary.scalar("model/bits_per_dim", self.bits_per_dim)
            tf.summary.scalar("model/dec_log_stdv", self.dec_log_stdv)
            self.summary_op = tf.summary.merge_all()
        else:
            self.train_op = tf.no_op()

        if mode in ["train", "eval"]:
            with tf.name_scope(None):  # This is needed due to EMA implementation silliness.
                # keep track of moving average
                ema = tf.train.ExponentialMovingAverage(decay=0.999)
                self.train_op = tf.group(*[self.train_op, ema.apply(tf.trainable_variables())])
                self.avg_dict = ema.variables_to_restore()
'''


def model_spec_iaf(x, hps, mode, gpu=-1):
        # x should already be a float tensor
        # x = tf.to_float(x)
        # x = tf.clip_by_value((x + 0.5) / 256.0, 0.0, 1.0) - 0.5

        dec_log_stdv = tf.get_variable('dec_log_stdv', initializer=tf.constant(0.0))

        # Input images are repeated k times on the input.
        # This is used for Importance Sampling loss (k is number of samples).
        bs = hps.init_batch_size if mode == 'init' else hps.batch_size
        data_size = bs * hps.k

        x = repeat(x, hps.k)
        orig_x = x
        h_size = hps.nr_filters

        with arg_scope([conv2d, deconv2d], init=(mode == "init")):
            layers = []
            for i in range(hps.depth):
                layers.append([])
                for j in range(hps.num_blocks):
                    downsample = (i > 0) and (j == 0)
                    layers[-1].append(IAFLayer(hps, mode, downsample))

            h = conv2d("x_enc", x, h_size, [5, 5], [2, 2])  # -> [16, 16]
            for i, layer in enumerate(layers):
                for j, sub_layer in enumerate(layer):
                    with tf.variable_scope("IAF_%d_%d" % (i, j)):
                        h = sub_layer.up(h)

            # top->down
            h_topi = h_top = tf.get_variable("h_top", [h_size], initializer=tf.zeros_initializer())
            h_top = tf.reshape(h_top, [1, -1, 1, 1])
            h = tf.tile(h_top, [data_size, 1, int(hps.image_size / 2 ** len(layers)), int(hps.image_size / 2 ** len(layers))])
            kl_cost = kl_obj = 0.0

            for i, layer in reversed(list(enumerate(layers))):
                for j, sub_layer in reversed(list(enumerate(layer))):
                    with tf.variable_scope("IAF_%d_%d" % (i, j)):
                        h, cur_obj, cur_cost = sub_layer.down(h)
                        kl_obj += cur_obj
                        kl_cost += cur_cost

                        # if gpu == hps.nr_gpu - 1:
                        #     tf.summary.scalar("vae/kl_obj_%s_%02d_%02d" % (mode, i, j), tf.reduce_mean(cur_obj))
                        #     tf.summary.scalar("vae/kl_cost_%s_%02d_%02d" % (mode, i, j), tf.reduce_mean(cur_cost))

            x = tf.nn.elu(h)
            x = deconv2d("x_dec", x, 3, [5, 5])
            x = tf.clip_by_value(x, -0.5 + 1 / 512., 0.5 - 1 / 512.)

        if mode == 'init' : 
            return h_topi

        log_pxz = discretized_logistic(x, dec_log_stdv, sample=orig_x)
        obj = tf.reduce_sum(kl_obj - log_pxz)

        # if gpu == hps.num_gpus - 1:
        #     tf.summary.scalar("vae/%s_log_pxz" % mode, -tf.reduce_mean(log_pxz))
        #     tf.summary.scalar("vae/%s_kl_obj" % mode, tf.reduce_mean(kl_obj))
        #     tf.summary.scalar("vae/%s_kl_cost" % mode, tf.reduce_mean(kl_cost))
        # loss = tf.reduce_sum(compute_lowerbound(log_pxz, kl_cost, hps.k))
        
        return x, log_pxz, kl_cost
