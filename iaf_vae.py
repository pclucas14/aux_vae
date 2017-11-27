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
from tensorflow.contrib import layers
from debug_utils import *

DEBUGGER = get_debugger()


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
            is_training = True if self.mode == 'train' else False 
            # with tf.variable_scope('bn_up'):
            #     x = tf.layers.batch_normalization(inputs=x, axis=1, training=is_training)
            x = conv2d("up_conv1", x, 2 * z_size + 2 * h_size, stride=stride)
            self.qz_mean, self.qz_logsd, self.up_context, h = split(x, 1, [z_size, z_size, h_size, h_size])
            if 'up_mean' not in DEBUGGER.keys():
                DEBUGGER['up_mean'] = self.qz_mean
                DEBUGGER['up_std'] = self.qz_logsd
            h = tf.nn.elu(h)
            h = conv2d("up_conv3", h, h_size)
            if self.downsample:
                input = resize_nearest_neighbor(input, 0.5)
            return input + 0.1 * h

    def down(self, input):
        global DEBUGGER 
        hps = self.hps
        h_size = hps.nr_filters
        z_size = hps.z_size

        with arg_scope([conv2d, ar_multiconv2d]):
            x = tf.nn.elu(input)

            is_training = True if self.mode == 'train' else False 
            # with tf.variable_scope('bn_down'):
            #     x = tf.layers.batch_normalization(inputs=x, axis=1, training=is_training)
            x = conv2d("down_conv1", x, 4 * z_size + h_size * 2)
            pz_mean, pz_logsd, rz_mean, rz_logsd, down_context, h_det = split(x, 1, [z_size] * 4 + [h_size] * 2)

            if 'down_mean' not in DEBUGGER.keys():
                DEBUGGER['down_mean'] = rz_mean
                DEBUGGER['down_std'] = rz_logsd
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
                logq_f = logqs # remove this
                z_f = z        # remove thos 
                x = ar_multiconv2d("ar_multiconv2d", z, context, [h_size, h_size], [z_size, z_size])
                arw_mean, arw_logsd = x[0] * 0.1, x[1] * 0.1
                # arw_mean, arw_logsd = x_0 * 0.1, x_1 * 0.1
                
                """" THIS LINE (exp(log_sd)) causes overflow, which causes nan """
                z = (z - arw_mean) / tf.exp(arw_logsd)
                logqs += arw_logsd
                logps = prior.logps(z)

                kl_cost = logqs - logps

                if self.mode != 'init' : 
                    cnt = sum(['logps' in k for k in get_debug_keys()])
                    cnt = str(min(3, cnt))
                    DEBUGGER['logps'+cnt] = logps
                    DEBUGGER['arw_mean'+cnt] = arw_mean
                    DEBUGGER['arw_std'+cnt] = arw_logsd
                    DEBUGGER['z'+cnt] = z
                    DEBUGGER['kl_og'+cnt] = kl_cost
                
                if hps.kl_min > 0:
                    # [0, 1, 2, 3] -> [0, 1] -> [1] / (b * k)
                    activations = tf.reduce_sum(kl_cost, [2,3])
                    kl_ave = tf.reduce_mean(activations, [0], keep_dims=True)
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
            return output, kl_obj, kl_cost, activations


def model_spec_iaf(x, hps, mode, gpu=-1):
        global DEBUGGER
        # x should already be a float tensor
        # x = tf.to_float(x)
        # x = tf.clip_by_value((x + 0.5) / 256.0, 0.0, 1.0) - 0.5
        '''
        input is in range [-1,1] --> divide by 2 to get [-.5, .5]
        '''
        # x = x / 2.

        dec_log_stdv = tf.get_variable('dec_log_stdv', initializer=tf.constant(0.0))
        DEBUGGER['dec_log_stdv'] = dec_log_stdv
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

            DEBUGGER['h_pre_kl'] = h
            # top->down
            h_topi = h_top = tf.get_variable("h_top", [h_size], initializer=tf.zeros_initializer())
            h_top = tf.reshape(h_top, [1, -1, 1, 1])
            h = tf.tile(h_top, [data_size, 1, int(hps.image_size / 2 ** len(layers)), int(hps.image_size / 2 ** len(layers))])
            kl_cost = kl_obj = 0.0
            acts = []

            DEBUGGER['h_pre_kl_2'] = h

            for i, layer in reversed(list(enumerate(layers))):
                for j, sub_layer in reversed(list(enumerate(layer))):
                    with tf.variable_scope("IAF_%d_%d" % (i, j)):
                        if hps.kl_min > 0 : 
                            h, cur_obj, cur_cost, act = sub_layer.down(h)
                            acts.append(act)
                        else : 
                            h, cur_obj, cur_cost      = sub_layer.down(h)
                        kl_obj += cur_obj
                        kl_cost += cur_cost
                        DEBUGGER["IAF_%d_%d_h_down" % (i, j)] = h
                        DEBUGGER["IAF_%d_%d_kl_obj" % (i, j)] = cur_obj
                        DEBUGGER["IAF_%d_%d_kl_cost" % (i, j)] = cur_cost
            
            if hps.kl_min > 0 :
                acts = tf.reduce_mean(tf.add_n(acts), [0])
            
            x = tf.nn.elu(h)
            x = deconv2d("x_dec", x, 3, [5, 5])
            # x = tf.clip_by_value(x, -0.5 + 1 / 512., 0.5 - 1 / 512.)

        if mode == 'init' : 
            return h_topi

        log_pxz = discretized_logistic(x, dec_log_stdv, sample=orig_x)
        obj = tf.reduce_sum(kl_obj - log_pxz)

        # TODO : remove this
        DEBUGGER['dec_log_stdv'] = dec_log_stdv
        DEBUGGER['kl_obj'] = kl_obj
        DEBUGGER['acts'] = acts

        return (x, log_pxz, kl_cost, kl_obj) if hps.kl_min > 0. else (x, log_pxz, kl_cost)
