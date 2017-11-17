
"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr-gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf

import pixel_cnn_pp.nn as nn
import pixel_cnn_pp.plotting as plotting
from pixel_cnn_pp.model import model_spec
import data.cifar10_data as cifar10_data
import data.imagenet_data as imagenet_data

""" IAF imports """
from iaf_vae import model_spec_iaf
from tf_utils.adamax import AdamaxOptimizer
from tf_utils.distributions import compute_lowerbound

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='/tmp/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='save_dir/default_model/',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str,
                    default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=10,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true',
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu',
                    help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional',
                    action='store_true', help='Condition generative model on labels?')
# optimization
parser.add_argument('-l', '--learning_rate', type=float,
                    default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=16,
                    help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100,
                    help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5,
                    help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=8,
                    help='How many GPUs to distribute the training across?')
parser.add_argument('-w', '--z_size', type=int,
                    default=32, help='latent dim size')
parser.add_argument('-k', '--k', type=int,
                    default=1, help='number of samples drawn from posterior')
parser.add_argument('-v', '--kl_min', type=float,
                    default=.25, help='number of free bit/nats')
parser.add_argument('-f', '--depth', type=int,
                    default=2, help='number of downsampling blocks')
parser.add_argument('-j', '--num_blocks', type=int,
                    default=2, help='number of residual blocks IN THE VAE')
parser.add_argument('-u', '--image_size', type=int,
                    default=32, help='size of image')
parser.add_argument('-L', '---Lambda', type=float, 
                    default=10., help='lambda of kl term, as described in AUX VAE paper')

# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995,
                    help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4,
                                  separators=(',', ':')))  # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# initialize data loaders for train/test splits
DataLoader = {'cifar': cifar10_data.DataLoader,
              'imagenet': imagenet_data.DataLoader}[args.data_set]
train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu,
                        rng=rng, shuffle=True, return_labels=args.class_conditional)
test_data = DataLoader(args.data_dir, 'test', args.batch_size *
                       args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
obs_shape = train_data.get_observation_size()  # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'

###  data place holders  ###
# for data dependant initialization
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)

# for regular training
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape)
      for i in range(args.nr_gpu)]

# no conditioning right now, simply set up unconditional place holders
h_init = None
h_sample = [None] * args.nr_gpu
hs = h_sample

# other placeholders
tf_lr = tf.placeholder(tf.float32, shape=[])

# useful constants
num_pixels = 3 * args.image_size ** 2

### model creation/init  ###
""" VAE with IAF model creation """
vae_iaf    = tf.make_template('vae_iaf', model_spec_iaf)
vae_init   = vae_iaf(tf.transpose(x_init, perm=[0,3,1,2]), args, "init")
saver_iaf  = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, \
                           scope='vae_iaf'))

"""  PixelCNN  model  creation """
model_opt = {'nr_filters': args.nr_filters,
             'nr_logistic_mix': args.nr_logistic_mix, 
             'resnet_nonlinearity': args.resnet_nonlinearity,
             'nr_resnet': args.nr_resnet}

pixel_cnn  = tf.make_template('pixel_cnn', model_spec)
pcnn_out   = pixel_cnn(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)
saver_pcnn = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, \
                           scope='pixel_cnn'))
all_params = tf.trainable_variables()
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

###    training loop     ###
opt = AdamaxOptimizer(args.learning_rate)
grads = []
losses_pcnn_train   , losses_pcnn_test     = [], []
losses_log_pxz_train, losses_log_pxz_test  = [], []
losses_kl_train,      losses_kl_test       = [], []
elbo_train,           elbo_test            = [], []
elbo_full_train,      elbo_full_test       = [], []
full_objs_train,      full_objs_test       = [], []
acts_train,           acts_test            = [], []

replace_none = lambda x : [0. if i==None else i for i in x]

for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # forward pass
        vae_in                                      = tf.transpose(xs[i], perm=[0,3,1,2])
        # x = vae_iaf(vae_in, args, 'train', gpu=i)
        # import pdb; pdb.set_trace()
        vae_out, log_pxz_train, kl_train, act_train = vae_iaf(vae_in, args, "train", gpu=i)
        vae_out, log_pxz_test,  kl_test,  act_test  = vae_iaf(vae_in, args, "test" , gpu=i)

        pcnn_in       = tf.transpose(vae_out, perm=[0,2,3,1])
        pcnn_out      = pixel_cnn(pcnn_in, hs[i], dropout_p=args.dropout_p, **model_opt)
        pcnn_out_test = pixel_cnn(pcnn_in, hs[i], dropout_p=0., **model_opt)
        
        # compute losses
        loss_pcnn_train = nn.discretized_mix_logistic_loss(xs[i], pcnn_out)
        loss_pcnn_test  = nn.discretized_mix_logistic_loss(xs[i], pcnn_out_test)

        # save losses
        losses_pcnn_train.append(loss_pcnn_train); losses_pcnn_test.append(loss_pcnn_test)
        losses_kl_train.append(kl_train); losses_kl_test.append(kl_test)
        acts_train.append(act_train); acts_test.append(act_test)
        losses_log_pxz_train.append(log_pxz_train)
        losses_log_pxz_test.append(log_pxz_test) 
        elbo_train.append(tf.reduce_sum(compute_lowerbound(log_pxz_train, kl_train, args.k)))
        elbo_test.append(tf.reduce_sum(compute_lowerbound(log_pxz_test, kl_test, args.k)))
        elbo_full_train.append(tf.reduce_sum(compute_lowerbound(loss_pcnn_train, kl_train, args.k)))
        elbo_full_test.append(tf.reduce_sum(compute_lowerbound(loss_pcnn_test, kl_test, args.k)))

        # full_obj_train = compute_lowerbound(log_pxz_train + loss_pcnn_train, args.Lambda*kl_train, args.k)
        full_obj_train = -log_pxz_train + loss_pcnn_train + args.Lambda*kl_train
        full_obj_train = tf.reduce_sum(full_obj_train)
        # full_obj_test  = compute_lowerbound(log_pxz_test + loss_pcnn_test, args.Lambda*kl_test, args.k)
        full_obj_test = -log_pxz_test + loss_pcnn_test + args.Lambda*kl_train
        full_obj_test  = tf.reduce_sum(full_obj_train)
        full_objs_train.append(full_obj_train); full_objs_test.append(full_obj_test)

        grads.append(replace_none(tf.gradients(full_obj_train, all_params)))

# accumulate gradient over GPUs
with tf.device('/gpu:0'):
    for i in range(1, args.nr_gpu):
        full_objs_train[0] += full_objs_train[i]
        full_objs_test[0]  += full_objs_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    optimizer = tf.group(nn.adam_updates(
        all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995))

# convert loss to bits/dim
bits_per_dim = losses_pcnn_train[
    0] / (args.nr_gpu * np.log(2.) * np.prod(obs_shape) * args.batch_size)
bits_per_dim_test = losses_pcnn_test[
    0] / (args.nr_gpu * np.log(2.) * np.prod(obs_shape) * args.batch_size)

# log to Tensorboard
ds, ts = [], [] # dev and test summaries
ds = [
tf.summary.scalar('vae/recon_train', tf.reduce_sum(tf.stack(losses_log_pxz_train))), 
tf.summary.scalar('vae/kl_train', tf.reduce_sum(tf.stack(losses_kl_train))),
tf.summary.scalar('comb/elbo_full_train', tf.reduce_sum(tf.stack(elbo_full_train))),
tf.summary.scalar('com/lambda_elbo_full_train', tf.reduce_sum(tf.stack(full_objs_train))),
tf.summary.scalar('pcnn/recon_train', tf.reduce_sum(tf.stack(losses_pcnn_train))),
tf.summary.scalar('comb/elbo_full_train', tf.add_n(elbo_full_train) / \
                        (np.log(2.)*num_pixels*args.batch_size*args.nr_gpu)),
tf.summary.scalar('vae/elbo_train', tf.add_n(elbo_train) / \
                        (np.log(2.)*num_pixels*args.batch_size*args.nr_gpu)),
tf.summary.histogram('vae/iaf_act_train', (tf.add_n(acts_train)/len(acts_train)))
]
 
ts = [
tf.summary.scalar('vae/recon_test', tf.reduce_sum(tf.stack(losses_log_pxz_test))),
tf.summary.scalar('vae/kl_test', tf.reduce_sum(tf.stack(losses_kl_test))),
tf.summary.scalar('comb/lambda_elbo_full_test', tf.reduce_sum(tf.stack(full_objs_test))),
tf.summary.scalar('pcnn/recon_test', tf.reduce_sum(tf.stack(losses_pcnn_test))),

tf.summary.scalar('comb/elbo_full_test', tf.add_n(elbo_full_test) / \
                        (np.log(2.)*num_pixels*args.batch_size*args.nr_gpu)),
tf.summary.scalar('vae/elbo_test', tf.add_n(elbo_test) / \
                        (np.log(2.)*num_pixels*args.batch_size*args.nr_gpu)),
tf.summary.histogram('vae/iaf_act_test', (tf.add_n(acts_test)/len(acts_test)))
]

merged_train, merged_test = tf.summary.merge(ds), tf.summary.merge(ts)

def make_feed_dict(data, init=False):
    if type(data) is tuple:
        x, y = data
    else:
        x = data
        y = None
    # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    x = np.cast[np.float32]((x - 127.5) / 127.5)
    if init:
        feed_dict = {x_init: x}
        if y is not None:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if y is not None:
            y = np.split(y, args.nr_gpu)
            feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
    return feed_dict

# sample from the model
new_x_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # vae_in  = tf.transpose(xs[i], perm=[0,3,1,2])
        # vae_out, _, _, _= vae_iaf(vae_in, args, "test" , gpu=i)
        # pcnn_in       = tf.transpose(vae_out, perm=[0,2,3,1])
        gen_par = pixel_cnn(xs[i], h_sample[i], dropout_p=0, **model_opt)
        new_x_gen.append(nn.sample_from_discretized_mix_logistic(
            gen_par, args.nr_logistic_mix))

def sample_from_model(sess):
    x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32)
             for i in range(args.nr_gpu)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(
                new_x_gen, {xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            for i in range(args.nr_gpu):
                x_gen[i][:, yi, xi, :] = new_x_gen_np[i][:, yi, xi, :]
    return np.concatenate(x_gen, axis=0)


# //////////// perform training //////////////
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print('starting training')
test_bpd = []
lr = args.learning_rate
index_train, index_test = 0, 0
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(args.save_dir + '/train', sess.graph)
    test_writer  = tf.summary.FileWriter(args.save_dir + '/test' , sess.graph)
    for epoch in range(args.max_epochs):
        begin = time.time()

        # init
        if epoch == 0:
            # manually retrieve exactly init_batch_size examples
            feed_dict = make_feed_dict(
                train_data.next(args.init_batch_size), init=True)
            train_data.reset()  # rewind the iterator back to 0 to do one full epoch
            sess.run(vae_init.initializer)
            sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
            print('initializing the model...')
            if args.load_params:
                ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
                print('restoring parameters from', ckpt_file)
                saver.restore(sess, ckpt_file)
        
        # train for one epoch
        train_losses = []
        i = 0
        for d in train_data:
            feed_dict = make_feed_dict(d)
            # forward/backward/update model on each gpu
            lr *= args.lr_decay
            feed_dict.update({tf_lr: lr})
            if i % args.write_every : 
                l, summary, _ = sess.run([bits_per_dim, merged_train, optimizer], feed_dict)
                train_writer.add_summary(summary, index_train)
                index_train += 1
            else : 
                l, _ = sess.run([bits_per_dim, optimizer], feed_dict)
            train_losses.append(l)
            i += 1

        train_loss_gen = np.mean(train_losses)

        # compute likelihood over test data
        test_losses = []
        i = 0
        for d in test_data:
            feed_dict = make_feed_dict(d)
            l, summary = sess.run([bits_per_dim_test, merged_test], feed_dict)
            test_losses.append(l)
            test_writer.add_summary(summary, index_test)
            index_test += 1
            i += 1
        test_loss_gen = np.mean(test_losses)
        test_bpd.append(test_loss_gen)

        # log progress to console
        print("Iteration %d, time = %ds, train bits_per_dim = %.4f, test bits_per_dim = %.4f" % (
            epoch, time.time() - begin, train_loss_gen, test_loss_gen))
        sys.stdout.flush()
        
        if epoch % args.save_interval == 0:

            # generate samples from the model
            sample_x = sample_from_model(sess)
            img_tile = plotting.img_tile(sample_x[:int(np.floor(np.sqrt(
                args.batch_size * args.nr_gpu))**2)], aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
            plotting.plt.savefig(os.path.join(
                args.save_dir, '%s_sample%d.png' % (args.data_set, epoch)))
            plotting.plt.close('all')
            print('saved image')

            # save params
            saver.save(sess, args.save_dir + '/params_' +
                       args.data_set + '.ckpt')
            np.savez(args.save_dir + '/test_bpd_' + args.data_set +
                     '.npz', test_bpd=np.array(test_bpd))
