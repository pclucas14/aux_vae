import os
import sys
import time
import json
import argparse
from debug_utils import * 

import numpy as np
import tensorflow as tf

import pixel_cnn_pp.nn as nn
import pixel_cnn_pp.plotting as plotting
from pixel_cnn_pp.model import * #model_spec
import data.cifar10_data as cifar10_data
import data.imagenet_data as imagenet_data


DEGUGGER = get_debugger()

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='/tmp/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='save_dir/default_model/',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar10', help='Can be either cifar10|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=10,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-T', '--write_every', type=int, default=25,
                    help='Every how many iterations to write to tensorboard?')
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
parser.add_argument('-b', '--batch_size', type=int, default=12,
                    help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100,
                    help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5,
                    help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--num_gpus', type=int, default=2,
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
parser.add_argument('-M', '--mode', type=str, 
                    default='train', help='mode, either train or eval')

# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995,
                    help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4,
                                  separators=(',', ':')))  # pretty print args

'''
Important notes : vae implementation takes x e (-.5, .5), while
pixelcnn takes floats from in (-1, 1), so always x2 pcnn input
'''

### DATA STREAMS ###
DataLoader = {'cifar10': cifar10_data.DataLoader,
              'imagenet': imagenet_data.DataLoader}[args.dataset]
train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.num_gpus,
                        shuffle=True, return_labels=False)
test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.num_gpus, 
                        shuffle=False, return_labels=False)
obs_shape = (3, args.image_size, args.image_size)
num_pixels = 3 * args.image_size * args.image_size

### PLACEHOLDERS ###
x_init   = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape)
              for i in range(args.num_gpus)]
tf_lr = tf.placeholder(tf.float32, shape=[])

h_init = None
h_sample = [None] * args.num_gpus
hs = h_sample

# TODO : remove this
vs = hs # [tf.transpose(x, perm=[0,2,3,1]) for x in xs]

# for PIXELCNN initialization
model_opt = {'nr_filters': args.nr_filters,
             'nr_logistic_mix': args.nr_logistic_mix, 
             'resnet_nonlinearity': args.resnet_nonlinearity,
             'nr_resnet': args.nr_resnet}

with tf.name_scope('pixel_cnn'):
    pixel_cnn   = tf.make_template('pixel_cnn', model_spec)
    x_init_pcnn = tf.transpose(x_init, perm=[0,2,3,1])
    pcnn_out    = pixel_cnn(x_init_pcnn, h_init, vae_out=x_init_pcnn, 
                    init=True, dropout_p=args.dropout_p, **model_opt)
    pcnn_params = [v for v in tf.trainable_variables() if 'pixel_cnn' in v.name] 
    ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
    maintain_averages_op = tf.group(ema.apply(pcnn_params))


global_step = tf.get_variable("global_step", [], tf.int32, 
                    initializer=tf.zeros_initializer(), trainable=False)
recons_pcnn_train, recons_pcnn_test   = [], []
updates = []


for i in range(args.num_gpus):
    with tf.device('/gpu:%d' % i):
        
        ''' Second Pass : PixelCNN '''
        pcnn_input = tf.transpose(xs[i], perm=[0,2,3,1])
        pcnn_out_train = pixel_cnn(pcnn_input, hs[i], vae_out=vs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
        pcnn_out_test  = pixel_cnn(pcnn_input, hs[i], vae_out=vs[i], ema=None, dropout_p=0., **model_opt)
        target_pcnn = tf.transpose(xs[i], perm=[0,2,3,1])

        recons_pcnn_train  += [nn.discretized_mix_logistic_loss(target_pcnn, pcnn_out_train)]
        recons_pcnn_test   += [nn.discretized_mix_logistic_loss(target_pcnn, pcnn_out_test)]

        updates_pcnn += [tf.train.AdamOptimizer(tf_lr).minimize(recons_pcnn_train[-1], global_step=global_step, 
            var_list=pcnn_params)]
        

# calculate likelihood
denominator = (np.log(2.) * num_pixels * args.batch_size * args.num_gpus) 
bpd_pcnn_train = tf.add_n(recons_pcnn_train) / denominator
bpd_pcnn_test = tf.add_n(recons_pcnn_test)  / denominator

if args.mode == "train":
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.name_scope('pixel_cnn'):

            # training op
            pcnn_op = tf.group(*updates_pcnn) 
    
    train_summaries = [
    tf.summary.scalar('train/recon_pcnn', tf.add_n(recons_pcnn_train) / denominator),
    ]

    test_summaries = [
    tf.summary.scalar('test/recon_pcnn', tf.add_n(recons_pcnn_test) / denominator),
    ]

    merged_train, merged_test = [tf.summary.merge(x) for x in [train_summaries, test_summaries]]


var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pixel_cnn')
opt_list = [x for x in var_list if 'adam' in x.name.lower()]
var_list = [x for x in var_list if 'adam' not in x.name.lower()]
saver_pcnn = tf.train.Saver(var_list=var_list, max_to_keep=None)
saver_opt  = tf.train.Saver(var_list=opt_list, max_to_keep=None)

new_x_gen = []
for i in range(args.num_gpus):
    with tf.device('/gpu:%d' % i):
        pcnn_input = tf.transpose(xs[i], perm=[0,2,3,1])
        gen_par = pixel_cnn(pcnn_input, h_sample[i], ema=None, dropout_p=0, **model_opt)
        new_x_gen.append(nn.sample_from_discretized_mix_logistic(
        gen_par, args.nr_logistic_mix))

def sample_from_model(sess):
    x_gen = [np.zeros((args.batch_size,) + obs_shape[::-1], dtype=np.float32)
             for i in range(args.num_gpus)]
    for yi in range(obs_shape[-1]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(
                new_x_gen, {xs[i]: x_gen[i].transpose(0,3,1,2) for i in range(args.num_gpus)})
            for i in range(args.num_gpus):
                x_gen[i][:, yi, xi, :] = new_x_gen_np[i][:, yi, xi, :]
    return np.concatenate(x_gen, axis=0)

total_size = 0
for v in tf.trainable_variables():
    total_size += np.prod([int(s) for s in v.get_shape()])
print("Num trainable variables: %d" % total_size)

local_step = 0
begin = time.time()

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    print('setting up tensorboard')
    train_writer = tf.summary.FileWriter(args.save_dir, sess.graph)
    test_writer  = tf.summary.FileWriter(args.save_dir, sess.graph)
    lr = args.learning_rate
    for epoch in range(10000):
        if epoch == 0 :
            print('initalizing weights')
            data_init = train_data.next(args.init_batch_size).transpose(0,3,1,2)
            feed_dict = {x_init:data_init}
            sess.run(tf.global_variables_initializer(), feed_dict)
            
            if args.load_params : 
                print('trying to load from checkpoint')
                saver_pcnn.restore(sess, args.save_dir + '/params_pcnn.ckpt')
                saver_opt.restore(sess, args.save_dir + '/params_pcnn_opt.ckpt')
                print('restored!')               
            
            print('training!')

        ''' training set '''
        train_err = []
        for d in list(train_data): 
            lr *= args.lr_decay
            
            x_data = d.transpose(0, 3, 1, 2) 
            x_data = np.cast[np.float32]((x_data-127.5) / 127.5)
            x_data = np.split(x_data, args.num_gpus)
            
            feed_dict = {xs[i]: x_data[i] for i in range(args.num_gpus)}
            feed_dict.update({tf_lr : lr})
            fetches = [bpd_pcnn_train, global_step, pcnn_op]
            
            should_compute_summary = ((local_step  + 1) % args.write_every == 0)
            if should_compute_summary : 
                fetches += [merged_train]
            
            fetched = sess.run(fetches, feed_dict)
            train_err.append(fetched[0])
            
            if  local_step < 10 or should_compute_summary:
                print("Iteration %d, time = %.2fs, train bpd pcnn %.4f" % (
                      fetched[1], time.time() - begin, fetched[0]))
                print('kl : %s' % fetched[2])
                if should_compute_summary : train_writer.add_summary(fetched[-1], fetched[1])
                begin = time.time()
            
            local_step += 1
        print('training error at epoch %s : %s' % (epoch, np.array(train_err).mean()))
        
        
        ''' test set '''
        test_err = []
        for d in list(test_data): 
            x_data = d.transpose(0, 3, 1, 2) # feed_dict = make_feed_dict(d)
            x_data = np.cast[np.float32]((x_data-127.5) / 127.5)
            x_data = np.split(x_data, args.num_gpus)
            feed_dict = {xs[i]: x_data[i] for i in range(args.num_gpus)}
            fetches = [bpd_pcnn_test, global_step]
            should_compute_summary = ((local_step + 1) % args.write_every == 0)

            if should_compute_summary : 
                fetches += [merged_test]

            fetched = sess.run(fetches, feed_dict)
            test_err.append(fetched[0])

            if  local_step < 10 or should_compute_summary:
                print("Iteration %d, time = %.2fs, test bpd pcnn %.4f" % (
                      fetched[1], time.time() - begin, fetched[0]))
                if should_compute_summary : test_writer.add_summary(fetched[-1], fetched[1])
                begin = time.time()
            
            local_step += 1
        print('test error at epoch %s : %s' % (epoch, np.array(test_err).mean()))
        
        if (epoch + 1) % args.save_interval == 0 : 
            print('saving parameters')
            saver_pcnn.save(sess, args.save_dir + '/params_pcnn.ckpt')
            saver_opt.save(sess, args.save_dir + '/params_pcnn_opt.ckpt')
            print('parameters saved')
        
        if (epoch + 1) % 8 == 0 : 
            print('making sample')
            sample_x = sample_from_model(sess)
            img_tile = plotting.img_tile(sample_x[:int(np.floor(np.sqrt(
                args.batch_size * args.num_gpus))**2)], aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=args.dataset + ' samples')
            plotting.plt.savefig(os.path.join(
                args.save_dir, '%s_sample%d.png' % (args.dataset, epoch)))
            plotting.plt.close('all')
            print('sample saved as %s_sample%d.png' % (args.dataset, epoch)) 

