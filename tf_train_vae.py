import json
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
from debug_utils import * 

import pixel_cnn_pp.nn as nn
import pixel_cnn_pp.plotting as plotting
from pixel_cnn_pp.model import model_spec
from iaf_vae import * 
import data.cifar10_data as cifar10_data
import data.imagenet_data as imagenet_data
import argparse


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
parser.add_argument('-q', '--nr_resnet', type=int, default=2, #5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, #60,
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

def get_default_hparams():
    return HParams(
        batch_size=16,        # Batch size on one GPU.
        eval_batch_size=100,  # Batch size for evaluation.
        num_gpus=2,           # Number of GPUs (effectively increases batch size).
        learning_rate=0.001,  # Learning rate.
        z_size=32,            # Size of z variables.
        h_size=160,           # Size of resnet block.
        nr_filters=160,
        kl_min=0.25,          # Number of "free bits/nats".
        depth=2,              # Number of downsampling blocks.
        num_blocks=2,         # Number of resnet blocks for each downsampling layer.
        k=1,                  # Number of samples for IS objective.
        dataset="cifar10",    # Dataset name.
        image_size=32,        # Image size.
    )


### DATA STREAMS ###
hps = get_default_hparams()  
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

# for regular training
with tf.name_scope('vae_iaf'):
    vae_iaf  = tf.make_template('vae_iaf', model_spec_iaf)
    # init_vae = vae_iaf(x_init, args, 'init', -1) 


recons_vae_train, recons_vae_test     = [], []
kls_cost_vae_train, kls_cost_vae_test = [], []
kls_obj_vae_train, kls_obj_vae_test   = [], []
elbos_vae_train, elbos_vae_test       = [], []
losses_vae_train, losses_vae_test     = [], []

vae_outs_train,    pcnns_out_train    = [], []
vae_outs_test,    pcnns_out_test      = [], []
updates_vae = []

dec_log_stdv = tf.get_variable("dec_log_stdv", initializer=tf.constant(0.0))
global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer(),
                                    trainable=False)

for i in range(args.num_gpus):
    with tf.device('/gpu:%d' % i):
        # rescale input to be in [-.5, .5]
        xs_scaled = xs[i]  / 2.

        ''' First pass : VAE '''
        vae_out_train, recon, kl_cost, kl_obj_train = vae_iaf(xs_scaled, args, 'train', i)
        vae_outs_train     += [vae_out_train]
        recons_vae_train   += [tf.reduce_sum(recon)]
        kls_cost_vae_train += [tf.reduce_sum(kl_cost)]
        kls_obj_vae_train  += [tf.reduce_sum(kl_obj_train)]
        losses_vae_train   += [tf.reduce_sum(kl_cost - recon)]
        elbos_vae_train    += [tf.reduce_sum(kl_obj_train -  recon)]

        vae_out_test, recon, kl_cost, kl_obj_test  = vae_iaf(xs_scaled, args, 'eval', i)
        vae_outs_test      += [vae_out_test]
        recons_vae_test    += [tf.reduce_sum(recon)]
        kls_cost_vae_test  += [tf.reduce_sum(kl_cost)]
        kls_obj_vae_test   += [tf.reduce_sum(kl_obj_test)]
        losses_vae_test    += [tf.reduce_sum(kl_cost - recon)]
        elbos_vae_test     += [tf.reduce_sum(kl_obj_test - recon)]
        
        
        updates_vae += [tf.train.AdamOptimizer(tf_lr).minimize(losses_vae_train[-1], 
                global_step=global_step)]
        

# calculate likelihood
denominator = (np.log(2.) * num_pixels * args.batch_size * args.num_gpus) 

bpd_vae_train = tf.add_n(elbos_vae_train)    / denominator 
bpd_vae_test = tf.add_n(elbos_vae_test)      / denominator
 
if args.mode == "train":
    with tf.name_scope('vae_iaf'):
        vae_op = tf.group(*updates_vae) 
    # with tf.name_scope('pixel_cnn'):
    #     pcnn_op = tf.group(*[tf.train.AdamOptimizer(tf_lr).minimize(obj, global_step=global_step) for obj in recons_pcnn_train]) 
    
    # train_op = tf.group(*updates_all)

    train_summaries = [
    tf.summary.scalar('train/recon_vae', tf.add_n(recons_vae_train) / denominator),
    tf.summary.scalar('train/kl_obj_vae', tf.add_n(kls_obj_vae_train) / denominator),
    tf.summary.scalar('train/elbo_vae', tf.add_n(elbos_vae_train) / denominator),
    tf.summary.scalar('train/obj_vae', tf.add_n(losses_vae_train) / denominator),
    tf.summary.scalar('train/bpd_vae', bpd_vae_train),
    tf.summary.image( 'train/image_vae', tf.transpose(vae_outs_train[-1], perm=[0,2,3,1]))
    ]

    test_summaries = [
    tf.summary.scalar('test/recon_vae', tf.add_n(recons_vae_test) / denominator),
    tf.summary.scalar('test/kl_obj_vae', tf.add_n(kls_obj_vae_test) / denominator),
    tf.summary.scalar('test/elbo_vae', tf.add_n(elbos_vae_test) / denominator),
    tf.summary.scalar('test/obj_vae', tf.add_n(losses_vae_test) / denominator),
    tf.summary.scalar('test/bpd_vae', bpd_vae_test),
    tf.summary.image( 'test/image_vae', tf.transpose(vae_outs_test[-1], perm=[0,2,3,1]))
    ]

    merged_train, merged_test = [tf.summary.merge(x) for x in [train_summaries, test_summaries]]

vae_var_list       = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae_iaf')
vae_opt_list       = [x for x in vae_var_list if 'adam' in x.name.lower()]
vae_var_list       = [x for x in vae_var_list if 'adam' not in x.name.lower()]
saver_vae          = tf.train.Saver(var_list=vae_var_list, max_to_keep=None)
saver_vae_opt      = tf.train.Saver(var_list=vae_opt_list, max_to_keep=None)

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
            # sess.run(init_vae.initializer, {x_init:data_init})
            sess.run(tf.global_variables_initializer()) # , feed_dict)

        if args.load_params : 
            print('trying to load from checkpoint')
            saver_vae.restore(sess, args.save_dir + '/params_vae.ckpt')
            saver_vae_opt.restore(sess, args.save_dir + '/params_vae_opt.ckpt')
            print('restored!')               
        
        print('training!')
        
        ''' training set '''
        train_err =[]
        for d in list(train_data): 
            lr *= args.lr_decay
            
            x_data = d.transpose(0, 3, 1, 2) 
            x_data = np.cast[np.float32]((x_data-127.5) / 127.5) 
            x_data = np.split(x_data, args.num_gpus)
            feed_dict = {xs[i]: x_data[i] for i in range(args.num_gpus)}
            feed_dict.update({tf_lr : lr})
            
            op = vae_op 
            fetches = [bpd_vae_train,           # 0
                       kls_obj_vae_train[-1],   # 1
                       kls_cost_vae_train[-1],  # 2
                       global_step,             # 3
                       dec_log_stdv,            # 4
                       vae_op,                  # 5
                       elbos_vae_train[-1],     # 6
                       vae_out_train]           # 7
            
            should_compute_summary = ((local_step  + 1) % 25 == 0)
            if should_compute_summary : 
                fetches += [merged_train]
            
            fetched = sess.run(fetches, feed_dict)
            # import pdb; pdb.set_trace()
            train_err.append(fetched[0])
            
            if  local_step < 10 or should_compute_summary:
                print("Iteration %d, time = %.2fs, train bpd vae %.4f, dec_log_stdv = %.4f" % (
                      fetched[3], time.time() - begin, fetched[0], fetched[4]))
                print('kl : %s' % fetched[2])
                if should_compute_summary : train_writer.add_summary(fetched[-1], fetched[3])
                begin = time.time()
            
            if np.isnan(fetched[2]).any():
                print("NAN detected!")
                import pdb; pdb.set_trace()
                break

            local_step += 1
        
        
        ''' test set '''
        test_err = []
        for d in list(test_data): 
            
            x_data = d.transpose(0, 3, 1, 2) 
            x_data = np.cast[np.float32]((x_data-127.5) / 127.5) 
            x_data = np.split(x_data, args.num_gpus)
            feed_dict = {xs[i]: x_data[i] for i in range(args.num_gpus)}
           
            fetches = [bpd_vae_test, kls_obj_vae_test[-1], global_step, dec_log_stdv]
            should_compute_summary = ((local_step + 1) % 25 == 0)

            if should_compute_summary : 
                fetches += [merged_test]

            fetched = sess.run(fetches, feed_dict)
            test_err.append(fetched[0])

            if  local_step < 10 or should_compute_summary:
                print("Iteration %d, time = %.2fs, test bpd vae %.4f, dec_log_stdv = %.4f" % (
                      fetched[2], time.time() - begin, fetched[0], fetched[3]))
                print('kl : %s' % fetched[1])
                if should_compute_summary : test_writer.add_summary(fetched[-1], fetched[3])
                begin = time.time()
            if np.isnan(fetched[2]).any():
                print("NAN detected!")
                import pdb; pdb.set_trace()
                break
            
            local_step += 1
       
       
        if (epoch + 1) % args.save_interval == 0 : 
            print('saving parameters')
            saver_vae.save(sess, args.save_dir + '/params_vae.ckpt')
            saver_vae_opt.save(sess, args.save_dir + '/params_vae_opt.ckpt')
            print('parameters saved')
        
        '''
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

        '''  
