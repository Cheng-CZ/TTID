import os

import numpy as np
import tensorflow as tf
from tqdm import trange

import scipy
from model import Model
from utils import *
from six.moves import reduce, xrange


"""
Trainer: 

1. Initializes model
2. Train
3. Test
"""

class Trainer(object):
    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.task = config.task
        self.model_dir = config.model_dir
        self.gpu_memory_fraction = config.gpu_memory_fraction
        self.checkpoint_secs = config.checkpoint_secs
        self.log_step = config.log_step
        self.num_epoch = config.num_epochs
        
        ## import data Loader ##
        data_dir = config.data_dir
        dataset_name = config.task
        batch_size = config.batch_size
        num_time_steps = config.num_time_steps
        feat_in = config.feat_in
        num_node = config.num_node

        self.adj_data = sample_data_loader(data_dir, dataset_name, num_time_steps)
        # adj_data dim: 10 * 5 * 1031 * 1031

        self.att_data = attr_data_loader(data_dir, dataset_name, num_time_steps)
        self.att_data = np.reshape(self.att_data, (num_time_steps, num_node, feat_in))
        # att_data dim: 10 * 1031 * 5 

        
        
        ## define model ##
        self.model = Model(config)
        
        ## model saver / summary writer ##
        self.saver = tf.train.Saver()
        # self.model_saver = tf.train.Saver(self.model.model_vars)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        
        sv = tf.train.Supervisor(logdir=self.model_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_summaries_secs=300,
                                 save_model_secs=self.checkpoint_secs,
                                 global_step=self.model.model_step)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.gpu_memory_fraction,
            allow_growth=True)  # seems to be not working
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        
    def train(self):
        
        print("[*] Checking if previous run exists in {}"
              "".format(self.model_dir))
        latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
        if tf.train.latest_checkpoint(self.model_dir) is not None:
            print("[*] Saved result exists! loading...")
            self.saver.restore(
                self.sess,
                latest_checkpoint
            )
            print("[*] Loaded previously trained weights")
            self.b_pretrain_loaded = True
        else:
            print("[*] No previous result")
            self.b_pretrain_loaded = False
            
        print("[*] Training starts...")
        self.model_summary_writer = None
        
        ##Training
        for n_epoch in trange(self.num_epoch, desc="Training[epoch]"):
            batch_x = self.att_data
            batch_adj = self.adj_data 
            batch_x = batch_x.reshape([self.config.num_time_steps,
                                       self.config.num_node,
                                       self.config.feat_in])
            

            # batch_x: 10 * 1031 * 5
            # batch_adj: 10 * 5 * 1031 * 1031
            feed_dict = {
                self.model.rnn_input: batch_x,
                self.model.rnn_adj: batch_adj
            } 

            res = self.model.train(self.sess, feed_dict, self.model_summary_writer,
                                   with_output=True)
            
            self.model_summary_writer = self._get_summary_writer(res)

            # print res['loss']
            if n_epoch % 100 == 0:
                self.saver.save(self.sess, save_path=self.model_dir+'/')

                
        #Testing
        feed_dict = {self.model.rnn_input: batch_x,
                     self.model.rnn_adj: batch_adj}

        res = self.model.test(self.sess, feed_dict, self.model_summary_writer,
                                  with_output=True)
        # print 'adj para', res['adj']
        
        print 'Test output tensor shape is ', res['output'].shape

        with file(self.model_dir+'/test_output.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(res['output'].shape))
            for data_slice in res['output']:
                np.savetxt(outfile, data_slice, fmt='%-7.5f')
        outfile.close()

            
                
    def _get_summary_writer(self, result):
        if result['step'] % self.log_step == 0:
            return self.summary_writer
        else:
            return None
        
        
