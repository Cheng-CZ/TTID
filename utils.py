import tensorflow as tf
import numpy as np
import math
import pickle
import os
import json
from datetime import datetime
import tensorflow.contrib.slim as slim
from scipy.sparse import coo_matrix, load_npz

def attr_data_loader(data_dir, dataset_name, num_time_steps):
    load_file_name = os.path.join(data_dir, dataset_name, 'lda.txt')
    lda_2d = np.loadtxt(load_file_name)
    return lda_2d



def sample_data_loader(data_dir, dataset_name, num_time_steps):
    file_lists = ['edge_user_mention_user', 'edge_user_quote', 
              'edge_user_retweet', 'edge_user_hashtag']

    A = []
    for t_ in range(num_time_steps):
        file_l = []
        for file_ in file_lists:
            load_file_name = os.path.join(data_dir, dataset_name, file_+str(t_)+'.npz')
            file_l.append(load_npz(load_file_name).todense())
        A.append(file_l)
    print load_file_name, 'loaded'
    
    for t_ in range(num_time_steps):
        for file_ in ['following']:
            load_file_name = os.path.join(data_dir, dataset_name, file_ +'.npz')
            file_l = load_npz(load_file_name).todense()
            A[t_].append(file_l)
    print load_file_name, 'loaded'
    print 'adj_data shape is ', np.array(A).shape
    return np.array(A)

def save_config(model_dir, config):
    '''
    save config params in a form of param.json in model directory
    '''
    param_path = os.path.join(model_dir, "params.json")

    print("[*] PARAM path: %s" %param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def prepare_dirs(config):
    if config.load_path:
        config.model_name = "{}_{}".format(config.task, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.task, get_time())

    config.model_dir = os.path.join(config.log_dir, config.model_name)

    for path in [config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Directory '%s' created" %path)
            