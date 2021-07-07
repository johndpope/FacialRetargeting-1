#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 16:52:01 2018
"""
import os
from src.model import moji, triplemoji
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', action='store', dest="gpu", default='3',
                    help="gpu id")
parser.add_argument('-e', action='store', dest="epoch", default=2,
                    help="training epoch")
parser.add_argument('-lr', action='store', dest="l_rate", default=0.0001,
                    help="learning rate")
parser.add_argument('-m', action='store', dest="mode", default= '2moji',
                    help="training mode")
parser.add_argument('-l', action='store_true', dest="load",
                    help="load pretrained model")
parser.add_argument('-t', action='store_false', dest="train",
                    help="use -t to switch to testing step")
parser.add_argument('-s', action='store', dest="suffix",default='',
                    help="suffix of filename")
parser.add_argument('-p', action='store', dest="test_people_id",default=141,
                    help="id of test people")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
epoch = int(args.epoch)
l_rate = float(args.l_rate)
load = args.load
train = args.train
mode = args.mode
suffix = args.suffix
people_id = int(args.test_people_id)
# filename prefix
prefix = mode

#if not train:
if True:
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config) 
    KTF.set_session(sess)
    
if people_id>140:
    filename = 'test'
else:
    filename = 'train'
    
print('Config: Load pretrained<-{}, suffix<-{}, epoch<-{}, lr<-{}'.format(load, suffix, epoch, l_rate))


if mode == 'triplemoji':
    # Hint: u should always consider where is the singular point for the whole objective function!
    # dim of input vector
    input_dim = 11510*9
    # dim of output vector
    output_dim = 1895*9
    # dim of per feature
    feature_dim = 9
    prefix = 'mery'
    if suffix == '':
        suffix = 'triple'
    net = triplemoji(input_dim, output_dim, prefix, suffix, l_rate, load, feature_dim = feature_dim, batch_size=1, MAX_DEGREE=2)
    if train:
        net.train_easytriplet(epoch)
    else:
        #net.make_sure()
        net.test()
if not train:
    import shutil, os
else:
    import matplotlib.pyplot as plt
    import numpy as np
    log = np.load('log.npy')
    test_log = np.load('testlog.npy')

    plt.switch_backend('agg')
    plt.plot(log, 'r-')
    plt.plot(test_log, 'g-')
    plt.savefig(suffix)