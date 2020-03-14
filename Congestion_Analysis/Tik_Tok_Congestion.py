#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import argparse
from os import path
from data_load import *
from DF_Model import *
from sklearn.utils import shuffle
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adamax
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import random
random.seed(583004949)


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('--congestion', type=str, required= True, choices = ['slow', 'fast'],\
                    help='i) Slow cirtuits as test set, or ii) fast circuits as test set')
parser.add_argument('--dataset', type=str, choices=['Undefended', 'WTF-PAD', 'Onion-Sites'],\
                    required=True,help='choose any of the dataset category: Undefended, WTF-PAD, Onion-Sites')
parser.add_argument('--data_rep', type=str, required= True,choices = ['D', 'RT', 'DT'],\
                    help='type of data representation to be used. D: direction, RT: Raw Timing, and DT: Directional Timing')

args = parser.parse_args()

congestion_type = args.congestion
dataset = args.dataset

if args.data_rep == 'D':
    data_rep = 'direction'
elif args.data_rep == 'RT':
    data_rep = 'raw_timing'
else:
    data_rep = 'directional_timing'

data_dir = os.getcwd() + '/' + 'datasets/' + str(congestion_type) + '/' + str(dataset) + '/' + str(data_rep) + '/'

try:
    data_dir = data_dir
except:
    os.makedirs(data_dir)

# In[ ]:


model_files = ['X_tr', 'Y_tr', 'X_vl', 'Y_vl', 'X_te', 'Y_te']
count_m_files = 0
for f in model_files:
    f = data_dir + f + '.pkl'
    if path.exists(f):
        count_m_files +=1
if count_m_files == 6:
    X_train, y_train, X_valid, y_valid, X_test, y_test = data_load(data_dir)
else:
    print('Not enough files ...')


# In[ ]:

#X_train, y_train, X_valid, y_valid, X_test, y_test = data_load(data_dir)
K.set_image_dim_ordering("tf")

if dataset == 'Undefended':
    num_classes = 95
    num_epochs = 30
elif dataset == 'WTF-PAD':
    num_classes = 95
    num_epochs = 30
else:
    num_classes = 538
    num_epochs = 150
    
# Convert data as float32 type
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('float32')

# we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]
X_test = X_test[:, :,np.newaxis]

print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to categorical classes matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_valid = np_utils.to_categorical(y_valid, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

exp_type = args.data_rep
run_trial = 1 # change run_trial > 1 to get a standard deviation of the accuracy.
seq_length = 5000 # 8 timing features x 20 bins = 160 features values.


VERBOSE = 2
df_res = [None] * run_trial
for j in range(run_trial):
    df_res[j] = df_accuracy(exp_type, num_classes, num_epochs, seq_length,VERBOSE, X_train, y_train, X_valid, y_valid, X_test, y_test)

if run_trial !=1:
    print('Mean Acc: ',np.mean(df_res))
    print('STD of Mean: ',np.std(df_res))


# In[ ]:




