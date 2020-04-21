#!/usr/bin/env python
# coding: utf-8

# Code for the paper Tik-Tok: The Utility of Packet Timing in Website Fingerprinting Attacks accepted in PETS 2020.
# Mohammad Saidur Rahman - saidur.rahman@mail.rit.edu
# Global Cybersecurity Institute, Rochester Institute of Technology



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from final_features_process import *
from DF_Model import *
import pickle
import os
import numpy as np
import time
from sklearn.utils import shuffle
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adamax
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from os import path
import argparse
import random

random.seed(583004949)


# In[ ]:


datasets = ['Undefended', 'WTF-PAD', 'W-T-Simulated', 'W-T-Real', 'Onion-Sites']

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='Undefended, WTF-PAD, W-T-Simulated, W-T-Real, Onion-Sites')

args = parser.parse_args()

dataset = args.dataset


# In[ ]:


# Provide Data Path in place of data_root
data_root = 'data_root/' # 
save_path = os.getcwd() + '/' + 'save_data/' + str(dataset) + '/'

try:
    os.stat(save_path)
except:
    os.makedirs(save_path)


# In[ ]:


# Check whether the files for training the model already exists
# If not, it will process the raw data and create the files.
model_files = ['X_tr', 'Y_tr', 'X_vl', 'Y_vl', 'X_te', 'Y_te']
count_m_files = 0
for f in model_files:
    f = save_path + f + '.pkl'
    if path.exists(f):
        count_m_files +=1
if count_m_files == 6:
    X_train, y_train, X_valid, y_valid, X_test, y_test = final_data_load(save_path)
else:
    # Option 1: Processing raw data . Takes a long time. Uncomment if you choose Option 1.
    # Download the raw data from the google drive and put the data into data_root.
    X_train, y_train, X_valid, y_valid, X_test, y_test = final_process(dataset, data_root, save_path)
    
    # Option 2: Download the processed data from google drive and put those into the save_path
    #X_train, y_train, X_valid, y_valid, X_test, y_test = final_data_load(save_path)


# In[ ]:


#K.set_image_dim_ordering("tf")
K.image_data_format() # In new version of Keras, the method has be renamed to image_data_format

if dataset == datasets[0]:
    num_classes = 95
elif dataset == datasets[1]:
    num_classes = 95
elif dataset ==  datasets[2]:
    num_classes = 100
elif dataset == datasets[3]:
    num_classes = 100
else:
    num_classes = 538
    
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


run_trial = 1 # change run_trial > 1 to get a standard deviation of the accuracy.
seq_length = 160 # 8 timing features x 20 bins = 160 features values.
num_epochs = 100 # 100 epochs for experiments with timing_features and onion_sites
VERBOSE = 2
df_res = [None] * run_trial
for j in range(run_trial):
    df_res[j] = df_accuracy(num_classes, num_epochs, seq_length,VERBOSE, X_train, y_train, X_valid, y_valid, X_test, y_test)

if run_trial !=1:
    print('Mean Acc: ',np.mean(df_res))
    print('STD of Mean: ',np.std(df_res))

