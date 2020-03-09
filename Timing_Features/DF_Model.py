from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import dill
import os
import shutil
import numpy as np
import h5py
import time
import json
import threading
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Activation, ZeroPadding1D, \
    GlobalAveragePooling1D, Add, Concatenate, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import Input
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.advanced_activations import ELU
from keras.initializers import glorot_uniform
from keras.optimizers import Adamax
from sklearn.utils import shuffle

import random
random.seed(583004949)

# DF model used for non-defended dataset
class DFNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        #Block1
        filter_num = ['None',32,64,128,256]
        kernel_size = ['None',8,8,8,8]
        conv_stride_size = ['None',1,1,1,1]
        pool_stride_size = ['None',4,4,4,4]
        pool_size = ['None',8,8,8,8]

        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=input_shape,
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act1'))
        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                               padding='same', name='block1_pool'))
        model.add(Dropout(0.1, name='block1_dropout'))

        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block2_act1'))

        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block2_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                               padding='same', name='block2_pool'))
        model.add(Dropout(0.1, name='block2_dropout'))

        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                         name='block3_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block3_act1'))
        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                         name='block3_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block3_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                               padding='same', name='block3_pool'))
        model.add(Dropout(0.1, name='block3_dropout'))

        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                         name='block4_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block4_act1'))
        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                         name='block4_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block4_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                               padding='same', name='block4_pool'))
        model.add(Dropout(0.1, name='block4_dropout'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='fc1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc1_act'))

        model.add(Dropout(0.7, name='fc1_dropout'))

        model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='fc2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc2_act'))

        model.add(Dropout(0.5, name='fc2_dropout'))

        model.add(Dense(classes, kernel_initializer=glorot_uniform(seed=0), name='fc3'))
        model.add(Activation('softmax', name="softmax"))
        return model
    
    
def df_accuracy(num_classes, num_epochs, seq_length, VERBOSE, X_tr, Y_tr, X_vl, Y_vl, X_te, Y_te):
    OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer
    input_shape = (seq_length,1)
    # Building and training model

    #print ("Building and training DF model")
    model = DFNet.build(input_shape=input_shape, classes=num_classes)

    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
        metrics=["accuracy"])
    #print ("Model compiled")

    # Start training
    history = model.fit(X_tr, Y_tr, batch_size=128, epochs=num_epochs, verbose=VERBOSE, validation_data=(X_vl, Y_vl))

    # Start evaluating model with testing data
    score_test = model.evaluate(X_te, Y_te, verbose=0)
    print("Testing accuracy:", score_test[1])
    return score_test[1]