from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, EarlyStopping
import random
from keras.utils import np_utils
from keras.optimizers import Adamax
import numpy as np
import sys
import os
from timeit import default_timer as timer
from pprint import pprint
import argparse
import json
from data_utils import *


# define the ConvNet
class ConvNet:
    @staticmethod
    def build(classes,
              input_shape,
              activation_function=("relu", "relu", "relu", "relu", "relu", "relu"),
              dropout=(0.1, 0.1, 0.1, 0.1, 0.7, 0.7),
              filter_num=(32, 64, 128, 256),
              kernel_size=8,
              conv_stride_size=1,
              pool_stride_size=4,
              pool_size=8,
              fc_layer_size=(512, 512)):

        # confirm that parameter vectors are acceptable lengths
        assert len(filter_num) + len(fc_layer_size) <= len(activation_function)
        assert len(filter_num) + len(fc_layer_size) <= len(dropout)

        # Sequential Keras model template
        model = Sequential()

        # add convolutional layer blocks
        for block_no in range(0, len(filter_num)):
            if block_no == 0:
                model.add(Conv1D(filters=filter_num[block_no],
                                 kernel_size=kernel_size,
                                 input_shape=input_shape,
                                 strides=conv_stride_size,
                                 padding='same',
                                 name='block{}_conv1'.format(block_no)))
            else:
                model.add(Conv1D(filters=filter_num[block_no],
                                 kernel_size=kernel_size,
                                 strides=conv_stride_size,
                                 padding='same',
                                 name='block{}_conv1'.format(block_no)))

            model.add(BatchNormalization())

            model.add(Activation(activation_function[block_no], name='block{}_act1'.format(block_no)))

            model.add(Conv1D(filters=filter_num[block_no],
                             kernel_size=kernel_size,
                             strides=conv_stride_size,
                             padding='same',
                             name='block{}_conv2'.format(block_no)))

            model.add(BatchNormalization())

            model.add(Activation(activation_function[block_no], name='block{}_act2'.format(block_no)))

            model.add(MaxPooling1D(pool_size=pool_size,
                                   strides=pool_stride_size,
                                   padding='same',
                                   name='block{}_pool'.format(block_no)))

            model.add(Dropout(dropout[block_no], name='block{}_dropout'.format(block_no)))

        # flatten output before fc layers
        model.add(Flatten(name='flatten'))

        # add fully-connected layers
        for layer_no in range(0, len(fc_layer_size)):
            model.add(Dense(fc_layer_size[layer_no],
                            kernel_initializer=glorot_uniform(seed=0),
                            name='fc{}'.format(layer_no)))

            model.add(BatchNormalization())
            model.add(Activation(activation_function[len(filter_num)+layer_no],
                                 name='fc{}_act'.format(layer_no)))

            model.add(Dropout(dropout[len(filter_num)+layer_no],
                              name='fc{}_drop'.format(layer_no)))

        # add final classification layer
        model.add(Dense(classes, kernel_initializer=glorot_uniform(seed=0), name='fc_final'))
        model.add(Activation('softmax', name="softmax"))

        # compile model with Adamax optimizer
        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        return model


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train and test the DF model against the W-T prototype dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--traces',
                        type=str,
                        required=True,
                        metavar='<root_directory>',
                        help='Path to the directory where the data is stored.')
    parser.add_argument('-a', '--attack',
                        type=int,
                        default=0,
                        metavar='<attack_type>',
                        help='Type of attack: (0) direction (1) tik-tok (2) timing')
    parser.add_argument('-o', '--output',
                        type=str,
                        default='trained_model.h5',
                        metavar='<model_output>',
                        help='Location to store the file.')
    return parser.parse_args()


def attack(X_train, y_train, X_test, y_test, unmon_label, args, VERBOSE=1):
    """
    Perform WF training and testing
    """
    classes = len(set(list(y_train)))
    print(classes)

    # shuffle and split for val
    s = np.arange(X_test.shape[0])
    np.random.shuffle(s)
    sp = X_test.shape[0]//2
    X_va = X_test[s][sp:]
    y_va = y_test[s][sp:]
    X_test = X_test[s][:sp]
    y_test = y_test[s][:sp]

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, classes)
    y_va = np_utils.to_categorical(y_va, classes)

    # Build and compile model
    print("Compiling model...")
    model = ConvNet.build(classes=classes, input_shape=(5000, 1))

    # Train the model
    filepath = args.output
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='auto', restore_best_weights=True)
    callbacks_list = [checkpoint, early_stopping]

    history = model.fit(X_train, y_train,
                        epochs=30,
                        verbose=VERBOSE,
                        validation_data=(X_va, y_va),
                        callbacks=callbacks_list)

    # Save & reload model
    model.save(filepath)
    del model
    model = load_model(filepath)

    X_test_mon = X_test[y_test != unmon_label]
    y_test_mon = y_test[y_test != unmon_label]
    X_test_unmon = X_test[y_test == unmon_label]
    y_test_unmon = y_test[y_test == unmon_label]

    # Test the model
    y_test = np_utils.to_categorical(y_test, classes)
    score = model.evaluate(X_test, y_test,
                           verbose=VERBOSE)
    score_train = model.evaluate(X_train, y_train,
                                 verbose=VERBOSE)


    y_test_mon = np_utils.to_categorical(y_test_mon, classes)
    y_test_unmon = np_utils.to_categorical(y_test_unmon, classes)

    all_acc = score[1]
    print("\n=> Train score:", score_train[0])
    print("=> Train accuracy:", score_train[1])
    print("\n=> Test score:", score[0])
    print("=> Test accuracy:", score[1])
    score = model.evaluate(X_test_mon, y_test_mon,
                           verbose=VERBOSE)
    print("\n=> Test_mon score:", score[0])
    print("=> Test_mon accuracy:", score[1])
    mon_acc = score[1]
    score = model.evaluate(X_test_unmon, y_test_unmon,
                           verbose=VERBOSE)
    print("\n=> Test_unmon score:", score[0])
    print("=> Test_unmon accuracy:", score[1])
    umon_acc = score[1]
    return all_acc, mon_acc, umon_acc


def main():
    """
    """
    att_types = ['direction','tiktok','timing']
    mon_dirs = ['1-normal', '2-normal', '3-normal', '4-normal', '5-normal']
    unmon_dirs = ['1-inverse', '2-inverse', '3-inverse', '4-inverse', '5-inverse']
    dirs = list(zip(mon_dirs, unmon_dirs))

    # Parse arguments
    args = parse_arguments()

    # Load the dataset
    print("Loading dataset as type {}...".format(att_types[args.attack]))

    # load dataset folds
    Xs, ys, class_map = [], [], dict()
    for mon_dir, unmon_dir in dirs:
        Xm, ym, class_map = load_data(os.path.join(args.traces, mon_dir), typ=att_types[args.attack], unmon=False, class_map=class_map)
        Xu, _, _ = load_data(os.path.join(args.traces, unmon_dir), typ=att_types[args.attack], unmon=True)
        unmon_label = (np.amax(ym) + 1)
        yu = np.ones((Xu.shape[0])) * unmon_label

        X = np.concatenate((Xm, Xu))
        y = np.concatenate((ym, yu))
        Xs.append(X)
        ys.append(y)

    res = []
    for i in range(len(Xs)):
        print("======================")
        print("-- Fold {}".format(i))
        print("======================")

        # build train data folds
        X_tr, y_tr = [], []
        for j in range(len(Xs)):
            if j != i:
                X_tr.append(Xs[j])
                y_tr.append(ys[j])
        X_tr = np.concatenate(X_tr)
        y_tr = np.concatenate(y_tr)
        # build test data folds
        X_te, y_te = Xs[i], ys[i]

        # perform attack
        acc = attack(X_tr, y_tr, X_te, y_te, unmon_label, args)
        print(acc)
        res.append([acc])

    print("======================")
    print("-- Summary")
    print("======================")
    print(res)


if __name__ == "__main__":
    # execute only if run as a script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
