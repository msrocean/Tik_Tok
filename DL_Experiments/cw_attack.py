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

random.seed(0)


# define the ConvNet
class ConvNet:
    @staticmethod
    def build(classes,
              input_shape,
              activation_function=("elu", "relu", "relu", "relu", "relu", "relu"),
              dropout=(0.1, 0.1, 0.1, 0.1, 0.5, 0.7),
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
    parser = argparse.ArgumentParser(description='Train and test the DeepFingerprinting model in the Closed-World setting.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--traces',
                        type=str,
                        required=True,
                        metavar='<path/to/traffic>',
                        help='Path to the directory where the traffic data is stored.')
    parser.add_argument('-a', '--attack',
                        type=int,
                        default=0,
                        metavar='<attack_type>',
                        help='Type of attack: (0) direction (1) tik-tok (2) timing')
    parser.add_argument('-o', '--output',
                        type=str,
                        default='trained_model_cw.h5',
                        metavar='<output>',
                        help='Location to store the file.')
    parser.add_argument('-f', '--folds',
                        type=int,
                        default=5,
                        metavar='<num_folds>',
                        help='Number of folds to use for cross-validation.')
    return parser.parse_args()


def attack(X_train, y_train, X_valid, y_valid, X_test, y_test, args, VERBOSE=1):
    """
    """
    # convert class vectors to binary class matrices
    classes = len(set(list(y_train)))
    y_train = np_utils.to_categorical(y_train, classes)
    y_valid = np_utils.to_categorical(y_valid, classes)
    y_test = np_utils.to_categorical(y_test, classes)

    # # # # # # # # 
    # Build and compile model
    # # # # # # # # 
    print("Compiling model...")
    model = ConvNet.build(classes=classes, input_shape=(5000, 1))

    # # # # # # # # 
    # Train the model
    # # # # # # # # 
    filepath = args.output
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='auto', restore_best_weights=True)
    callbacks_list = [checkpoint, early_stopping]

    history = model.fit(X_train, y_train,
                        epochs=40,
                        verbose=VERBOSE,
                        validation_data=(X_valid, y_valid),
                        callbacks=callbacks_list)

    # Save & reload model
    model.save(filepath)
    del model
    model = load_model(filepath)

    # # # # # # # # 
    # Test the model
    # # # # # # # # 
    score = model.evaluate(X_test, y_test,
                           verbose=VERBOSE)
    score_train = model.evaluate(X_train, y_train,
                                 verbose=VERBOSE)

    # # # # # # # # 
    # Print results
    # # # # # # # # 
    print("\n=> Train score:", score_train[0])
    print("=> Train accuracy:", score_train[1])

    print("\n=> Test score:", score[0])
    print("=> Test accuracy:", score[1])

    return score[1]


def main():
    """
    """

    # # # # # # # # 
    # Parse arguments
    # # # # # # # # 
    args = parse_arguments()

    # # # # # # # # 
    # Load the dataset
    # # # # # # # # 
    print("Loading dataset as type {}...".format(args.attack))

    X, y = load_data(args.traces, typ=args.attack)

    res = []

    folds = args.folds
    count = len(list(y))
    for i in range(folds):
        print("======================")
        print("-- Fold {}".format(i))
        print("======================")
        chunk_start = i*(count//folds)
        chunk_end = (i+1)*(count//folds)
        X_te = X[chunk_start:chunk_end]
        y_te = y[chunk_start:chunk_end]
        if i == 0:
            X_tr = X[chunk_end:, :, :]
            y_tr = y[chunk_end:]
        elif i == folds:
            X_tr = X[:chunk_start, :, :]
            y_tr = y[:chunk_start]
        else:
            X_tr = np.concatenate((X[chunk_end:, :, :], X[:chunk_start, :, :]))
            y_tr = np.concatenate((y[chunk_end:], y[:chunk_start]))
        acc = attack(X_tr, y_tr, X_te[:(count//folds)//2], y_te[:(count//folds)//2], X_te[(count//folds)//2:], y_te[(count//folds)//2:], args, VERBOSE=1)
        res.append(acc)

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
