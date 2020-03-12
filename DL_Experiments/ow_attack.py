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
    parser = argparse.ArgumentParser(description='Train and test the DeepFingerprinting model in the Open-World setting.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mon',
                        type=str,
                        required=True,
                        metavar='<path/to/mon>',
                        help='Path to the directory where the unmonitored traffic is stored.')
    parser.add_argument('-u', '--unmon',
                        type=str,
                        required=True,
                        metavar='<path/to/unmon>',
                        help='Path to the directory where the unmonitored traffic is stored.')
    parser.add_argument('-a', '--attack',
                        type=int,
                        default=0,
                        metavar='<attack_type>',
                        help='Type of attack: (0) direction (1) tik-tok (2) timing')
    parser.add_argument('-w', '--world_size',
                        type=int,
                        default=16000,
                        metavar='<world_size>',
                        help='Number of sites to include in the unmonitored testing set.')
    parser.add_argument('-o', '--output',
                        type=str,
                        default='trained_model_ow.h5',
                        metavar='<output>',
                        help='Location to store the file.')
    return parser.parse_args()


def train(X_train, y_train, classes, filepath, VERBOSE=1):
    """
    """
    print("Compiling model...")
    model = ConvNet.build(classes=classes+1, input_shape=(5000, 1))

    history = model.fit(X_train, y_train,
                        epochs=30,
                        verbose=VERBOSE)

    # Save & reload model
    model.save(filepath)
    del model
    model = load_model(filepath)

    return model


def ow_evaluation(model, mon_samples, unmon_samples, unmon_label):
    """
    """
    upper_bound = 1.0
    thresholds = upper_bound - upper_bound / np.logspace(0.05, 2, num=15, endpoint=True)
    #threshold = np.linspace(0.75, 0.85, num=5)

    fmt_str = '{}:\t{}\t{}\t{}\t{}\t{}\t{}'
    print(fmt_str.format('TH  ', 'TP   ', 'TN   ', 'FP   ', 'FN   ', 'Pre. ', 'Rec. '))
    fmt_str = '{:.2f}:\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}'

    # evaluate model performance at different thresholds
    # high threshold will yield higher precision, but reduced recall
    results = []
    for TH in thresholds:
        TP, FP, TN, FN = 0, 0, 0, 0

        # Test with Monitored testing instances
        for s in range(mon_samples.shape[0]):
            test_example = mon_samples[s]
            predict_prob = model.predict(np.array([test_example]))
            best_n = np.argsort(predict_prob[0])[-1:]
            if best_n[0] != unmon_label:
                if predict_prob[0][best_n[0]] >= TH:
                    TP = TP + 1
                else:
                    FN = FN + 1
            else:
                FN = FN + 1

        # Test with Unmonitored testing instances
        for s in range(unmon_samples.shape[0]):
            test_example = unmon_samples[s]
            predict_prob = model.predict(np.array([test_example]))
            best_n = np.argsort(predict_prob[0])[-1:]
            if best_n[0] != unmon_label:
                if predict_prob[0][best_n[0]] >= TH:
                    FP = FP + 1
                else:
                    TN = TN + 1
            else:
                TN = TN + 1

        res = [TH, TP, TN, FP, FN, float(TP)/(TP+FP), float(TP)/(TP+FN)]
        print(fmt_str.format(*res))
        results.append(res)

    return results


def main():
    """
    """
    # Parse arguments
    args = parse_arguments()

    # Load the dataset
    print("Loading dataset as type {}...".format(args.attack))
    X_mon, y_mon = load_data(args.mon, typ=args.attack, unmon=False)
    X_unmon, _ = load_data(args.unmon, typ=args.attack, unmon=True)
    unmon_label = np.amax(y_mon) + 1
    y_unmon = np.ones((X_unmon.shape[0],)) * unmon_label
    y_mon = np_utils.to_categorical(y_mon, unmon_label+1)
    
    # split data into train and test
    sr = 0.8
    X_mon_tr = X_mon[int(X_mon.shape[0]*sr):]
    y_mon_tr = y_mon[int(X_mon.shape[0]*sr):]
    X_unmon_tr = X_unmon[args.world_size:]
    y_unmon_tr = np_utils.to_categorical(y_unmon[args.world_size:], unmon_label+1)

    X_mon_te = X_mon[:int(X_mon.shape[0]*sr)]
    X_unmon_te = X_unmon[:args.world_size]

    # train ow model
    X_train = np.concatenate([X_mon_tr, X_unmon_tr])
    y_train = np.concatenate([y_mon_tr, y_unmon_tr])
    model = train(X_train, y_train, unmon_label, args.output)

    # test in ow setting
    ow_evaluation(model, X_mon_te, X_unmon_te, unmon_label)

if __name__ == "__main__":
    # execute only if run as a script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
