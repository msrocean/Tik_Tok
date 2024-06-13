import os
import numpy as np
import re
import json


def load_data(directory, delimiter='\t', file_split="-", length=5000, typ=2, unmon=False):
    """
    Load data from ascii files
    """
    X, y = [], []
    class_counter = 0
    for root, dirs, files in os.walk(directory):
        if not unmon:
            files = [fname for fname in files if len(fname.split(file_split)) == 2]
        for fname in files:
            try:
                trace_class = None
                if unmon:
                    trace_class = -1
                else:
                    cls, inst  = fname.split(file_split)
                    trace_class = int(cls)

                # build direction sequence
                sequence = load_trace(os.path.join(root, fname), seperator=delimiter)

                # use time direction
                if typ==1:
                    sequence = [sequence[0][i]*sequence[1][i] for i in range(len(sequence[0]))] 

                # use time only
                elif typ==2:
                    sequence = sequence[0]  

                # use direction only
                else:
                    sequence = sequence[1]  

                # add sequence and label
                sequence = np.array(sequence)
                if len(sequence) < length:
                    sequence = np.hstack((sequence, np.zeros(((length-len(sequence),)))))
                sequence.resize((length, 1))
                X.append(sequence)
                y.append(trace_class)
            except Exception as e:
                print(e)
                pass

    # wrap as numpy array
    X, Y = np.array(X), np.array(y)

    # shuffle
    s = np.arange(Y.shape[0])
    np.random.seed(0)
    np.random.shuffle(s)
    X, Y = X[s], Y[s]
    return X, Y


def load_trace(path, seperator="\t"):
    """
    loads data to be used for predictions
    """
    file = open(path, 'r')
    sequence = [[], []]
    for line in file:
        try:
            pieces = line.strip("\n").split(seperator)
            if int(pieces[1]) == 0:
                break
            timestamp = float(pieces[0])
            length = abs(int(pieces[1]))
            direction = int(pieces[1]) // length
            sequence[0].append(timestamp)
            sequence[1].append(direction)
        except Exception as e:
            print(e)
            print("Error when trying to read packet sequence from %s!" % path)
            return None
    return sequence


