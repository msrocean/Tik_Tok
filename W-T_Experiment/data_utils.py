import os
import numpy as np
import re
import json


def load_data(directory, delimiter='\t', file_split="_", length=5000, typ=2, unmon=False, class_map=dict()):
    """
    Load data from ascii files
    """
    X, y = [], []
    class_counter = 0
    for root, dirs, files in os.walk(directory):
        if not unmon:
            files = [fname for fname in files if len(fname.split(file_split)) == 3]
        for fname in files:
            try:
                trace_class = None
                if unmon:
                    trace_class = -1
                else:
                    _, url, _  = fname.split(file_split)
                    if url not in class_map.keys():
                        if len(class_map.values()) > 0:
                            class_map[url] = max(class_map.values()) + 1
                        else:
                            class_map[url] = 0
                    trace_class = class_map[url]

                # build direction sequence
                sequence = load_trace(os.path.join(root, fname), seperator=delimiter)

                # use direction only
                if typ=='direction':
                    sequence = sequence[1]  

                # use time direction
                elif typ=='tiktok':
                    sequence = [sequence[1][i]*sequence[0][i] for i in range(len(sequence[0]))] 

                # use time only
                elif typ=='timing':
                    sequence = sequence[0]  

                else:
                    print("Bad type argument!")

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
    return X, Y, class_map


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
            sequence[0].append(direction)
            sequence[1].append(timestamp)
        except Exception as e:
            print(e)
            print("Error when trying to read packet sequence from %s!" % path)
            return None
    return sequence


