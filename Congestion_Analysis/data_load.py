import pickle as pickle
import numpy as np
import random
random.seed(583004949)

def data_load(save_path):
    
    with open(save_path + 'X_tr.pkl', 'rb') as handle:
        X_train = pickle.load(handle)
    with open(save_path + 'Y_tr.pkl', 'rb') as handle:
        y_train = pickle.load(handle)
    with open(save_path + 'X_vl.pkl', 'rb') as handle:
        X_valid = pickle.load(handle)
    with open(save_path + 'Y_vl.pkl', 'rb') as handle:
        y_valid = pickle.load(handle)
    with open(save_path + 'X_te.pkl', 'rb') as handle:
        X_test = pickle.load(handle)
    with open(save_path + 'Y_te.pkl', 'rb') as handle:
        y_test = pickle.load(handle)
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test