import pandas as pd
import numpy as np
import pickle
from sklearn.utils import shuffle
from common import *
from features import *
import time
import random
random.seed(583004949)


def data_details(dataset):
    datasets = ['Undefended', 'WTF-PAD', 'W-T-Simulated', 'W-T-Real', 'Onion-Sites']
    if dataset == datasets[0]: # Undefended
        num_sites = 95
        num_instances = 1000
        tr_ins = 800
        vl_ins = 100
        te_ins = 100
        bin_size = 20
    elif dataset == datasets[1]: # WTF_PAD
        num_sites = 95
        num_instances = 1000
        tr_ins = 800
        vl_ins = 100
        te_ins = 100
        bin_size = 20   
    elif dataset == datasets[2]: # Walkie-Talkie (Simulated)
        num_sites = 100
        num_instances = 900
        tr_ins = 720
        vl_ins = 90
        te_ins = 90
        bin_size = 20
    elif dataset == datasets[3]: # Walkie-Talkie (Real)
        num_sites = 100
        num_instances = 750
        tr_ins = 600
        vl_ins = 75
        te_ins = 75
        bin_size = 20
    else: # Onion Sites
        num_sites = 538
        num_instances = 77
        tr_ins = 61
        vl_ins = 8
        te_ins = 8
        bin_size = 20
        
    return num_sites, tr_ins, vl_ins, te_ins, bin_size


def gen_save_feats(dataset, data_path, save_path):

    num_sites, tr_ins, vl_ins, te_ins, bin_size = data_details(dataset)
    
    for i in range(3):
        #print('Iteration: ', i)
        features = {
            "MED": {},
            "IBD_FF": {},
            "IBD_IFF": {},
            "IBD_LF": {},
            "IBD_OFF": {},
            "Burst_Length": {},
            "IMD": {},
            "Variance": {},
        }
        
        if i == 0:
            print('Processing Training Data ...')
            st_ind = 0
            en_ind = tr_ins
        elif i == 1:
            print('Processing Validation Data ...')
            st_ind = tr_ins
            en_ind = tr_ins + vl_ins
        else:
            print('Processing Testing Data ...')
            st_ind = tr_ins + vl_ins
            en_ind = tr_ins + vl_ins + te_ins
        
        labels_instances = []
        
        #s_count = 0
        for site in range(0, num_sites):
            for label in range(st_ind, en_ind):
                file_name = str(site) + "-" + str(label)
                
                if dataset == 'W-T-Simulated':
                    final_fname = file_name + '.cell'
                else:
                    final_fname = file_name
                # Directory of the raw data
                with open(data_path + final_fname , "r") as file_pt:
                    traces = []
                    for line in file_pt:
                        x = line.strip().split('\t')
                        x[0] = float(x[0])
                        x[1] = 1 if float(x[1]) > 0 else -1
                        traces.append(x)
                    bursts, direction_counts = extract_bursts(traces)
                    features["MED"][file_name] = MED(bursts)
                    features["IBD_FF"][file_name] = \
                        IBD_FF(bursts)
                    features["IBD_IFF"][file_name] = \
                        IBD_IFF(bursts)
                    features["IBD_LF"][file_name] = \
                        IBD_LF(bursts)
                    features["IBD_OFF"][file_name] = \
                        IBD_OFF(bursts)
                    features["Burst_Length"][file_name] = Burst_Length(bursts)
                    features["IMD"][file_name] = IMD(bursts)
                    features["Variance"][file_name] = Variance(bursts)
                    labels_instances.append(file_name)
            
            #if s_count%5 == 0:
                #print ('Done for ', s_count, ' sites.')
            #s_count +=1


        feature_bins = {
            "MED": bin_size,
            "IBD_FF": bin_size,
            "IBD_IFF": bin_size,
            "IBD_LF": bin_size,
            "IBD_OFF": bin_size,
            "Burst_Length": bin_size,
            "IMD": bin_size,
            "Variance": bin_size,
        }
        # Create bins for each feature, extract bin counts and normalize them
        if i == 0:
            print ("Extracting Training Features ...")
            output_file = 'training'
        elif i == 1:
            print ("Extracting Validation Features ...")
            output_file = 'validation'
        else:
            print ("Extracting Testing Features ...")
            output_file = 'testing'

        for feature in features:
            features[feature] = normalize_data(features[feature], feature_bins[feature])

        feature_names = features.keys()
        #Up to this point, you have all of the features, and feature names.

        if i == 0:
            print ("Saving Training Features ...")
        elif i == 1:
            print ("Saving Validation Features ...")
        else:
            print ("Saving Testing Features ...")
        with open(save_path + output_file, "w") as out:
            for label in labels_instances:
                data = []
                data.extend(values
                            for f in feature_names
                            for values in features[f][label])

                row = ",".join([str(val) for val in data]) + "," + label.split("-")[
                    0] + "\n"
                out.write(row)

        print('Done ...')



def read_data(file_name):
    """
    :param file_name: absolute path to the data file
    :return:
    """
    data = pd.read_csv(file_name, header=None)
    return data[data.columns[:-1]].as_matrix(), \
        data[data.columns[-1]].as_matrix()


def making_matrx(X, Y, f_name, f_dir):
    """
    :param X:
    :param Y:
    :return:
    """

    m, n = X.shape
    
    X_ = np.zeros(shape=(m, n))
    Y_ = np.zeros(shape=(m,))

    labels = np.unique(Y)
    ind1 = 0

    for i in np.arange(labels.size):
        indices = np.where(Y == labels[i])[0]

        splt_nbr = int(round(indices.size))
        X_[ind1:ind1+splt_nbr, :] = X[indices[:splt_nbr], :]
        Y_[ind1:ind1+splt_nbr] = Y[indices[:splt_nbr]]

        ind1 += splt_nbr

    X_final, Y_final = shuffle(X_, Y_)
    
    # Dumping X_train, Y_train, X_test, Y_test, X_val, Y_val in pickle files
    
    data_dir_out = f_dir
    pickle_file_X = data_dir_out + 'X_' + f_name + '.pkl'
    with open(pickle_file_X, 'wb') as handle:
        pickle.dump(X_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_file_Y = data_dir_out + 'Y_' + f_name + '.pkl'
    with open(pickle_file_Y, 'wb') as handle:
        pickle.dump(Y_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #print "Pickle Files Done!!..................."

    return X_final, Y_final



def final_process(dataset, data_root, save_path):
    #st_time = time.time()
    print('Processing ', dataset,' data.')
    gen_save_feats(dataset, data_root, save_path)
    #print('Features Processing Completed in ', (time.time() - st_time)/60, ' mins.')
    train_file = 'training'
    valid_file = 'validation'
    test_file = 'testing'
    X_tr, Y_tr = read_data(save_path + train_file)
    X_vl, Y_vl = read_data(save_path + valid_file)
    X_te, Y_te = read_data(save_path + test_file)
    
    
    X_train, y_train = making_matrx(X = X_tr, Y = Y_tr, f_name = 'tr', f_dir = save_path)

    X_valid, y_valid = making_matrx(X = X_vl, Y = Y_vl, f_name = 'vl', f_dir = save_path)

    X_test, y_test = making_matrx(X = X_te, Y = Y_te, f_name = 'te', f_dir = save_path)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test
    


def final_data_load(save_path):
    
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