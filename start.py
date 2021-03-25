import os
import numpy as np
import time
from scripts.utils import *
from scripts.models import *
from scripts.mismatch import *
from scripts.spectrum import *
from scripts.substring import *
if __name__ == "__main__":
    """
    Load Data, compute kernels, and then generate the predictions.
    """

    # load data
    data = DataLoader()

    # ===== KERNEL GENERATION =====
    path = 'Kernel/'
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

            
    # spectrum Kernel
    K_list = [3, 4, 5, 6]

    for k in K_list:
        for i in range(3):
            t1 = time.time()
            X_train = data.raw_data['train'][i]
            X_test = data.raw_data['test'][i]
            sk = SpectrumKernel(k, True)
            K_train = sk.fit(X_train)
            K_test = sk.fit(X_test)
            name_train = '{}_spectrum_N_train_{}'.format(i, k)
            name_test = '{}_spectrum_N_test_{}'.format(i, k)
            print('{}_{} tps = {}'.format(k, i, time.time() - t1))
            np.save('./Kernel/' + name_train, K_train)
            np.save('./Kernel/' + name_test, K_test)

    # Mismatch Kernel
    M_list = [1, 2]
    K_list = [6, 7, 8, 9, 10, 11, 12, 13]

    for m in M_list:
        for k in K_list:
            for i in range(3):
                t1 = time.time()
                X_train = data.raw_data['train'][i]
                X_test = data.raw_data['test'][i]
                mk = MisKernel2(k, m, True)
                K_train = mk.fit(X_train)
                K_test = mk.predict(X_test)
                name_train = '{}_mismatch_N_train_{}_{}'.format(i, k, m)
                name_test = '{}_mismatch_N_test_{}_{}'.format(i, k, m)
                print('{}_{} tps = {}'.format(k, i, time.time() - t1))
                np.save('./Kernel/' + name_train, K_train)
                np.save('./Kernel/' + name_test, K_test)

    # Substring Kernel
    k = 11
    l_list = [0.5, 0.75]
    for l in l_list:
        for i in range(3):
            t1 = time.time()
            X_train = data.raw_data['train'][i][:10]
            X_test = data.raw_data['test'][i][:10]
            K_train = compute_kernel(X_train, X_train, k, l)
            K_test = compute_kernel(X_train, X_test, k, l)
            for k in range(K_train.shape[-1]):
                name_train = '{}_substring_N_train_{}_{}'.format(i, k, l)
                name_test = '{}_substring_N_test_{}_{}'.format(i, k, l)
                print('{}_{} tps = {}'.format(l, i, time.time() - t1), flush=True)
                np.save('./Kernel/'+name_train, K_train[:, :, k])
                np.save('./Kernel/'+name_test, K_test[:, :, k])                
    # ==== GENERATE RESULTS ====

    # best parameters
    parameters = [{'C': 0.12,
                   'seq_length_mm': [np.arange(7, 13)],
                   'mismatch_nb': [[1, 2]],
                   'kernel_name': ['mismatch'],
                   'data_index': [0]},
                  {'C': 0.07,
                   'seq_length_sp': [3],
                   'seq_length_mm': [np.arange(8, 14)],
                   'mismatch_nb': [[1, 2]],
                   'lambda_ss': [0.5],
                   'seq_length_ss': [np.arange(6, 12)],
                   'kernel_name': ['sum'],
                   'data_index': [1]},
                  {'C': 0.21,
                   'seq_length_mm': [np.arange(5, 14)],
                   'mismatch_nb': [[1, 2]],
                   'kernel_name': ['mismatch'],
                   'data_index': [2]}]

    # init grid search
    GS = GridSearch(SupportVectorMachine(), parameters, data.raw_data, data.labels, data_type='raw')
    # run grid search to generate performances on each dataset
    GS.grid_search()
    # print performances for each dataset and overall
    GS.return_results()
    # save performances
    GS.save_performances()
    # write predictions in the file: results/Yte.csv
    GS.write_csv()
