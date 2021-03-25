import os
import sys
from utils import *
import sys
import time
sys.path.append('/home/diai_samuel/string-kernels/src')
import numpy as np
from ssk import compute_kernel_rectangular, compute_kernel_triangular
import multiprocessing as mp
from functools import partial


def compute_kernel(train, test, k, l):
    K = np.zeros((train.shape[0], test.shape[0], k))
    if train.shape != test.shape :
        with mp.Pool(mp.cpu_count()) as p:
            res = p.map(partial(compute_kernel_rectangular, train = train, test = test, k = k, l = l), range(train.shape[0]))

        for idx_train, elem_train in enumerate(train):
            K[idx_train, :, :] = res[idx_train]
        return K

    else :
        n = train.shape[0]
        bijection = []
        for i in range(train.shape[0]):
            for j in range(i + 1):
                bijection.append((i, j))
        with mp.Pool(mp.cpu_count()) as p:
            res = p.map(partial(compute_kernel_triangular, dataset = train, bijection = bijection, k = k, l = l), range(n * (n+1) //2))

        for idx in range(len(res)):
            i, j = bijection[idx]
            K[i, j, :] = res[idx]
            K[j, i, :] = res[idx]
        return K

if __name__ == "__main__":
    data = DataLoader()
    l = 0.5
    k = 12
    for i in range(3):
        t1 = time.time()
        X_train = data.raw_data['train'][i][:10]
        X_test = data.raw_data['test'][i][:10]
        print(X_train.shape, X_test.shape)

        K_train = compute_kernel(X_train, X_train, k, l)
        K_test = compute_kernel(X_train, X_test, k, l)
        for k in range(K_train.shape[-1]):
            name_train = '{}_substring_N_train_{}_{}'.format(i, k, l)
            name_test = '{}_substring_N_test_{}_{}'.format(i, k, l)
            print('{}_{} tps = {}'.format(l, i, time.time() - t1), flush=True)
            np.save('./Kernel/'+name_train, K_train[:, :, k])
            np.save('./Kernel/'+name_test, K_test[:, :, k])

