from scipy import optimize
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
from cvxopt import matrix, solvers
# from strkernel.mismatch_kernel import MismatchKernel
# from strkernel.mismatch_kernel import preprocess
import re
import os
from collections.abc import Iterable

# import kernels
from scripts.kernels import *

"""
Mother Class
"""


class KernelMethod:
    def __init__(self, kernel_name='gaussian', data_index=None, sigma=1., linear_c=1., seq_length_sp=3,
                 seq_length_mm=6, mismatch_nb=1, seq_length_ss=6, lambda_ss=0.5, kappa=1., theta=1., d=1):
        """
        :param kernel_name: string, kernel type
        :param data_index: string, index of the ARN dataset (between 0 and 2)
        :param data_path: string, if not None, load data at the given path
        """
        # data parameters
        self.data_index = data_index
        self.X_train = None
        self.y_train = None
        self.preds = None  # predictions
        self.bias = 0  # bias

        # kernel parameters
        self.K = None
        self.kernel_name = kernel_name
        self.alpha = None
        self.custom_kernels = ['spectrum', 'mismatch', 'substring']

        self.sigma = sigma
        self.linear_c = linear_c
        # spectrum parameters
        self.seq_length_sp = seq_length_sp
        # mismatch parameters
        self.seq_length_mm = seq_length_mm  # sequence length
        self.mismatch_nb = mismatch_nb  # mismatch
        # substring parameters
        self.seq_length_ss = seq_length_ss
        self.lambda_ss = lambda_ss

        self.kappa = kappa
        self.theta = theta
        self.d = d  # polynomial degree

        # useful parameters
        self.common_params = dict()
        self.params = dict()
        self.update_params()

    def set_data_index(self, data_index):
        self.data_index = data_index

    def update_params(self):
        self.common_params = {'kernel_name': self.kernel_name, 'data_index': self.data_index}
        self.params = {
            'gaussian': {**self.common_params, 'sigma': self.sigma},
            'linear': {**self.common_params, 'linear_c': self.linear_c},
            'sigmoid': {**self.common_params, 'kappa': self.kappa, 'theta': self.theta},
            'polynomial': {**self.common_params, 'd': self.d},
            'sum': {**self.common_params, 'seq_length_sp':  self.seq_length_sp, 'seq_length_mm': self.seq_length_mm,
                    'mismatch_nb': self.mismatch_nb, 'seq_length_ss': self.seq_length_ss, 'lambda_ss': self.lambda_ss},
            'spectrum': {**self.common_params, 'seq_length_sp':  self.seq_length_sp},
            'mismatch': {**self.common_params, 'seq_length_mm': self.seq_length_mm, 'mismatch_nb': self.mismatch_nb},
            'substring': {**self.common_params, 'seq_length_ss': self.seq_length_ss, 'lambda_ss': self.lambda_ss}
        }

    def _fit(self, X_train=None, y_train=None):
        """
        Function to call before calling fit function.
        X_train may be either:
        - features if kernel is computed
        - indexes if kernel is loaded

        :param X_train: array
        :param y_train: array, labels
        """
        self.X_train = X_train
        self.y_train = y_train

        # labels formatting
        self.y_train = self.y_train.reshape(-1, 1)
        self.y_train = -1 + 2 * self.y_train.astype(float)
        self.y_train = self.y_train.squeeze()

        # kernel computation/loading
        self.K = self.kernel(self.X_train, dataset='train')

    def predict(self, X=None, true_value=False, dataset='train'):
        """
        Prediction function.

        === dataset parameter useful when kernel is loaded (already computed) ===
        Set dataset to 'train' when cross-validation is performed.
        Set dataset to 'test' to generate final predictions on test dataset.

        :param X: array, input data, with shape (n_samples, n_features) or indexes for already computed kernels.
        It can be None for already computed kernels (i.e. custom kernels).
        :param true_value: boolean
        :param dataset: string, 'train' or 'test ; might be useless on not ARN data.
        """
        self.preds = self.kernel(X, dataset, prediction=True).dot(self.alpha) + self.bias
        if true_value:
            return self.preds
        return (self.preds > 0).astype(int)

    def kernel(self, X=None, dataset='train', prediction=False):
        """
        Compute kernel or values of K(x_i,x).
        === if K is computed ===
        If X is None: compute gaussian kernel K. Otherwise, compute values of K(x_i, x) for x in X (test set).
        === if K is loaded ===
        'dataset' parameter is used to determine which data to load.
        X corresponds to the index of lines/columns to keep (to split between train/val datasets)

        :param X: array
        :param dataset: string, 'train' or 'test'
        :param prediction: boolean, True for prediction, False for training
        """
        if X is None:
            # Compute kernel K
            X = self.X_train

        if self.kernel_name == 'gaussian':
            X_base = np.tile(self.X_train, (X.shape[0], 1, 1))
            K = np.exp(- np.linalg.norm(X_base - X[:, np.newaxis, :], ord=2, axis=2) / (2 * self.sigma ** 2))
        elif self.kernel_name == 'linear':
            # separation
            self.w = np.sum(self.alpha.reshape((-1, 1)) * self.X_train, axis=0)
            K = X.dot(self.X_train.T) + self.linear_c
        elif self.kernel_name == 'sigmoid':
            K = np.tanh(self.kappa * X.dot(self.X_train.T) + self.theta)
        elif self.kernel_name == 'polynomial':
            K = np.power(X.dot(self.X_train.T) + 1, self.d)
        elif self.kernel_name == 'sum':
            K = self.sum_kernel(self.X_train, custom_kernels=None, dataset=dataset, prediction=prediction)
        elif self.kernel_name in self.custom_kernels:
            K = self.sum_kernel(self.X_train, custom_kernels=self.kernel_name, dataset=dataset, prediction=prediction)
        else:
            raise Exception("No kernel found")
        return K

    def fetch_files(self, kernel_name, dataset):
        """
        Fetch file names to load corresponding precomputed kernels.

        :param kernel_name: stirng, name of the kernel
        :param dataset: string, 'train' or 'test'
        """
        files = []

        def int_to_list(d):
            if isinstance(d, Iterable):
                return d
            return [d]

        if kernel_name == 'spectrum':
            for k in np.array(int_to_list(self.seq_length_sp)):
                file_regex = '{}_{}_N_{}_{}.npy'.format(self.data_index, 'spectrum', dataset, k)
                files = files + [file for file in os.listdir('Kernel') if re.search(file_regex, file)]
        elif kernel_name == 'mismatch':
            for k in int_to_list(self.seq_length_mm):
                for m in int_to_list(self.mismatch_nb):
                    file_regex = '{}_{}_N_{}_{}_{}.npy'.format(self.data_index, 'mismatch', dataset, k, m)
                    files = files + [file for file in os.listdir('Kernel') if re.search(file_regex, file)]
        elif kernel_name == 'substring':
            for k in int_to_list(self.seq_length_ss):
                for lambda_ in int_to_list(self.lambda_ss):
                    file_regex = '{}_{}_N_{}_{}_{}.npy'.format(self.data_index, 'substring', dataset, k, str(lambda_))
                    files = files + [file for file in os.listdir('Kernel') if re.search(file_regex, file)]
        else:
            raise Exception("No custom kernel found")

        return files

    def sum_kernel(self, indexes, custom_kernels=None, dataset='train', prediction=False):
        """
        Sum precomputed kernels.

        :param indexes: array, indexes of training samples
        :param custom_kernels: str, or list of str: custom kernels to use
        :param dataset: string, 'train' or 'test'
        :param prediction: boolean, True for prediction, False otherwise
        """

        # define custom_kernels
        if custom_kernels is None:
            custom_kernels = self.custom_kernels
        if type(custom_kernels) != list:
            custom_kernels = [custom_kernels]

        # var
        K = None  # kernel

        # generate kernel
        for custom_kernel in custom_kernels:
            # fetch files names
            files = self.fetch_files(custom_kernel, dataset)
            print(files)
            for k, file in enumerate(files):
                # load precomputed kernels
                if K is None:
                    K = np.load('Kernel/' + file)
                    if K.shape[0] > K.shape[1]:
                        K = K.T
                else:
                    K_ = np.load('Kernel/' + file)
                    if K_.shape[0] > K_.shape[1]:
                        K += K_.T
                    else:
                        K += K_

        # truncate kernel if necessary
        if dataset == 'train':
            if prediction:
                training_indexes = np.arange(K.shape[0])
                training_indexes = np.array([k for k in training_indexes if k not in indexes])
                return K[training_indexes, :][:, indexes]
            else:
                return K[indexes, :][:, indexes]
        elif dataset == 'test':
            return K
        else:
            raise Exception('No dataset found!')

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.update_params()
        return self


'''
Kernel Ridge Regression
'''


class KernelRidgeRegression(KernelMethod):

    def __init__(self, lambda_=0.05, **params):
        super().__init__(**params)
        self.lambda_ = lambda_
        self.model_name = 'krr'

    def fit(self, X_train=None, y_train=None):
        """
        Fit function.
        :param X_train: array
        :param y_train: labels
        :param data_index: int, index of the dataset
        """
        self._fit(X_train, y_train)

        inv = np.linalg.inv(self.K + self.lambda_ * self.X_train.shape[0] * np.eye(self.X_train.shape[0]))
        self.alpha = np.dot(inv, self.y_train).reshape(-1, 1).squeeze()

    def get_params(self, deep=True):
        return {**self.params[self.kernel_name], 'lambda_': self.lambda_}


'''
SVM
'''


class SupportVectorMachine(KernelMethod):
    def __init__(self, C=5., **params):
        """
        :param C: float, regularization constant
        """
        super().__init__(**params)
        # verbose
        # solvers.options['MOSEK'] = {iparam.log: 0}
        # parameters
        #        print(C)
        #        assert type(C) is float, 'C must be a float'
        self.model_name = 'svm'
        self.C = C
        self.K = None  # computed kernel matrix
        self.epsilon = 1e-6  # precision

    def fit(self, X_train=None, y_train=None):
        """
        Fit function.

        :param X_train: array
        :param y_train: labels
        """
        self._fit(X_train, y_train)
        N = len(self.y_train)

        # QP solver
        P = matrix(self.K)
        q = -1 * matrix(self.y_train)
        G = np.vstack([-np.diag(self.y_train), np.diag(self.y_train)])
        G = matrix(G)
        h = matrix([0] * N + [self.C] * N)
        self.sol = solvers.qp(P, q, G, h)
        self.alpha = np.array(self.sol['x']).squeeze()
        # keep only support vectors (set non-support vectors coefs to 0)
        self.alpha[(np.abs(self.alpha) < self.epsilon).squeeze()] = 0  # * (np.abs(self.alpha) > self.C - self.epsilon)

    def get_params(self, deep=True):
        return {**self.params[self.kernel_name], 'C': self.C}


class SupportVectorMachine2(KernelMethod):
    def __init__(self, C=5., **params):
        """
        :param C: float, regularization constant
        """
        super().__init__(**params)
        self.model_name = 'svm'
        self.C = C
        self.K = None  # computed kernel matrix
        self.epsilon = 1e-6  # precision

    def fit(self, X_train=None, y_train=None):
        """
        Fit function.

        :param X_train: array
        :param y_train: labels
        """
        self._fit(X_train, y_train)
        N = len(self.y_train)

        # QP solver
        Q = np.multiply(self.y_train[np.newaxis, :], self.y_train[:, np.newaxis]) * self.K
        e = -np.ones((N, 1))
        G = np.vstack((np.eye(N), -np.eye(N)))
        h = np.vstack((self.C * np.ones((N, 1)), np.zeros((N, 1))))
        A = self.y_train.astype('float').reshape(-1, 1).T
        b_mat = 0.
        Q_mat = matrix(Q)
        e_mat = matrix(e)
        G_mat = matrix(G)
        h_mat = matrix(h)
        A_mat = matrix(A)
        b_mat = matrix(b_mat)
        self.sol = solvers.qp(Q_mat, e_mat, G_mat, h_mat, A_mat, b_mat)
        self.alpha = np.array(self.sol['x']).squeeze()
        # keep only support vectors (set non-support vectors coefs to 0)
        self.alpha[(np.abs(self.alpha) < self.epsilon).squeeze()] = 0  # * (np.abs(self.alpha) > self.C - self.epsilon)
        self.alpha[(np.abs(self.C - self.alpha) < self.epsilon).squeeze()] = self.C

        # support = np.where(np.logical_and(self.alpha > 0, self.alpha < self.C))[0]
        # grad = np.dot(Q, self.alpha.reshape(-1, 1)) + e
        # b = - np.sum(self.y_train[support].reshape(-1, 1) * grad[support]) / len(support)

        self.alpha = self.alpha * self.y_train
        # compute bias
        self.compute_bias()

    def compute_bias(self):
        """
        Compute bias.
        """
        # support vectors indexes
        support_vectors = self.alpha != 0
        # vectors on +/-1 margins indexes
        margin_vectors = support_vectors * (np.abs(self.alpha) < self.C - self.epsilon)
        bias = 1 / margin_vectors.sum() * np.sum(self.y_train[margin_vectors] - np.sum(self.alpha[support_vectors] \
                                                                                       .reshape((1, -1)) * self.K[
                                                                                                               margin_vectors][
                                                                                                           :,
                                                                                                           support_vectors],
                                                                                       axis=-1))
        self.bias = bias

    def get_params(self, deep=True):
        return {**self.params[self.kernel_name], 'C': self.C}
