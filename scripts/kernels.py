import numpy as np
from collections import Counter

"""
Spectrum Kernel
"""


class SpectrumKernel:
    def __init__(self, X_train, seq_length):
        """
        :param X_train: array
        :param seq_length: int, length of subsequences to consider inside ARN
        """
        self.X_train = X_train
        self.seq_length = seq_length

    def sort_substring(self, seq):
        res = [seq[i:i + self.seq_length] for i in range(len(seq) - self.seq_length + 1)]
        return Counter(res)

    def spectrum_ps(self, C1, C2):
        ps = 0
        for parameter, value in C1.items():
            if parameter in C2:
                ps += value * C2[parameter]
        return ps

    def embed(self, normalization=False):
        count_list = []
        for i in range(len(self.X_train)):
            count_list.append(self.sort_substring(self.X_train[i]))
        K = np.zeros((self.X_train.shape[0], self.X_train.shape[0]))
        for i in range(0, self.X_train.shape[0]):
            for j in range(0, i + 1):
                ps = self.spectrum_ps(count_list[i], count_list[j])
                K[i, j] = ps
                if i != j:
                    K[j, i] = ps
        K_norm = np.zeros((self.X_train.shape[0], self.X_train.shape[0]))
        # normalization
        if normalization:
            for i in range(0, self.X_train.shape[0]):
                for j in range(i + 1):
                    norm = K[i, j] / np.sqrt(K[i, i] * K[j, j])
                    K_norm[i, j] = norm
                    if i != j:
                        K_norm[j, i] = norm
            return K_norm
        return K

    def create_dict(self, seq_length):
        if seq_length == 1:
            d = {'C': 0, 'A': 0, 'T': 0, 'G': 0}
        else:
            d_k = self.create_dict(seq_length - 1)
            d = {}
            for parameter, value in d_k.items():
                d[parameter + 'C'] = 0
                d[parameter + 'A'] = 0
                d[parameter + 'T'] = 0
                d[parameter + 'G'] = 0
        return d

    def lookup_table(self, alpha, normalization=False):
        table = self.create_dict(self.seq_length)
        norm = 1.
        for i in range(self.X_train.shape[0]):
            if alpha[i] != 0:
                seq = self.X_train[i]
                C = Counter([seq[i:i + self.seq_length] for i in range(len(seq) - self.seq_length + 1)])
                if normalization:
                    norm = np.sqrt(self.spectrum_ps(C, C))
                for subseq, value in C.items():
                    table[subseq] += value * alpha[i].item() / norm
        self.table = table

    def predict(self, X, true_value=False):
        """
        :param X: array, data to make predictions on.
        :param true_value: bool, if True, return float values, if False, return binary array
        """
        label = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            for j in range(len(X[i]) - self.seq_length + 1):
                label[i] += self.table[X[i][j:j + self.seq_length]]
        return label

