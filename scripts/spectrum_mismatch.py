import numpy as np
from collections import Counter

class SpectrumKernel:
    def __init__(self, seq_length, normalization=True):
        """
        :param X_train: array\n",
        :param seq_length: int, length of subsequences to consider inside ARN\n",
        """
        self.seq_length = seq_length
        self.normalization = normalization

    def sort_substring(self, seq):
        res = [seq[i:i + self.seq_length] for i in range(len(seq) - self.seq_length + 1)]
        return Counter(res)

    def spectrum_ps(self, C1, C2):
        ps = 0
        for parameter, value in C1.items():
            if parameter in C2:
                ps += value * C2[parameter]
        return ps

    def embed(self, X1, X2):
            count_list1 = []
            count_list2 = []
            for i in range(len(X1)):
                count_list1.append(self.sort_substring(X1[i]))
            for i in range(len(X2)):
                count_list2.append(self.sort_substring(X2[i]))

            K = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(0, X1.shape[0]):
                for j in range(0, X2.shape[0]):
                    ps = self.spectrum_ps(count_list1[i], count_list2[j])
                    K[i, j] = ps

            # normalization
            if self.normalization:
                K_norm = np.zeros((X1.shape[0], X2.shape[0]))
                for i in range(0, X1.shape[0]):
                    norm_i = self.spectrum_ps(count_list1[i], count_list1[i])
                    for j in range(X2.shape[0]):
                        norm_j = self.spectrum_ps(count_list2[j], count_list2[j])
                        K_norm[i, j] = K[i, j] / np.sqrt(norm_i * norm_j)
                return K_norm
            return K

    def fit(self, X_train):
        self.X_train = X_train
        self.K = self.embed(X_train, X_train)
        return self.K

    def predict(self, X_test):
        self.K_test = self.embed(X_test, self.X_train)
        return self.K_test

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

#     def predict(self, X, true_value=False):
#         """
#         :param X: array, data to make predictions on.
#         :param true_value: bool, if True, return float values, if False, return binary array
#         """
#         label = np.zeros(X.shape[0])
#         for i in range(X.shape[0]):
#             for j in range(len(X[i]) - self.seq_length + 1):
#                 label[i] += self.table[X[i][j:j + self.seq_length]]
#         return label

class MismatchKernel:

    def __init__(self, seq_length, m, normalization=True):
        self.seq_length = seq_length
        self.normalization = normalization
        self.fitted = False
        self.m = m

    def compute_dist(self, s1 ,s2):
        dist = 0
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                dist+=1
                if dist>self.m:
                    return -1
        return dist

    def sort_substring(self, seq):
        res = [seq[i:i + self.seq_length] for i in range(len(seq) - self.seq_length + 1)]
        return Counter(res)

    def spectrum_ps(self, C1, C2):
        ps = 0
        for parameter, value in C1.items():
            if parameter in C2:
                ps += value * C2[parameter]
        return ps

    def mismatch_counter(self, seq):
        C = self.sort_substring(seq)
        C2 = C.copy()

        for key1, value1 in C.items():
            for key2, value2 in C2.items():
                d = self.compute_dist(key1, key2)
                if d>0:
                    C2[key2] += value1
        return C2

    def embed(self, X1, X2):
        count_list1 = []
        count_list2 = []
        for i in range(len(X1)):
            count_list1.append(self.mismatch_counter(X1[i]))

        if self.fitted:
            for i in range(len(X2)):
                count_list2.append(self.mismatch_counter(X2[i]))
        else:
            count_list2 = count_list1

        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(0, X1.shape[0]):
            if self.fitted:
                for j in range(0, X2.shape[0]):
                    K[i, j] = self.spectrum_ps(count_list1[i], count_list2[j])
            else:
                for j in range(0, i+1):
                    K[i, j] = self.spectrum_ps(count_list1[i], count_list2[j])
                    K[j, i] = K[i, j]

        # normalization\n",
        if self.normalization:
            K_norm = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(0, X1.shape[0]):
                norm_i = self.spectrum_ps(count_list1[i], count_list1[i])
                if self.fitted:
                    for j in range(X2.shape[0]):
                        norm_j = self.spectrum_ps(count_list2[j], count_list2[j])
                        K_norm[i, j] = K[i, j] / np.sqrt(norm_i * norm_j)
                else:
                    for j in range(i+1):
                        norm_j = self.spectrum_ps(count_list2[j], count_list2[j])
                        K_norm[i, j] = K[i, j] / np.sqrt(norm_i * norm_j)
                        K_norm[j, i] = K_norm[i, j]
            return K_norm
        return K

    def fit(self, X_train):
        self.X_train = X_train
        self.K = self.embed(X_train, X_train)
        self.fitted=True
        return self.K

    def predict(self, X_test):
        self.K_test = self.embed(X_test, self.X_train)
        return self.K_test

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
