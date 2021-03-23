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
        """
        Return a counter of all k-mers from a sequence
        """
        res = [seq[i:i + self.seq_length] for i in range(len(seq) - self.seq_length + 1)]
        return Counter(res)

    def spectrum_ps(self, C1, C2):
        """
        Scalar product between two counters
        """
        ps = 0
        for parameter, value in C1.items():
            if parameter in C2:
                ps += value * C2[parameter]
        return ps

    def embed(self, X1, X2):
        """
        Return a matrix K(x_i, x_j) with x_i in X1 and x_j in X2
        """
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
        """
        Return the train matrix
        """
        self.X_train = X_train
        self.K = self.embed(X_train, X_train)
        return self.K

    def predict(self, X_test):
        """
        Return the test compute_occurence_matrix
        """
        self.K_test = self.embed(X_test, self.X_train)
        return self.K_test

    def create_dict(self, seq_length):
        """
        Create a mapping vector
        """
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
        """
        Create a lookup table of all possible k-mers
        """
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
