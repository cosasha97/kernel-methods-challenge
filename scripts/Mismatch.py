import numpy as np
from collections import Counter


class Trie2:
    def __init__(self, deep, node_letter=None, parent=None, prefix='', predict=False, count=None, adv=None):
        self.node_letter=node_letter
        self.parent = {}
        self.children = {}
        self.deep = deep
        self.kmers={}
        self.kmers2={}
        self.prefix = prefix
        if node_letter != None:
            self.prefix+=node_letter
        self.letters = ['A', 'C', 'G', 'T']
        self.predict=predict
        if count==None:
          self.count = np.array(0.)
          self.adv = np.array(1/100)
        else:
          self.count = count
          self.adv = adv

    def is_leaf(self):
        return len(self.children==0)

    def add_child(self, letter):
        self.children[letter] = Trie(self.deep+1, node_letter=letter, parent=self, prefix=self.prefix)

    def generate_subsequence(self, X, k, predict=False, X2=None):
        if self.deep == 0:
            for i in range(X.shape[0]):
                self.kmers[i] = [(j, 0) for j in range(len(X[i]) -k + 1)]

            if predict:
                for i in range(X2.shape[0]):
                    self.kmers2[i] = [(j, 0) for j in range(len(X2[i]) -k + 1)]


    def update_kmers(self, X, kmers, k, m):
      kmers_temp = {}
      for index, value in kmers.items():
        kmers_temp[index] = [(kmers[index][i][0], kmers[index][i][1] + 1*(X[index][kmers[index][i][0]+self.deep-1] != self.node_letter)) for i in range(len(kmers[index]))]

      idx_to_pop = []
      for index, value in kmers_temp.items():
          L_temp = [(kmers_temp[index][i][0], kmers_temp[index][i][1]) for i in range(len(kmers_temp[index])) if kmers_temp[index][i][1]<=m ]
          kmers_temp[index] = L_temp
          if len(kmers_temp[index]) == 0:
              idx_to_pop.append(index)
      for idx in idx_to_pop:
          kmers_temp.pop(idx)
      return kmers_temp

    def process_child(self, letter, k, m, predict, X_train, X_test=None):
        T_child = Trie2(self.deep+1, node_letter=letter, prefix=self.prefix, count=self.count, adv=self.adv)
        T_child.kmers =  T_child.update_kmers(X_train, self.kmers.copy(), k, m)
        T_child.predict = self.predict
        if predict:
            if len(self.kmers2) == 0:
                T_child.kmers2 = {}
            else:
                T_child.kmers2 = T_child.update_kmers(X_test, self.kmers2.copy(), k, m)

        return T_child

    def update_kernel(self, K, Knorm, Knorm2):
        if self.predict:
            if len(self.kmers) > 0:
                for key, value in self.kmers.items():
                    Knorm[key] += len(value)**2
            if len(self.kmers2) > 0:
                for key2, value2 in self.kmers2.items():
                    Knorm2[key2] += len(value2)**2

            if (len(self.kmers2) > 0) and (len(self.kmers) > 0):
                for key2, value2 in self.kmers2.items():
                    for key, value in self.kmers.items():
                        K[key2, key] += len(value2) * len(value)

        else:
            if len(self.kmers) > 0:
                for key1, value1 in self.kmers.items():
                    for key2, value2 in self.kmers.items():
                        K[key1, key2] += len(value1) * len(value2)

    def construct(self, X, K, k, m, predict=False, Knorm=None, Knorm2=None, X_test=None):

        if self.deep<k:

            for letter in self.letters:

                T_child = self.process_child(letter, k, m, predict, X, X_test)
                if (len(T_child.kmers)>0) or (len(T_child.kmers2)>0):
                    T_child.parent = self
                    self.children[letter] = T_child
                    self.children[letter].construct(X, K, k, m, predict=predict, Knorm=Knorm, Knorm2=Knorm2, X_test=X_test)
                else:
                  self.count += 4**(k-self.deep-1)

                if letter in self.children:
                  del self.children[letter]
        else:
            self.update_kernel(K, Knorm, Knorm2)
            self.count+=1
            if self.count/(4**k)>=self.adv:
              print('{}%'.format(self.adv*100))
              self.adv+=1/100


class MisKernel2:

    def __init__(self, k, m, normalization=True):
        self.k=k
        self.m=m
        self.normalization=normalization

    def fit(self, X):
        self.X_train = X
        n = X.shape[0]
        K = np.zeros((n, n))
        T = Trie2(0, predict=False)
        T.generate_subsequence(X, self.k, predict=False)
        T.construct(X, K, self.k, self.m, predict=False)

        self.K = K
        if self.normalization:
            K_norm = np.zeros((n, n))
            np.fill_diagonal(K_norm, 1)
            for i in range(0, X.shape[0]):
                for j in range(i):
                    K_norm[i, j] = K[i, j] / np.sqrt(K[i, i] * K[j, j])
                    K_norm[j, i] = K_norm[i, j]
            self.K_norm = K_norm
            return K_norm
        else:
            return K

    def predict(self, X_test):
        n_test = X_test.shape[0]
        n_train = self.X_train.shape[0]
        K = np.zeros((n_test, n_train))
        K_norm1 = np.zeros(n_train)
        K_norm2 = np.zeros(n_test)
        T = Trie2(0, predict = True)
        T.generate_subsequence(self.X_train, self.k, predict = True, X2 = X_test)
        T.construct(self.X_train, K, self.k, self.m, predict=True, Knorm=K_norm1, Knorm2=K_norm2, X_test=X_test)
        self.K_test = K
        if self.normalization:
            K_norm = np.zeros((n_test, n_train))
            for i in range(n_train):
                for j in range(n_test):
                    K_norm[j, i] = K[j, i] / np.sqrt(K_norm1[i] * K_norm2[j])
            self.K_test_norm = K_norm
            return K_norm
        else:
            return K


if __name__ == "__main__":
    t1 = time.time()

m=2
k = 12
K_list = [9,10,11,12]

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
      np.save('./Kernel/'+name_train, K_train)
      np.save('./Kernel/'+name_test, K_test)
