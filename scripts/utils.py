"""
Useful functions
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
import csv
import random


class DataLoader:
    def __init__(self, path='data'):
        """
        Load all data and store it.
        :param data: string, path to data folder
        """
        self.nb_data = 3
        self.path = path
        self.data_train_name = 'Xtr'
        self.data_test_name = 'Xte'
        self.features_name = '_mat100'
        self.label_train_name = 'Ytr'
        self.label_test_name = 'Ytr'
        # load raw data
        self.raw_data = {'train': self.load_data(self.data_train_name),
                         'test': self.load_data(self.data_test_name)}
        # load data features
        self.data_features = {'train': self.load_data(self.data_train_name, self.features_name, type_='features'),
                              'test': self.load_data(self.data_test_name, self.features_name, type_='features')}
        # load labels
        self.labels = {'train': self.load_data(self.label_train_name),
                       'test': self.load_data(self.label_test_name)}

        # toy data
        self.toy_data_functions = {
            'blobs': blobs,
            'two_moons': two_moons
        }
        self.toy_data = dict()

    def load_dataframe(self, path, type_='raw'):
        """
        Load dataframe and convert it to numpy array.
        Only keep relevant columns (no index).

        :param path: string, path of data to load
        :param type_: string, type of data to load
        """
        if type_ == 'raw':
            return pd.read_csv(path).to_numpy()[:, 1]
        elif type_ == 'features':
            return pd.read_csv(path, sep=' ', header=None).to_numpy()
        else:
            raise Exception("No type found!")

    def load_data(self, prename="", postname="", type_='raw'):
        """
        Load all the datasets for given prename and postname.
        :param prename: string, data name before numerotation
        :param postname: string, data name after numerotation
        :param type_: string, type of data to load (ie features or raw)
        """
        return [self.load_dataframe(os.path.join(self.path, prename + str(k) + postname + '.csv'), type_)
                for k in range(self.nb_data)]

    def load_toy_data(self, num_samples=500, type='blobs', split=0.8):
        """
        Generate and load toy data to test models on.
        :param num_samples: int, number of samples
        :param type: string, type of toy data
        :param split: float, percentage of data in the train set
        """
        X, Y = self.toy_data_functions[type](num_samples)
        split_index = int(split * num_samples)
        indexes = np.arange(len(Y))
        random.shuffle(indexes)
        self.toy_data['train'] = {
            'X': X[indexes[:split_index]],
            'y': Y[indexes[:split_index]]
        }
        self.toy_data['test'] = {
            'X': X[indexes[split_index:]],
            'y': Y[indexes[split_index:]]
        }


# data creation
# code for blobs and two_moons come from the class of Graphs in Machine Learning
def blobs(num_samples, n_blobs=2, blob_var=0.15, surplus=0):
    """
    Creates N gaussian blobs evenly spaced across a circle.

    :param num_samples: number of samples to create in the dataset
    :param n_blobs:      how many separate blobs to create
    :param blob_var:    gaussian variance of each blob
    :param surplus:     number of extra samples added to first blob to create unbalanced classes
    :return: X,  (num_samples, 2) matrix of 2-dimensional samples
             Y,  (num_samples, ) vector of "true" cluster assignment
    """
    # data array
    X = np.zeros((num_samples, 2))
    # array containing the indices of the true clusters
    Y = np.zeros(num_samples, dtype=np.int32)

    # generate data
    block_size = (num_samples - surplus) // n_blobs

    # blob centers
    blob_centers = np.array([[1.2, 0], [0.5, 0.5]])

    for ii in range(1, n_blobs + 1):
        start_index = (ii - 1) * block_size
        end_index = ii * block_size
        if ii == n_blobs:
            end_index = num_samples
        Y[start_index:end_index] = ii - 1
        nn = end_index - start_index

        X[start_index:end_index, 0] = np.cos(2 * np.pi * ii / n_blobs) + blob_var * np.random.randn(nn) + blob_centers[ii-1][0]
        X[start_index:end_index, 1] = np.sin(2 * np.pi * ii / n_blobs) + blob_var * np.random.randn(nn) + blob_centers[ii-1][1]
    return X, Y


def two_moons(num_samples, moon_radius=2.0, moon_var=0.02):
    """
    Creates two intertwined moons

    :param num_samples: number of samples to create in the dataset
    :param moon_radius: radius of the moons
    :param moon_var:    variance of the moons
    :return: X,  (num_samples, 2) matrix of 2-dimensional samples
             Y,  (num_samples, ) vector of "true" cluster assignment
    """
    X = np.zeros((num_samples, 2))

    for i in range(int(num_samples / 2)):
        r = moon_radius + 4 * i / num_samples
        t = i * 3 / num_samples * np.pi
        X[i, 0] = r * np.cos(t)
        X[i, 1] = r * np.sin(t)
        X[i + int(num_samples / 2), 0] = r * np.cos(t + np.pi)
        X[i + int(num_samples / 2), 1] = r * np.sin(t + np.pi)

    X = X + np.sqrt(moon_var) * np.random.normal(size=(num_samples, 2))
    Y = np.ones(num_samples)
    Y[:int(num_samples / 2) + 1] = 0
    return [X, Y.astype(int)]


class GridSearch:

    def __init__(self, model, parameters, data, data_labels, verbose=0, data_type=None):
        """
        Grid Search initialization.

        :param model: classification model (SVM, logistic regression etc)
        :param parameters: list of dictionaries ; each dictionary corresponds to a set of parameters for each dataset
        :param data_type: string, if 'raw': replace data by list of indexes
        """
        self.model = model
        self.model_name = model.model_name
        self.parameters = parameters
        self.data = data
        self.data_type = data_type
        self.data_labels = data_labels
        self.n_pred = data['test'][0].shape[0]
        self.verbose = verbose
        self.results = []
        self.pred = np.zeros(3 * self.n_pred)

    def atomic_grid_search(self, i):
        # initialize grid search
        clf = GridSearchCV(self.model, self.parameters[i], cv=5, scoring='accuracy', verbose=self.verbose) #, njobs=-1)

        # run grid search
        if self.data_type == 'raw':
            # input indexes rather than raw data
            n = len(self.data_labels['train'][i])
            clf.fit(np.arange(n), self.data_labels['train'][i])
        else:
            clf.fit(self.data['train'][i], self.data_labels['train'][i])

        self.results.append(clf.cv_results_)
        self.model = self.model.set_params(**clf.best_params_)

        # final training
        if self.data_type == 'raw':
            self.model.fit(np.arange(n), self.data_labels['train'][i])
        else:
            self.model.fit(self.data['train'][i], self.data_labels['train'][i])

        # prediction
        Y_res = self.model.predict(self.data['test'][i], dataset='test')
        self.pred[i * self.n_pred:(i + 1) * self.n_pred] = Y_res.astype(np.int)

    def grid_search(self):
        for i in range(3):
            self.atomic_grid_search(i)

    def get_complete_results(self):
        return pd.DataFrame(self.results[0]), pd.DataFrame(self.results[1]), pd.DataFrame(self.results[2])

    def return_results(self):
        br0 = self.results[0]['mean_test_score'].max()
        br1 = self.results[1]['mean_test_score'].max()
        br2 = self.results[2]['mean_test_score'].max()
        print('Results set 1 = {}, set 2 = {}, set 3 = {}, avg = {}'.format(br0, br1, br2, (br0 + br1 + br2) / 3))

    def write_csv(self):
        filename = 'results/results_' + self.model_name + '.csv'

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['Id', 'Bound']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(self.pred.shape[0]):
                writer.writerow({'Id': i, 'Bound': np.int(self.pred[i].item())})

    def save_performances(self):
        """
        Save performances and best obtained parameters in a csv.
        """
        nb_datasets = len(self.results)
        resu = [[] for k in range(nb_datasets)]

        # fetch results
        for k in range(nb_datasets):
            best = np.argmax(self.results[k]['mean_test_score'])
            resu[k].append(('score', self.results[k]['mean_test_score'][best]))
            resu[k] = resu[k] + list(self.results[k]['params'][best].items())

        # write results in csv
        for k, resu in enumerate(resu):
            with open('results/final_results_{}.csv'.format(k), 'a') as file:
                writer = csv.writer(file)
                writer.writerow(resu)


def accuracy(y_true, y_computed):
    return (y_true == y_computed).sum()/len(y_true)
