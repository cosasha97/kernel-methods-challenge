import os
import numpy as np
from scripts.utils import *
from scripts.models import *

if __name__ == "__main__":
    # load data
    data = DataLoader()

    # best parameters
    parameters = [{'C': 0.12,
                   'seq_length_mm': [np.arange(7,13)],
                   'mismatch_nb':[[1, 2]],
                   'kernel_name': ['mismatch'],
                   'data_index': [0]},
                  {'C': 0.07,
                   'seq_length_sp': [3],
                   'seq_length_mm': [np.arange(8,14)],
                   'mismatch_nb':[[1, 2]],
                   'lambda_ss': [0.5],
                   'seq_length_ss': [np.arange(6, 12)],
                   'kernel_name': ['sum'],
                   'data_index': [1]},
                  {'C': 0.21,
                   'seq_length_mm': [np.arange(5,14)],
                   'mismatch_nb':[[1, 2]],
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
    # write predictions
    GS.write_csv()