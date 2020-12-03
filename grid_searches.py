"""
Script used to do the grid search over hyperparameters of the different methods.
We use the dataset with [...tranformation...].
The methods that we are testing are linear SVC, kernel SVM, logistic regression,
random forest and neural network.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import *
from KernelSVM import *
from LinearSVC import *
from LogReg import *
from RandomForest import *
from NeuralNetworks import *


# y, X = final transform freq/clr/ilr + feature_expansion or not + k-mer=4/5/6
# data_char = 'clr_exp_k5' # characteristic describing the data used
# to change !!!
y, X, ids = load_csv_data("Counts_n10000_k5_s5000.csv")
data_char = 'freq_noexp_k5'
# to change !!!

seed = 42
cv = 5
verbose = 2

df = grid_search_linSVC(	X, y, seed=seed, cv=cv, verbose=verbose, data_char)
df = grid_search_SVC(X, y, seed=seed, cv=cv, verbose=verbose, data_char)
df = grid_search_LogReg(X, y, seed=seed, cv=cv, verbose=verbose, data_char)
df = grid_search_RF(X, y, seed=seed, cv=cv, verbose=verbose, data_char)
df = grid_search_NN(X, y, seed=seed, cv=cv, verbose=verbose, data_char)

