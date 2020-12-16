#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to do a grid search over hyperparameters of different sklerarn methods:
– linear SVC : tune the penalization parameter C
– kernel SVM : tune the penalization parameter C and the kernel coefficient gamma
– logistic regression : tune the penalization parameter C
– random forest : tune the maximum depth and the forest size
– neural network : tune the number of nodes in the hidden layer
For each method, we use the dataset transformation chosen in the previous step.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import *
from grid_search import *


FILENAME = "../data/Counts_n10000_k5_s5000.csv"

seed = 63
cv = 5
verbose = 2

#######################################
y, X, _ = load_csv_data(FILENAME)
X_freq = FREQ_transform(X)
X_clr = CLR_transform(X_freq)

k_list = np.arange(start=X.shape[1]//16, stop=X.shape[1], step=X.shape[1]//16)
#######################################

# define X, y and data_char for the correct transformation for linSVC
grid_search_linSVC(X_clr, y, 'LinearSVC_CLR', cv, seed, verbose)

grid_search_SVC(X_clr, y, 'KernelSVC_CLR', cv, seed, verbose)

for k in k_list:
    print(k)
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(X_freq.T)
    kX = create_kmeans_data(X_freq, kmeans.labels_)
    grid_search_SVC(kX, y, 'KernelSVC_Kmeans_k={}'.format(k), cv, seed, verbose)
    grid_search_LogReg(kX, y, 'LogReg_Kmeans_k={}'.format(k), cv, seed, verbose)
    grid_search_linSVC(kX, y, 'LinearSVC_Kmeans_k={}'.format(k), cv, seed, verbose)

# define X, y and data_char for the correct transformation for RF
grid_search_RF(X_freq, y, 'RandomForest_Freq', cv, seed, verbose)

# define X, y and data_char for the correct transformation for NN
grid_search_NN(X_clr, y, 'NeuralNetworks_CLR', cv, seed, verbose)
