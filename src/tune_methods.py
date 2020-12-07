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
#######################################
cv = 2
#######################################
verbose = 2


### to adapt to final transform
y, X, _ = load_csv_data(FILENAME)
X, y = X[:100], y[:100]
data_char = 'freq_k5'

grid_search_linSVC(X, y, data_char, cv, seed, verbose)

grid_search_SVC(X, y, data_char, cv, seed, verbose)

grid_search_LogReg(X, y, data_char, cv, seed, verbose)

grid_search_RF(X, y, data_char, cv, seed, verbose)

grid_search_NN(X, y, data_char, cv, seed, verbose)
