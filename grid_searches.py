#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to do a grid search over hyperparameters of different sklerarn methods:
– linear SVC : tune the penalization parameter C
– kernel SVM : tune the penalization parameter C and the kernel coefficient gamma
– logistic regression : tune the penalization parameter C
– random forest : tune the maximum depth and the forest size
– neural network : tune the number of nodes in the hidden layer
We use the dataset with [...chosen tranformation in step 1, 2 and 3...].
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


# y, X = final transform freq/clr/ilr + exp/noexp + k4/k5/k6
# data_char = 'clr_exp_k5'

seed = 63
cv = 5
verbose = 2

df = grid_search_linSVC(X, y, seed=seed, cv=cv, verbose=verbose, data_char)
df = grid_search_SVC(	X, y, seed=seed, cv=cv, verbose=verbose, data_char)
df = grid_search_LogReg(X, y, seed=seed, cv=cv, verbose=verbose, data_char)
df = grid_search_RF(	X, y, seed=seed, cv=cv, verbose=verbose, data_char)
df = grid_search_NN(	X, y, seed=seed, cv=cv, verbose=verbose, data_char)

