#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import *
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer


def grid_search_SVC(X, y, seed, n_jobs=None, cv=5, verbose=0,  data_char='freq_noexp_k5'):
    '''
    Performs a cross validation grid search of SVC for different values of
    parameters C and gamma. It computes the global accuracy, as well as the
    accuracy of each class. The learning time of each method is also stored.

    :return: panda DataFrame containing the cross-validation accuracy and the mean time used to learn
    '''
    # define the ranges
    c_range = np.logspace(0, 10, num=5)
    gamma_range = np.logspace(0, 10, num=10)
    param_grid = {'C': c_range, 'gamma':gamma_range}

    # define the scoring functions
    scorings = {'accuracy': make_scorer(balanced_accuracy_score),
            'eukaryote_accuracy':make_scorer(euk_accuracy),
            'prokaryote_accuracy':make_scorer(pro_accuracy)}

    # grid search
    svc = SVC(random_state=seed)
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=cv,
                                scoring=scorings, refit='accuracy', verbose=verbose)
    grid_search.fit(X, y)

    # store the results in a dataframe
    df = pd.DataFrame(columns=['C', 'gamma',
                    'accuracy', 'eukaryote accuracy', 'prokaryote accuracy',
                    'learning time', 'prediction time'])
    for i, trial in enumerate(grid_search.cv_results_['params']):
        trial = grid_search.cv_results_['params'][i]
        trial['accuracy'] = grid_search.cv_results_['mean_test_accuracy'][i]
        trial['eukaryote accuracy'] = grid_search.cv_results_['mean_test_eukaryote_accuracy'][i]
        trial['prokaryote accuracy'] = grid_search.cv_results_['mean_test_prokaryote_accuracy'][i]
        trial['learning time'] = grid_search.cv_results_['mean_fit_time'][i]
        trial['prediction time'] = grid_search.cv_results_['mean_score_time'][i]

        df = df.append(trial, ignore_index=True)

    # save dataframe
    df.to_csv('gs_results/SVM_{}.csv'.format(data_char), index=False)

    # plot results
    suptitle = 'Support Vector Machine with dataset characteristics {}\n \
                Accuracy and computation time with respect to gamma'.format(data_char)
    axtitle = 'SVM with C = {}'
    figtitle = 'gs_plots/SVM_{}.pdf'.format(data_char)
    plot_2param(df, 'gamma', 'C', suptitle, axtitle, figtitle, 'Gamma')
    display(df)

    return df
