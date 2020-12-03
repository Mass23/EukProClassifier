#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import *
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer

def grid_search_linSVC(X, y, seed, cv=5, verbose=0, data_char='freq_noexp_k5'):
    '''
    Performs a cross validation grid search of Linear SVC
    for different values of parameter C. It computes
    the global accuracy, as well as the accuracy of each class. The learning time
    of each method is also stored.

    :return: panda DataFrame containing the cross-validation accuracies
            and the mean time used to learn and predict
    '''
    # define the grid
    c_range = np.logspace(-2, 10, 10)
    param_grid = {'C': c_range}

    # define the scoring functions
    scorings = {'accuracy': make_scorer(balanced_accuracy_score),
            'eukaryote_accuracy':make_scorer(euk_accuracy),
            'prokaryote_accuracy':make_scorer(pro_accuracy)}

    # perform the grid search
    svc = LinearSVC(random_state=seed, max_iter=10000)
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=cv,
                               scoring=scorings, refit='accuracy', verbose=verbose)
    grid_search.fit(X, y)

    # store the result in a dataframe
    df = pd.DataFrame(columns=['C',
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
    df.to_csv('gs_results/linSVC_{}.csv'.format(data_char), index=False)

    # plot results
    suptitle = 'Linear SVC with dataset characteristics {}'.format(data_char)
    axtitle = 'Accuracy and computation time with respect to regularization parameter C'
    figtitle = 'gs_plots/linSVC_{}.pdf'.format(data_char)
    plot_1param(df, 'C', suptitle, axtitle, figtitle, 'C')
    display(df)

    return df
