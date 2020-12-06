#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer


def grid_search_LogReg(X, y, seed, cv=5, verbose=0, data_char='freq_noexp_k5'):
	'''
	Performs a cross validation grid search of LogisticRegression for different inverse of
	regularization strength values C. It computes the global accuracy, as well as the accuracy
	of each class. The learning and prediction time of each method is also stored. The results,
	as well as the associated plots, are saved into, respectively, a csv and a pdf file.

	Parameters
	----------
	X, y: the datapoints and associated labels
	seed: int, controls the pseudo random number generation for shuffling the data for probability estimates
	cv: int, number of cross-validation folds
	verbose: int, controls the verbosity: the higher, the more messages
	data_char: str, describes the dataset

	Returns
	-------
	df: panda DataFrame containing the cross-validation accuracies and time used to learn and predict
	'''
	# define the grid
	Cs = np.logspace(0, 3, 20)
	param_grid = {'C': Cs}

	# define the scoring functions
	scorings = {'accuracy': make_scorer(balanced_accuracy_score),
		'eukaryote_accuracy':make_scorer(euk_accuracy),
		'prokaryote_accuracy':make_scorer(pro_accuracy)}

	# perform the grid search
	lr = LogisticRegression(max_iter=10000, random_state=seed, n_jobs=n_jobs)
	grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=cv,
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
	df.to_csv('gs_results/LogReg_{}.csv'.format(data_char), index=False)

	# plot results
	suptitle = 'Logistic Regression with dataset characteristics {}'.format(data_char)
	axtitle = 'Accuracy and computation time with respect to regularization parameter C'
	figtitle = 'gs_plots/LogReg_{}.pdf'.format(data_char)
	plot_1param(df, 'C', suptitle, axtitle, figtitle, 'C')
	display(df)

	return df
