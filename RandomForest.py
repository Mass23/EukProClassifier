#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer


def grid_search_RF(X, y, seed, cv=5, verbose=0, data_char='freq_noexp_k5'):
	'''
	Performs a cross validation grid search of RandomForestClassifiers for different number
	of trees of different maximum depth. It computes the global accuracy, as well as the accuracy
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
	# define the grids
	nb_trees = [20, 80, 100, 150, 200]
	depths = [5, 10, 15, 20, 35, 50]
	param_grid = {'n_estimators': nb_trees, 'max_depth':depths}

	# define the scoring functions
	scorings = {'accuracy': make_scorer(balanced_accuracy_score),
			'eukaryote_accuracy':make_scorer(euk_accuracy),
			'prokaryote_accuracy':make_scorer(pro_accuracy)}

	# perform the grid search
	rf = RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
	grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv,
								scoring=scorings, refit='accuracy', verbose=verbose)
	grid_search.fit(X, y)

	# store the result in a dataframe
	df = pd.DataFrame(columns=['n_estimators', 'max_depth',
						'accuracy',	'eukaryote accuracy', 'prokaryote accuracy',
						'learning time', 'prediction time'])
	for i, trial in enumerate(grid_search.cv_results_['params']):
		trial = grid_search.cv_results_['params'][i]
		trial['accuracy'] = grid_search.cv_results_['mean_test_accuracy'][i]
		trial['eukaryote accuracy'] = grid_search.cv_results_['mean_test_eukaryote_accuracy'][i]
		trial['prokaryote accuracy'] = grid_search.cv_results_['mean_test_prokaryote_accuracy'][i]
		trial['learning time'] = grid_search.cv_results_['mean_fit_time'][i]
		trial['prediction time'] = grid_search.cv_results_['mean_score_time'][i]

		df = df.append(trial, ignore_index=True)

	df['n_estimators'] = df['n_estimators'].astype(int)
	df['max_depth'] = df['max_depth'].astype(int)

	# save dataframe
	df.to_csv('gs_results/RF_{}.csv'.format(data_char), index=False)

	# plot results
	suptitle = 'Random Forest with dataset characteristics {}'.format(data_char) \
			+ '\nAccuracy and computation time with respect to maximum tree depth'
	axtitle = 'Forest of size {}'
	figtitle = 'gs_plots/RF_{}.pdf'.format(data_char)
	plot_2param(df, 'max_depth', 'n_estimators', suptitle, axtitle, figtitle, 'Maximum depth')
	display(df)

	return df
