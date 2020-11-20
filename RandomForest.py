#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer


def euk_accuracy(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    class_ac = matrix.diagonal() / matrix.sum(axis=1)
    return class_ac[1]

def pro_accuracy(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    class_ac = matrix.diagonal() / matrix.sum(axis=1)
    return class_ac[0]

def grid_search_RF(X, y, seed, n_jobs, cv=5, verbose=None):
	'''
	Performs a cross validation grid search of RandomForestClassifiers
	for different number of trees of different maximum depth. It computes
	the global accuracy, as well as the accuracy of each class. The learning time
	of each method is also stored.

	:return: panda DataFrame containing the cross-validation accuracy and the mean time used to learn
	'''
	# define the grids and the scoring functions
	nb_trees = [10, 100, 500, 1000]
	depths = [5, 10, 15, 20, None]
	param_grid = {'n_estimators': nb_trees, 'max_depth':depths}
	scorings = {'accuracy': make_scorer(accuracy_score),
			'eukaryote_accuracy':make_scorer(euk_accuracy),
			'procaryote_accuracy':make_scorer(pro_accuracy)}

	# perform the grid search
	rf = RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
	grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv,
								scoring=scorings, refit='accuracy', verbose=verbose)
	grid_search.fit(X, y)

	# store the result
	df = pd.DataFrame(columns=['n_estimators', 'max_depth', 'accuracy',
					'procaryote accuracy', 'eukaryote accuracy', 'learning time'])
	for i, trial in enumerate(grid_search.cv_results_['params']):
		trial = results['params'][i]
		trial['learning time'] = grid_search.cv_results_['mean_fit_time'][i]
		trial['accuracy'] = grid_search.cv_results_['mean_test_accuracy'][i]
		trial['procaryote accuracy'] = grid_search.cv_results_['mean_test_procaryote_accuracy'][i]
		trial['eukaryote accuracy'] = grid_search.cv_results_['mean_test_eukaryote_accuracy'][i]
		df = df.append(trial, ignore_index=True)

	df['n_estimators'] = df['n_estimators'].astype(int)
	df['max_depth'] = df['max_depth'].astype(int)

	return df
