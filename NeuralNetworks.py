#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer


def grid_search_NN(X, y, seed, cv=5, verbose=0, data_char='freq_noexp_k5'):
	'''
	Performs a cross validation grid search of MLP classifier for different number
	of nodes and hidden layers architecture. It computes the global accuracy, as well as the accuracy
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
	hidden_layer_size = [(50,), (100,), (150,), (200,), (250,), (300,), (350,), (400,), (450,), (500,)]
	param_grid = {'hidden_layer_sizes': hidden_layer_size}

	# define the scoring functions
	scorings = {'accuracy': make_scorer(balanced_accuracy_score),
			'eukaryote_accuracy':make_scorer(euk_accuracy),
			'prokaryote_accuracy':make_scorer(pro_accuracy)}

	# perform the grid search
	nn = MLPClassifier(random_state=seed, solver='adam', max_iter=500)
	grid_search = GridSearchCV(estimator=nn, param_grid=param_grid, cv=cv,
								scoring=scorings, refit='accuracy', verbose=verbose)
	grid_search.fit(X, y)

	# store the result in a dataframe
	df = pd.DataFrame(columns=['hidden_layer_size',
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

	df['hidden_layer_size'] = df['hidden_layer_size'].astype(int)

	# save dataframe
	df.to_csv('gs_results/NN_{}.csv'.format(data_char), index=False)

	# plot results
	suptitle = 'Neural Network with dataset characteristics {}'.format(data_char)
	axtitle = 'Accuracy and computation time with respect to hidden layer size'
	figtitle = 'gs_plots/NN_{}.pdf'.format(data_char)
	plot_1param(df, 'hidden_layer_size', suptitle, axtitle, figtitle, "Number of nodes in the hidden layer")
	display(df)

	return df
