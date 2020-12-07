#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to perform cross-validation grid searches.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def grid_search(X, y, estimator, param_grid, cv, verbose=0):
	"""
	Performs a cross validation grid search of the given estimator for the given parameter
	grid param_grid. It computes the global accuracy, as well as the accuracy of each class.
	The learning and prediction time of each method is also stored. The results,

	Parameters
	----------
	X, y: the datapoints and associated labels
	estimator: estimator object, the estimator to perform the grid search with
	param_grid: dict, dictionnary with paramaeters names (str) as keys and lists of
		parameter setting to try as values
	cv: int, number of cross-validation folds
	verbose: int, controls the verbosity: the higher, the more messages

	Returns
	-------
	df: panda DataFrame containing the cross-validation accuracies and time used to learn and predict
	"""
	# define the scoring functions
	scorings = {'accuracy': make_scorer(balanced_accuracy_score),
			'eukaryote_accuracy':make_scorer(euk_accuracy),
			'prokaryote_accuracy':make_scorer(pro_accuracy)}

	# perform the grid_search
	grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv,
								scoring=scorings, refit='accuracy', verbose=verbose)
	grid_search.fit(X, y)

	# store the result in a dataframe
	df = pd.DataFrame(columns=list(param_grid.keys()) +
						['accuracy',	'eukaryote accuracy', 'prokaryote accuracy',
						'learning time', 'prediction time'])
	for i, trial in enumerate(grid_search.cv_results_['params']):
		trial = grid_search.cv_results_['params'][i]
		trial['accuracy'] = grid_search.cv_results_['mean_test_accuracy'][i]
		trial['eukaryote accuracy'] = grid_search.cv_results_['mean_test_eukaryote_accuracy'][i]
		trial['prokaryote accuracy'] = grid_search.cv_results_['mean_test_prokaryote_accuracy'][i]
		trial['learning time'] = grid_search.cv_results_['mean_fit_time'][i]
		trial['prediction time'] = grid_search.cv_results_['mean_score_time'][i]

		df = df.append(trial, ignore_index=True)

	return df

def grid_search_linSVC(X, y, data_char, cv, seed, verbose=0):
	"""
	Performs a cross validation grid search of LinearSVC for different values
	of the penalization parameter C. The results, as well as the associated
	plots, are saved into, respectively, a csv and a pdf file.

	Parameters
	----------
	X, y: the datapoints and associated labels
	data_char: str, describes the dataset
	cv: int, number of cross-validation folds
	seed: int, controls the pseudo random number generation for shuffling the data for probability estimates
	verbose: int, controls the verbosity: the higher, the more messages
	"""
	# define the estimator
	lin_svc = LinearSVC(random_state=seed, max_iter=10000)

	# define the grid
	c_range = np.logspace(-2, 3, 20)
#######################################
	c_range = np.logspace(0, 3, 2)
#######################################
	param_grid = {'C': c_range}

	# get the grid search results
	df = grid_search(X, y, lin_svc, param_grid, cv, verbose)

	# save dataframe
	df.to_csv('../output/{}_linSVC.csv'.format(data_char), index=False)

	# plot results
	suptitle = 'Linear SVC with dataset characteristics {}'.format(data_char)
	axtitle = 'Accuracy and computation time with respect to regularization parameter C'
	figtitle = '../output/plots/{}_linSVC.pdf'.format(data_char)
	plot_1param(df, 'C', suptitle, axtitle, figtitle, 'C')


def grid_search_SVC(X, y, data_char, cv, seed, verbose=0):
	"""
	Performs a cross validation grid search of SVC for different values of the parameters
    C and gamma. The results, as well as the associated plots, are saved into, respectively,
    a csv and a pdf file.

	Parameters
	----------
	X, y: the datapoints and associated labels
	data_char: str, describes the dataset
	cv: int, number of cross-validation folds
	seed: int, controls the pseudo random number generation for shuffling the data for probability estimates
	verbose: int, controls the verbosity: the higher, the more messages
	"""
	# define the estimator
	svc = SVC(random_state=seed)

	# define the grid
	c_range = [0.01, 0.1, 10, 100]
	gamma_range = [0.001, 0.01, 0.1, 1, 10, 100]
#######################################
	c_range = [0.1, 10]
	gamma_range = [1, 10]
#######################################
	param_grid = {'C': c_range, 'gamma':gamma_range}

	# get the grid search results
	df = grid_search(X, y, svc, param_grid, cv, verbose)

	# save dataframe
	df.to_csv('../output/{}_SVM.csv'.format(data_char), index=False)

	# plot results
	suptitle = 'Support Vector Machine with dataset characteristics {}\n \
	            Accuracy and computation time with respect to gamma'.format(data_char)
	axtitle = 'SVM with C = {}'
	figtitle = '../output/plots/{}_SVM.pdf'.format(data_char)
	plot_2param(df, 'gamma', 'C', suptitle, axtitle, figtitle, 'Gamma')


def grid_search_LogReg(X, y, data_char, cv, seed, verbose=0):
	"""
	Performs a cross validation grid search of LogisticRegression for different inverse of
	regularization strength values C.  The results, as well as the associated plots, are
	saved into, respectively, a csv and a pdf file.

	Parameters
	----------
	X, y: the datapoints and associated labels
	data_char: str, describes the dataset
	cv: int, number of cross-validation folds
	seed: int, controls the pseudo random number generation for shuffling the data for probability estimates
	verbose: int, controls the verbosity: the higher, the more messages
	"""
	# define the estimator
	lr = LogisticRegression(max_iter=10000, random_state=seed)

	# define the grid
	Cs = np.logspace(0, 3, 20)
#######################################
	Cs = np.logspace(0, 3, 2)
#######################################
	param_grid = {'C': Cs}

	# get the grid search results
	df = grid_search(X, y, lr, param_grid, cv, verbose)

	# save dataframe
	df.to_csv('../output/{}_LogReg.csv'.format(data_char), index=False)

	# plot results
	suptitle = 'Logistic Regression with dataset characteristics {}'.format(data_char)
	axtitle = 'Accuracy and computation time with respect to regularization parameter C'
	figtitle = '../output/plots/{}_LogReg.pdf'.format(data_char)
	plot_1param(df, 'C', suptitle, axtitle, figtitle, 'C')


def grid_search_RF(X, y, data_char, cv, seed, verbose=0):
	"""
	Performs a cross validation grid search of RandomForestClassifier for different number
	of trees of different maximum depth. The results, as well as the associated plots, are
	saved into, respectively, a csv and a pdf file.

	Parameters
	----------
	X, y: the datapoints and associated labels
	data_char: str, describes the dataset
	cv: int, number of cross-validation folds
	seed: int, controls the pseudo random number generation for shuffling the data for probability estimates
	verbose: int, controls the verbosity: the higher, the more messages
	"""
	# define the estimator
	rf = RandomForestClassifier(random_state=seed)

	# define the grid
	nb_trees = [20, 80, 100, 150, 200]
	depths = [5, 10, 15, 20, 35, 50]
#######################################
	nb_trees = [5, 10]
	depths = [5, 7]
#######################################
	param_grid = {'n_estimators': nb_trees, 'max_depth':depths}

	# get the grid search results
	df = grid_search(X, y, rf, param_grid, cv, verbose)
	df['n_estimators'] = df['n_estimators'].astype(int)
	df['max_depth'] = df['max_depth'].astype(int)

	# save dataframe
	df.to_csv('../output/{}_RF.csv'.format(data_char), index=False)

	# plot results
	suptitle = 'Random Forest with dataset characteristics {}'.format(data_char) \
			+ '\nAccuracy and computation time with respect to maximum tree depth'
	axtitle = 'Forest of size {}'
	figtitle = '../output/plots/{}_RF.pdf'.format(data_char)
	plot_2param(df, 'max_depth', 'n_estimators', suptitle, axtitle, figtitle, 'Maximum depth')


def grid_search_NN(X, y, data_char, cv, seed, verbose=0):
	"""
	Performs a cross validation grid search of MLP classifier for different number
	of nodes and hidden layers architecture. The results, as well as the associated
	plots, are saved into, respectively, a csv and a pdf file.

	Parameters
	----------
	X, y: the datapoints and associated labels
	data_char: str, describes the dataset
	cv: int, number of cross-validation folds
	seed: int, controls the pseudo random number generation for shuffling the data for probability estimates
	verbose: int, controls the verbosity: the higher, the more messages
	"""
	# define the estimator
	nn = MLPClassifier(random_state=seed, solver='adam', max_iter=500)

	# define the grid
	hidden_layer_size = [(50,), (100,), (150,), (200,), (250,), (300,), (350,), (400,), (450,), (500,)]
#######################################
	hidden_layer_size = [(5,), (10,)]
#######################################
	param_grid = {'hidden_layer_sizes': hidden_layer_size}

	# get the grid search results
	df = grid_search(X, y, nn, param_grid, cv, verbose)
	df['hidden_layer_sizes'] = list(map(lambda x:x[0], df['hidden_layer_sizes']))

	# save dataframe
	df.to_csv('../output/{}_NN.csv'.format(data_char), index=False)

	# plot results
	suptitle = 'Neural Network with dataset characteristics {}'.format(data_char)
	axtitle = 'Accuracy and computation time with respect to hidden layer size'
	figtitle = '../output/plots/{}_NN.pdf'.format(data_char)
	plot_1param(df, 'hidden_layer_sizes', suptitle, axtitle, figtitle, "Number of nodes in the hidden layer")
