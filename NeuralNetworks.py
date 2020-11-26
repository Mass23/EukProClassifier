#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

def plot_NN(df):
	fig,ax = plt.subplots()

	ax.plot(df['hidden_layer_size'], df['mean test accuracy'], color="black", marker="o")
	ax.plot(df['hidden_layer_size'], df['mean_test_prokaryote_accuracy'], ':', color="red", marker="o")
	ax.plot(df['hidden_layer_size'], df['mean_test_eukaryote_accuracy'], ':', color="green", marker="o")
	ax.set_xlabel("Node number in the hidden layer",fontsize=14)
	ax.set_ylabel("Mean accuracy (10-fold CV)",color="black",fontsize=14)

	ax2=ax.twinx()
	ax2.plot(df['hidden_layer_size'], df['learning time'],color="blue",marker="o")
	ax2.set_ylabel("Computing time [sec]",color="blue",fontsize=14)
	fig.savefig('NN_n_nodes.pdf',bbox_inches='tight')

def grid_search_NN(X, y, seed, n_jobs=None, cv=5, verbose=None):
	'''
	Performs a cross validation grid search of MLP classifier
	for different number of nodes and hidden layers architecture. It computes
	the global accuracy, as well as the accuracy of each class. The learning time
	of each method is also stored.

	:return: panda DataFrame containing the cross-validation accuracy and the mean time used to learn
	'''
	# define the grids
	hidden_layer_size = [(50,), (100,), (150,), (200,), (250,), (300,), (350,), (400,), (450,), (500,)]

	# cannot have None value because of df.astype(int) ==> to handle
	param_grid = {'hidden_layer_sizes': hidden_layer_size}

	# define the scoring functions
	scorings = {'accuracy': make_scorer(accuracy_score),
			'eukaryote_accuracy':make_scorer(euk_accuracy),
			'prokaryote_accuracy':make_scorer(pro_accuracy)}

	# perform the grid search
	rf = MLPClassifier(random_state=seed, solver='adam', max_iter=500, n_jobs=n_jobs)
	grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv,
								scoring=scorings, refit='accuracy', verbose=verbose)
	grid_search.fit(X, y)

	# store the result in a dataframe
	df = pd.DataFrame(columns=['hidden_layer_size', 'max_depth', 'accuracy',
					'prokaryote accuracy', 'eukaryote accuracy', 'learning time'])
	for i, trial in enumerate(grid_search.cv_results_['params']):
		trial = grid_search.cv_results_['params'][i]
		# trial['n_estimators'] = int(trial['n_estimators'])
		# trial['max_depth'] = int(trial['max_depth']) if trial['max_depth'] else trial['max_depth']

		trial['learning time'] = grid_search.cv_results_['mean_fit_time'][i]
		trial['accuracy'] = grid_search.cv_results_['mean_test_accuracy'][i]
		trial['prokaryote accuracy'] = grid_search.cv_results_['mean_test_prokaryote_accuracy'][i]
		trial['eukaryote accuracy'] = grid_search.cv_results_['mean_test_eukaryote_accuracy'][i]

		df = df.append(trial, ignore_index=True)

	df['hidden_layer_size'] = df['hidden_layer_size'].astype(int)

	plot_NN(df)
	return df
