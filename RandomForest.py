#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

def plot(df):
	depths = df['max_depth'].unique()
	nb_trees = df['n_estimators'].unique()

	fig, axs = plt.subplots(len(nb_trees), figsize=(1+len(depths), 4*len(nb_trees)), constrained_layout=True)
	fig.suptitle('RandomForest accuracies for different forest sizes and different maximum tree depths', fontsize=20)

	for i, n in enumerate(nb_trees):
		axs[i].set_title('RandomForest of size {}'.format(n), fontsize=16)

		glo_acc = df.loc[df['n_estimators'] == n]['accuracy'].to_numpy()
		euk_acc = df.loc[df['n_estimators'] == n]['eukaryote accuracy'].to_numpy()
		pro_acc = df.loc[df['n_estimators'] == n]['procaryote accuracy'].to_numpy()
		time = df.loc[df['n_estimators'] == n]['learning time'].to_numpy()

		axs[i].plot(depths, glo_acc, color="red", marker='o', label='Accuracy global')
		axs[i].plot(depths, euk_acc, color="orange", marker="o", label='Accuracy on eukaryotes')
		axs[i].plot(depths, pro_acc, color="yellow", marker="o", label='Accuracy on prokaryotes')
		axs[i].set_xlabel("Maximum depth",fontsize=14)
		axs[i].set_ylabel("Accuracy", fontsize=14)

		ax2 = axs[i].twinx()
		ax2.plot(depths, time, color="blue", marker="^")
		ax2.set_ylabel("Learning time [sec]",color="blue",fontsize=14)

		axs[i].legend()

	plt.show()
	fig.savefig('RF_size_depth.pdf', bbox_inches='tight')

def grid_search_RF(X, y, seed, n_jobs=None, cv=5, verbose=None):
	'''
	Performs a cross validation grid search of RandomForestClassifiers
	for different number of trees of different maximum depth. It computes
	the global accuracy, as well as the accuracy of each class. The learning time
	of each method is also stored.

	:return: panda DataFrame containing the cross-validation accuracy and the mean time used to learn
	'''
	# define the grids
	nb_trees = [20, 80, 100, 150, 200]
	depths = [5, 10, 15, 20, 35, 50]
# cannot have None value because of df.astype(int) ==> to handle
	# nb_trees = [10, 20]
	# depths = [5, 7]
	param_grid = {'n_estimators': nb_trees, 'max_depth':depths}

	# define the scoring functions
	scorings = {'accuracy': make_scorer(accuracy_score),
			'eukaryote_accuracy':make_scorer(euk_accuracy),
			'procaryote_accuracy':make_scorer(pro_accuracy)}

	# perform the grid search
	rf = RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
	grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv,
								scoring=scorings, refit='accuracy', verbose=verbose)
	grid_search.fit(X, y)

	# store the result in a dataframe
	df = pd.DataFrame(columns=['n_estimators', 'max_depth', 'accuracy',
					'procaryote accuracy', 'eukaryote accuracy', 'learning time'])
	for i, trial in enumerate(grid_search.cv_results_['params']):
		trial = grid_search.cv_results_['params'][i]
		# trial['n_estimators'] = int(trial['n_estimators'])
		# trial['max_depth'] = int(trial['max_depth']) if trial['max_depth'] else trial['max_depth']

		trial['learning time'] = grid_search.cv_results_['mean_fit_time'][i]
		trial['accuracy'] = grid_search.cv_results_['mean_test_accuracy'][i]
		trial['procaryote accuracy'] = grid_search.cv_results_['mean_test_procaryote_accuracy'][i]
		trial['eukaryote accuracy'] = grid_search.cv_results_['mean_test_eukaryote_accuracy'][i]

		df = df.append(trial, ignore_index=True)

	df['n_estimators'] = df['n_estimators'].astype(int)
	df['max_depth'] = df['max_depth'].astype(int)

	plot(df)
	return df
