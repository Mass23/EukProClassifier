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


def plot_LogReg(df, figtitle='plots/LogReg.pdf'):
	Cs = df['C'].unique()

	fig, ax = plt.subplots(figsize=(15, 10))
	fig.suptitle('LogisticRegression accuracies for different penalization terms', fontsize=20)

	glo_acc = df['accuracy'].to_numpy()
	euk_acc = df['eukaryote accuracy'].to_numpy()
	pro_acc = df['procaryote accuracy'].to_numpy()
	time = df['learning time'].to_numpy()

	ax.plot(Cs, glo_acc, color="red", marker='o', label='Accuracy global')
	ax.plot(Cs, euk_acc, color="orange", marker="o", label='Accuracy on eukaryotes')
	ax.plot(Cs, pro_acc, color="yellow", marker="o", label='Accuracy on prokaryotes')
	ax.set_xlabel("penalization C",fontsize=14)
	ax.set_ylabel("Accuracy", fontsize=14)
	ax.legend()

	ax2 = ax.twinx()
	ax2.plot(Cs, time, color="blue", marker="^")
	ax2.set_ylabel("Learning time [sec]",color="blue", fontsize=14)

	plt.show()
	fig.savefig(figtitle, bbox_inches='tight')

def grid_search_LogReg(X, y, seed, n_jobs=None, cv=5, verbose=0, figtitle='plots/LogReg.pdf'):
	'''
	Performs a cross validation grid search of LogisticRegressions
	for different inverse of regularization strength values C. It computes
	the global accuracy, as well as the accuracy of each class. The learning time
	of each method is also stored.

	:return: panda DataFrame containing the cross-validation accuracies and the mean time used to learn
	'''
	# define the grid
	Cs = np.logspace(0, 3, 20)
	param_grid = {'C': Cs}

	# define the scoring functions
	scorings = {'accuracy': make_scorer(balanced_accuracy_score),
		'eukaryote_accuracy':make_scorer(euk_accuracy),
		'procaryote_accuracy':make_scorer(pro_accuracy)}

	# perform the grid search
	lr = LogisticRegression(max_iter=10000, random_state=seed, n_jobs=n_jobs)
	grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=cv,
								scoring=scorings, refit='accuracy', verbose=verbose)
	grid_search.fit(X, y)

	# store the result in a dataframe
	df = pd.DataFrame(columns=['C', 'accuracy', 'procaryote accuracy', 'eukaryote accuracy', 'learning time'])
	for i, trial in enumerate(grid_search.cv_results_['params']):
		trial = grid_search.cv_results_['params'][i]
		trial['learning time'] = grid_search.cv_results_['mean_fit_time'][i]
		trial['accuracy'] = grid_search.cv_results_['mean_test_accuracy'][i]
		trial['procaryote accuracy'] = grid_search.cv_results_['mean_test_procaryote_accuracy'][i]
		trial['eukaryote accuracy'] = grid_search.cv_results_['mean_test_eukaryote_accuracy'][i]

		df = df.append(trial, ignore_index=True)

	plot_LogReg(df, figtitle)
	return df
