#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to compare the performance of different sklearn estimators
with respect to three differents metrics :
– the frequencies of each k-mer
– the clr transformation of the frequencies
– the K-means transformation of the frequencies
We use the raw dataset with k = 5. The accuracies and timings are
measured using cross validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from helpers import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer
from sklearn.base import clone

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from time import time


FILENAME = "../data/Counts_n10000_k5_s5000.csv"

seed = 42
k_mean_seed = 27
cv = 5
############################################
cv = 2
############################################
verbose = 1

# define the methods
euk_rep = LinearSVC(C=100, max_iter=10000, random_state=seed)
lin_svc = LinearSVC(max_iter=10000, random_state=seed)
ker_svc = SVC(max_iter=10000, random_state=seed)
log_reg = LogisticRegression(C=1000, max_iter=10000, random_state=seed) # unpenalized (big C)
rf = RandomForestClassifier(max_depth=10, random_state=seed)
nn = MLPClassifier(solver='adam', max_iter=10000, random_state=seed)
methods = {'EukRep': euk_rep, 'linear svc':lin_svc, 'kernel svc':ker_svc,
	'logistic regression':log_reg, 'random forest':rf, 'neural network':nn}

# define the scoring functions
scorings = {'accuracy': make_scorer(balanced_accuracy_score),
		'eukaryote_accuracy':make_scorer(euk_accuracy),
		'prokaryote_accuracy':make_scorer(pro_accuracy)}

# define the datasets with different transformations
datas = {}
y, X, _ = load_csv_data(FILENAME)
############################################
X, y = X[:100], y[:100]
methods = {'linear svc':lin_svc}
############################################
X_freq = FREQ_transform(X)
X_clr = CLR_transform(X_freq)
datas['freq'] = (X_freq, y)
datas['clr'] = (X_clr, y)
datas['k-means'] = (X_freq, y)

# store accuracies of different methods and transformations in panda dataframe
df = pd.DataFrame(columns=['method', 'transformation', 'accuracy', 'euk_acc', 'pro_acc',
				'learning time', 'prediction time'])
for m in methods:
	clf = clone(methods[m])
	for t in datas:
		if verbose:
			print('Testing count transformation for {} with {}'.format(m, t))

		if t == 'k-means':
			X, y = datas[t]
			res = kmeans_optimisation(X, y, clf, scoring=scorings, cv=cv, seed=k_mean_seed, verbose=verbose)
			best_k, bal_acc, euk_acc, pro_acc, learn_time, predi_time = res
			transfo = t + '(k=' + str(best_k) + ')'
		else:
			X, y = datas[t]
			scores = cross_validate(clf, X, y, cv=cv, verbose=verbose, scoring=scorings)
			res = {i:np.mean(scores[i]) for i in scores.keys()}

			bal_acc = res['test_accuracy']
			euk_acc = res['test_eukaryote_accuracy']
			pro_acc = res['test_prokaryote_accuracy']
			learn_time = res['fit_time']
			predi_time = res['score_time']
			transfo = t

			if verbose:
				print('      - accuracy=' + str(bal_acc))

		df = df.append({'method':m, 'transformation':transfo,
			'accuracy':bal_acc, 'euk_acc':euk_acc, 'pro_acc':pro_acc,
			'learning time':learn_time, 'prediction time':predi_time}
			, ignore_index=True)

df.to_csv('../output/transform.csv', index=False)
