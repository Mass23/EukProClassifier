#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to compare the performance of different sklearn estimators
with respect to three differents metrics :
– the frequencies of each k-mer
– the clr transformation of the frequencies
– the K-means transformation of the frequencies
We use the raw dataset with k = 5.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from helpers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer

from time import time


seed = 42
k_mean_seed = 27
verbose = 1
filename = "Counts_n10000_k5_s5000.csv"


# define the methods
euk_rep = svm.LinearSVC(C=100, max_iter=10000, random_state=seed)
lin_svc = svm.LinearSVC(max_iter=10000, random_state=seed)
ker_svc = svm.SVC(max_iter=10000, random_state=seed)
log_reg = LogisticRegression(C=1000, max_iter=10000, random_state=seed) # unpenalized (big C)
rf = RandomForestClassifier(max_depth=10, random_state=seed)
nn = MLPClassifier(solver='adam', max_iter=10000, random_state=seed)
methods = {'EukRep': euk_rep, 'linear svc':lin_svc, 'kernel svc':ker_svc,
	'logistic regression':log_reg, 'random forest':rf, 'neural network':nn}

# define the datasets with different transformations
datas = {}

y, X, _ = load_csv_data(filename)

#to rm
X, y = X[:100], y[:100]
methods = {'linear svc':lin_svc}


X_freq = FREQ_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_freq, y, random_state=seed)
datas['freq'] = (X_train, X_test, y_train, y_test)

X_clr = CLR_transform(X_freq)
X_train, X_test, y_train, y_test = train_test_split(X_clr, y, random_state=seed)
datas['clr'] = (X_train, X_test, y_train, y_test)

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
			res = kmeans_optimisation(X, y, clf, k_mean_seed, seed, verbose)
			# df = df.append({'method':m, 'transformation': t + '(k=' + str(res[0]) + ')',
			#'accuracy':res[1], 'euk_acc':res[2], 'pro_acc':res[2],
			#'learning time':res[3], 'prediction time':res[4]}
		#, ignore_index=True)
			best_k, bal_acc, euk_acc, pro_acc, learn_time, predi_time = res
			transfo = t + '(k=' + str(best_k) + ')'
		else:
			X_train, X_test, y_train, y_test = datas[t]
			t1 = time()
			clf.fit(X_train, y_train)
			t2 = time()
			y_pred = clf.predict(X_test)
			t3 = time()

			bal_acc = balanced_accuracy_score(y_test, y_pred)
			print('      - accuracy=' + str(bal_acc))
			euk_acc = euk_accuracy(y_test, y_pred)
			pro_acc = pro_accuracy(y_test, y_pred)

			transfo = t
			learn_time = t2 - t1
			predi_time = t3 - t2

		df = df.append({'method':m, 'transformation':transfo,
			'accuracy':bal_acc, 'euk_acc':euk_acc, 'pro_acc':pro_acc,
			'learning time':learn_time, 'prediction time':predi_time}
			, ignore_index=True)

df.to_csv('Compare_counts.csv', index=False)
