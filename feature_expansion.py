#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to compare the performance of different sklearn estimators
with respect to different polynomial feature expansions. We use the sklearn
PolynomailFeature method to perform polynomial feature expansions until degree max_degree
We use the raw dataset with k=5.
"""

import pandas as pd

from helpers import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from time import time


seed = 42
verbose = 1
max_degree = 3
filename = "Counts_n10000_k5_s5000.csv"

# define the methods
lin_svc = LinearSVC(random_state=seed)
svc = SVC(random_state=seed)
log_reg = LogisticRegression(max_iter=10000, random_state=seed)
rf = RandomForestClassifier(max_depth=10, random_state=seed)
nn = MLPClassifier(solver='adam', max_iter=500, random_state=seed)
methods = {'linear svc':lin_svc, 'rbf svc':svc,
    'logistic regression':log_reg, 'random forest':rf, 'neural network':nn}

# define the datasets with different feature expansions
y, X, _ = load_csv_data(filename)
datas = dict()
for i in range(0, max_degree + 1):
    poly = PolynomialFeatures(i)
    new_X = poly.fit_transform(X)
    datas[i] = new_X

# store accuracies of different methods and transformations in panda dataframe
df = pd.DataFrame(columns=['method', 'degree', 'accuracy', 'euk_acc', 'pro_acc',
                           'learning time', 'prediction time'])
for m in methods:
    clf = clone(methods[m])
    for deg in datas:
        if verbose:
            print("Testing feature expansion for {} with degree {}".format(m, deg))

        X_p = datas[deg]
        X_train, X_test, y_train, y_test = train_test_split(X_p, y, random_state=seed)

        t1 = time()
        clf.fit(X_train, y_train)
        t2 = time()
        y_pred = clf.predict(X_test)
        t3 = time()

        bal_acc = balanced_accuracy_score(y_test, y_pred)
        euk_acc = euk_accuracy(y_test, y_pred)
        pro_acc = pro_accuracy(y_test, y_pred)

        df = df.append({'method' : m, 'degree' : deg,
                'accuracy' : bal_acc, 'euk_acc' : euk_acc, 'pro_acc' : pro_acc,
                'learning time' : (t2-t1), 'prediction time' : (t3 - t2)}
                , ignore_index = True)

df.to_csv('2_feature_expansion.csv', index=False)