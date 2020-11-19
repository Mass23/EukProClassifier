#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from helpers import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from time import time
import pydot


seed = 42

# load data
y, X, ids = load_csv_data("Counts_n10000_k5_s5000.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)


# visualisation of 1 small tree
def visualisation():
	rf_small = RandomForestClassifier(n_estimators=10, max_depth=3)
	rf_small.fit(X_train, y_train)
	tree_small = rf_small.estimators_[5]
	export_graphviz(tree_small, out_file = 'small_tree.dot', rounded=True, precision=1)
	(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
	graph.write_png('small_tree.png')

def grid_search(nb_trees, depths):
	for d in depths:
		for n in nb_trees:
			print('Forest of {n} trees of maximum depth {d}'.format(n=n, d=d))

			# learning
			rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=seed)
			t1 = time()
			rf.fit(X_train, y_train);
			t2 = time()
			print('\tlearning time =', round(t2 - t1, 5))

			# prediction
			y_pred = rf.predict(X_test)

			# error
			accuracy = accuracy_score(y_test, y_pred)
			print('\taccuracy =', round(accuracy, 5))

nb_trees = [100, 500, 1000]
depths = [7, 10, 13]

grid_search(nb_trees, depths)

# print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Eukaryote', 'Prokaryote']))


