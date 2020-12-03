import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from helpers import *
from skbio.stats.composition import clr, ilr
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from time import time



seed = 42
verbose = 1
filename = "Counts_n10000_k5_s5000.csv"


### define the methods
lin_svc = svm.LinearSVC(random_state=seed)
ker_svc = svm.SVC(random_state=seed)
log_reg = LogisticRegression(max_iter=10000, random_state=seed)
rf = RandomForestClassifier(max_depth=10, random_state=seed)
nn = MLPClassifier(solver='adam', max_iter=500, random_state=seed)
methods = {'linear svc':lin_svc, 'kernel svc':ker_svc,
	'logistic regression':log_reg, 'random forest':rf, 'neural network':nn}

### define the datasets with different transformations
datas = {}

y, X_freq, _ = load_csv_data(filename)
X_train, X_test, y_train, y_test = train_test_split(X_freq, y, random_state=seed)
datas['freq'] = (X_train, X_test, y_train, y_test)

X_clr = clr(X_freq)
X_train, X_test, y_train, y_test = train_test_split(X_clr, y, random_state=seed)
datas['clr'] = (X_train, X_test, y_train, y_test)

X_ilr = ilr(X_freq)
X_train, X_test, y_train, y_test = train_test_split(X_ilr, y, random_state=seed)
datas['ilr'] = (X_train, X_test, y_train, y_test)



### store accuracies of different methods and transformations in panda dataframe
df = pd.DataFrame(columns=['method', 'transformation',
                           'accuracy', 'euk_acc', 'pro_acc',
                           'learning time', 'prediction time'])
for m in methods:
    for t in datas:
        if verbose:
            print('{} with {}'.format(m, t))

        X_train, X_test, y_train, y_test = datas[t]

        t1 = time()
        methods[m].fit(X_train, y_train)
        t2 = time()
        y_pred = methods[m].predict(X_test)
        t3 = time()

        bal_acc = balanced_accuracy_score(y_test, y_pred)
        euk_acc = euk_accuracy(y_test, y_pred)
        pro_acc = pro_accuracy(y_test, y_pred)

        df = df.append({'method':m, 'transformation':t,
               'accuracy':bal_acc, 'euk_acc':euk_acc, 'pro_acc':pro_acc,
               'learning time':(t2 - t1), 'prediction time':(t3 - t2)}
               , ignore_index=True)

df.to_csv('1_counts.csv', index=False)
