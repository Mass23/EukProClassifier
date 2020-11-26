import numpy as np
from sklearn.metrics import confusion_matrix

def load_csv_data(data_path, n_min = 1000):
    """Loads data and returns y (class labels), X (features) and ids (event ids)"""
    print('Loading data...')
    labels = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = data[:, 0].astype(np.int)
    X = data[:, 2:]

    # convert class labels from strings to binary (0,1)
    y = np.ones(len(labels))
    y[np.where(labels=='Prokaryote')] = 0

    # Remove rows having less than n_min count and change counts to frequencies
    print('Removing rows with less than {n} counts...'.format(n=n_min))
    to_delete = [i for i in range(X.shape[0]) if np.sum(X[i,]) < n_min]
    y   = np.delete(y,   to_delete, axis=0)
    ids = np.delete(ids, to_delete, axis=0)
    X   = np.delete(X,   to_delete, axis=0)

    print('Convert counts to frequencies...')
    row_sums = np.sum(X, axis=0)
    X = X / X.sum(axis=1, keepdims=True)

    print('Data loaded!')
    return y, X, ids

def euk_accuracy(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    class_ac = matrix.diagonal() / matrix.sum(axis=1)
    return class_ac[1]

def pro_accuracy(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    class_ac = matrix.diagonal() / matrix.sum(axis=1)
    return class_ac[0]
