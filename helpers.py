import numpy as np
from sklearn.metrics import confusion_matrix

def CLR_transform(X, scale):
    assert 0 < CLR_scale and CLR_scale < 1
    minval = np.min(X[np.nonzero(X)])
    X[X == 0] = minval * scale
    X = np.log(X)
    X = X - np.mean(X, axis = 0)
    return(X)

def load_csv_data(data_path, n_min=1000, CLR_scale=None):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    print('Loading data...')
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = data[:, 0].astype(np.int)
    X = data[:, 2:]

    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y=='Prokaryote')] = 0

    # Remove rows having less than n_min counts
    print('Removing rows with less than n_min counts...')
    to_delete = [i for i in range(X.shape[0]) if np.sum(X[i,]) < n_min]
    yb   = np.delete(yb,   to_delete, axis=0)
    ids = np.delete(ids, to_delete, axis=0)
    X   = np.delete(X,   to_delete, axis=0)

    if CLR_scale:
        print('Counts to CLR transformed...')
        X = CLR_transform(X, CLR_scale)

        print('Data loaded!')
        return yb, X, ids

    print('Counts to frequencies...')
    X = X / X.sum(axis=1, keepdims=True)
    print('Data loaded!')
    return yb, X, ids

def euk_accuracy(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    class_ac = matrix.diagonal() / matrix.sum(axis=1)
    return class_ac[1]

def pro_accuracy(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    class_ac = matrix.diagonal() / matrix.sum(axis=1)
    return class_ac[0]
