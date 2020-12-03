#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Some usueful functions to load the data, compute label accuracies
and plot grid search results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def load_csv_data(data_path, n_min=1000):
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

def plot_2param(df, param1, param2, suptitle, axtitle, figtitle, x_label):
    P1 = df[param1].unique()
    P2 = df[param2].unique()

    fig, axs = plt.subplots(len(P2), figsize=(1+len(P1), 7*len(P2)), constrained_layout=True)
    fig.suptitle(suptitle, fontsize=20)

    for i, p2 in enumerate(P2):
        axs[i].set_title(axtitle.format(p2), fontsize=16)

        glo_acc = df.loc[df[param2] == p2]['accuracy'].to_numpy()
        euk_acc = df.loc[df[param2] == p2]['eukaryote accuracy'].to_numpy()
        pro_acc = df.loc[df[param2] == p2]['prokaryote accuracy'].to_numpy()
        time = df.loc[df[param2] == p2]['learning time'].to_numpy()
        learn_time = df.loc[df[param2] == p2]['learning time'].to_numpy()
        predict_time = df.loc[df[param2] == p2]['prediction time'].to_numpy()

        # plot accuracies
        axs[i].plot(P1, glo_acc, color="red", marker='o', label='Accuracy (global)')
        axs[i].plot(P1, euk_acc, color="orange", marker="o", label='Accuracy on eukaryotes')
        axs[i].plot(P1, pro_acc, color="yellow", marker="o", label='Accuracy on prokaryotes')
        axs[i].set_xlabel(x_label, fontsize=14)
        axs[i].set_ylabel("Accuracy", fontsize=14)
        axs[i].legend(bbox_to_anchor=(1.04,1), loc="upper left")

        # plot times
        ax2 = axs[i].twinx()
        ax2.plot(P1, learn_time, ls='--', color="blue", marker="^", label='Learning time')
        ax2.plot(P1, predict_time, ls='--', color="cyan", marker="^", label='Prediction time')
        ax2.set_ylabel("Time [sec]", color="blue", fontsize=14)
        ax2.legend(bbox_to_anchor=(1.04, 0), loc="lower left")

    fig.tight_layout(pad=1)
    fig.savefig(figtitle, bbox_inches='tight')

def plot_1param(df, param, suptitle, axtitle, figtitle, x_label):
    P = df[param].unique()
    glo_acc = df['accuracy'].to_numpy()
    euk_acc = df['eukaryote accuracy'].to_numpy()
    pro_acc = df['prokaryote accuracy'].to_numpy()
    learn_time = df['learning time'].to_numpy()
    predict_time = df['prediction time'].to_numpy()

    fig, ax = plt.subplots(figsize=(1 + len(P), 7), constrained_layout=True)
    fig.suptitle(suptitle, fontsize=20, y=0)
    ax.set_title(axtitle, fontsize=16)

    # plot accuracies
    ax.plot(P, glo_acc, color="red", marker='o', label='Accuracy (global)')
    ax.plot(P, euk_acc, color="orange", marker="o", label='Accuracy on eukaryotes')
    ax.plot(P, pro_acc, color="yellow", marker="o", label='Accuracy on prokaryotes')
    ax.set_xlabel(x_label,fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")

    # plot times
    ax2 = ax.twinx()
    ax2.plot(P, learn_time, ls='--', color="blue", marker="^", label='Learning time')
    ax2.plot(P, predict_time, ls='--', color="cyan", marker="^", label='Prediction time')
    ax2.set_ylabel("Time [sec]", color="blue", fontsize=14)
    ax2.legend(bbox_to_anchor=(1.04, 0), loc="lower left")

    fig.savefig(figtitle, bbox_inches='tight')
