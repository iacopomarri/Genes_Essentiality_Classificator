from sklearn.linear_model import Perceptron
import csv
import numpy as np
import copy
import random
from random import randint
from sklearn.model_selection import KFold
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def SplitX_y(dati):
    X=copy.deepcopy(dati)
    y=[]
    for i in range(len(X)):
        y.append(X[i].pop())
    return X, y

#IN TUTTO IL DATASET CI SONO 769 GENI ESSENIALI
def setSplit(Train_dim, essPositiveNum, X):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    j=0
    essCount=0
    random.shuffle(X)

    while (essCount<essPositiveNum and essPositiveNum<700 and essPositiveNum<Train_dim):
        if X[j][14]==1.0:
            X_train.insert(0,X.pop(j))
            essCount+=1
        j+=1
    if essPositiveNum>=700:
        print("Richiesti troppi geni essenziali nel train, impostarli ad un valore minore di 700")
    if essPositiveNum>Train_dim:
        print("Il numero di geni positivi dev'essere minore della dimensione del Train set")

    random.shuffle(X)

    while(len(X_train)<Train_dim):
        X_train.insert(0, X.pop())
    while(len(X_test)<3500-Train_dim):
        X_test.insert(0, X.pop())

    random.shuffle(X_train)
    random.shuffle(X_test)

    # genero y train
    for i in range(len(X_train)):
        y_train.append(X_train[i].pop())

    # genero y test
    for i in range(len(X_test)):
        y_test.append(X_test[i].pop())

    return X_train, y_train, X_test, y_test


def plot_curve(train_sizes, train_scores, test_scores, invert_score=True, title='', ylim=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    plt.grid(True)
    if invert_score:
        train_scores = 1-train_scores
        test_scores = 1-test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test error")
    plt.legend(loc="best")


def Count(list):
    u=0
    z=0
    for z in range(len(list)):
        if list[z]==1:
            u+=1
    return u, z-u