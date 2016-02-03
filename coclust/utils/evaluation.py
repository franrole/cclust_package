# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:23:46 2015

@author: frole
"""

from __future__ import print_function
import argparse
import numpy as np
import scipy.sparse as sp
import scipy
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score
import itertools
from scipy.io import loadmat


def plot_reorganized_matrix(X,model,precision=0.8, markersize=0.9):
    row_indices=np.argsort(model.row_labels_)
    col_indices=np.argsort(model.column_labels_)
    X_reorg=X[row_indices,:]
    X_reorg=X_reorg[:,col_indices]
    plt.spy(X_reorg,precision=precision, markersize=markersize)
    plt.show()

def plot_convergence(criteria, criterion_name,marker='o'):
    plt.plot(criteria,marker=marker)
    plt.ylabel(criterion_name)
    plt.xlabel('Iterations')
    plt.show()
    plt.show()

def print_NMI_and_ARI(true_labels, predicted_labels) :
     print("NMI:", nmi(true_labels, predicted_labels))
     print("ARI:", adjusted_rand_score(true_labels, predicted_labels))


def print_accuracy(cm,n_rows,n_classes) :
    total=0
    for i in range(n_classes):
        if len(cm) ==0 : break
        max_value=np.amax(cm)
        r_indices,c_indices = np.where(cm==max_value)
        total = total + max_value
        cm = np.delete(cm,r_indices[0], 0)
        cm = np.delete(cm, c_indices[0], 1)
    accuracy=(total)/(n_rows*1.)
    print("ACCURACY:" + str(accuracy))



## Better version used in the benchmark code
## To use it you need to install the munkres package
##from munkres import Munkres, make_cost_matrix
##def get_accuracy(X, nb_clusters, true_row_labels, predicted_row_labels):
##    m = Munkres()
##    cm = confusion_matrix(true_row_labels, predicted_row_labels)
##    s = np.max(cm)
##    cost_matrix = make_cost_matrix(cm, lambda cost: s - cost)
##    indexes = m.compute(cost_matrix)
##    total = 0
##    for row, column in indexes:
##        value = cm[row][column]
##        total += value
##    return(total*1./np.sum(cm))


        
