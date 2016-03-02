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
    
def plot_confusion_matrix(cm,colormap=plt.cm.jet , labels='012') :
  import numpy as np
  import matplotlib.pyplot as plt

  conf_arr=np.array(cm)

  norm_conf_arr = []
  for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    print(a)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf_arr.append(tmp_arr)

  fig = plt.figure()
  plt.clf()
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  res = ax.imshow(np.array(norm_conf_arr), cmap=plt.cm.jet, 
                interpolation='nearest')

  width, height = conf_arr.shape

  for x in np.arange(width):
    for y in np.arange(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')
  cb = fig.colorbar(res)
  plt.xticks(range(width), labels[:width])
  plt.yticks(range(height), labels[:height])

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


        
