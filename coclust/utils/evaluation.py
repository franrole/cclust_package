# -*- coding: utf-8 -*-

"""
Created on Mon Dec 14 14:23:46 2015

@author: frole
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib as mpl


def plot_reorganized_matrix(X, model, precision=0.8, markersize=0.9):
    row_indices = np.argsort(model.row_labels_)
    col_indices = np.argsort(model.column_labels_)
    X_reorg = X[row_indices, :]
    X_reorg = X_reorg[:, col_indices]
    plt.spy(X_reorg, precision=precision, markersize=markersize)
    plt.show()


def plot_convergence(criteria, criterion_name, marker='o'):
    plt.plot(criteria, marker=marker)
    plt.ylabel(criterion_name)
    plt.xlabel('Iterations')
    plt.show()
    plt.show()


def plot_confusion_matrix(cm, colormap=plt.cm.jet, labels='012'):
    conf_arr = np.array(cm)

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

    fig.colorbar(res)
    plt.xticks(range(width), labels[:width])
    plt.yticks(range(height), labels[:height])


def plot_delta_kl(delta, model, colormap=plt.cm.jet, labels='012'):

    delta_arr = np.round(np.array(delta), decimals=3)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(delta_arr), cmap=colormap,
                    interpolation='nearest')

    width, height = delta_arr.shape

    for x in np.arange(width):
        for y in np.arange(height):
            nb_docs = len(model.get_row_indices(x))
            nb_terms = len(model.get_col_indices(y))
            ax.annotate(str(delta_arr[x][y]) + "\n(%d,%d)" %
                        (nb_docs, nb_terms),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    fig.colorbar(res)
    plt.xticks(range(width), labels[:width])
    plt.yticks(range(height), labels[:height])


def print_NMI_and_ARI(true_labels, predicted_labels):
    print("NMI:", nmi(true_labels, predicted_labels))
    print("ARI:", adjusted_rand_score(true_labels, predicted_labels))


def print_accuracy(cm, n_rows, n_classes):
    total = 0
    for i in range(n_classes):
        if len(cm) == 0:
            break
        max_value = np.amax(cm)
        r_indices, c_indices = np.where(cm == max_value)
        total = total + max_value
        cm = np.delete(cm, r_indices[0], 0)
        cm = np.delete(cm, c_indices[0], 1)
    accuracy = (total)/(n_rows*1.)
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

# %matplotlib inline


def plot_all():
    grid_size = (6, 10)
    ax_1 = plt.subplot2grid(grid_size, (0, 0), rowspan=2, colspan=10)
    ax_2 = plt.subplot2grid(grid_size, (2, 0), rowspan=4, colspan=5)
    ax_3 = plt.subplot2grid(grid_size, (2, 5), rowspan=2, colspan=5)
    ax_4 = plt.subplot2grid(grid_size, (4, 5), rowspan=2, colspan=5)

    ax_1.get_xaxis().set_visible(False)
    ax_2.get_xaxis().set_visible(False)
    ax_3.get_xaxis().set_visible(False)
    ax_1.get_yaxis().set_visible(False)
    ax_2.get_yaxis().set_visible(False)
    ax_3.get_yaxis().set_visible(False)
    plt.subplots_adjust(hspace=-.6, wspace=0.4)

    grid = np.array([[1,8,13,29,17,26,10,4],[16,25,31,5,21,30,19,15]])
    ax_1.plot(np.arange(20))
    ax_1.set_title('EEEEE')
    ax_2.imshow(grid, interpolation ='none', aspect = 'auto')
    ax_2.set_title('EEEEE')
    cmap = mpl.colors.ListedColormap(['white', 'white'])
    bounds = [0., 1.]
    imshow_data = np.array([[0.125, 0.268, 0.230]])
    ax_3.imshow(imshow_data, interpolation='none', aspect='auto', cmap=cmap)
    ax_3.text(0., 0.,'ACC\n\n 0.986', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10} , va='center', ha='center', fontsize=24, fontweight='bold')
    ax_3.text(1., 0.,'ARI\n\n 0.960', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10} , va='center', ha='center', fontsize=24 , fontweight='bold')
    ax_3.text(2., 0.,'NMI\n\n 0.932',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10} , va='center', ha='center', fontsize=24, fontweight='bold')
    ax_3.set_title('EEEEE')

    ax_4.imshow(grid, interpolation='none', aspect='auto')

    fig = ax_1.get_figure()
    fig.set_size_inches(12, 12)
#    fig.suptitle('Evaluation Report', fontsize=20, fontweight='bold')
    plt.tight_layout(h_pad=3, w_pad=0.8)
    plt.axis('off')
