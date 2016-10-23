# -*- coding: utf-8 -*-

"""
Evaluation
"""

# Author: Francois Role <francois.role@gmail.com>
#         Stanislas Morbieu <stanislas.morbieu@gmail.com>

# License: BSD 3 clause

from __future__ import print_function
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

from coclust.CoclustMod import CoclustMod

plt.style.use('ggplot')

def best_partition(in_data, nbr_clusters_range):
    """Evaluation the best partition over a range of number of cluster
    using co-clustering by direct maximization of graph modularity.

    Parameters
    ----------
    in_data : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
        Matrix to be analyzed
    nbr_clusters_range : number of clusters to be evaluated

    Values
    ----------
    tmp_best_model: model with highest final modularity
    tmp_max_modularities: final modularities for all evaluated partitions
    """
    tmp_best_model = None
    tmp_max_modularities = [np.nan] * len(nbr_clusters_range)
    eps_best_model = 1e-4
    
    # Set best final modularity to -inf
    modularity_begin = float("-inf")

    print("CoClust ../.. number of clusters =", end = ' ')
    for tmp_n_clusters in nbr_clusters_range:
        print("... %d " % (tmp_n_clusters), end = ' ')
        # Create and fit a model with tmp_n_clusters co-clusters
        tmp_model = CoclustMod(n_clusters = tmp_n_clusters, random_state = 0)
        tmp_model.fit(in_data)

        modularity_end = tmp_model.modularity
        # Check if the final modularity is better with tolerance
        if((modularity_end - modularity_begin) > eps_best_model):
            tmp_best_model = tmp_model
            modularity_begin = modularity_end

        tmp_max_modularities[(tmp_n_clusters)-min(nbr_clusters_range)] = tmp_model.modularity

    print(" All done !")
    return [tmp_best_model, tmp_max_modularities,]

def plot_max_modularities(max_modularities, range_n_clusters):
    """Plot all max modularities obtained after a series of evaluations. The best partition is
    indicated in the graph and main title

    Parameters
    ----------
    max_modularities: final modularities for all evaluated partitions
    """
    # Prepare a subplot and set the axis tick values and labels
    fig, ax = plt.subplots()
    fig.canvas.draw()
    labels = np.arange(1,(len(max_modularities)), 1)
    plt.xticks(np.arange(0, len(max_modularities)+1, 1))
    labels = range_n_clusters
    ax.set_xticklabels(labels)
    # Plot all max modularities
    plt.plot(max_modularities, marker='o')
    # Set the axis titles
    plt.ylabel("Final Modularity", size=10)
    plt.xlabel("Number of clusters", size=10)
    # Set the axis limits
    plt.xlim(-0.5, (len(max_modularities)-0.5))
    plt.ylim((min(max_modularities)-0.05*min(max_modularities)),
             (max(max_modularities)+0.05*max(max_modularities)))
    # Set the main plot titlee
    plt.title ("\nMax. modularity for %d clusters (%.4f)\n" %
               (range_n_clusters[max_modularities.index(max(max_modularities))],
                                                              max(max_modularities)), size=12)
    # Remove automatic ticks
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right='off', left='off')
    # Plot a dashed vertical line at best partition
    plt.axvline(np.argmax(max_modularities), linestyle = "dashed")
    plt.show()

def plot_intermediate_modularities(model):
    """Plot all intermediate modularities for a model.

    Parameters
    ----------
    model: fitted model
    """
    # Prepare a subplot and set the axis tick values and labels
    fig, ax = plt.subplots()
    fig.canvas.draw()
    labels = np.arange(1,(len(model.modularities)+1), 1)
    plt.xticks(np.arange(0, len(model.modularities)+1, 1))
    ax.set_xticklabels(labels)
    # Plot all intermdiate modularities
    plt.plot(model.modularities, marker='o')
    # Set the axis titles
    plt.ylabel("Modularities", size=10)
    plt.xlabel("Iterations", size=10)
    # Set the axis limits
    plt.xlim(-0.5, (len(model.modularities)-0.5))
    plt.ylim((min(model.modularities)-0.05*min(model.modularities)),
             (max(model.modularities)+0.05*max(model.modularities)))
    # Set the main plot titlee
    plt.title ("\nIntermediate modularities for %d clusters\n" % (model.n_clusters),
               size=12)
    # Remove automatic ticks
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right='off', left='off')
    # Plot a dashed horizontal line around max modularity
    plt.axhline(max(model.modularities), linestyle = "dashed")
    plt.axhline((max(model.modularities)-model.tol), linestyle = "dashed")
    plt.show()

def plot_cluster_top_terms(in_data, all_terms, nb_top_terms, model):
    """Plot the top terms for each cluster

    Parameters
    ----------
    in_data : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
    all_terms: list of all terms from the original data set
    nb_top_terms, model: number of top terms to be displayed per cluster
    """
    x_label="number of occurences"
    plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(hspace=0.200)
    plt.suptitle("      Top %d terms" % nb_top_terms, size=15)
    number_of_subplots = model.n_clusters

    for i,v in enumerate(range(number_of_subplots)):
        # Get the row/col indices corresponding to the given cluster
        row_indices, col_indices = model.get_indices(v)
        # Get the submatrix corresponding to the given cluster
        cluster = model.get_submatrix(in_data, v)
        # Count the number of each term
        p = cluster.sum(0)
        t = p.getA().flatten()
        # Obtain all term names for the given cluster
        tmp_terms = np.array(all_terms)[col_indices]
        # Get the first n terms
        max_indices = t.argsort()[::-1][:nb_top_terms]

        pos = np.arange(nb_top_terms)
        
        v = v+1
        ax1 = plt.subplot(number_of_subplots,1,v)
        ax1.barh(pos, t[max_indices][::-1])
        ax1.set_title("Cluster %d (%d terms)" % (v,len(col_indices)), size=11)

        plt.yticks(.4 + pos, tmp_terms[max_indices][::-1], size=9.5)
        plt.xlabel(x_label, size = 9)
        plt.margins(y=0.05)
        #_remove_ticks()
        plt.tick_params(axis='both', which='both', bottom='on', top='off',
                        right='off', left='off')

    # Tight layout often produces nice results
    # but requires the title to be spaced accordingly
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    plt.show()


##def _remove_ticks():
##    plt.tick_params(axis='both', which='both', bottom='off', top='off',
##                    right='off', left='off')
##
##
##def plot_criterion(values, ylabel):
##    plt.plot(values, marker='o')
##    plt.ylabel(ylabel)
##    plt.xlabel('Iterations')
##    _remove_ticks()
##    plt.show()
##
##
##def plot_reorganized_matrix(X, model, precision=0.8, markersize=0.9):
##    row_indices = np.argsort(model.row_labels_)
##    col_indices = np.argsort(model.column_labels_)
##    X_reorg = X[row_indices, :]
##    X_reorg = X_reorg[:, col_indices]
##    plt.spy(X_reorg, precision=precision, markersize=markersize)
##    _remove_ticks()
##    plt.show()
