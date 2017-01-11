# -*- coding: utf-8 -*-

"""
The :mod:`coclust.visualization.coClusteringMeasures` module provides
functions to visualize co-clustering related measures.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_max_modularities(max_modularities, range_n_clusters):
    """Plot all max modularities obtained after a series of evaluations. The
    best partition is indicated in the graph and main title.

    Parameters
    ----------
    max_modularities: list of float
        final modularities for all evaluated partitions
    range_n_clusters: list

    Example
    -------
    >>> plot_max_modularities(all_max_modularities, range_n_clusters)

    .. plot::

        from coclust.visualization.coClusteringMeasures import plot_max_modularities
        from coclust.io.io import load_doc_term_data
        from coclust.evaluation.partitionEvaluation import best_modularity_partition

        path = '../../datasets/classic3_coclustFormat.mat'
        doc_term_data = load_doc_term_data(path)

        min_cluster_nbr = 2
        max_cluster_nbr = 9
        range_n_clusters = range(min_cluster_nbr, (max_cluster_nbr + 1))

        all_final_modularities = [None] * len(range_n_clusters)

        best_coclustMod_model, all_max_modularities = \
            best_modularity_partition(doc_term_data['doc_term_matrix'],
                                      range_n_clusters, 1)

        plot_max_modularities(all_max_modularities, range_n_clusters)
    """

    # Prepare a subplot and set the axis tick values and labels
    fig, ax = plt.subplots()
    fig.canvas.draw()
    labels = np.arange(1, (len(max_modularities)), 1)
    plt.xticks(np.arange(0, len(max_modularities) + 1, 1))
    labels = range_n_clusters
    ax.set_xticklabels(labels)

    # Plot all max modularities
    plt.plot(max_modularities, marker='o')

    # Set the axis titles
    plt.ylabel("Final Modularity", size=10)
    plt.xlabel("Number of clusters", size=10)

    # Set the axis limits
    plt.xlim(-0.5, (len(max_modularities) - 0.5))
    plt.ylim((min(max_modularities) - 0.05 * min(max_modularities)),
             (max(max_modularities) + 0.05 * max(max_modularities)))

    # Set the main plot titlee
    plt.title("\nMax. modularity for %d clusters (%.4f)\n" %
              (range_n_clusters[max_modularities.index(max(max_modularities))],
               max(max_modularities)), size=12)

    # Remove automatic ticks
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right='off', left='off')

    # Plot a dashed vertical line at best partition
    plt.axvline(np.argmax(max_modularities), linestyle="dashed")
    plt.show()


def plot_intermediate_modularities(model):
    """Plot all intermediate modularities for a model.

    Parameters
    ----------
    model: fitted model

    Example
    -------
    >>> plot_intermediate_modularities(model)

    .. plot::

        from coclust.visualization.coClusteringMeasures import plot_intermediate_modularities
        from coclust.io.io import load_doc_term_data
        from coclust.coclustering.CoclustMod import CoclustMod
        model = CoclustMod(n_clusters=3)
        matrix = load_doc_term_data('../../datasets/classic3.csv')['doc_term_matrix']
        model.fit(matrix)
        plot_intermediate_modularities(model)
    """

    # Prepare a subplot and set the axis tick values and labels
    fig, ax = plt.subplots()
    fig.canvas.draw()
    labels = np.arange(1, (len(model.modularities) + 1), 1)
    plt.xticks(np.arange(0, len(model.modularities) + 1, 1))
    ax.set_xticklabels(labels)

    # Plot all intermdiate modularities
    plt.plot(model.modularities, marker='o')

    # Set the axis titles
    plt.ylabel("Modularities", size=10)
    plt.xlabel("Iterations", size=10)

    # Set the axis limits
    plt.xlim(-0.5, (len(model.modularities)-0.5))
    plt.ylim((min(model.modularities) - 0.05 * min(model.modularities)),
             (max(model.modularities) + 0.05 * max(model.modularities)))

    # Set the main plot titlee
    plt.title("\nIntermediate modularities for %d clusters\n"
              % (model.n_clusters), size=12)

    # Remove automatic ticks
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right='off', left='off')

    # Plot a dashed horizontal line around max modularity
    plt.axhline(max(model.modularities), linestyle="dashed")
    plt.axhline((max(model.modularities)-model.tol), linestyle="dashed")
    plt.show()
