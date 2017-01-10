# -*- coding: utf-8 -*-

"""
Visualize co-clustering related measures (modularity, NMI, Acc...)
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_max_modularities(max_modularities, range_n_clusters):
    """Plot all max modularities obtained after a series of evaluations. The
    best partition is indicated in the graph and main title.

    Parameters
    ----------
    max_modularities: final modularities for all evaluated partitions
    """
    # Prepare a subplot and set the axis tick values and labels
    fig, ax = plt.subplots()
    fig.canvas.draw()
    labels = np.arange(1, (len(max_modularities)), 1)
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
    """
    # Prepare a subplot and set the axis tick values and labels
    fig, ax = plt.subplots()
    fig.canvas.draw()
    labels = np.arange(1, (len(model.modularities)+1), 1)
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
    plt.title("\nIntermediate modularities for %d clusters\n" % (model.n_clusters),
              size=12)
    # Remove automatic ticks
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right='off', left='off')
    # Plot a dashed horizontal line around max modularity
    plt.axhline(max(model.modularities), linestyle="dashed")
    plt.axhline((max(model.modularities)-model.tol), linestyle="dashed")
    plt.show()
