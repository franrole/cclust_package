# -*- coding: utf-8 -*-

"""
The :mod:`coclust.evaluation.external` module provides functions
to evaluate clustering or co-clustering results with external information
such as the true labeling of the clusters.
"""


import logging
import numpy as np

from sklearn.metrics import confusion_matrix


def accuracy(X, nb_clusters, true_row_labels, predicted_row_labels):
    """Get the best accuracy.

    Log an error message if the best accuracy cannot be computed due to missing
    dependencies.

    Parameters
    ----------
    X:
        The data matrix
    nb_clusters: int
        The number of clusters
    true_row_labels: array-like
        The true row labels, given as external information
    predicted_row_labels: array-like
        The row labels predicted by the model

    Returns
    -------
    float
        Best value of accuracy

    """

    try:
        accuracy = _true_accuracy(X, nb_clusters, true_row_labels,
                                  predicted_row_labels)
        return accuracy
    except ImportError as e:
        logging.error(e)
        logging.error("Fallback to approximate accuracy, install Munkres for "
                      "true accuracy.")
        return _approximate_accuracy(X, nb_clusters, true_row_labels,
                                     predicted_row_labels)


def _true_accuracy(X, nb_clusters, true_row_labels, predicted_row_labels):
    from munkres import Munkres, make_cost_matrix
    m = Munkres()
    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    s = np.max(cm)
    cost_matrix = make_cost_matrix(cm, lambda cost: s - cost)
    indexes = m.compute(cost_matrix)
    total = 0
    for row, column in indexes:
        value = cm[row][column]
        total += value

    return(total * 1. / np.sum(cm))


def _approximate_accuracy(X, nb_clusters, true_row_labels,
                          predicted_row_labels):
    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    n_rows = len(true_row_labels)
    total = 0
    for i in range(nb_clusters):
        if len(cm) == 0:
            break
        max_value = np.amax(cm)
        r_indices, c_indices = np.where(cm == max_value)
        total = total + max_value
        cm = np.delete(cm, r_indices[0], 0)
        cm = np.delete(cm, c_indices[0], 1)
    accuracy = (total) / (n_rows * 1.)

    return accuracy
