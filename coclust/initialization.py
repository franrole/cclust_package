# -*- coding: utf-8 -*-

"""
The :mod:`coclust.initialization` module provides functions to initialize
clustering or co-clustering algorithms.
"""

# Author: Francois Role <francois.role@gmail.com>
#         Stanislas Morbieu <stanislas.morbieu@gmail.com>

# License: BSD 3 clause

import numpy as np
from sklearn.utils import check_random_state


def random_init(n_clusters, n_cols, random_state=None):
    """Create a random column cluster assignment matrix.

    Each row contains 1 in the column corresponding to the cluster where the
    processed data matrix column belongs, 0 elsewhere.

    Parameters
    ----------
    n_clusters: int
        Number of clusters
    n_cols: int
        Number of columns of the data matrix (i.e. number of rows of the
        matrix returned by this function)
    random_state : int or :class:`numpy.RandomState`, optional
        The generator used to initialize the cluster labels. Defaults to the
        global numpy random number generator.

    Returns
    -------
    matrix
        Matrix of shape (``n_cols``, ``n_clusters``)
    """

    random_state = check_random_state(random_state)
    W_a = random_state.randint(n_clusters, size=n_cols)
    W = np.zeros((n_cols, n_clusters))
    W[np.arange(n_cols), W_a] = 1
    return W


def random_init_clustering(n_clusters, n_rows, random_state=None):
    """Create a random row cluster assignment matrix.

    Each row contains 1 in the column corresponding to the cluster where the
    processed data matrix row belongs, 0 elsewhere.

    Parameters
    ----------
    n_clusters: int
        Number of clusters
    n_rows: int
        Number of rows of the data matrix (i.e. also the number of rows of the
        matrix returned by this function)
    random_state : int or :class:`numpy.RandomState`, optional
        The generator used to initialize the cluster labels. Defaults to the
        global numpy random number generator.

    Returns
    -------
    matrix
        Matrix of shape (``n_rows``, ``n_clusters``)
    """

    random_state = check_random_state(random_state)
    Z_a = random_state.randint(n_clusters, size=n_rows)
    Z = np.zeros((n_rows, n_clusters))
    Z[np.arange(n_rows), Z_a] = 1
    return Z
