# -*- coding: utf-8 -*-

"""
The :mod:`coclust.coclustering.coclust_fuzzy_mod` module provides an
implementation of a fuzzy co-clustering algorithm by approximation of the
modularity matrix.
"""

# Author: Mira Ait Saada <>
#         Alexandra Benamar <benamar.alexandra@gmail.com>

# License: BSD 3 clause

import numpy as np
from sklearn.utils import check_random_state, check_array

from .base_diagonal_coclust import BaseDiagonalCoclust
from ..io.input_checking import check_positive


class CoclustFuzzyMod(BaseDiagonalCoclust):
    """Fuzzy Co-clustering based on modularity maximization.

    Parameters
    ----------
    n_clusters : int, optional, default: 2
        Number of co-clusters to form

    max_iter : int, optional, default: 20
        Maximum number of iterations

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of `n_init`
        consecutive runs in terms of inertia.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    tol : float, default: 1e-9
        Relative tolerance with regards to criterion to declare convergence

    Attributes
    ----------
    row_labels_ : array-like, shape (n_rows,)
        Bicluster label of each row

    column_labels_ : array-like, shape (n_cols,)
        Bicluster label of each column

    References
    ----------
    * Yongli L., Jingli C., Hao C., A Fuzzy Co-Clustering Algorithm via Modularity Maximization.
    Journal: Mathematical Problems in Engineering - Year: 2018 - Pages 1-11. 
    """

    def __init__(self, n_clusters=2, init=None, max_iter=20, n_init=1,
                 tol=1e-9, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state

        self.row_labels_ = None
        self.column_labels_ = None
        self.modularity = -np.inf
        self.modularities = []

    def fit(self, X, y=None):
        """Perform fuzzy co-clustering based on modularity maximization.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        
