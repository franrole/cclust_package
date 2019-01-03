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

    def __init__(self, n_clusters=2, Tu = 1, Tv = 1, init=None, max_iter=20, n_init=1,
                 tol=1e-9, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.Tu = Tu
        self.Tv = Tv

        self.U=None
        self.V=None

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

        random_state = check_random_state(self.random_state)

        check_array(X, accept_sparse=True, dtype="numeric", order=None,
                    copy=False, force_all_finite=True, ensure_2d=True,
                    allow_nd=False, ensure_min_samples=self.n_clusters,
                    ensure_min_features=self.n_clusters,
                    warn_on_dtype=False, estimator=None)
        
        if type(X) == np.ndarray:
            X = np.matrix(X)

        X = X.astype(float)

        modularity = self.modularity
        modularities = []
        row_labels_ = None
        column_labels_ = None

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            self._fit_single(X, seed, y)
            if np.isnan(self.modularity):
                raise ValueError("matrix may contain unexpected NaN values")
            # remember attributes corresponding to the best modularity
            if (self.modularity > modularity):
                modularity = self.modularity
                modularities = self.modularities
                row_labels_ = self.row_labels_
                column_labels_ = self.column_labels_

        # update attributes
        self.modularity = modularity
        self.modularities = modularities
        self.row_labels_ = row_labels_
        self.column_labels_ = column_labels_

        return self

    def _fit_single(self, X, random_state, y=None):
        """Perform one run of co-clustering by direct maximization of graph
        modularity.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """

        # (3) init U and V 

        # Compute the modularity matrix
        row_sums = np.matrix(X.sum(axis=1))
        col_sums = np.matrix(X.sum(axis=0))
        N = float(X.sum())
        indep = (row_sums.dot(col_sums)) / N

        # B is a numpy matrix
        B = X - indep

        self.modularities = []

        # Loop
        obj_begin = float("-inf")
        change = True
        iteration = 0
        # (4)
        while change:
            change = False

            # Reassign rows 
            # (5)

            # Reassign columns
            # (6)

            #(7)
            Q = np.trace((U.T).dot(BV)) / N

            entropy_u = Tu * np.trace(np.dot(U.T, np.log(U))) 
            entropy_v = Tv * np.trace(np.dot(V.T, np.log(V))) 

            obj_end = Q - entropy_u - entropy_v
            iteration += 1 
            
            # (8)
            if (np.abs(obj_end - obj_begin) > self.tol and
                    iteration < self.max_iter):
                self.modularities.append(obj_end)
                obj_begin = obj_end 
                change = True

        self.row_labels_ = np.argmax(Z, axis=1).tolist()
        self.column_labels_ = np.argmax(W, axis=1).tolist()
        self.btu = BtU
        self.bv = BV
        self.modularity = obj_end
        self.nb_iterations = iteration 
        
    def get_assignment_matrix(self, kind, i):
        """Returns the indices of 'best' i cols of an assignment matrix
        (row or column).

        Parameters
        ----------
        kind : string
             Assignment matrix to be used: rows or cols

        Returns
        -------
        numpy array or scipy sparse matrix
            Matrix containing the i 'best' columns of a row or column
            assignment matrix
        """
        if kind == "rows":
            s_bw = np.argsort(self.bw)
            return s_bw[:, -1:-(i+1):-1]
        if kind == "cols":
            s_btz = np.argsort(self.btz)
            return s_btz[:, -1:-(i+1):-1]