# -*- coding: utf-8 -*-

"""
The :mod:`coclust.coclustering.coclust_dcc` module provides an implementation
of a model-based von Mises-Fisher co-clustering with a Conscience.
"""

# Author: Mira Ait Saada <aitsaadamira@gmail.com>

# License: BSD 3 clause

import numpy as np
from sklearn.utils import check_random_state, check_array
from scipy.sparse.csr import csr_matrix
from scipy.sparse import issparse

from ..initialization import random_init
from .base_diagonal_coclust import BaseDiagonalCoclust


def _dcc_sample_function(prob):
    """Performs a stochastic column assignment

    Parameters
    ----------
    prob : numpy array, shape (n_features, )
            Relative probability given to each column

    Returns
    -------
    int
        Randomly selected column label
    """

    normalizer = 1 / np.sum(prob)
    prob = prob * normalizer
    return np.random.choice(np.arange(len(prob)), p=prob)


class CoclustDCC(BaseDiagonalCoclust):
    """Directional Co-clustering with a Conscience.

    Parameters
    ----------
    n_clusters : int, optional, default: 2
        Number of co-clusters to form

    row_init : numpy array or scipy sparse matrix, \
        shape (n_samples, n_clusters), optional, default: None
        Initial row labels

    col_init : numpy array or scipy sparse matrix, \
        shape (n_features, n_clusters), optional, default: None
        Initial column labels

    max_iter : int, optional, default: 100
        Maximum number of iterations

    max_stoch_iter : int, optional, default: 70
        Maximum number of stochastic iterations
        Those iterations are used to avoid bad local solutions
        Must be smaller than max_iter

    n_init : int, optional, default: 1
        Number of time the algorithm will be run with different
        initializations. The final results will be the best output of `n_init`
        consecutive runs in terms of criterion value.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    tol : float, default: 1e-9
        Relative tolerance with regards to objective function to declare convergence

    Attributes
    ----------
    row_labels_ : array-like, shape (n_rows,)
        Bicluster label of each row

    column_labels_ : array-like, shape (n_cols,)
        Bicluster label of each column

    criterion : float
        Final value of the objective function

    criterion_evolution : list
        Record of all computed criterion values for all iterations

    References
    ----------
    * Salah, A. and Nadif, M. Model-based von mises-fisher co-clustering \
    with a conscience. SIAM : 246–254.
    """

    def __init__(self, n_clusters=2, row_init=None, col_init=None, max_iter=20, max_stoch_iter = 10, n_init=1,
                 tol=1e-9, random_state=None):
        self.n_clusters = n_clusters
        self.row_init = row_init
        self.col_init = col_init
        self.max_iter = max_iter
        self.max_stoch_iter = max_stoch_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.epsilon = 1 / (np.finfo("float32").max)

        self.row_labels_ = None
        self.column_labels_ = None
        self.criterion = -np.inf
        self.criterion_evolution = []

    def fit(self, X, y=None):
        """Performs Directional Co-clustering with a Conscience.

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

        # In case X is in matrix format or scipy.sparse
        if type(X) != np.ndarray:
            X = X.A

        X = X.astype(float)

        X = ((X.T)/(self.epsilon + np.sqrt(((X * X).sum(axis=1))))).T
        X = csr_matrix(X)

        criterion = self.criterion
        criterion_evolution = []
        row_labels_ = None
        column_labels_ = None

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            self._fit_single(X, seed, y)
            if np.isnan(self.criterion):
                raise ValueError("matrix may contain unexpected NaN values")
            # Remember attributes corresponding to the best criterion
            if (self.criterion > criterion):
                criterion = self.criterion
                criterion_evolution = self.criterion_evolution
                row_labels_ = self.row_labels_
                column_labels_ = self.column_labels_
                Zt = self.Zt
                Wt = self.Wt

        # Update attributes
        self.criterion = criterion
        self.criterion_evolution = criterion_evolution
        self.row_labels_ = row_labels_
        self.column_labels_ = column_labels_
        self.Zt = Zt
        self.Wt = Wt

        return self

    def _fit_single(self, X, random_state, y=None):
        """Performs one run of Directional Co-clustering with a Conscience.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """

        if self.row_init is None:
            Z = random_init(self.n_clusters, X.shape[0], random_state)
        else:
            Z = self.row_init
        if self.col_init is None:
            W = random_init(self.n_clusters, X.shape[1], random_state)
        else:
            W = self.col_init

        self.criterion_evolution = []

        c_begin = float("-inf")
        
        Z = csr_matrix(Z)
        W = csr_matrix(W)

        # Compute initial row centroids 
        MU_z = np.diag(1/ (self.epsilon + np.array(np.sqrt(Z.sum(axis = 0)))).flatten())
        MU_z = csr_matrix(MU_z)
        # Compute initial column centroids
        MU_w = np.diag(1/ (self.epsilon + np.array(np.sqrt(W.sum(axis = 0)))).flatten())
        MU_w = csr_matrix(MU_w)

        for iteration in range(self.max_iter):
            # Row partitionning
            Zt = np.dot(X, W).dot(np.dot(MU_w, MU_z))
            row_partition = np.array(np.argmax(Zt, axis = 1)).flatten()
            Z = csr_matrix(np.eye(self.n_clusters)[row_partition])
            
            # Update row centroids
            MU_z = np.diag(1/ (self.epsilon + np.array(np.sqrt(Z.sum(axis = 0)))).flatten())
            MU_z = csr_matrix(MU_z)
            
            # Column partitionning
            Wt = X.T.dot(Z.dot(MU_z.dot(MU_w)))

            if iteration + 1 < self.max_stoch_iter:
                # Perform stochastic column assignement to avoid bad local solutions
                col_partition = np.apply_along_axis(_dcc_sample_function, axis=1, arr=Wt.A)
            else:
                col_partition = np.array(np.argmax(Wt, axis=1)).flatten()

            W = csr_matrix(np.eye(self.n_clusters)[col_partition])
            
            # Update column centroids
            MU_w = np.diag(1/ (self.epsilon + np.array(np.sqrt(W.sum(axis = 0)))).flatten())
            MU_w = csr_matrix(MU_w)
            
            # Evaluate the criterion
            c_end = (Z.multiply(np.dot(np.dot(X, W), np.dot(MU_w, MU_z)))).sum()

            if np.abs(c_end - c_begin) > self.tol:
                c_begin = c_end
            else:
                break

            self.criterion_evolution.append(c_end)

        self.row_labels_ = row_partition
        self.column_labels_ = col_partition
        self.criterion = c_end
        self.Zt = Zt
        self.Wt = Wt
        self.nb_iterations = iteration + 1
