# -*- coding: utf-8 -*-

"""
The :mod:`coclust.coclustering.coclust_mod` module provides an implementation
of a co-clustering algorithm by direct maximization of graph modularity.
"""

# Author: Francois Role <francois.role@gmail.com>
#         Stanislas Morbieu <stanislas.morbieu@gmail.com>

# License: BSD 3 clause

import numpy as np
from sklearn.utils import check_random_state

from ..io.input_checking import check_array, check_numbers
from ..initialization import random_init
from .base_diagonal_coclust import BaseDiagonalCoclust


class CoclustMod(BaseDiagonalCoclust):
    """Co-clustering by direct maximization of graph modularity.

    Parameters
    ----------
    n_clusters : int, optional, default: 2
        Number of co-clusters to form

    init : numpy array or scipy sparse matrix, \
        shape (n_features, n_clusters), optional, default: None
        Initial column labels

    max_iter : int, optional, default: 20
        Maximum number of iterations

    n_init : int, optional, default: 1
        Number of time the algorithm will be run with different
        initializations. The final results will be the best output of `n_init`
        consecutive runs in terms of modularity.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    tol : float, default: 1e-9
        Relative tolerance with regards to modularity to declare convergence

    Attributes
    ----------
    row_labels_ : array-like, shape (n_rows,)
        Bicluster label of each row

    column_labels_ : array-like, shape (n_cols,)
        Bicluster label of each column

    modularity : float
        Final value of the modularity

    modularities : list
        Record of all computed modularity values for all iterations

    References
    ----------
    * Ailem M., Role F., Nadif M., Co-clustering Document-term Matrices by \
    Direct Maximization of Graph Modularity. CIKM 2015: 1807-1810
    """

    def __init__(self, n_clusters=2, init=None, max_iter=20, n_init=1,
                 tol=1e-9, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = check_random_state(random_state)

        self.row_labels_ = None
        self.column_labels_ = None
        self.modularity = -np.inf
        self.modularities = []

    def fit(self, X, y=None):
        """Perform co-clustering by direct maximization of graph modularity.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """

        check_numbers(X, self.n_clusters)

        check_array(X)

        check_numbers(X, self.n_clusters)

        if type(X) == np.ndarray:
            X = np.matrix(X)

        X = X.astype(float)

        modularity = self.modularity

        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            self.random_state = seed
            self._fit_single(X, y)
            if np.isnan(self.modularity):
                raise ValueError("matrix may contain unexpected NaN values")
            # remember attributes corresponding to the best modularity
            if (self.modularity > modularity):
                modularity = self.modularity
                modularities = self.modularities
                row_labels_ = self.row_labels_
                column_labels_ = self.column_labels_

        self.random_state = random_state
        # update attributes
        self.modularity = modularity
        self.modularities = modularities
        self.row_labels_ = row_labels_
        self.column_labels_ = column_labels_

    def _fit_single(self, X, y=None):
        """Perform one run of co-clustering by direct maximization of graph
        modularity.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """

        if self.init is None:
            W = random_init(self.n_clusters, X.shape[1], self.random_state)
        else:
            W = np.matrix(self.init, dtype=float)

        Z = np.zeros((X.shape[0], self.n_clusters))

        # Compute the modularity matrix
        row_sums = np.matrix(X.sum(axis=1))
        col_sums = np.matrix(X.sum(axis=0))
        N = float(X.sum())
        indep = (row_sums.dot(col_sums)) / N

        # B is a numpy matrix
        B = X - indep

        self.modularities = []

        # Loop
        m_begin = float("-inf")
        change = True
        iteration = 0
        while change:
            change = False

            # Reassign rows
            BW = B.dot(W)
            for idx, k in enumerate(np.argmax(BW, axis=1)):
                Z[idx, :] = 0
                Z[idx, k] = 1

            # Reassign columns
            BtZ = (B.T).dot(Z)
            for idx, k in enumerate(np.argmax(BtZ, axis=1)):
                W[idx, :] = 0
                W[idx, k] = 1

            k_times_k = (Z.T).dot(BW)
            m_end = np.trace(k_times_k)
            iteration += 1
            if (np.abs(m_end - m_begin) > self.tol and
                    iteration < self.max_iter):
                self.modularities.append(m_end/N)
                m_begin = m_end
                change = True

        self.row_labels_ = np.argmax(Z, axis=1).tolist()
        self.column_labels_ = np.argmax(W, axis=1).tolist()
        self.btz = BtZ
        self.bw = BW
        self.modularity = m_end / N
        self.nb_iterations = iteration

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators

        Returns
        -------
        dict
            Mapping of string to any parameter names mapped to their values
        """
        return {"init": self.init,
                "n_clusters": self.n_clusters,
                "max_iter": self.max_iter,
                "n_init": self.n_init,
                "tol": self.tol,
                "random_state": self.random_state
                }

    def set_params(self, **parameters):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        CoclustMod
            self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

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
