# -*- coding: utf-8 -*-

"""
The :mod:`coclust.coclustering.coclust_spec_mod` module provides an
implementation of a co-clustering algorithm by spectral approximation of the
modularity matrix.
"""

# Author: Francois Role <francois.role@gmail.com>
#         Stanislas Morbieu <stanislas.morbieu@gmail.com>

# License: BSD 3 clause

import numpy as np
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans

from ..io.input_checking import check_array, check_numbers
from .base_diagonal_coclust import BaseDiagonalCoclust


class CoclustSpecMod(BaseDiagonalCoclust):
    """Co-clustering by spectral approximation of the modularity matrix.

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
    * Labiod L., Nadif M., ICONIP'11 Proceedings of the 18th international \
    conference on Neural Information Processing - Volume Part II Pages 700-708
    """

    def __init__(self, n_clusters=2, max_iter=20, tol=1e-9, n_init=1,
                 random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.row_labels_ = None
        self.column_labels_ = None

    def fit(self, X, y=None):
        """Perform co-clustering by spectral approximation of the modularity
        matrix

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """

        check_array(X)

        check_numbers(X, self.n_clusters)

        X = X.astype(float)

        # Compute diagonal matrices D_r and D_c

        D_r = np.diag(np.asarray(X.sum(axis=1)).flatten())
        D_c = np.diag(np.asarray(X.sum(axis=0)).flatten())

        try:

            # Compute weighted X
            with np.errstate(divide='ignore'):
                D_r **= (-1./2)
                D_r[D_r == np.inf] = 0

                D_c = D_c**(-1./2)
                D_c[D_c == np.inf] = 0

            D_r = np.matrix(D_r)
            D_c = np.matrix(D_c)

            X_tilde = D_r * X * D_c

            # Compute the g-1 largest eigenvectors of X_tilde

            U, s, V = svds(X_tilde, k=self.n_clusters)
            V = V.transpose()

            # Form matrices U-tilde and V_tilde and stack them to form Q

            U = D_r * U
            # TODO:
            # verifier type U  nd-array ou matrice ??? Convertir en csr ?
            # D_r vaut ici D_r_initial **-1/2 alors que doit etre D_r**1/2 ???

            norm = np.linalg.norm(U, axis=0)
            U_tilde = U/norm

            V = D_c * V
            # TODO:
            # verifier type U  nd-array ou matrice ??? Convertir en csr ?
            # D_r vaut ici D_r_initial **-1/2 alors que doit etre D_r**1/2

            norm = np.linalg.norm(V, axis=0)
            V_tilde = V/norm

            Q = np.concatenate((U_tilde, V_tilde), axis=0)

            # kmeans

            k_means = KMeans(init='k-means++',
                             n_clusters=self.n_clusters,
                             n_init=self.n_init,
                             max_iter=self.max_iter,
                             tol=self.tol,
                             random_state=self.random_state)
            k_means.fit(Q)
            k_means_labels = k_means.labels_

            nb_rows = X.shape[0]

            self.row_labels_ = k_means_labels[0:nb_rows].tolist()
            self.column_labels_ = k_means_labels[nb_rows:].tolist()

        except:
            raise ValueError("matrix may contain unexpected NaN values")

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
        return {"n_clusters": self.n_clusters,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "n_init": self.n_init,
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
        CoclustSpecMod
            self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
