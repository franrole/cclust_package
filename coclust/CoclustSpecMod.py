# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans


class CoclustSpecMod(object):
    """Co-clustering by spectral approximation of the modularity matrix.

    Parameters
    ----------

    n_clusters : int, optional, default: 2
        Number of co-clusters to form

    max_iter : int, optional, default: 20
        Maximum number of iterations

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different centroid
        seeds. The final results will be the best output of `n_init` consecutive
        runs in terms of inertia.

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

    def __init__(self, n_clusters=2, max_iter=20, n_init=10, epsilon=1e-9,
                 n_runs=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.epsilon = epsilon
        self.n_runs = n_runs

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

        if not sp.issparse(X):
            X = np.matrix(X)

        # Compute diagonal matrices D_r and D_c

        D_r = np.diag(np.asarray(X.sum(axis=1)).flatten())
        D_c = np.diag(np.asarray(X.sum(axis=0)).flatten())

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
                         n_init=self.n_runs,
                         max_iter=self.max_iter,
                         tol=self.epsilon)
        k_means.fit(Q)
        k_means_labels = k_means.labels_

        nb_rows = X.shape[0]

        self.row_labels_ = k_means_labels[0:nb_rows].tolist()
        self.column_labels_ = k_means_labels[nb_rows:].tolist()

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
                "n_init": self.n_init,
                "epsilon": self.epsilon,
                "n_runs": self.n_runs
                }

    def set_params(self, **parameters):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        CoclustSpecMod.CoclustSpecMod
            self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_indices(self, i):
        """Give the row and column indices of the i’th co-cluster.

        Parameters
        ----------
        i : integer
            Index of the co-cluster

        Returns
        -------
        (list, list)
            (row indices, column indices)
        """
        row_indices = [index for index, label in enumerate(self.row_labels_)
                       if label == i]
        column_indices = [index for index, label
                          in enumerate(self.column_labels_) if label == i]
        return (row_indices, column_indices)

    def get_shape(self, i):
        """Give the shape of the i’th co-cluster.

        Parameters
        ----------
        i : integer
            Index of the co-cluster

        Returns
        -------
        (int, int)
            (number of rows, number of columns)
        """
        row_indices, column_indices = self.get_indices(i)
        return (len(row_indices), len(column_indices))

    def get_submatrix(self, i, data):
        """Returns the submatrix corresponding to bicluster `i`.

        Works with sparse matrices. Only works if ``rows_`` and
        ``columns_`` attributes exist.
        """
        row_ind, col_ind = self.get_indices(i)
        return data[row_ind[:, np.newaxis], col_ind]
