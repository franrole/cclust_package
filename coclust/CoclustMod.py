# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp

# from __future__ import absolute_import


from .utils.initialization import random_init
print(__name__)


class CoclustMod(object):

    """ Co-clustering by approximate cut minimization
    Parameters
    ----------
    X : numpy array or scipy sparse matrix, shape (n_samples, n_features)
        The matrix to be analyzed

    n_clusters : int, optional, default: 2
        The number of co-clusters to form

    init : numpy array or scipy sparse matrix, shape (n_features, n_clusters),
           optional, default: None
        The initial column labels

    max_iter : int, optional, default: 20
        The maximum number of iterations

    Attributes
    ----------
    row_labels_ : array-like, shape (n_rows,)
        The bicluster label of each row.

    column_labels_ : array-like, shape (n_cols,)
        The bicluster label of each column.

    modularity : float
        The final value of the modularity.
    References
    ----------
    * Ailem M.,  Role F., Nadif M., 2015. `Co-clustering Document-term Matrices
    by Direct Maximization of Graph Modularity`

      <http://....>`__.
    Notes
    -----
    To be added
    """
    def __init__(self, n_clusters=2, init=None, max_iter=20):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.row_labels_ = None
        self.column_labels_ = None
        self.modularity = float("NaN")

    def fit(self, X, y=None):
        """ Perform Approximate Cut co-clustering
        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
        """

        if not sp.issparse(X):
            X = np.matrix(X)

        if self.init is None:
            W = random_init(self.n_clusters, X.shape[1])
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

        # Loop
        m_begin = float("-inf")
        change = True
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

            if np.abs(m_end - m_begin) > 1e-9:
                m_begin = m_end
                change = True

        self.row_labels_ = np.argmax(Z, axis=1).tolist()
        self.column_labels_ = np.argmax(W, axis=1).tolist()

        print "Modularity", m_end / N
        self.modularity = m_end / N

    def get_indices(self, i):  # Row and column indices of the i’th bicluster.
        row_indices = [index for index, label in enumerate(self.row_labels_)
                       if label == i]
        column_indices = [index for index, label
                          in enumerate(self.column_labels_) if label == i]
        return (row_indices, column_indices)

    def get_shape(self, i):         # Shape of the i’th bicluster.
        row_indices, column_indices = self.get_indices(i)
        return (len(row_indices), len(column_indices))

    def get_submatrix(self, i):
        pass
