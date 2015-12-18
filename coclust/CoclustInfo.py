# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
from .utils.initialization import random_init
from sklearn.utils import check_random_state


class CoclustInfo(object):
    """Co-clustering.

    Parameters
    ----------
    n_row_clusters : int, optional, default: 2
        Number of row clusters to form

    n_col_clusters : int, optional, default: 2
        Number of column clusters to form

    init : numpy array or scipy sparse matrix, shape (n_features, n_clusters), \
        optional, default: None
        Initial column labels

    max_iter : int, optional, default: 20
        Maximum number of iterations

    n_init : int, optional, default: 1
        Number of time the algorithm will be run with different initializations.
        The final results will be the best output of `n_init` consecutive runs.

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
    """

    def __init__(self, n_row_clusters=2, n_col_clusters=2, init=None,
                 max_iter=20, n_init=1, tol=1e-9, random_state=None):
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = check_random_state(random_state)

        self.row_labels_ = None
        self.column_labels_ = None
        self.criterions = []
        self.criterion = -np.inf

    def fit(self, X, y=None):
        """Perform co-clustering.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        criterion = self.criterion

        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            self.random_state = seed
            self._fit_single(X, y)

            # remember attributes corresponding to the best criterion
            if (self.criterion > criterion):
                criterion = self.criterion
                criterions = self.criterions
                row_labels_ = self.row_labels_
                column_labels_ = self.column_labels_

        self.random_state = random_state

        # update attributes
        self.criterion = criterion
        self.criterions = criterions
        self.row_labels_ = row_labels_
        self.column_labels_ = column_labels_

    def _fit_single(self, X, y=None):
        """Perform one run of co-clustering.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        K = self.n_row_clusters
        L = self.n_col_clusters

        if not sp.issparse(X):
            X = np.matrix(X)

        if self.init is None:
            W = random_init(L, X.shape[1], self.random_state)
        else:
            W = np.matrix(self.init, dtype=float)

#        Z = np.zeros((X.shape[0], K))

        # TODO
        X = sp.csr_matrix(X)

        N = float(X.sum())
        X = X.multiply(1. / N)
#        nb_rows = X.shape[0]
#        nb_cols = X.shape[1]

        Z = sp.lil_matrix(random_init(K, X.shape[0], self.random_state))
#        W = sp.lil_matrix(random_init(L, nb_cols))

        W = sp.csr_matrix(W)

        # Initial delta
        p_il = X * W
        # p_il = p_il     # matrice m,l ; la colonne l' contient les p_il'
        p_kj = X.T * Z  # matrice j,k

        p_kd = p_kj.sum(axis=0)  # array contenant les p_k.
        p_dl = p_il.sum(axis=0)  # array contenant les p_.l

        p_kd_times_p_dl = p_kd.T * p_dl  # p_k. p_.l ; transpose because p_kd is "horizontal"
        min_p_kd_times_p_dl = np.min(
            p_kd_times_p_dl[
                np.nonzero(p_kd_times_p_dl)])
        p_kd_times_p_dl[p_kd_times_p_dl == 0.] = min_p_kd_times_p_dl * 0.01
        p_kd_times_p_dl_inv = 1. / p_kd_times_p_dl

        p_kl = (Z.T * X) * W
        delta_kl = p_kl.multiply(p_kd_times_p_dl_inv)

        change = True
        news = []

        n_iters = self.max_iter
        pkl_mi_previous = float(-np.inf)

        # Loop
        while change and n_iters > 0:
            change = False

            # Update Z
            p_il = X * W  # matrice m,l ; la colonne l' contient les p_il'
            delta_kl[delta_kl == 0.] = 0.0001  # to prevent log(0)
            log_delta_kl = np.log(delta_kl.T)
            log_delta_kl = sp.lil_matrix(log_delta_kl)
            # p_il * (d_kl)T ; on examine chaque cluster
            Z1 = p_il * log_delta_kl
            Z1 = Z1.toarray()
            Z = np.zeros_like(Z1)
            # Z[(line index 1...), (max col index for 1...)]
            Z[np.arange(len(Z1)), Z1.argmax(1)] = 1
            Z = sp.lil_matrix(Z)

            # Update delta
            p_kj = X.T * Z      # matrice d,k ; la colonne k' contient les p_jk'
            # p_il unchanged
            p_dl = p_il.sum(axis=0)  # array l contenant les p_.l
            p_kd = p_kj.sum(axis=0)  # array k contenant les p_k.

            p_kd_times_p_dl = p_kd.T * p_dl  # p_k. p_.l ; transpose because p_kd is "horizontal"
            min_p_kd_times_p_dl = np.min(
                p_kd_times_p_dl[
                    np.nonzero(p_kd_times_p_dl)])
            p_kd_times_p_dl[p_kd_times_p_dl == 0.] = min_p_kd_times_p_dl * 0.01
            p_kd_times_p_dl_inv = 1. / p_kd_times_p_dl
            p_kl = (Z.T * X) * W
            delta_kl = p_kl.multiply(p_kd_times_p_dl_inv)

            # Update W
            p_kj = X.T * Z  # matrice m,l ; la colonne l' contient les p_il'
            delta_kl[delta_kl == 0.] = 0.0001  # to prevent log(0)
            log_delta_kl = np.log(delta_kl)
            log_delta_kl = sp.lil_matrix(log_delta_kl)
            W1 = p_kj * log_delta_kl  # p_kj * d_kl ; on examine chaque cluster
            W1 = W1.toarray()
            W = np.zeros_like(W1)
            W[np.arange(len(W1)), W1.argmax(1)] = 1
            W = sp.lil_matrix(W)

            # Update delta
            p_il = X * W     # matrice d,k ; la colonne k' contient les p_jk'
            # p_kj unchanged
            p_dl = p_il.sum(axis=0)  # array l contenant les p_.l
            p_kd = p_kj.sum(axis=0)  # array k contenant les p_k.

            p_kd_times_p_dl = p_kd.T * p_dl  # p_k. p_.l ; transpose because p_kd is "horizontal"
            min_p_kd_times_p_dl = np.min(
                p_kd_times_p_dl[
                    np.nonzero(p_kd_times_p_dl)])
            p_kd_times_p_dl[p_kd_times_p_dl == 0.] = min_p_kd_times_p_dl * 0.01
            p_kd_times_p_dl_inv = 1. / p_kd_times_p_dl
            p_kl = (Z.T * X) * W

            delta_kl = p_kl.multiply(p_kd_times_p_dl_inv)
            # to prevent log(0) when computing criterion
            delta_kl[delta_kl == 0.] = 0.0001

            # Criterion
            pkl_mi = sp.lil_matrix(p_kl).multiply(
                sp.lil_matrix(np.log(delta_kl)))
            pkl_mi = pkl_mi.sum()

            if np.abs(pkl_mi - pkl_mi_previous) > self.tol:
                pkl_mi_previous = pkl_mi
                change = True
                news.append(pkl_mi)
                n_iters -= 1

        self.criterions = news
        self.criterion = pkl_mi

        self.row_labels_ = Z.toarray().argmax(axis=1).tolist()
##        print( self.row_labels_)
##        self.row_labels_ = [item for sublist in self.row_labels_
##                            for item in sublist]
        
        self.column_labels_ = W.toarray().argmax(axis=1).tolist()
        

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
                "n_row_clusters": self.n_row_clusters,
                "n_col_clusters": self.n_col_clusters,
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
        CoclustMod.CoclustMod
            self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_row_indices(self, i):
        """Give the row indices of the i’th co-cluster.

        Parameters
        ----------
        i : integer
            Index of the i'th row cluster

        Returns
        -------
        list
            list of row indices
        """
        row_indices = [index for index, label in enumerate(self.row_labels_)
                       if label == i]
        return row_indices

    def get_col_indices(self, i):
        """Give the column indices of the i’th co-cluster.

        Parameters
        ----------
        i : integer
            Index of the i'th column cluster

        Returns
        -------
        list
            list of column indices
        """
        col_indices = [index for index, label in enumerate(self.column_labels_)
                       if label == i]
        return col_indices



#    def get_shape(self, i):
#        """Give the shape of the i’th co-cluster.
#
#        Parameters
#        ----------
#        i : integer
#            Index of the co-cluster
#
#        Returns
#        -------
#        (int, int)
#            (number of rows, number of columns)
#        """
#        row_indices, column_indices = self.get_indices(i)
#        return (len(row_indices), len(column_indices))
#
#    def get_submatrix(self, i, data):
#        """Returns the submatrix corresponding to bicluster `i`.
#
#        Works with sparse matrices. Only works if ``rows_`` and
#        ``columns_`` attributes exist.
#        """
#        row_ind, col_ind = self.get_indices(i)
#        return data[row_ind[:, np.newaxis], col_ind]
