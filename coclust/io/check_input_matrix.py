# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp


def check_array(a, pos=True):
    if not (type(a) == np.ndarray or type(a) == np.matrix or sp.issparse(a)):
        raise TypeError("Input data must be a Numpy/SciPy array or matrix")

    if (not np.issubdtype(a.dtype.type, np.integer) and
            not np.issubdtype(a.dtype.type, np.floating)):
        raise TypeError("Input array or matrix must be of a numeric type")

    if not sp.issparse(a):
        a = np.matrix(a)

        if len(np.where(~a.any(axis=0))[0]) > 0:
            raise ValueError("Zero-valued columns in data")
        if len(np.where(~a.any(axis=1))[1]) > 0:
            raise ValueError("Zero-valued rows in data")
        if pos:
            if (a < 0).any():
                raise ValueError("Negative values in data")
        if np.isnan(a).any():
            raise ValueError("NaN in data")


def check_numbers(a, n_clusters):
    if a.shape[0] < n_clusters or a.shape[1] < n_clusters:
        raise ValueError("data matrix has not enough rows or columns")


def check_numbers_non_diago(a, n_row_clusters, n_col_clusters):
    if a.shape[0] < n_row_clusters or a.shape[1] < n_col_clusters:
        raise ValueError("data matrix has not enough rows or columns")


def check_numbers_clustering(a, n_clusters):
    if a.shape[0] < n_clusters:
        raise ValueError("data matrix has not enough rows")
