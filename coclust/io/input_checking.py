# -*- coding: utf-8 -*-

"""
The :mod:`coclust.io.input_checking` module provides functions to check
input matrices.
"""

import numpy as np
import scipy.sparse as sp


def check_array(a, pos=True):
    """Check if an array contains numeric values with non empty rows nor
    columns.

    Parameters
    ----------
    a:
        The input array
    pos: bool
        If ``True``, check if the values are positives

    Raises
    ------
    TypeError
        If the array is not a Numpy/SciPy array or matrix or if the values are
        not numeric.

    ValueError
        If the array contains empty rows or columns or contains NaN values, or
        negative values (if ``pos`` is ``True``).
    """

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


def check_numbers(matrix, n_clusters):
    """Check if the given matrix has enough rows and columns for the given
    number of co-clusters.

    Parameters
    ----------
    matrix:
        The input matrix

    n_clusters: int
        Number of co-clusters

    Raises
    ------
    ValueError
        If the data matrix has not enough rows or columns.
    """

    if matrix.shape[0] < n_clusters or matrix.shape[1] < n_clusters:
        raise ValueError("data matrix has not enough rows or columns")


def check_numbers_non_diago(matrix, n_row_clusters, n_col_clusters):
    """Check if the given matrix has enough rows and columns for the given
    number of row and column clusters.

    Parameters
    ----------
    matrix:
        The input matrix

    n_row_clusters: int
        Number of row clusters

    n_col_clusters: int
        Number of column clusters

    Raises
    ------
    ValueError
        If the data matrix has not enough rows or columns.
    """

    if matrix.shape[0] < n_row_clusters or matrix.shape[1] < n_col_clusters:
        raise ValueError("data matrix has not enough rows or columns")


def check_numbers_clustering(matrix, n_clusters):
    """Check if the given matrix has enough rows and columns for the given
    number of clusters.

    Parameters
    ----------
    matrix:
        The input matrix

    n_clusters: int
        Number of clusters

    Raises
    ------
    ValueError
        If the data matrix has not enough rows or columns.
    """

    if matrix.shape[0] < n_clusters:
        raise ValueError("data matrix has not enough rows")
