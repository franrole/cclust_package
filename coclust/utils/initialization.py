"""
Initialization and sanity checking routines
"""

# Author: Francois Role <francois.role@gmail.com>
#         Stanislas Morbieu <stanislas.morbieu@gmail.com>

# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state


def random_init(n_clusters, n_cols, random_state=None):
    """ Random Initialization
    """
    random_state = check_random_state(random_state)
    W_a = random_state.randint(n_clusters, size=n_cols)
    W = np.zeros((n_cols, n_clusters))
    W[np.arange(n_cols), W_a] = 1
    return W


def check_array(a):
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
