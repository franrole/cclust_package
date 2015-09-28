###############################################################################
# Initialization methods

import numpy as np


def random_init(n_clusters, n_cols):
    """ Random Initialization
    """
    W_a = np.random.randint(n_clusters, size=n_cols)
    W = np.zeros((n_cols, n_clusters))
    W[np.arange(n_cols), W_a] = 1
    return W


def smart_init(X, K):
    """Initialize by ....
    Parameters
    -----------
    X: array or sparse matrix, shape (n_samples, n_features)
    k : number of requested co-clusters
    """
    pass
