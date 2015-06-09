###############################################################################
# Initialization methods

import numpy as np
from numpy import *
import scipy.sparse as sp


def random_init(n_clusters,n_cols):
    """ Random Initialization
    """
    W_a=np.random.randint(n_clusters,size=n_cols)
    W=np.zeros((n_cols,n_clusters))
    W[np.arange(n_cols) , W_a]=1
    W=sp.lil_matrix(W,dtype=float)
    return W
    

def smart_init(X, K):
    """Initialize by ....
    Parameters
    -----------
    X: array or sparse matrix, shape (n_samples, n_features)
    k : number of requested co-clusters
    """
    pass
