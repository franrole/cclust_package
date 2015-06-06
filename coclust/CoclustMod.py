import sys
import getopt
import re
import os,glob
from math import * 
import numpy as np
from numpy import *
from collections import *
import scipy.sparse as sp
import marshal
import cPickle
import pickle
import itertools
from scipy.io import loadmat, savemat

from ..utils.initialization import random_init, smart_init



class CoclustMod(X,k,init_method):
    """ Co-clustering by graph modularity maximization
    Parameters
    ----------
    X : X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
        The matrix to be analyzed
        
    n_coclusters : int, optional, default: 2
        The number of co-clusters to form

    init :  {'random','smart'}, , optional, default: 'random'
        The initialization method to be used

    max_iter : int, optional, default: 20
        The maximum number of iterations
    Notes
    -----
    To be added 
    """
    def __init__(self, n_coclusters=2, init='random', max_iter=20) :
            self.n_clusters = n_clusters
            self.init = init
            self.max_iter = max_iter

    def fit(self, X, y=None):
        """ Perform Approximate Cut co-clustering
        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
        """
        print X.todense()

