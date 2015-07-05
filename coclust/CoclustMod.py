# -*- coding: utf-8 -*-
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


from .utils.initialization import random_init, smart_init

class CoclustMod(object):
    """ Co-clustering by approximate cut minimization
    Parameters
    ----------
    X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
        The matrix to be analyzed
        
    n_coclusters : int, optional, default: 2
        The number of co-clusters to form

    init :  {'random','smart'}, , optional, default: 'random'
        The initialization method to be used

    max_iter : int, optional, default: 20
        The maximum number of iterations

    Attributes
    ----------
    row_labels_ : array-like, shape (n_rows,)
        The bicluster label of each row.
    column_labels_ : array-like, shape (n_cols,)
        The bicluster label of each column.
    References
    ----------
    * Ailem M.,  Role F., Nadif M., 2015. `Co-clustering Document-term Matrices by Direct
Maximization of Graph Modularity`
      
      <http://....>`__.
    Notes
    -----
    To be added 
    """
    def __init__(self, n_clusters=2, init='random', max_iter=20,corpus='cstr') :
            self.n_clusters = n_clusters
            self.init = init
            self.max_iter = max_iter
            self.corpus=corpus

    def fit(self, X, y=None):
        """ Perform Approximate Cut co-clustering
        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
        """
        if self.init == 'random' :
            W = random_init(self.n_clusters, X.shape[1] )
            
        else :
            pass

        # Compute the modularity matrix
        row_sums=sp.lil_matrix(X.sum(axis=1))
        col_sums=sp.lil_matrix(X.sum(axis=0))
        N=float(X.sum())
        indep= (row_sums * col_sums) / N
        B=m-indep  # lil - lil = csr ...
        rcut_begin=  1e9 #float("-inf") 

        # Loop
        m_begin=float("-inf")
        change=True
        while (change):
            change=False
            
            # Reassign rows
            BW=B * W
            BW_a=BW.toarray()
            for idx , k  in enumerate(np.argmax(BW_a, axis=1)) :
                Z[idx,:]=0
                Z[idx,k]=1
                ##    Z[:,:]=0
                ##    Z[np.arange(nb_rows) , np.argmax(BW, axis=1)]=1

            # Reassign columns
            BtZ=(B.T) * Z
            BtZ_a=BtZ.toarray()
            for idx , k  in enumerate(np.argmax(BtZ_a, axis=1)) :
                W[idx,:]=0
                W[idx,k]=1
                ##    W[:,:]=0
                ##    W[np.arange(nb_cols) , np.argmax(BtZ, axis=1)]=1

            k_times_k= (Z.T) * BW
            m_end=np.trace(k_times_k.todense())# pas de trace pour sp ...

            if np.abs(m_end - m_begin) > 1e-9 :
                m_begin=m_end
                change=True


        print "Modularity" , m_end

    
    def fit(X) :  # Creates a biclustering for X.
        pass	
    def get_indices(i):  # Row and column indices of the i’th bicluster.
        pass
    def get_shape(i) :         # Shape of the i’th bicluster.
        pass
    def get_submatrix(i) :
        pass

    

