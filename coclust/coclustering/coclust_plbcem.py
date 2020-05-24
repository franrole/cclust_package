# -*- coding: utf-8 -*-

"""
PLBcem
"""

# Author: Melissa Ailem <melissa.ailem@parisdescartes.fr>
#         Francois Role <francois.role@parisdescartes.fr>

# License: BSD 3 clause
import itertools
from math import *
from scipy.io import loadmat, savemat
import sys
import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state, check_array
#from coclust.utils.initialization import (random_init, check_numbers,check_array)
# use sklearn instead FR 08-05-19
from .base_non_diagonal_coclust import BaseNonDiagonalCoclust
from ..initialization import random_init
from ..io.input_checking import check_positive

# from pylab import *


class CoclustPLBcem:
    """Clustering.
    Parameters
    ----------
    n_row_clusters : int, optional, default: 2
        Number of clusters to form
    n_col_clusters : int, optional, default: 2
        Number of clusters to form
    init : numpy array or scipy sparse matrix, \
        shape (n_features, n_clusters), optional, default: None
        Initial column or row labels
    max_iter : int, optional, default: 100
        Maximum number of iterations
    n_init : int, optional, default: 1
        Number of time the algorithm will be run with different
        initializations. The final results will be the best output of `n_init`
        consecutive runs.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    tol : float, default: 1e-9
        Relative tolerance with regards to criterion to declare convergence
    Attributes
    ----------
    row_labels_ : array-like, shape (n_rows,)
        cluster label of each row
    column_labels_ : array-like, shape (n_cols,)
        Bicluster label of each column
    criterion : float
        criterion obtained from the best run
    criterions : list of floats
        sequence of criterion values during the best run
    """

    def __init__(self, n_row_clusters=2, n_col_clusters=2, init=None,
                 max_iter=100, n_init=1, tol=1e-9, random_state=None):
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters=n_col_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.row_labels_ =None
        self.column_labels_=None
        self.criterions = []
        self.criterion = -np.inf
        
    def fit(self, X, y=None):
        """Perform clustering.
        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """

        check_array(X, accept_sparse=True, dtype="numeric", order=None,
                    copy=False, force_all_finite=True, ensure_2d=True,
                    allow_nd=False, ensure_min_samples=self.n_row_clusters,
                    ensure_min_features=self.n_col_clusters, estimator=None)

        check_positive(X)

        criterion = self.criterion

        X = sp.csr_matrix(X)
        
        #X = X.astype(float)
        X=X.astype(int)

        random_state = check_random_state(self.random_state) 
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            self._fit_single(X, seed, y)
            if np.isnan(self.criterion):
                raise ValueError("matrix may contain negative or unexpected NaN values")
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

        
                
        
    def _fit_single(self, X, random_state, y=None) :
        # X=X.astype(int)
        K = self.n_row_clusters
        L = self.n_col_clusters
        
        N = X.sum()
        const = 1./(1.*N*N) ##safety parameter to avoid log(0) and  division by zero
        
        #init of row labels
        if self.init is None:
            Z = random_init(K, X.shape[0], random_state)   
        else:
            Z = np.matrix(self.init, dtype=float)

       
        Z=sp.lil_matrix(Z) # random_init function returns a nd_array
        
        # init of column labels
        if self.init is None:
            W = random_init(L, X.shape[1], random_state)
        else:
            W = np.matrix(self.init, dtype=float)
        
        W = sp.lil_matrix(W)
        X = sp.lil_matrix(X)

        #initial pik row proportion 
        n = Z.sum()
        pik = Z.sum(axis=0)
        pik = pik/n
        pik = np.squeeze(np.asarray(pik))

        #initial pl column proportions 
        d = W.sum()
        pl = W.sum(axis=0)
        pl = pl/d
        pl = np.squeeze(np.asarray(pl))
        
        ##########gamakl

        Xw = X*W
        Xz = X.T*Z        
        Xw_l = Xw.sum(axis=0)
        Xz_k = Xz.sum(axis=0)


        den =  Xz_k.T * Xw_l
        den = 1/(den+const)
        gammakl = ((Z.T * X) * W).multiply(den)
        
        
        #######loop
        change = True
        c_init = float(-np.inf)
        c_list = []
        iteration = 0

        while change :
            change = False
            
            ########### rows
            Xw = X * W
    
            ### CE step
            gammakl = sp.csr_matrix(gammakl)
            gammakl.data = np.log((gammakl).data)
	
            Z1 = Xw * gammakl.T
            Z1 = Z1 + np.log(pik+const)
            Z1 = sp.csr_matrix(Z1)
            Z = sp.lil_matrix((Z1.shape[0],K))
            Z[np.arange(Z1.shape[0]), Z1.argmax(1).A1] = 1	
          
    
            ### M step
            ### proportions
            n=Z.sum()
            pik=Z.sum(axis=0)
            pik=pik/n
            pik=np.squeeze(np.asarray(pik))

            #### parameters gammakl
            Xz=X.T * Z 
            Xw_l=Xw.sum(axis=0)
            Xz_k=Xz.sum(axis=0)
            Num=Z.T * Xw
            Den= Xz_k.T * Xw_l 
            Den=1/(Den+const)
            gammakl=Num.multiply(Den) 
            # !!! gammakl has been transformed to a (non-subscriptable)
            # COO matrix. Convert it back to CSR  FR 08-05-19
            gammakl = gammakl.tocsr()
            
            #####avoid zero in gammakl matrix
            
            minval = np.min(gammakl[np.nonzero(gammakl)]) 
            gammakl[gammakl == 0] = minval*0.00000001

            
            

            ################################################## Columns
            Xz= X.T * Z 
            
            ### CE step

            gammakl=sp.csr_matrix(gammakl)
            gammakl.data=np.log((gammakl).data)
	
            W1 = Xz * gammakl
            W1 = W1+np.log(pl+const)

            W1 = sp.csr_matrix(W1)
            W = sp.lil_matrix((W1.shape[0],L))
            W[np.arange(W1.shape[0]), W1.argmax(1).A1] = 1


            ### M step
            # proportions
            d=W.sum()
            pl=W.sum(axis=0)
            pl=pl/d
            pl=np.squeeze(np.asarray(pl))

            #### gammakl        
            Xw = X * W 
            Xw_l = Xw.sum(axis=0)    
            Xz_k=Xz.sum(axis=0)
            Num= W.T* Xz 
            Den=Xz_k.T  * Xw_l
            Den=1/(Den+const)    
            gammakl=(Num.T).multiply(Den)
            gammakl = gammakl.tocsr()
            minval=np.min(gammakl[np.nonzero(gammakl)]) 
            gammakl[gammakl == 0] = minval*0.00000001


            #criterion
            sumk=Z.sum(axis=0) * np.log(pik+const)[:,None]
            suml=W.sum(axis=0) * np.log(pl+const)[:,None]
            logamma=np.log(gammakl.todense()) # convert to dense because of log FR 08-05-19
            c=sp.lil_matrix(logamma).multiply(sp.lil_matrix(Num.T))
            c=c.sum() + sumk.sum() + suml.sum()
                    

            iteration += 1
            if (np.abs(c - c_init)  > self.tol and iteration < self.max_iter): 
                c_init=c
                change=True
                c_list.append(c)

        

        self.max_iter = iteration
        self.criterion=c
        self.criterions=c_list
        self.row_labels_ = Z.toarray().argmax(axis=1).tolist()
        self.column_labels_ = W.toarray().argmax(axis=1).tolist()
		
		
		
		
		
		
