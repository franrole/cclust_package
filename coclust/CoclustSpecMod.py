# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


class CoclustSpecMod(object):

    """ Co-clustering by spectral approximation of the  modularity matrix
    Parameters
    ----------
    X : numpy array or scipy sparse matrix, shape (n_samples, n_features)
        The matrix to be analyzed
    n_clusters : int, optional, default: 2
        The number of co-clusters to form
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
    * Labiod L., Nadif M., ICONIP'11 Proceedings of the 18th international conference on Neural Information Processing - Volume Part II
Pages 700-708 
    """
    def __init__(self, n_clusters=2, init=None, max_iter=20):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.row_labels_ = None
        self.column_labels_ = None

    def fit(self, X, y=None):
        """ Perform co-clustering by spectral approximation of the  modularity matrix
        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
        """

        if not sp.issparse(X):
            X = np.matrix(X)

        #### Compute diagonal matrices D_r and D_c 

        D_r=np.diag(np.asarray(X.sum(axis=1)).flatten())
        D_c=np.diag(np.asarray(X.sum(axis=0)).flatten()) 

        #### Compute weighted X

        D_r**=(-1./2)
        D_r[D_r == inf] = 0

        D_c=D_c**(-1./2)
        D_c[D_c == inf] = 0

        X_tilde= D_r * X * D_c

        #### Compute the g-1 largest eigenvectors of X_tilde

        U, s, V=svds(A, k=g)
        V=V.transpose()

        #### Form matrices U-tilde and V_tilde and stack them to form Q

        U=D_r * U   # verifier type U  nd-array ou matrice ??? Convertir en csr ?
                    # D_r vaut ici D_r_initial **-1/2 alors que doit etre D_r**1/2 ???
        norm=np.linalg.norm(U, axis=0)
        U_tilde=U/norm

        V=D_c * V   # verifier type U  nd-array ou matrice ??? Convertir en csr ?
                    # D_r vaut ici D_r_initial **-1/2 alors que doit etre D_r**1/2
        norm=np.linalg.norm(V, axis=0)
        V_tilde=V/norm

        Q=numpy.concatenate((U_tilde, V_tilde), axis=0)


        #### kmeans

        k_means = KMeans(init='k-means++', n_clusters=g, n_init=10)
        k_means.fit(Q)
        k_means_labels = k_means.labels_
        
        nb_rows=X.shape[0]
        
        self.row_labels_ = k_means_labels[0:nb_rows]
        self.column_labels_ = k_means_labels[nb_rows:]




        
    def get_params(self, deep=True):
        """ Perform ...
        ----------
        deep : boolean
        """
        return {"init": self.init,
                "n_clusters": self.n_clusters,
                "max_iter": self.max_iter
                }

    def set_params(self, **parameters):
        """ ...
        ----------
        parameters
        """
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def get_indices(self, i):  # R
        """ Give the row and column indices of the i’th bicluster.
        ----------
        i : integer
        """
        row_indices = [index for index, label in enumerate(self.row_labels_)
                       if label == i]
        column_indices = [index for index, label
                          in enumerate(self.column_labels_) if label == i]
        return (row_indices, column_indices)

    def get_shape(self, i):         
        """ # Give the shape of the i’th bicluster.
        ----------
        i : integer
        """
        row_indices, column_indices = self.get_indices(i)
        return (len(row_indices), len(column_indices))

    def get_submatrix(self, i):
        pass
