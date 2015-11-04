# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import multiprocessing
import ctypes
from scipy.io import loadmat
import sys
import os

if __name__ == '__main__' :
    def random_init(n_cols,n_clusters,):
        """ Random Initialization
        """
        W_a=np.random.randint(n_clusters,size=n_cols)
        W=np.zeros((n_cols,n_clusters))
        W[np.arange(n_cols) , W_a]=1
        W=multiprocessing.Array(ctypes.c_double, W.flatten())
        W = np.ctypeslib.as_array(W.get_obj())
        W = W.reshape(n_cols ,n_clusters)
        return W

    n_clusters=3
    file_name="../datasets/cstr.mat"  # cstr.mat fea ; ng20.mat dtm ; classic3.mat A
    key="fea"
    matlab_dict=loadmat(file_name)

    print file_name.upper()

    X=matlab_dict[key]

    # 3 co-clusters 
    X=sp.lil_matrix([ [ 1, 2,3,4, 0,0,0,0 ],
                    [ 1, 2,3,4, 0,0,0,0 ],
                    [ 0,0,0,0 , 2,1,0,0 ],
                    [ 0,0,0,0 , 2,2,0,0 ],
                    [ 0,0,0,0 , 0,0,3,2 ],
                    [ 0,0,0,0 , 0,0,3,2 ],
                    ], dtype=float)
                    

    n_rows=X.shape[0]
    n_cols=X.shape[1]


    nb_process=2

    if not sp.issparse(X) :
        X=sp.lil_matrix(X)

        
    # Shared versions of W and Z
    W = random_init(n_cols,n_clusters)
    print "Initial W" ,W
    print
    #Z=np.zeros((n_rows,n_clusters))
    Z=multiprocessing.Array(ctypes.c_double, [0.] * ( n_rows * n_clusters ))
    Z = np.ctypeslib.as_array(Z.get_obj())
    Z = Z.reshape(n_rows ,n_clusters)

    # Compute the modularity matrix B
    row_sums = sp.lil_matrix(X.sum(axis=1))
    col_sums = sp.lil_matrix(X.sum(axis=0))
    N = float(X.sum())
    indep = (row_sums * col_sums) / N

    B = X - indep  # lil - lil = csr ...
    B=B.toarray()

    # Shared versions of B, BW, BtZ, k_times_k
    B=multiprocessing.Array(ctypes.c_double, B.flatten())
    B= np.ctypeslib.as_array(B.get_obj())
    B = B.reshape(X.shape[0] ,X.shape[1])
    print B.base.base
    #print type(B.base.base)

    BW=multiprocessing.Array(ctypes.c_double, X.shape[0] *  W.shape[1] )
    BW= np.ctypeslib.as_array(BW.get_obj())
    BW = BW.reshape(X.shape[0] , W.shape[1])

    BtZ=multiprocessing.Array(ctypes.c_double, X.shape[1] *  W.shape[1] )
    BtZ= np.ctypeslib.as_array(BtZ.get_obj())
    BtZ = BtZ.reshape(X.shape[1] , W.shape[1])

    k_times_k=multiprocessing.Array(ctypes.c_double, [0.] * ( n_clusters * n_clusters ))
    k_times_k= np.ctypeslib.as_array(k_times_k.get_obj())
    k_times_k = k_times_k.reshape(n_clusters ,n_clusters)

    def mult_BW(t):
        print "mult_BW" , os.getpid()
        begin=t[0]
        end=t[1]
        print "BW[begin:end,:] - before", os.getpid() ,  BW[begin:end,:]
        print
        print "W - before", os.getpid() ,  W
        BW[begin:end,:] = B[begin:end,:].dot(W)
        print "W - after", os.getpid() ,  W
        print
        print "BW[begin:end,:] - after",os.getpid() , BW[begin:end,:]

    def mult_BtZ(t):
        print "mult_BtZ" , os.getpid()
        begin=t[0]
        end=t[1]
        print "BtZ[begin:end,:] - before", os.getpid() ,  BtZ[begin:end,:]
        print "Z - before", os.getpid() ,  Z
        BtZ[begin:end,:] = (B.T)[begin:end,:].dot(Z)
        print "Z - after", os.getpid() ,  Z
        print "BtZ[begin:end,:] - after",os.getpid() , BtZ[begin:end,:]

    def mult_ZtBW(t):
        begin=t[0]
        end=t[1]
        k_times_k[begin:end,:] = (Z.T)[begin:end,:].dot(BW)



    # Loop
    m_begin = float("-inf")
    change = True
    while change:
        change = False
        print("ITERATION")

        pool = multiprocessing.Pool(processes=2)
        # Reassign rows
        pool.map(mult_BW, [(0,int(float(X.shape[0] / 2))), (int(float(X.shape[0] / 2)),X.shape[0] ) ] )
        #pool.map(mult_BW, [(0,237), (237,475 ) ] )    # compute BW
        for idx, k in enumerate(np.argmax(BW, axis=1)):
            Z[:,:]=0
            Z[np.arange(n_rows) , np.argmax(BW, axis=1)]=1
        print "Z is now" , Z
        print
        # Reassign columns
        pool.map(mult_BtZ, [(0,int(float(X.shape[1] / 2))), (int(float(X.shape[1] / 2)),X.shape[0] ) ] )    # compute BtZ
        for idx, k in enumerate(np.argmax(BtZ, axis=1)):
            W[:,:]=0
            W[np.arange(n_cols) , np.argmax(BtZ, axis=1)]=1
        print "*********** Pool map for BtZ starts **********\n"

        # Compute k_times_k matrix and its trace
        pool.map(mult_ZtBW, [(0,n_clusters/2), (n_clusters/2,n_clusters) ] )    # compute k_times_k

        m_end = np.trace(k_times_k)
        if np.abs(m_end - m_begin) > 1e-9:
            m_begin = m_end
            change = True

    print "Modularity", m_end/N
        
    print
    print("W")
    print W
    print
    print("Z")
    print Z

    ##from sklearn.metrics.cluster import normalized_mutual_info_score
    ##print "NMI" , normalized_mutual_info_score(labels, predicted)

    ##B= sp.lil_matrix(B)
    ##W= sp.lil_matrix([[ 0. , 1. , 0.],
    ## [ 0. , 0.,  1.],
    ## [ 1. , 0. , 0.],
    ## [ 1. , 0. , 0.],
    ## [ 1. , 0. , 0.],
    ## [ 1. , 0. , 0.],
    ## [ 1. , 0.,  0.],
    ## [ 1. , 0. , 0.]])
    ##
    ##print (B * W).todense()

