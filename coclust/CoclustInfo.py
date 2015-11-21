import numpy as np
from numpy import *

import scipy.sparse as sp
 
import scipy
 
from random import randint
import itertools
from scipy.io import loadmat, savemat


mat= loadmat("../datasets/classic3.mat")
X = mat['A']



X=sp.csr_matrix(X)

N = float(X.sum())
X = X.multiply(1./N)
print("X")
print(X.todense())
nb_rows=X.shape[0]
nb_cols=X.shape[1]

K=3

## Z, W
Z=np.zeros((nb_rows,K))
Z_a=np.random.randint(K,size=nb_rows)
Z=np.zeros((nb_rows,K))
Z[np.arange(nb_rows) , Z_a]=1
Z=sp.lil_matrix(Z)


W=np.zeros((nb_cols,K))
W_a=np.random.randint(K,size=nb_cols)
W=np.zeros((nb_cols,K))
W[np.arange(nb_cols) , W_a]=1
W=sp.lil_matrix(W)

print("Z")
print(Z.todense())
print("W")
print(W.todense())
(X*W).todense()


## Initial delta
p_il=X*W  
p_il=p_il # matrice m,l ; la colonne l' contient les p_il'
p_kj=X.T*Z # matrice j,k
print("p_kj")
print(p_kj.todense())
print("p_il")
print(p_il.todense())

p_kd=p_kj.sum(axis=0)  # array contenant les p_k.
p_dl=p_il.sum(axis=0)      # array  contenant les p_.l
print("p_kd")
print(p_kd)
print("p_dl")
print(p_dl)
p_kd_times_p_dl=  p_kd.T * p_dl # p_k. p_.l ; transpose because p_kd is "horizontal"
min_p_kd_times_p_dl=np.min(p_kd_times_p_dl[np.nonzero(p_kd_times_p_dl)])
p_kd_times_p_dl[p_kd_times_p_dl == 0.] = min_p_kd_times_p_dl * 0.01
print("p_kd_times_p_dl")
print(p_kd_times_p_dl)
p_kd_times_p_dl_inv=1./p_kd_times_p_dl

p_kl= (Z.T * X ) * W
print("p_kl")
print(p_kl.todense())
delta_kl=p_kl.multiply(p_kd_times_p_dl_inv)

print("First delta")
print(delta_kl)



change=True
news=[]

n_iters=20
pkl_mi_previous=float(-inf)

## Loop 
while change and n_iters > 0:
    change=False
    
    # Update Z
    #print("Current Z")
    #print(Z.todense())
    p_il=X*W  # matrice m,l ; la colonne l' contient les p_il'
    delta_kl[delta_kl==0.]=0.0001 # to prevent log(0)
    log_delta_kl=log(delta_kl.T)
    log_delta_kl=sp.lil_matrix(log_delta_kl)
    Z1=p_il * log_delta_kl  # p_il * (d_kl)T ; on examine chaque cluster 
    Z1=Z1.toarray()
    Z =np.zeros_like(Z1)
    Z[np.arange(len(Z1)), Z1.argmax(1)] = 1 # Z[(line index 1...), (max col index for 1...)]
    Z=sp.lil_matrix(Z)
    #print("New Z")
    #print(Z.todense())

    
    # Update delta
    p_kj=X.T*Z      # matrice d,k ; la colonne k' contient les p_jk'
    # p_il unchanged
    p_dl=p_il.sum(axis=0)      # array l contenant les p_.l
    p_kd=p_kj.sum(axis=0)  # array k contenant les p_k.
    
    p_kd_times_p_dl=  p_kd.T * p_dl # p_k. p_.l ; transpose because p_kd is "horizontal"
    min_p_kd_times_p_dl=np.min(p_kd_times_p_dl[np.nonzero(p_kd_times_p_dl)])
    p_kd_times_p_dl[p_kd_times_p_dl == 0.] = min_p_kd_times_p_dl * 0.01
    p_kd_times_p_dl_inv=1./p_kd_times_p_dl
    p_kl= (Z.T * X ) * W
    #print(p_kl.todense())
    delta_kl=p_kl.multiply(p_kd_times_p_dl_inv)
    #print(delta_kl)
    #print("delta after modifying Z")
    #print(delta_kl)
    
    # Update W
    #print("Current W")
    #print(W.todense())
    p_kj=X.T*Z  # matrice m,l ; la colonne l' contient les p_il'
    delta_kl[delta_kl==0.]=0.0001 # to prevent log(0)
    log_delta_kl=log(delta_kl)
    log_delta_kl=sp.lil_matrix(log_delta_kl)
    W1=p_kj * log_delta_kl  # p_kj * d_kl ; on examine chaque cluster 
    W1=W1.toarray()
    W =np.zeros_like(W1)
    W[np.arange(len(W1)), W1.argmax(1)] = 1
    W=sp.lil_matrix(W)
    #print("New W")
    #print(W.todense())

    # Update delta
    #print("Current delta")
    #print(delta_kl)
    p_il=X*W     # matrice d,k ; la colonne k' contient les p_jk'
    # p_kj unchanged
    p_dl=p_il.sum(axis=0)      # array l contenant les p_.l
    p_kd=p_kj.sum(axis=0)  # array k contenant les p_k.

    p_kd_times_p_dl=  p_kd.T * p_dl # p_k. p_.l ; transpose because p_kd is "horizontal"
    min_p_kd_times_p_dl=np.min(p_kd_times_p_dl[np.nonzero(p_kd_times_p_dl)])
    p_kd_times_p_dl[p_kd_times_p_dl == 0.] = min_p_kd_times_p_dl * 0.01
    p_kd_times_p_dl_inv=1./p_kd_times_p_dl
    p_kl= (Z.T * X ) * W

    delta_kl=p_kl.multiply(p_kd_times_p_dl_inv)
    delta_kl[delta_kl==0.]=0.0001 # to prevent log(0) when computing criterion
    #print("delta after modifying W")
    #print(delta_kl)

    # Criterion
    #print(p_kl.todense())
    #print(log(delta_kl))
    pkl_mi=sp.lil_matrix(p_kl).multiply(sp.lil_matrix(log(delta_kl)))
    pkl_mi=pkl_mi.sum()
    print(pkl_mi)


    if np.abs(pkl_mi - pkl_mi_previous)  > 1e-9 :
        pkl_mi_previous=pkl_mi
        change=True
        news.append(pkl_mi)
        n_iters-=1




		
import matplotlib.pyplot as plt
import itertools
from scipy.io import loadmat, savemat

from sklearn.metrics import confusion_matrix
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.feature_extraction.text import *
from sklearn.metrics.cluster import adjusted_rand_score
		
labels=mat['labels']
labels=labels.tolist()
labels = list(itertools.chain.from_iterable(labels))

part=Z.todense().argmax(axis=1).tolist()
part=[item for sublist in part for item in sublist]

part2=W.todense().argmax(axis=1).tolist()
part2=[item for sublist in part2 for item in sublist]

n=normalized_mutual_info_score(labels, part)
ari=adjusted_rand_score(labels, part)

cm=confusion_matrix(labels, part)
cm=np.matrix(cm)
print (cm)

cm1=cm

cml=np.array(cm).tolist()
	

cml = list(itertools.chain(*cml))

total=0
for i in range(0,K):
	if len(cml) != 0:
		ma=max(cml)
		if (ma  in cm1): 
			index = np.where(cm1==ma)
			total = total +ma
			cml.remove(ma)
			cm1 = scipy.delete(cm1,index[0][0,0], 0)
			cm1 = scipy.delete(cm1, index[1][0,0], 1)
			cml=np.array(cm1).tolist()

			cml = list(itertools.chain(*cml))

purity=(total)/(nb_rows*1.)
print( "Accuracy ==>" + str(purity))
print ("nmi ==>" + str(n))
print ("adjusted rand index ==>" + str(ari))


plt.plot(news,marker='o')
plt.ylabel('Lc')
plt.xlabel('Iterations')
plt.show()


