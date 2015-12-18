# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:23:46 2015

@author: frole
"""

from __future__ import print_function
import argparse
import numpy as np
import scipy.sparse as sp
import sys
import matplotlib.pyplot as plt


from scipy.io import loadmat

def plot_reorganized_matrix(X,model,precision=0.8, markersize=0.9):
    row_indices=np.argsort(model.row_labels_)
    col_indices=np.argsort(model.column_labels_)
    X_reorg=X[row_indices,:]
    X_reorg=X_reorg[:,col_indices]
    plt.spy(X_reorg,precision=precision, markersize=markersize)
    plt.show()

def plot_convergence(criteria, criterion_name,marker='o'):
    plt.plot(criteria,marker=marker)
    plt.ylabel(criterion_name)
    plt.xlabel('Iterations')
    plt.show()
    plt.show()

def print_NMI_and_ARI(true_labels, predicted_labels) :
     from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
     from sklearn.metrics.cluster import adjusted_rand_score
     print("NMI:", nmi(true_labels, predicted_labels))
     print("ARI:", adjusted_rand_score(true_labels, predicted_labels))


def show_terms(model,terms,idx,limit=20,t='d') :
     print("TERMS:")
     i=0
     if t == 'd' :
        for idx in model.get_indices(idx)[1] :
            print(terms[idx])
            i+=1
            if i >= limit : break
     else :
        for idx in model.get_row_indices(idx):
            print(terms[idx])
            i+=1
            if i >= limit : break
        
import json
##    with open("d:/recherche/cclust_package/datasets/classic3-terms.json",'w') as f :
##        json.dump([matlab_dict['ms'][i][0][0] for i in np.arange(len(matlab_dict['ms']))] ,f)
terms=[]
with open("d:/recherche/cclust_package/datasets/classic3-terms.json",'r') as f :
    terms=json.load(f)
    print(len(terms))

def main_coclust_demo():
    
    print("########## CoclustMod Usage (CSTR dataset) #############")
    
    #from coclust.CoclustMod import CoclustMod
    
    from .CoclustMod import CoclustMod
    
    # Retrieve the CSTR  document-term matrix from a matlab file
    print("1) Loading data")
    file_name = "../datasets/cstr.mat"
    matlab_dict = loadmat(file_name)
    X = matlab_dict['fea']
    
    # Create and fit a model with 4 co-clusters
    print("2) Co-clustering") 
    model = CoclustMod(n_clusters=4,n_init=4)
    model.fit(X)

    
    
    # Plot modularities
    plot_convergence(model.modularities,'Modularities',marker='o')

    # Print best modularity value
    print("MODULARITY:", model.modularity)
    
    # Use the computed and true row labels to compute and print the NMI
    predicted_row_labels = model.row_labels_
    true_row_labels = matlab_dict['gnd'].flatten()
    print_NMI_and_ARI( true_row_labels, predicted_row_labels)

    # Plot reorganized matrix
    plot_reorganized_matrix(X,model)


    print("\n\n########## CoclustMod Usage (classic3.mat) #############")
    
    #from coclust.CoclustMod import CoclustMod
    
    from .CoclustMod import CoclustMod
    
    # Retrieve the classic  document-term matrix from a matlab file
    print("1) Loading data")
    file_name = "../datasets/classic3.mat"
    matlab_dict = loadmat(file_name)
    X = matlab_dict['A']
    # Create and fit a model with 3 co-clusters
    print("2) Co-clustering") 
    model = CoclustMod(n_clusters=3,n_init=4)
    model.fit(X)

    # Print best modularity
    print("MODULARITY:",model.modularity)
    # Use the computed and true row labels to compute the NMI
    true_row_labels=matlab_dict['labels'].flatten()
    predicted_row_labels = model.row_labels_
    print_NMI_and_ARI( true_row_labels, predicted_row_labels)

    # Show terms
    show_terms(model,terms,1,limit=20,t='d')
    

    
    print("\n\n#######   CoclustInfo (classic3.mat) ##########################")

    from coclust.CoclustInfo import CoclustInfo
    # Retrieve the Classic3  document-term matrix from a matlab file
    print("1) Loading data")
    file_name = "../datasets/classic3.mat"
    matlab_dict = loadmat(file_name)
    X = matlab_dict['A']
    print("Keys")
    print( matlab_dict.keys())
    

    print("2) Co-clustering")
    # Create and fit a model with 3 co-clusters
    model = CoclustInfo(n_row_clusters=3, n_col_clusters=3, n_init=4)
    model.fit(X)
    
    # Print criterion 
    print(model.criterion)

    # Plot criteria
    #plot_convergence(model.criterions, 'P_KL MI',marker='o')
    
    
    # Use the computed and true row labels to compute the NMI
    # Print best modularity
    print("CRITERION:",model.criterion)
    # Use the computed and true row labels to compute the NMI
    true_row_labels=matlab_dict['labels'].flatten()
    predicted_row_labels = model.row_labels_
    print_NMI_and_ARI( true_row_labels, predicted_row_labels)

    #plot_reorganized_matrix(X,model)

    # Display terms
    print("TERMS:")
    show_terms(model,terms,0,limit=10,t='nd')

    
    
##    print("\n\n ####### CoclustSpecMod Usage (classic3.csv) ###############")
##    
##
##    import csv
##    from coclust.CoclustSpecMod import CoclustSpecMod
##    print("1) Loading data")
##    file_name = "d:/recherche/cclust_package/datasets/classic3.csv"
##    csv_file = open(file_name, 'r')
##    csv_reader = csv.reader(csv_file, delimiter=",")
##    
##    for i, row in enumerate(csv_reader):
##        if i == 0 :
##            nb_row, nb_col = map(int,row)
##            X = sp.lil_matrix((nb_row, nb_col))
##        else:
##            i, j, v = map(int,row)
##            X[i, j] = v
##            
##    print("2) Co-clustering")        
##    model = CoclustSpecMod(n_clusters=3)
##    model.fit(X)
##    
##    predicted_row_labels = model.row_labels_
##    import json
##    with open("../datasets/classic3.labels",'r') as f :
##        true_row_labels=np.array(json.load(f))
##    
##    # Use the computed and true row labels to compute the NMI
##    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
##    print("NMI:", nmi(true_row_labels, predicted_row_labels))
##    
##    for i in range(3):
##        number_of_rows, number_of_columns = model.get_shape(i)
##        print("Cluster", i, "has", number_of_rows, "rows and",
##              number_of_columns, "columns.")
##
##    from sklearn.metrics import confusion_matrix
##    cm=confusion_matrix(true_row_labels,predicted_row_labels, )
##    cm=np.matrix(cm)
##    print("Confusion Matrix")
##    print(cm)
##    
##
##
####    print("########## CoclustInfo Usage (CSTR dataset  4x7) #############")
####    
####    #from coclust.CoclustMod import CoclustMod
####    
####    from coclust.CoclustInfo import CoclustInfo
####    
####    # Retrieve the CSTR  document-term matrix from a matlab file
####    print("1) Loading data")
####    file_name = "../datasets/cstr.mat"
####    matlab_dict = loadmat(file_name)
####    X = matlab_dict['fea']
####    
####    # Create and fit a model with 4x7 co-clusters
####    model = CoclustInfo(n_row_clusters=4, n_col_clusters=7, n_init=4)
####    model.fit(X)
####    
####    # Print criterion 
####    print(model.criterion)
####    print(model.criterions)
####    
####    predicted_row_labels = model.row_labels_
####    
####    # Use the computed and true row labels to compute the NMI
####    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
####    true_row_labels = matlab_dict['gnd'].flatten()
####    print("NMI:", nmi(true_row_labels, predicted_row_labels))
####
####    print(model.row_labels_)
####    print(model.column_labels_)
####    
##
##    
##    
##    print("\n\n ###########  Scikit Pipeline Demo ########################")
##    
##    
##    from coclust.CoclustMod import CoclustMod
##    
##    from sklearn.datasets import fetch_20newsgroups
##    from sklearn.pipeline import Pipeline
##    from sklearn.feature_extraction.text import CountVectorizer
##    from sklearn.feature_extraction.text import TfidfTransformer
##    from sklearn.metrics.cluster import normalized_mutual_info_score
##    
##    categories = [
##        'rec.motorcycles',
##        'rec.sport.baseball',
##        'comp.graphics',
##        'sci.space',
##        'talk.politics.mideast'
#   ]
    
##    ng5 = fetch_20newsgroups(categories=categories, shuffle=True)
##    
##    true_labels = ng5.target
##    
##    pipeline = Pipeline([
##        ('vect', CountVectorizer()),
##        ('tfidf', TfidfTransformer()),
##        ('coclust', CoclustMod()),
##    ])
##    
##    pipeline.set_params(coclust__n_clusters=5)
##    pipeline.fit(ng5.data)
##    
##    predicted_labels = pipeline.named_steps['coclust'].row_labels_
##    
##    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
##    
##    print(nmi)
