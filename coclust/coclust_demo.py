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


from scipy.io import loadmat

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
    model = CoclustMod(n_clusters=4)
    model.fit(X)
    
    # Print modularity 
    print(model.modularity)
    predicted_row_labels = model.row_labels_
    
    # Use the computed and true row labels to compute the NMI
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
    true_row_labels = matlab_dict['gnd'].flatten()
    print("NMI:", nmi(true_row_labels, predicted_row_labels))
    
    print("\n\n#######   CoclustInfo (classic3.mat) ##########################")

    from coclust.CoclustInfo import CoclustInfo
    # Retrieve the Classic3  document-term matrix from a matlab file
    print("1) Loading data")
    file_name = "../datasets/classic3.mat"
    matlab_dict = loadmat(file_name)
    true_row_labels=matlab_dict['labels'].flatten()
    X = matlab_dict['A']
    

    print("2) Co-clustering")
    # Create and fit a model with 3 co-clusters
    model = CoclustInfo(n_row_clusters=3, n_col_clusters=3, n_init=4)
    model.fit(X)
    
    # Print criterion 
    print(model.criterion)
    print(model.criterions)
    predicted_row_labels = model.row_labels_
    
    # Use the computed and true row labels to compute the NMI
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
    print("NMI:", nmi(true_row_labels, predicted_row_labels))
    
    print("\n\n ####### CoclustSpecMod Usage (classic3.csv) ###############")
    

    import csv
    from coclust.CoclustSpecMod import CoclustSpecMod
    print("1) Loading data")
    file_name = "d:/recherche/cclust_package/datasets/classic3.csv"
    csv_file = open(file_name, 'r')
    csv_reader = csv.reader(csv_file, delimiter=",")
    
    for i, row in enumerate(csv_reader):
        if i == 0 :
            nb_row, nb_col = map(int,row)
            X = sp.lil_matrix((nb_row, nb_col))
        else:
            i, j, v = map(int,row)
            X[i, j] = v
            
    print("2) Co-clustering")        
    model = CoclustSpecMod(n_clusters=3)
    model.fit(X)
    
    predicted_row_labels = model.row_labels_
    # Use the computed and true row labels to compute the NMI
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
    print("NMI:", nmi(true_row_labels, predicted_row_labels))
    
    for i in range(3):
        number_of_rows, number_of_columns = model.get_shape(i)
        print("Cluster", i, "has", number_of_rows, "rows and",
              number_of_columns, "columns.")
        
    

    
    
    print("\n\n ###########  Scikit Pipeline Demo ########################")
    
    
    from coclust.CoclustMod import CoclustMod
    
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.metrics.cluster import normalized_mutual_info_score
    
    categories = [
        'rec.motorcycles',
        'rec.sport.baseball',
        'comp.graphics',
        'sci.space',
        'talk.politics.mideast'
    ]
    
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
