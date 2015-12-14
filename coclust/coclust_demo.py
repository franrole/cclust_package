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
    file_name = "../datasets/cstr.mat"
    matlab_dict = loadmat(file_name)
    X = matlab_dict['fea']
    
    # Create and fit a model with 4 co-clusters
    model = CoclustMod(n_clusters=4)
    model.fit(X)
    
    # Print modularity 
    print(model.modularity)
    predicted_row_labels = model.row_labels_
    
    # Use the computed and true row labels to compute the NMI
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
    true_row_labels = matlab_dict['gnd'].flatten()
    print(nmi(true_row_labels, predicted_row_labels))
    
    
    
    print("\n\n #### CoclustSpecMod Usage (Classic3 dataset) #################")
    

    import csv
    from coclust.CoclustSpecMod import CoclustSpecMod
    
    file_name = "../datasets/classic3.csv"
    csv_file = open(file_name, 'rb')
    csv_reader = csv.reader(csv_file, delimiter=",")
    
    nb_row, nb_col, nb_clusters = map(int, csv_reader.next())
    X = sp.lil_matrix((nb_row, nb_col))
    
    for row in csv_reader:
        i, j, v = map(int, row)
        X[i, j] = v
    
    model = CoclustSpecMod(n_clusters=nb_clusters)
    model.fit(X)
    
    predicted_row_labels = model.row_labels_
    
    for i in range(nb_clusters):
        number_of_rows, number_of_columns = model.get_shape(i)
        print("Cluster", i, "has", number_of_rows, "rows and",
              number_of_columns, "columns.")
        
    print("\n\n####   CoclustInfo (?) ##############################")
    
    
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
    
    ng5 = fetch_20newsgroups(categories=categories, shuffle=True)
    
    true_labels = ng5.target
    
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('coclust', CoclustMod()),
    ])
    
    pipeline.set_params(coclust__n_clusters=5)
    pipeline.fit(ng5.data)
    
    predicted_labels = pipeline.named_steps['coclust'].row_labels_
    
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    
    print(nmi)