# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:26:09 2015

@author: frole
"""


from __future__ import absolute_import
from unittest import TestCase
import numpy as np
from scipy.io import loadmat
import sys


from coclust.CoclustInfo import CoclustInfo




class TestCstr(TestCase):
    @classmethod
    def setUpClass(cls):
        file_name = "datasets/cstr.mat"
        matlab_dict = loadmat(file_name)
        X = matlab_dict['fea']  # numpy.ndarray
        model = CoclustInfo(n_row_clusters=4, n_col_clusters=3)
        model.fit(X)
        cls.model = model

    def test_cstr_labels_range(self):
            not_in_range = [False for label in self.model.row_labels_ if label not in range( self.model.n_row_clusters)]
            self.assertTrue(len(not_in_range) == 0)
            not_in_range = [False for label in self.model.column_labels_ if label not in range( self.model.n_col_clusters)]
            self.assertTrue(len(not_in_range) == 0)


    def test_cstr_get_row_and_col_indices(self):
        all_row_indices = get_row_indices(self.model)
        all_col_indices = get_col_indices(self.model)
        if sys.version_info[0] < 3:
            self.assertItemsEqual(all_row_indices,
                                  range(len(self.model.row_labels_)))
            self.assertItemsEqual(all_col_indices,
                                  range(len(self.model.column_labels_)))                    
        else :
            self.assertCountEqual(all_row_indices,
                                  range(len(self.model.row_labels_)))
            self.assertCountEqual(all_col_indices,
                                  range(len(self.model.column_labels_)))
           
            
#
#
class TestClassic3(TestCase):
    @classmethod
    def setUpClass(cls):
        file_name = "datasets/classic3.mat"
        matlab_dict = loadmat(file_name)
        X = matlab_dict['A']  # scipy.sparse.csc.csc_matrix
        model = CoclustInfo(n_row_clusters=4, n_col_clusters=3)
        model.fit(X)
        cls.model = model
        
    def test_classic3_labels_range(self):
            not_in_range = [False for label in self.model.row_labels_ if label not in range( self.model.n_row_clusters)]
            self.assertTrue(len(not_in_range) == 0)
            not_in_range = [False for label in self.model.column_labels_ if label not in range( self.model.n_col_clusters)]
            self.assertTrue(len(not_in_range) == 0)


    def test_classic3_get_row_and_col_indices(self):
        all_row_indices = get_row_indices(self.model)
        all_col_indices = get_col_indices(self.model)
        if sys.version_info[0] < 3:
            self.assertItemsEqual(all_row_indices,
                                  range(len(self.model.row_labels_)))
            self.assertItemsEqual(all_col_indices,
                                  range(len(self.model.column_labels_)))                    
        else :
            self.assertCountEqual(all_row_indices,
                                  range(len(self.model.row_labels_)))
            self.assertCountEqual(all_col_indices,
                                  range(len(self.model.column_labels_)))



def get_row_indices(model):
    all_row_indices = []
    for index in range(model.n_row_clusters):
        row_indices = model.get_row_indices(index)
        all_row_indices.extend(row_indices)
    return all_row_indices
    
def get_col_indices(model):
    all_col_indices = []
    for index in range(model.n_col_clusters):
        col_indices = model.get_col_indices(index)
        all_col_indices.extend(col_indices)
    return all_col_indices