
from unittest import TestCase
import numpy as np
from scipy.io import loadmat
import sys

from coclust.coclustering import CoclustFuzzyMod


class TestCstr(TestCase):
    @classmethod
    def setUpClass(cls):
        file_name = "datasets/cstr.mat"
        matlab_dict = loadmat(file_name)
        X = matlab_dict['fea']
        model = CoclustFuzzyMod(n_clusters=4, Tu = 10, Tv = 1)
        model.fit(X)
        cls.model = model

    def test_cstr_labels_range(self):
        for labels in (self.model.row_labels_, self.model.column_labels_):
            not_in_range = [False for label in labels if label not in range(4)]
            self.assertTrue(len(not_in_range) == 0)

    def test_cstr_modularity(self):
        self.assertTrue(-1 <= self.model.modularity <= 1)

    def test_cstr_weighted_parameters(self):
        self.assertTrue(self.model.U.sum(axis=1)=np.ones(self.X.shape[0]))
        self.assertTrue(self.model.V.sum(axis=1)==np.ones(self.X.shape[1]))

    def test_cstr_get_indices(self):
        all_row_indices, all_column_indices = get_indices(self.model)
        if sys.version_info[0] < 3:
            self.assertItemsEqual(all_row_indices,
                                  range(len(self.model.row_labels_)))
            self.assertItemsEqual(all_column_indices,
                                  range(len(self.model.column_labels_)))
        else:
            self.assertCountEqual(all_row_indices,
                                  range(len(self.model.row_labels_)))
            self.assertCountEqual(all_column_indices,
                                  range(len(self.model.column_labels_)))


class TestClassic3(TestCase):
    @classmethod
    def setUpClass(cls):
        file_name = "datasets/classic3.mat"
        matlab_dict = loadmat(file_name)
        X = matlab_dict['A'] 
        model = CoclustFuzzyMod(n_clusters=3, Tu = 1, Tv = 10)
        model.fit(X)
        cls.model = model

    def test_classic3_labels_range(self):
        for labels in (self.model.row_labels_, self.model.column_labels_):
            not_in_range = [False for label in labels if label not in range(3)]
            self.assertTrue(len(not_in_range) == 0)

    def test_classic3_modularity(self):
        self.assertTrue(-1 <= self.model.modularity <= 1)
        self.assertTrue(round(self.model.modularity,4) == 0.3490)

    def test_classic3_weighted_parameters(self):
        self.assertTrue(self.model.U.sum(axis=1)=np.ones(self.X.shape[0]))
        self.assertTrue(self.model.V.sum(axis=1)==np.ones(self.X.shape[1]))

    def test_classic3_get_indices(self):
        all_row_indices, all_column_indices = get_indices(self.model)
        if sys.version_info[0] < 3:
            self.assertItemsEqual(all_row_indices,
                                  range(len(self.model.row_labels_)))
            self.assertItemsEqual(all_column_indices,
                                  range(len(self.model.column_labels_)))
        else:
            self.assertCountEqual(all_row_indices,
                                  range(len(self.model.row_labels_)))
            self.assertCountEqual(all_column_indices,
                                  range(len(self.model.column_labels_)))


def get_indices(model):
    all_row_indices = []
    all_column_indices = []

    for index in range(model.n_clusters):
        row_indices, column_indices = model.get_indices(index)
        all_row_indices.extend(row_indices)
        all_column_indices.extend(column_indices)

    return (all_row_indices, all_column_indices)
