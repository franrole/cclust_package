# -*- coding: utf-8 -*-

"""
The :mod:`coclust.coclustering.base_diagonal_coclust` module provides a class
with common methods for diagonal co-clustering algorithms.
"""

# Author: Francois Role <francois.role@gmail.com>
#         Stanislas Morbieu <stanislas.morbieu@gmail.com>

# License: BSD 3 clause

import numpy as np


class BaseDiagonalCoclust(object):
    def get_indices(self, i):
        """Give the row and column indices of the i’th co-cluster.

        Parameters
        ----------
        i : integer
            Index of the co-cluster

        Returns
        -------
        (list, list)
            (row indices, column indices)
        """
        row_indices = [index for index, label in enumerate(self.row_labels_)
                       if label == i]
        column_indices = [index for index, label
                          in enumerate(self.column_labels_) if label == i]
        return (row_indices, column_indices)

    def get_shape(self, i):
        """Give the shape of the i’th co-cluster.

        Parameters
        ----------
        i : integer
            Index of the co-cluster

        Returns
        -------
        (int, int)
            (number of rows, number of columns)
        """
        row_indices, column_indices = self.get_indices(i)
        return (len(row_indices), len(column_indices))

    def get_submatrix(self, m,  i):
        """Give the submatrix corresponding to co-cluster i.

        Parameters
        ----------
        m : X : numpy array or scipy sparse matrix
            Matrix from which the block has to be extracted
        i : integer
           index of the co-cluster

        Returns
        -------
        numpy array or scipy sparse matrix
            Submatrix corresponding to co-cluster i
        """
        row_ind, col_ind = self.get_indices(i)
        row_ind = np.array(row_ind)
        col_ind = np.array(col_ind)
        return m[row_ind[:, np.newaxis], col_ind]
