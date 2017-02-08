# -*- coding: utf-8 -*-

"""
The :mod:`coclust.io.data_transformation` module provides functions to
transform data.
"""

from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import TfidfTransformer


def cooccurence_to_binary(coocurence_sparse_matrix):
    """Convert cooccurence data to binary data. Each count higher than 0 is set
    to 1.

    Parameters
    ----------
    coocurence_sparse_matrix: :class:`scipy.sparse.csc.csc_matrix`
        a cooccurence matrix of shape (nrow = #documents, ncol = #words)

    Returns
    -------
    :class:`scipy.sparse.csc.csc_matrix`
        a binary matrix of shape (#documents, #words)
    """

    # Get the row and column index of non zero elements
    rowidx, colidx = coocurence_sparse_matrix.nonzero()

    # Set a 1D array full of ones
    tmp_array_ones = [1] * rowidx.shape[0]

    # Set a new scipy.sparse.csc.csc_matrix with all 0 but 1 at rowidx & colidx
    binary_sparse_matrix = csc_matrix((tmp_array_ones,
                                      (rowidx, colidx)),
                                      shape=coocurence_sparse_matrix.shape)

    return binary_sparse_matrix


def cooccurence_to_tfidf(coocurence_sparse_matrix):
    """Convert cooccurence data to tfidf data.

    The TF-IDF weighting scheme from scikit-learn is used.

    Parameters
    ----------
    coocurence_sparse_matrix: :class:`scipy.sparse.csc.csc_matrix`
        a cooccurence matrix of shape (nrow = #documents, ncol = #words)

    Returns
    -------
    :class:`scipy.sparse.csc.csc_matrix`
        a weighted TF-IDF matrix of shape (#documents, #words)
    """

    transformer = TfidfTransformer(smooth_idf=True, norm='l2')
    tfidf_sparse_matrix = transformer.fit_transform(coocurence_sparse_matrix)

    return tfidf_sparse_matrix
