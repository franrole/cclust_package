# -*- coding: utf-8 -*-

import os.path
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.io import whosmat, loadmat
from sklearn.feature_extraction.text import TfidfTransformer

"""
Load and process data from files of different types
"""

def load_cooccurence_data(file_path):
    """Load cooccurence data from a .csv, a .tsv or a .mat file.

        The expected formats are (if other extension, .tsv is assumed):

        .csv: three comma separated columns
            1st line:
                1st column, number of documents
                2nd column, number of words
            Other lines:
                1st column, document index
                2nd column, word index
                3rd column, word counts

        .tsv: three tab separated columns
            1st line:
                1st column, number of documents
                2nd column, number of words
            Other lines:
                1st column, document index
                2nd column, word index
                3rd column, word counts
                
        .mat: matlab file
            cooccurence data expected to be found under the key name 'cooccurence'.
            If the key name 'cooccurence' does not exist, the cooccurence table is the
            first matrix that can be interpreted as scipy.sparse.csc.csc_matrix by the
            'scipy.io.loadmat' method.
        
    Parameters
    ----------
    file_path: string
        Path to file that contains the cooccurence data

    Returns
    ----------
    a cooccurence matrix of type scipy.sparse.csc.csc_matrix of shape
    (nrow = #documents, ncol = #words)

    the list of term labels
    """
    # Check that file_name is a file path and correspond to an exisiting file
    if not os.path.isfile(file_path): raise Exception("[file_name] argument is not a file path or file does not exist")

    # Get the file extension
    _, file_extension = os.path.splitext(file_path)

    X = None
    term_labels = None
    if(file_extension == '.mat'):
        # Load cooccurence table from .mat file
        X, term_labels, doc_labels = _load_cooccurence_data_from_mat_(file_path)
    else:
        # Load and format cooccurence table from .xsv file (.csv or .tsv)
        X, term_labels, doc_labels = _load_cooccurence_data_from_xsv_(file_path)

    return [X, term_labels, doc_labels]

def cooccurence_to_binary(coocurence_sparse_matrix):
    """Convert cooccurence data to binary data. Each count higher than 0 is set to 1.

    Parameters
    ----------
    coocurence_sparse_matrix: scipy.sparse.csc.csc_matrix
        a cooccurence matrix of type scipy.sparse.csc.csc_matrix of shape
        (nrow = #documents, ncol = #words)
    
    Returns
    ----------
    a binary scipy.sparse.csc.csc_matrix of shape (#documents, #words)
    """
    # Get the row and column index of non zero elements
    rowidx, colidx = coocurence_sparse_matrix.nonzero()
    
    # Set a 1D array full of ones
    tmp_array_ones = [1]*rowidx.shape[0]
    
    # Set a new scipy.sparse.csc.csc_matrix with all 0 but 1 at rowidx & colidx
    binary_sparse_matrix = csc_matrix((tmp_array_ones,
                                       (rowidx, colidx)), shape=coocurence_sparse_matrix.shape)

    return binary_sparse_matrix

def cooccurence_to_tfidf(coocurence_sparse_matrix):
    """Convert cooccurence data to tfidf data. The TF-IDF weighting scheme from
        scikit-learn is used.

    Parameters
    ----------
    coocurence_sparse_matrix: scipy.sparse.csc.csc_matrix
        a cooccurence matrix of type scipy.sparse.csc.csc_matrix of shape
        (nrow = #documents, ncol = #words)
    
    Returns
    ----------
    a weighted TF-IDF scipy.sparse.csc.csc_matrix of shape (#documents, #words)
    """
    transformer = TfidfTransformer(smooth_idf=True, norm='l2')
    tfidf_sparse_matrix = transformer.fit_transform(coocurence_sparse_matrix)
    
    return tfidf_sparse_matrix

    
def _get_file_delimiter_(extension):
    switcher = {
        '.csv': ',',
        '.tsv': '\t',
    }
    return switcher.get(extension, "\t")

def _load_cooccurence_data_from_xsv_(path, extension = None):

    term_labels = None
    doc_labels = None
    
    # Get the file extension if needed
    if extension is None: _, extension = os.path.splitext(path)
        
    # --> get the delimeter from the extension
    file_delimiter = _get_file_delimiter_(extension)

    # --> build the matrix (it may take a few seconds)
    a = np.loadtxt(path, delimiter = file_delimiter, skiprows=1)
    
    return [(coo_matrix((a[:, 2], (a[:, 0].astype(int), a[:, 1].astype(int))))).tocsr(),
            term_labels, doc_labels]

def _load_cooccurence_data_from_mat_(path, key_name_data = 'cooccurence',
                                     key_name_term_labels = 'term_labels',
                                     key_name_doc_labels = 'doc_labels'):

    retX = None
    term_labels = None
    doc_labels = None
    
    # --> check that key name 'cooccurence' exists
    # --> if not, find the element with 2 dimensions > 1
    matlab_content = whosmat(path)
    
    # --> find the data in the matlab elements
    for idx in range(len(matlab_content)):
        if key_name_data in matlab_content[idx]:
            retX = loadmat(path)[key_name_data]
        else:
            tmp_name, tmp_shape, tmp_type = matlab_content[idx]
            tmp_row, tmp_col = tmp_shape

            if( tmp_row > 1 and  tmp_col > 1 and tmp_type == 'sparse'):
                retX = loadmat(path)[tmp_name]

    # --> find the term_labels in the matlab elements
    for idx in range(len(matlab_content)):
        if key_name_term_labels in matlab_content[idx]:
            term_labels = loadmat(path)[key_name_term_labels]
        else:
            tmp_name, tmp_shape, tmp_type = matlab_content[idx]
            tmp_row, tmp_col = tmp_shape

            if( tmp_row == retX.shape[1] and  tmp_type == 'cell'):
                term_labels = [str(x[0][0]) for x in loadmat(path)[tmp_name]]

    # --> find the doc_labels in the matlab elements
    for idx in range(len(matlab_content)):
        if key_name_doc_labels in matlab_content[idx]:
            doc_labels = loadmat(path)[key_name_doc_labels]
        else:
            tmp_name, tmp_shape, tmp_type = matlab_content[idx]
            tmp_row, tmp_col = tmp_shape

            if( tmp_row == retX.shape[0] and  tmp_type == 'double'):
                #doc_labels = [str(x[0][0]) for x in loadmat(path)[tmp_name]]
                doc_labels = loadmat(path)[tmp_name]
                
    return [retX, term_labels, doc_labels]
