# -*- coding: utf-8 -*-

"""
The :mod:`coclust.io.data_loading` module provides functions to load data
from files of different types.
"""

# Author: Severine Affeldt  <severine.affeldt@parisdescartes.fr>

import logging
import os.path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import whosmat, loadmat


logger = logging.getLogger(__name__)

# module variables
key_name_data = 'doc_term_matrix'
key_name_term_labels = 'term_labels'
key_name_doc_labels = 'doc_labels'


def load_doc_term_data(data_filepath,
                       term_labels_filepath=None,
                       doc_labels_filepath=None):

    """Load cooccurence data from a .[...]sv or a .mat file.

    The expected formats are:

    - ``(data_filepath).[...]sv``: three [...] separated columns:

        1st line:
            - 1st column: number of documents
            - 2nd column: number of words
        Other lines:
            - 1st column: document index
            - 2nd column: word index
            - 3rd column: word counts

    - ``(data_filepath).mat``: matlab file with fields:

        - ``'doc_term_matrix'``: :class:`scipy.sparse.csr_matrix` of
          shape (#docs, #terms)
        - ``'doc_labels'``: list of int (len = #docs)
        - ``'term_labels'``: list of string (len = #terms)

        If the key ``'doc_term_matrix'`` is not found, data loading fails.
        If the key ``'doc_labels'`` or ``'term_labels'`` are missing, a warning
        message is displayed.

    Term and doc labels can be separatly loaded from a one column
    .[x]sv|.txt file:

    - (term_labels_filepath).[x]sv|.txt:
        one column, one term label per row. The row index is assumed to
        correspond to the term index in the (columns of the) co-occurrence
        data matrix.

    - (doc_labels_filepath).[x]sv|.txt:
        one column, one document label per row. The row index is assumed to
        correspond to the non zero value number read by row from the
        co-occurrence data matrix.

    Parameters
    ----------
    file_path: string
        Path to file that contains the cooccurence data

    Returns
    -------
    a dictionnary:

        - ``'doc_term_matrix'``: :class:`scipy.sparse.csr_matrix` of shape
          (#docs, #terms)
        - ``'doc_labels'``: list of int (#docs)
        - ``'term_labels'``: list of string (#terms)

    Raises
    ------
    ValueError
        If the input file is not found or if its content is not correct.

    Example
    -------
    >>> dict = load_doc_term_data('../datasets/classic3.csv')
    >>> dict['doc_term_matrix'].shape
    (3891, 4303)

    """

    # Check that file_name is a file path and correspond to an exisiting file
    if not os.path.isfile(data_filepath):
        raise ValueError("[file_name] argument (%s) is not a file path or "
                         "file does not exist."
                         % os.path.abspath(data_filepath))

    # Get the file extension of the data_filepath
    _, file_extension = os.path.splitext(data_filepath)

    doc_term_dict = {}
    if file_extension == '.mat':
        # Load cooccurence table from .mat file
        doc_term_dict = _load_doc_term_data_from_mat_(data_filepath)
    else:
        # Load and format cooccurence table from .xsv file (.csv or .tsv)
        doc_term_dict = _load_doc_term_data_from_xsv_(data_filepath)

    # If doc_term_matrix is None, raise exception
    if doc_term_dict[key_name_data] is None:
        raise ValueError("doc_term matrix is None, check your input file "
                         "content or field names.")

    # Get the number of terms and docs
    n_term = doc_term_dict[key_name_data].shape[1]
    n_doc = doc_term_dict[key_name_data].shape[0]

    # If term|document labels are missing, load them from
    # term|doc_labels_filepath
    # --> terms
    if doc_term_dict[key_name_term_labels] is None:
        if term_labels_filepath is not None:
            tmp_term_labels = np.loadtxt(term_labels_filepath, dtype='str')\
                              .tolist()
            if len(tmp_term_labels) is not n_term:
                raise ValueError("Number of term labels (%d) not compatible "
                                 "with co-occurence matrix shape (%d, %d)"
                                 % (n_term, n_doc, len(tmp_term_labels)))
            else:
                doc_term_dict[key_name_term_labels] = tmp_term_labels
    # --> docs
    if doc_term_dict[key_name_doc_labels] is None:
        if doc_labels_filepath is not None:
            tmp_doc_labels = np.loadtxt(doc_labels_filepath, dtype='int')\
                             .tolist()
            if len(tmp_doc_labels) is not n_term:
                raise ValueError("Number of doc labels (%d) not compatible "
                                 "with the number of terms (%d)"
                                 % (len(tmp_doc_labels), n_term))
            doc_term_dict[key_name_doc_labels] = tmp_doc_labels

    if doc_term_dict[key_name_data] is None:
        raise ValueError("Co-occurence data cannot be loaded from .mat file: "
                         "no 'coccurrence' field found.")
    if doc_term_dict[key_name_term_labels] is None:
        logger.warning("Term labels cannot be loaded from .mat file. Use "
                       "input argument 'term_labels_filepath' if term labels "
                       "are available.")
    if doc_term_dict[key_name_doc_labels] is None:
        logger.warning("Document labels cannot be loaded  from .mat file. Use "
                       "input argument 'doc_labels_filepath' if doc labels "
                       "are available.")

    return doc_term_dict


def _get_file_delimiter_(extension):
    switcher = {
        '.csv': ',',
        '.tsv': '\t',
    }
    return switcher.get(extension, "\t")


def _load_doc_term_data_from_xsv_(path, extension=None):

    tmp_dict = {key: None for key in [key_name_data,
                                      key_name_term_labels,
                                      key_name_doc_labels]}

    # Get the file extension if needed
    if extension is None:
        _, extension = os.path.splitext(path)

    # --> get the delimeter from the extension
    file_delimiter = _get_file_delimiter_(extension)

    # --> build the matrix (it may take a few seconds)
    a = np.loadtxt(path, delimiter=file_delimiter, skiprows=1)

    # --> Set the co-occurrence data
    tmp_dict[key_name_data] = (coo_matrix((a[:, 2],
                                          (a[:, 0].astype(int),
                                           a[:, 1].astype(int))))).tocsr()

    return tmp_dict


def _load_doc_term_data_from_mat_(path):

    tmp_dict = {key: None for key in [key_name_data,
                                      key_name_term_labels,
                                      key_name_doc_labels]}

    # Get the fields from the matlab file
    matlab_content = whosmat(path)

    for index, element in enumerate(matlab_content):
        # if co-occurence data, load
        if element[0] == key_name_data:
            tmp_dict[key_name_data] = loadmat(path)[key_name_data]
        # if term label data, load and convert to list
        elif element[0] == key_name_term_labels:
            tmp_dict[key_name_term_labels] = loadmat(path)[key_name_term_labels]
            tmp_dict[key_name_term_labels] = tmp_dict[key_name_term_labels].tolist()
        # if doc label data, load, convert to list and take list inside...
        elif element[0] == key_name_doc_labels:
            tmp_dict[key_name_doc_labels] = loadmat(path)[key_name_doc_labels]
            tmp_dict[key_name_doc_labels] = tmp_dict[key_name_doc_labels].tolist()[0]

    return tmp_dict
