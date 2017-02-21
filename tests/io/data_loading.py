

from unittest import TestCase

from coclust.io.data_loading import load_doc_term_data


class DataLoadingTest(TestCase):
    def test_load_doc_term_data(self):
        file_name = "datasets/classic3.csv"
        term_labels_filepath = "datasets/classic3_terms.txt"
        data = load_doc_term_data(data_filepath=file_name,
                                  term_labels_filepath=term_labels_filepath)
        self.assertEqual(data['doc_term_matrix'].shape[1],
                         len(data['term_labels']))
