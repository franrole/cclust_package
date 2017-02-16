
from unittest import TestCase

from coclust.evaluation import external


class TestAccuracy(TestCase):
    def test_accuracy_1(self):
        nb_clusters = 2
        true_row_labels = [0, 0, 1, 1]
        predicted_row_labels = [0, 0, 1, 1]
        accuracy = external.accuracy(nb_clusters,
                                     true_row_labels,
                                     predicted_row_labels)

        self.assertEqual(accuracy, 1)

    def test_accuracy_2(self):
        nb_clusters = 2
        true_row_labels = [0, 0, 1, 1]
        predicted_row_labels = [1, 1, 0, 0]
        accuracy = external.accuracy(nb_clusters,
                                     true_row_labels,
                                     predicted_row_labels)

        self.assertEqual(accuracy, 1)

    def test_accuracy_3(self):
        nb_clusters = 2
        true_row_labels = [0, 1, 1, 0]
        predicted_row_labels = [1, 1, 0, 0]
        accuracy = external.accuracy(nb_clusters,
                                     true_row_labels,
                                     predicted_row_labels)

        self.assertEqual(accuracy, 0.5)
