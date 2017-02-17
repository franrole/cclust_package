
from unittest import TestCase

from coclust.evaluation import external


class TestAccuracy(TestCase):
    def test_accuracy_1(self):
        true_row_labels = [0, 0, 1, 1]
        predicted_row_labels = [0, 0, 1, 1]
        accuracy = external.accuracy(true_row_labels, predicted_row_labels)

        self.assertEqual(accuracy, 1)

    def test_accuracy_2(self):
        true_row_labels = [0, 0, 1, 1]
        predicted_row_labels = [1, 1, 0, 0]
        accuracy = external.accuracy(true_row_labels, predicted_row_labels)

        self.assertEqual(accuracy, 1)

    def test_accuracy_3(self):
        true_row_labels = [0, 1, 1, 0]
        predicted_row_labels = [1, 1, 0, 0]
        accuracy = external.accuracy(true_row_labels, predicted_row_labels)

        self.assertEqual(accuracy, 0.5)

    def test_accuracy_4(self):
        true_row_labels = [0, 0, 1, 1, 1]
        predicted_row_labels = [1, 1, 0, 1, 0]
        accuracy = external.accuracy(true_row_labels, predicted_row_labels)
        self.assertEqual(accuracy, 0.8)
