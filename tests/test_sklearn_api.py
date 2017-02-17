

from unittest import TestCase

from sklearn.utils.estimator_checks import check_estimator

from coclust.coclustering import CoclustMod, CoclustSpecMod, CoclustInfo


class TestSklearnApi(TestCase):

    def test_CoclustMod(self):
        check_estimator(CoclustMod)

    def test_CoclustSpecMod(self):
        check_estimator(CoclustSpecMod)

    def test_CoclustInfo(self):
        check_estimator(CoclustInfo)
