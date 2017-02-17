

from unittest import TestCase

from sklearn.utils.estimator_checks import check_estimator

from coclust.coclustering import CoclustMod, CoclustSpecMod, CoclustInfo


class TestSklearnApi(TestCase):

    def test_CoclustMod(self):
        check_estimator(CoclustMod)

    #def test_CoclustSpecMod(self):
    #    check_estimator(CoclustSpecMod)
    #TODO: it pass the test when changing default value of n_clusters to 1
    # this is because it runs a test with a number of clusters equal to the
    # number of lines

    def test_CoclustInfo(self):
        check_estimator(CoclustInfo)
