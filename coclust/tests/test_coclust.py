from __future__ import absolute_import
from unittest import TestCase
import numpy as np


from coclust.CoclustMod import CoclustMod


class TestCoclust(TestCase):
    def test_1(self):
        s = "abc"
        self.assertTrue(isinstance(s, basestring))

    def test_2(self):
        s = "abc"
        self.assertTrue(isinstance(s, basestring))
