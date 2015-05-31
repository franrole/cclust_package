from unittest import TestCase

import coclust

class TestJoke(TestCase):
    def test_is_string(self):
        s = "abc"
        self.assertTrue(isinstance(s, basestring))
