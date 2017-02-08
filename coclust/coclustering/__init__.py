# -*- coding: utf-8 -*-

"""
The :mod:`coclust.coclustering` module gathers implementations of co-clustering
algorithms.
"""

from .coclust_info import CoclustInfo
from .coclust_mod import CoclustMod
from .coclust_spec_mod import CoclustSpecMod
from base_diagonal_coclust import BaseDiagonalCoclust
from base_non_diagonal_coclust import BaseNonDiagonalCoclust

__all__ = ['CoclustInfo',
           'CoclustMod',
           'CoclustSpecMod',
           'BaseDiagonalCoclust',
           'BaseNonDiagonalCoclust']
