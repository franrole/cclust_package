import sys
import getopt
import re
import os,glob
from math import * 
import numpy as np
from numpy import *
from collections import *

import scipy.sparse as sp
import marshal
import cPickle
import pickle

import itertools
from scipy.io import loadmat, savemat


def cocluster():
        A=sp.lil_matrix(np.arange(12).reshape(4,3) ,dtype=float)
        print A.todense()

