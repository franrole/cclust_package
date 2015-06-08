import coclust.CoclustCut
import numpy as np

#from coclust.utils.initialization import random_init

c=coclust.CoclustCut()

c.fit(np.arange(12).reshape(4,3))
