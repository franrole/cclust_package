import coclust.CoclustCut
import numpy as np


c=coclust.CoclustCut()

c.fit(np.arange(12).reshape(4,3))
