import numpy as np
from sklearn.utils import check_random_state
import sys


def random_init(n_clusters, n_cols, random_state=None):
    """ Random Initialization
    """
    random_state = check_random_state(random_state)
    W_a = random_state.randint(n_clusters, size=n_cols)
    W = np.zeros((n_cols, n_clusters))
    W[np.arange(n_cols), W_a] = 1
    return W


def check_array(a) :

  if len(a[np.all(a == 0, axis=0)]) > 0 :
       print("ERROR: Zero-valued columns in data.")
       sys.exit(0)
  if len(a[np.all(a == 0, axis=1)]) > 0 :
       print("ERROR: Zero-valued rows in data.")
       sys.exit(0)
  if (a < 0).any():
        print("ERROR: Negative values in data")
        sys.exit(0)
  if np.isnan(a).any() :
        print("ERROR: NaN in data")
        sys.exit(0)
        
def check_numbers(a, n_clusters) :
    if a.shape[0] <  n_clusters or a.shape[1] <  n_clusters:
        print("ERROR: the data matrix has not enough rows or columns")
        sys.exit(0)

def check_numbers_non_diago(a,n_row_clusters,n_col_clusters) :
    if a.shape[0] < n_row_clusters or a.shape[1] < n_col_clusters :
        print("ERROR: the data matrix has not enough rows or columns")
        sys.exit(0)

        
    
