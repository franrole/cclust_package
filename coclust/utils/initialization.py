import numpy as np
from sklearn.utils import check_random_state


def random_init(n_clusters, n_cols, random_state=None):
    """ Random Initialization
    """
    random_state = check_random_state(random_state)
    W_a = random_state.randint(n_clusters, size=n_cols)
    W = np.zeros((n_cols, n_clusters))
    W[np.arange(n_cols), W_a] = 1
    return W
