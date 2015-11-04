Overview
========

Script
------

Python
------

Basic usage
'''''''''''

.. code-block:: python

    import coclust

Scikit-learn pipeline
'''''''''''''''''''''

.. code-block:: python

    from sklearn.pipeline import Pipeline
    import numpy as np
    from scipy.io import loadmat
    from coclust.CoclustMod import CoclustMod

    file_name = "/home/stan/recherche/github/cclust_package/datasets/cstr.mat"
    matlab_dict = loadmat(file_name)
    X = matlab_dict['fea']
    pipeline = Pipeline([
        ('clust', CoclustMod())
    ])
    pipeline.set_params(clust__n_coclusters=4)
    pipeline.fit(X)

