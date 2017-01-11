Visualization
=============

.. automodule:: coclust.visualization
    :members:

.. testsetup:: *

    from coclust.visualization.coClusteringMeasures import plot_intermediate_modularities
    from coclust.io.io import load_doc_term_data
    from coclust.coclustering.CoclustMod import CoclustMod
    model = CoclustMod()
    matrix = load_doc_term_data('../datasets/classic3.csv')['doc_term_matrix']
    model.fit(matrix)

Co-clustering measures
----------------------

.. automodule:: coclust.visualization.coClusteringMeasures
    :members:

Term clusters
-------------

.. automodule:: coclust.visualization.termClusters
    :members:
