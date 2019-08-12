Coclust: a Python package for co-clustering
===========================================

**Coclust** provides both a Python package which implements several diagonal
and non-diagonal co-clustering algorithms, and a ready to use script to
perform co-clustering.

Co-clustering (also known as biclustering), is an important extension of
cluster analysis since it allows to simultaneously groups objects and features
in a matrix, resulting in both row and column clusters.

The :doc:`script<scripts>` enables the user to process a dataset with
co-clustering algorithms without writing Python code.

The Python package provides an :doc:`API<api/index>` for Python developers.
This API allows to use the algorithms in a pipeline with the scikit-learn library
for example.

**coclust** is distributed under the 3-Clause BSD license.


.. toctree::
   :maxdepth: 1

   install
   examples
   api/index
   scripts

.. image:: img/logo_lipade.png
