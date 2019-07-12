Coclust: a Python package for co-clustering
===========================================

**Coclust** provides both a Python package which implements several diagonal
and non-diagonal co-clustering algorithms, and a ready to use script to
perform co-clustering.

Co-clustering (also known as biclustering), is an important extension of
cluster analysis since it allows to simultaneously groups objects and features
in a matrix, resulting in both row and column clusters.

The script enables the user to process a dataset with
co-clustering algorithms without writing Python code.

The Python package provides an API for Python developers.
This API allows to use the algorithms in a pipeline with scikit-learn library
for example.

**coclust** is distributed under the 3-Clause BSD license.

See the available `documentation`_ for details and usage samples.

If you use this package, please cite:

::

@article{JSSv088i07,
   author = {François Role and Stanislas Morbieu and Mohamed Nadif},
   title = {CoClust: A Python Package for Co-Clustering},
   journal = {Journal of Statistical Software, Articles},
   volume = {88},
   number = {7},
   year = {2019},
   keywords = {data mining; co-clustering; Python},
   issn = {1548-7660},
   pages = {1--29},
   doi = {10.18637/jss.v088.i07},
   url = {https://www.jstatsoft.org/v088/i07}
}




Changelog
~~~~~~~~~

0.2.0 - February, 2017
::::::::::::::::::::::

- Improved documentation
- Restructuring
- Evaluation, visualization and loading utilities
- Easier installation of optional dependencies


0.1.3 - April, 2016
:::::::::::::::::::

- New visualization methods in the utils module.
- New demos.
- Better PEP 8 conformance
- Improved documentation.

0.1.1 - March 07, 2016
:::::::::::::::::::::::

- First release.


Code
~~~~

You can check the latest sources with the command::

   git clone https://github.com/franrole/cclust_package.git


.. _`documentation`: http://coclust.readthedocs.org
