Installation
============

Ubuntu & Debian
---------------

Required
''''''''

The following libraries are required:

    - Numpy
    - SciPy

Optional
''''''''

Matplotlib is required for running examples involving visualizations.

Coclust can also be used with Scikit-learn, for example within a pipeline.

Performance
''''''''''''

OpenBLAS provides a fast multi-threaded implementation, you can install it with::

    sudo apt-get install libopenblas-base

If other implementations are installed on your system, you can select OpenBLAS with::

    sudo update-alternatives --config libblas.so.3

