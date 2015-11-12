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

For runnig the examples, Matplotlib is required.

Coclust can be used with Scikit-learn, for example within a pipeline.

Performances
''''''''''''

OpenBLAS provides a fast multi-threaded implementation, you can install it with::

    sudo apt-get install libopenblas-base

If other implementations are installed on your system, you can select OpenBLAS with::

    sudo update-alternatives --config libblas.so.3

