Installation
============

Required
''''''''

The following libraries are required:

    - Numpy
    - SciPy
    - scikit-learn

Optional
''''''''

Matplotlib is required for running examples involving visualizations.

If you are new to Python, we recommend using the Anaconda distribution since it will install everything for you.

Performance Note (Debian, Ubuntu)
'''''''''''''''''''''''''''''''''

OpenBLAS provides a fast multi-threaded implementation, you can install it with::

    sudo apt-get install libopenblas-base

If other implementations are installed on your system, you can select OpenBLAS with::

    sudo update-alternatives --config libblas.so.3
