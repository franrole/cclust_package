Description
============

**coclust** is a Python module which provides implementations for several co-clustering algorithms. Co-clustering (also known as biclustering), is an important extension of cluster analysis since it 
allows to simultaneously groups objects and features in a matrix, resulting in both row and column clusters. **coclust** is distributed under the 3-Clause BSD license.


Usage
=======

To use **coclust**, just use::

    >>> import coclust

See the available `HTML documentation`_ for details and usage samples.

Install
=======

**coclust** relies on the numpy and scipy libraries, and also on scikit and matplotlib for some of the demos included in the package.

If these libraries are already installed on your machine, you can install **coclust** by just entering::

 pip install coclust


If this is not the case, the following subsections show how to install the required libraries and then **coclust**.


On Windows
:::::::::::

The simplest method is to use a distribution which includes all the libraries. For example, when using the Continuum distro
go to the `download site`_ to get and double-click the graphical installer. Then, enter ``pip install coclust`` at the command line.

On Ubuntu, Debian
::::::::::::::::::

The easiest method is to use your package manager. For example, on Ubuntu::

   sudo apt-get install python-numpy python-scipy python-sklearn
   sudo pip install coclust

You can also try to compile from source, but compiling Scipy may be tricky, so it is not the recommended way. Try at your own risk::

   sudo apt-get install gfortran python-dev
   sudo apt-get install libopenblas-base
   sudo apt-get install liblapack-dev
   sudo pip install coclust


OpenBLAS provides a fast multi-threaded implementation. If other implementations are installed on your system, you can select OpenBLAS with::

   sudo update-alternatives --config libblas.so.3
   
Code
====
   
You can check the latest sources with the command::
   
   git clone https://github.com/franrole/cclust_package.git

.. _`download site`: https://www.continuum.io/downloads
.. _`HTML documentation`: http://coclust.readthedocs.org
