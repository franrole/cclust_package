Scripts
=======

The input matrix can be a Matlab file or a text file. For the Matlab file, the key
corresponding to the matrix must be given. For the text file, each line should describe an
entry of a matrix with three columns: the row index, the column index and the value. The
separator is given by a script parameter.


Perform co-clustering: the *coclust* script
-------------------------------------------

The *coclust* script can be used to run a particular co-clustering algorithm on a data matrix. The user has to select an algorithm which is given as a first argument to *coclust*.
The choices are:

* modularity
* specmodularity
* info

The following command line shows how to run the *CoclustMod* algorithm three times on a matrix contained in a Matlab file whose matrix key is the string 'fea'.
The computed row labels are to be stored in a file called cstr-rows.txt:

.. literalinclude:: coclust.txt

To have a list of all possible parameters for a given algorithm use the -h option as in the following example:

.. code-block:: bash

    coclust modularity -h



.. argparse::
   :module: coclust.coclust
   :func: get_coclust_parser
   :prog: coclust







Detect the best number of co-clusters: the *coclust-nb* script
--------------------------------------------------------------

*coclust-nb* detects the number of co-clusters giving the best modularity score. It therefore relies on the CoclustMod algorithm.
This is a simple yet often effective way to determine the appropriate number of co-clusters. A sample usage sample is given below:

.. literalinclude:: coclust-nb.txt


.. argparse::
   :module: coclust.coclust
   :func: get_coclust_nb_parser
   :prog: coclust-nb



