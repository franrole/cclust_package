Install
=======

You can install coclust using pip:

```
pip install coclust
```

Coclust relies on the numpy and scipy libraries, and also on scikit and matplotlib for some of the demos included in the package.

Situation 1: installing coclust on a machine where numpy, scipy, scikit and matplotlib are already installed
------------------------------------------------------------------------------------------------------------

In this case, coclust should install seamlessly. To install, just enter:

pip install coclust

Situation 2: install numpy, scipy, scikit and matplotlib first and then install coclust
----------------------------------------------------------------------------------------

### Step 1. Install numpy, scipy, matplotlib and scikit

We recommend using the Ananconda distribution, which has good installers both for Linux and Windows.
Note that, on Debian and Ubuntu, the libraries can also be obtained via the package manager. However, e still recommend using the Ananconda installer
 even on Ubutu, Debian boxes.
 
### Step 2. Install coclust

Just enter the following command line :

pip install coclust

Situation 3: compile from source
--------------------------------

You can also the "pip install coclust" command on a machine where the above libraries are not installed. 
The installer will then try to compile the libraries from source. 	
On Ubuntu, for the compilation to succeed the dev packages should be installed first using:

A COMPLETER



### Performance using OpenBlas (Debian, Ubuntu)

OpenBLAS provides a fast multi-threaded implementation, you can install it with:

```
sudo apt-get install libopenblas-base
```

If other implementations are installed on your system, you can select OpenBLAS with:

```
sudo update-alternatives --config libblas.so.3
```
