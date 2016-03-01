## Description

## Install

Coclust relies on the numpy and scipy libraries, and also on scikit and matplotlib for some of the demos included in the package.


### Situation 1: numpy, scipy, scikit and matplotlib are already installed

In this case, coclust should install seamlessly. To install, just enter:

```
pip install coclust
```

### Situation 2: install numpy, scipy, scikit and matplotlib first and then install coclust


#### Windows

* Install the required libraries, for example using the Ananconda distribution, which includes them all.
* Enter the following command line:

```
pip install coclust
```

#### Ubuntu, Debian 

##### Using the package manager

* sudo apt-get install python-numpy  python-scipy python-sklearn
* sudo pip install coclust

##### Compiling from source

* sudo apt-get install gfortran , python-dev
* sudo apt-get install libopenblas-base
* sudo apt-get install liblapack-dev
* pip install coclust : returns polate/src/_interpolate.o" failed with exit status 127






#### Performance using OpenBlas (Debian, Ubuntu)

OpenBLAS provides a fast multi-threaded implementation, you can install it with:

```
sudo apt-get install libopenblas-base
```

If other implementations are installed on your system, you can select OpenBLAS with:

```
sudo update-alternatives --config libblas.so.3
```
