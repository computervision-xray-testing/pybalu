pybalu
============

[![Build Status](https://travis-ci.com/mbucchi/pybalu.svg?branch=master)](https://travis-ci.com/mbucchi/pybalu)
[![Build status](https://ci.appveyor.com/api/projects/status/f010n1dwwyf5f2rk/branch/master?svg=true)](https://ci.appveyor.com/project/mbucchi/pybalu/branch/master)


Python implementation for Balu, a computer vision, pattern recognition and image processing library. Originally implemented in matlab by Domingo Mery.

## Setup for development and testing

### Requirements
Python 3.6 or higher is required to run setup and installation, together with the following packages:
- cython
- numpy
- scipy
- imageio
- scikit-image
- scikit-learn
- tqdm

Also, your sistem needs `C` compiling capabilities in order to compile the bits of the package that are written in low level languages.
### Setup process
To test locally, open a terminal on this projects root folder and run the following commands:
- `python setup.py build` in order to just build the package into a fully functional library. The code will be output to the `build` folder on the project's root
- `python setup.py install` installs the built package into the given python libsites, making it available for usage (`import pybalu`) while running python anywhere on the filesystem. If the build proceess was not executed beforehand, this step will perform both the build and install commands
### Upgrading to newer versions
If you've performed changes on the source code and are interested on installing the newer version, either the build folder must be cleaned, or the version number written on the `setup.py` file should be bumped up.
