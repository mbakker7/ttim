[![ttim](https://github.com/mbakker7/ttim/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/mbakker7/ttim/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/mbakker7/ttim/badge.svg?branch=master)](https://coveralls.io/github/mbakker7/ttim?branch=master)
![PyPI](https://img.shields.io/pypi/v/ttim?color=green)

# TTim, A Multi-Layer, Transient, Analytic Element Model

## Introduction

TTim is a computer program for the modeling of transient multi-layer flow with analytic elements.
TTim is based on the Laplace-transform analytic element method. The solution is computed analytically
in the Laplace domain and converted back to the time domain numerically usig the algorithm of De Hoog, Stokes, and Knight.
TTim may be applied to an arbitrary number of aquifers and leaky layers.
The head, flow, and leakage between aquifers may be computed semi-analytically at any point in space and time.
The design of TTim is object-oriented and has been kept simple and flexible.
New analytic elements may be added to the code without making any changes in the existing part of the code.
TTim is coded in Python and uses numba to speed up evaluation of the line elements and inverse laplace transforms.

## Latest version
New in version 0.6.6:
* Many documentation improvements: new look, better organization, tutorials, how-to guides, etc. Check it out [here](https://ttim.readthedocs.io)!

## Installation

**Python versions and dependencies**

TTim requires Python >= 3.8 and can be installed from PyPI.

Required packages: 
* numpy
* scipy
* matplotlib 
* numba
* lmfit

**Installation:**

To install TTim, open a command prompt or the anaconda prompt and type:

    pip install ttim

To update TTim type:

    pip install ttim --upgrade

To uninstall TTim type:

    pip uninstall ttim
    
## Documentation

* The documentation is hosted on [readthedocs](https://ttim.readthedocs.io).
* Example Notebooks are available from the notebooks directory on github, of from [here](https://github.com/mbakker7/ttim/tree/master/notebooks).

## Citation

Some of the papers that you may want to cite when using TTim are:

* M. Bakker. 2013. Semi-analytic modeling of transient multi-layer flow with TTim. Hydrogeology Journal, 21: 935�943.
* M .Bakker. 2013. Analytic modeling of transient multi-layer flow. In: Advances in Hydrogeology, edited by P Mishra and K Kuhlman, Springer, Heidelberg, 95-114.

