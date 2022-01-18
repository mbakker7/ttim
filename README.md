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
New in version 0.5:
* FORTRAN extension has been ported to Python and numba (many thanks to Davíd Brakenhoff)
* Python invlap routine (again with numba) ported from routine by Kris Kuhlman
* New invlap routine requires fewer terms in inverse Laplace transform (M=10 is usually enough)
* Calibrate now works on ranges of parameters.
* Calibrate now adjusts c values between layers when calibrating for hydraulic conductivity in Model3D

## Installation

**Python versions and dependencies**

TTim requires **Python** >= 3.6 and can be installed from PyPI.
Required packages: **NumPy** >=1.12, **Scipy** >=0.19 and **matplotlib** >=2.0, **numba>=0.4**, **lmfit>=1.0**

**Installation:**

To install TTim, open a command prompt or the anaconda prompt and type:

    pip install ttim

To update TTim type:

    pip install ttim --upgrade

To uninstall TTim type:

    pip uninstall ttim
    
## Documentation

* The manual is available from the docs directory or can be viewed [here](http://mbakker7.github.io/ttim/docs/builddocs/html/index.html).
* Example Notebooks are available from the notebooks directory on github, of from [here](https://github.com/mbakker7/ttim/tree/master/notebooks).

## Citation

Some of the papers that you may want to cite when using TTim are:

* M. Bakker. 2013. Semi-analytic modeling of transient multi-layer flow with TTim. Hydrogeology Journal, 21: 935�943.
* M .Bakker. 2013. Analytic modeling of transient multi-layer flow. In: Advances in Hydrogeology, edited by P Mishra and K Kuhlman, Springer, Heidelberg, 95-114.

