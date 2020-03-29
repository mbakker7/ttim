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

## Installation

**Python versions:**

TTim requires **Python** >= 3.6 and can be installed from PyPI.

**Dependencies:**

TTim requires **NumPy** >=1.12, **Scipy** >=0.19 and **matplotlib** >=2.0, **numba>=0.4**, **lmfit>=1.0**

**Installation:**

To install TTim, open a command prompt or the anaconda prompt and type:

    pip install ttim

To update TTim type:

    pip install ttim --upgrade

To uninstall TTi type:

    pip uninstall ttim
    
## Documentation

* The manual is available from the docs directory or can be viewed [here](http://mbakker7.github.io/ttim/docs/builddocs/html/index.html).
* Example Notebooks are available from the notebooks directory on github, of from [here](https://github.com/mbakker7/ttim/tree/master/notebooks).
    
##Testing installation:

    ipython
    import ttim.ttimtest
    
An example model is imported and two graphs are shown. When this is run from the regular Python prompt (not IPython), the
model is created and solved but the figure is probably not shown (depending on your default settings of matplotlib). 

## Citation

Some of the papers that you may want to cite when using TTim are

* M. Bakker. 2013. Semi-analytic modeling of transient multi-layer flow with TTim. Hydrogeology Journal, 21: 935Ð943.
* M .Bakker. 2013. Analytic modeling of transient multi-layer flow. In: Advances in Hydrogeology, edited by P Mishra and K Kuhlman, Springer, Heidelberg, 95-114.

