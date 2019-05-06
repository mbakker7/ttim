# TTim, A Multi-Layer, Transient, Analytic Element Model

## Introduction

TTim is a computer program for the modeling of transient multi-layer flow with analytic elements
and consists of a library of Python scripts and FORTRAN extensions.
TTim is based on the Laplace-transform analytic element method. The solution is computed analytically
in the Laplace domain and converted back to the time domain numerically usig the algorithm of De Hoog, Stokes, and Knight.
TTim may be applied to an arbitrary number of aquifers and leaky layers.
The head, flow, and leakage between aquifers may be computed semi-analytically at any point in space and time.
The design of TTim is object-oriented and has been kept simple and flexible.
New analytic elements may be added to the code without making any changes in the existing part of the code.
TTim is coded in Python; use is made of FORTRAN extensions to improve performance.

## TTim Changes

### Version 0.4
Version 0.4 is the new port of TTim to Python 3. The API has changed significantly and some of the API will undergo more changes
in future versions. Not all elements that were available in version 0.3 have been ported to version 0.4.

## Installation

**Python versions:**

TTim requires **Python** 3.6 and can be installed from PyPI.
The PyPI installation includes compiled versions of the FORTRAN extension
for both Windows and MaxOS.


**Dependencies:**

TimML requires **NumPy** 1.12 (or higher) and **matplotlib** 2.0 (or higher). 

**Installation for Python 3.6:**

To install TTim, open a command prompt and type:

    pip install ttim

To uninstall TTim type:

    pip uninstall ttim
    
**Installation for Python 3.7:**

There is no installer for Python 3.7. An installer is planned for the summer of 2019. There are two options to run TTim on Python 3.7:

* Install a virtual environment for Python 3.6. In a terminal, type
    
    conda create -n py36 python=3.6 anaconda
    
    conda activate py36
    
    pip install ttim

To activate the environment and start a jupyter notebook, type the following in a terminal:
    
    conda activate py36
    
    jupyter notebook

* If you have FORTRAN and C compilers installed (for example gfortran), you can go to the directory where TTim is installed and type

    f2py -c -m --fcompiler=gfortran bessel bessel.f95
    
    f2py -c -m --fcompiler=gfortran invlap invlap.f90

    
**Testing installation:**

    ipython
    import ttim.ttimtest
    
An example model is imported and two graphs are shown. When this is run from the regular Python prompt (not IPython), the
model is created and solved but the figure is probably not shown (depending on your default settings of matplotlib). 


## Documentation

* The manual is available from the docs directory or can be viewed [here](http://mbakker7.github.io/ttim/docs/builddocs/html/index.html).
* Example Notebooks are available from the notebooks directory on github, of from [here](https://github.com/mbakker7/ttim/tree/master/notebooks).

## Citation

Some of the papers that you may want to cite when using TTim are

* M. Bakker. 2013. Semi-analytic modeling of transient multi-layer flow with TTim. Hydrogeology Journal, 21: 935Ð943.
* M .Bakker. 2013. Analytic modeling of transient multi-layer flow. In: Advances in Hydrogeology, edited by P Mishra and K Kuhlman, Springer, Heidelberg, 95-114.

