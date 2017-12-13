.. ttim documentation master file, created by
   sphinx-quickstart on Mon Nov  6 09:36:59 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====
TTim
=====
TTim is a computer program for the modeling of transient multi-layer flow with analytic elements.
TTim may be applied to an arbitrary number of layers.
The Dupuit approximation is applied to aquifer layers, while flow in leaky layers is approximated as vertical.
The head, flow, and leakage between aquifer layers may be computed analytically at any point in the aquifer system and at any time.
TTim is coded in Python. Behind the scenes, use is made of FORTRAN extensions to improve performance.

This documentation is nearing completion.

Installation
------------
TTim is written for Python 3.
To install TTim, open a command prompt and type:

.. code-block:: python

  pip install ttim

To update TTim type:

.. code-block:: python

  pip install ttim --upgrade

To uninstall TTim type:

.. code-block:: python

  pip uninstall ttim
  
Main Approximations
-------------------

To be added. 

List of available elements
--------------------------

A list of available elements is available in the menu on the right under *elements*.

.. toctree::
    :maxdepth: 3
    :hidden:
    
    Models <models/modelindex>
    Elements <aems>
