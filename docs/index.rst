Introduction
============

TTim is a Python package for the modeling of transient multi-layer groundwater flow
with analytic elements.

The head, flow, and leakage between aquifer layers may be computed analytically at any
point in the aquifer system and at any time.

.. grid::

    .. grid-item-card:: Tutorials
        :link: 00tutorials/index
        :link-type: doc

        Tutorials for getting started with TTim.

    .. grid-item-card:: How-to guides
        :link: 01howto/index
        :link-type: doc

        How-to guides for more advanced modeling with TTim.

    .. grid-item-card:: Concepts
        :link: 02concepts/index
        :link-type: doc

        TTim basic concepts explained.

.. grid::

    .. grid-item-card:: Examples
        :link: 03examples/index
        :link-type: doc

        TTim example notebooks.

    .. grid-item-card:: Pumping tests
        :link: 04pumpingtests/index
        :link-type: doc

        TTim pumping test benchmark notebooks.

    .. grid-item-card:: Code reference
        :link: 05api/index
        :link-type: doc

        TTim code reference.


Quick Example
-------------

.. tab-set::

    .. tab-item:: Python

        In this example a well is modelled near a river in a single aquifer.

        .. code-block:: python

            # Import python packages
            import numpy as np
            import ttim

            # Create model
            ml = ttim.ModelMaq(kaq=10, z=[20, 0], Saq=[0.1], phreatictop=True, tmin=1e-3, tmax=100)
            
            # Add a river with a fixed water level
            yls = np.arange(-100.0, 101, 20)
            xls = 50.0 * np.ones_like(yls)
            river = ttim.HeadLineSinkString(ml, xy=list(zip(xls, yls)), tsandh='fixed')
            
            # Add a well
            well = ttim.Well(ml, 0.0, 0.0, rw=0.3, tsandQ=[(0, 1000)])
            
            # Solve model
            ml.solve()

            # Plot head contours at t=2 days
            ml.contour(win=[-30, 55, -30, 30], ngr=40, t=2, labels=True, decimals=1)
            

    .. tab-item:: Result

        In this example a well is modelled near a river in a single aquifer.

        .. figure:: _static/example_output.png
            :figwidth: 500px


Approximations
--------------

The Dupuit approximation is applied to aquifer layers, while flow in leaky layers is
approximated as vertical.


.. toctree::
    :maxdepth: 2
    :hidden:

    Tutorials <00tutorials/index>
    How-to guides <01howto/index>
    Concepts <02concepts/index>
    Examples <03examples/index>
    Pumping tests <04pumpingtests/index>
    Code reference <05api/index>
    Cite <06about/index>
