"""TTim is a computer program for the simulation of transient multi-layer flow with
analytic elements and consists of a library of Python scripts and FORTRAN extensions.

Copyright (C), 2017, Mark Bakker.

Mark Bakker, Delft University of Technology mark dot bakker at tudelft dot nl.
"""
# ruff : noqa
__name__ = "ttim"
__author__ = "Mark Bakker"
from .circareasink import CircAreaSink
from .fit import Calibrate
from .linedoublet import LeakyLineDoublet, LeakyLineDoubletString
from .linesink import (
    HeadLineSink,
    HeadLineSinkHo,
    HeadLineSinkString,
    LineSink,
    LineSinkDitchString,
)

# Import all classes and functions
from .model import Model3D, ModelMaq
from .trace import timtrace, timtraceline
from .version import __version__
from .well import DischargeWell, HeadWell, Well, WellTest
