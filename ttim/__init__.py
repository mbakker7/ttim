'''
Copyright (C), 2017, Mark Bakker.
Mark Bakker, Delft University of Technology
mark dot bakker at tudelft dot nl

TTim is a computer program for the simulation of transient
multi-layer flow with analytic elements and consists of a
library of Python scripts and FORTRAN extensions.
'''

#--version number
__name__='ttim'
__author__='Mark Bakker'
from .version import __version__

# Import all classes and functions
from .model import ModelMaq, Model3D
from .well import DischargeWell, HeadWell, Well, WellTest
from .linesink import LineSink, HeadLineSink, HeadLineSinkString, \
     LineSinkDitchString, HeadLineSinkHo
from .linedoublet import LeakyLineDoublet, LeakyLineDoubletString
from .circareasink import CircAreaSink
from .fit import Calibrate
from .trace import timtraceline, timtrace