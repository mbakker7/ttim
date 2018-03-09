'''
Copyright (C), 2017, Mark Bakker.
Mark Bakker, Delft University of Technology
mark dot bakker at tudelft dot nl

TTim is a computer program for the simulation of transient
multi-layer flow with analytic elements and consists of a
library of Python scripts and FORTRAN extensions.
'''

import os

#--version number
__name__='ttim'
__author__='Mark Bakker'
mypackage_root_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(mypackage_root_dir, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()

# Import all classes and functions
from .model import ModelMaq, Model3D
from .well import DischargeWell, HeadWell, Well, TestWell
from .linesink import LineSink, HeadLineSink, HeadLineSinkString, LineSinkDitchString
from .linedoublet import LeakyLineDoublet, LeakyLineDoubletString
from .circareasink import CircAreaSink
from .fit import Calibrate
