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
from .model import *
from .well import *
from .linesink import *
from .linedoublet import *
from .circareasink import *
from .fit import *

__all__=[s for s in dir() if not s.startswith("_")]
