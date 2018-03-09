import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize, fit_report
from ttim import *

observed = np.loadtxt('oudekorendijk_h30.dat')
to1 = observed[:,0] / 60.0 / 24.0
ho1 = -observed[:,1]
ro1 = 30.0
observed = np.loadtxt('oudekorendijk_h90.dat')
to2 = observed[:,0] / 60.0 / 24.0
ho2 = -observed[:,1]
ro2 = 90.0
Qo = 788.0
ml = ModelMaq(kaq=60, z=(-18,-25), Saq=1e-4, tmin=1e-5, tmax=1)
w = Well(ml, xw=0, yw=0, rw=0.1, tsandQ=[(0, 788)], layers=0)
ml.solve(silent='.')

cal = Calibrate(ml)
cal.parameter('kaq0', initial=10)
cal.parameter('Saq0', initial=1e-4)
cal.series('obs1', x=ro1, y=0, layer=0, t=to1, h=ho1)
#cal.series('obs2', x=ro2, y=0, layer=0, t=to2, h=ho2)
cal.fit()
        