import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize, fit_report
from ttim import *

observed = np.loadtxt('oudekorendijk_h30.dat')
to1 = observed[:,0] / 60.0 / 24.0
ho1 = -observed[:,1]
ro1 = 30.0
Qo = 788.0

def residual(p, t=None, rdata=None, data=None):
    vals = p.valuesdict()
    k =  vals['k']
    Ss =  vals['Ss']
    ml = ModelMaq(kaq=k, z=(-18,-25), Saq=Ss, tmin=1e-5, tmax=1)   
    w = Well(ml, xw=0, yw=0, rw=0.1, tsandQ=[(0, 788)], layers=0)
    ml.solve(silent='.')
    hm = ml.head(rdata, 0.0, t, layers=[0])
    if data is None:
        return hm[0]
    else:
        return hm[0] - data  # head in layer 0
    
p = Parameters()
p.add('k', value=10.0)
p.add('Ss', value=1e-4)

print('Parameter estimation using data at observation well 30 m')
p30 = minimize(residual, p, kws={'t':to1, 'rdata':ro1, 'data':ho1}, epsfcn=1e-4)
print()
print(fit_report(p30.params))


