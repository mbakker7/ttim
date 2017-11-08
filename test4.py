import numpy as np
from scipy.integrate import quad
from ttim import *

ml = ModelMaq(kaq=[1, 5], z=[3, 2, 1, 0], c=[10], Saq=[0.3, 0.01], Sll=[0.001], tmin=1e-4, tmax=1000, M=20)
w1 = HeadWell(ml, xw=0, yw=0, rw=0.3, tsandh=[(0, 1)], layers=0)
ml.solve()

def func1(tau, p0, p1, f):
    rv = np.exp(-tau) * np.cos(-p1 / p0 * tau) * f(tau / p0)
    return rv

def func2(tau, p0, p1, f):
    rv = np.exp(-tau) * np.sin(-p1 / p0 * tau) * f(tau / p0)
    return rv

def func(t):
    return w1.strength(t)

def quadfunc(p0, p1):
    f1 = quad(func1, 0, np.inf, args=(p0, p1, func))[0]
    f2 = quad(func2, 0, np.inf, args=(p0, p1, func))[0]
    return (f1 + 1j * f2) / p0

fp = np.zeros(41, 'D')
p = ml.p[-41:]
for i in range(41):
    p0 = p[i].real
    p1 = p[i].imag
    fp[i] = quadfunc(p0, p1)
    
#ml2 = ModelMaq(kaq=[1, 5], z=[3, 2, 1, 0], c=[10], Saq=[0.3, 0.01], Sll=[0.001], tmin=100, tmax=1000, M=20)
#w2 = TestWell(ml2, xw=0, yw=0, rw=0.3, tsandQ=[(0, 1)], layers=0)
#ml2.solve()