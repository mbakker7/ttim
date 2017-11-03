from ttim import *
from matplotlib.pyplot import *

ml = ModelMaq(kaq = [1.0, 5.0],
                   z = [3,2, 1,0],
                   c = [10.],
                   Saq = [0.3, 0.01],
                   Sll = [0.001],
                   tmin = 1e-3,
                   tmax = 1e6,
                   M = 15)
w1 = Well(ml, xw = 0, yw = 0, rw = 1e-5, tsandQ = [(0,1)], layers = 0)
ml.solve()

try:
    # equivalent to %matplotlib in IPython
    get_ipython().magic('matplotlib')
    t = np.logspace(-3, 6, 50)
    h0 = ml.head(0.2, 0, t)
    h1 = ml.head(10, 0, t)
    figure(figsize=(12,6))
    subplot(121)
    semilogx(t, h0[0], label = 'r=0.2, layer=0')
    semilogx(t, h0[1], label = 'r=0.2, layer=1')
    xlabel('time (d)')
    ylabel('head (m)')
    legend(loc = 'best')
    subplot(122)
    semilogx(t, h1[0], label = 'r=10, layer=0')
    semilogx(t, h1[1], label = 'r=10, layer=1')
    xlabel('time (d)')
    ylabel('head (m)')
    legend(loc = 'best')
except:
    print 'A figure is created when run from IPython'
    pass

