import numpy as np
import matplotlib.pyplot as plt

def xsection(ml,x1=0,x2=1,y1=0,y2=0,N=100,t=1,layers=0,color=None,lw=1,newfig=True,sstart=0):
    if newfig: plt.figure()
    x = np.linspace(x1,x2,N)
    y = np.linspace(y1,y2,N)
    s = np.sqrt( (x-x[0])**2 + (y-y[0])**2 ) + sstart
    h = ml.headalongline(x,y,t,layers)
    Nlayers,Ntime,Nx = h.shape
    for i in range(Nlayers):
        for j in range(Ntime):
            if color is None:
                plt.plot(s,h[i,j,:],lw=lw)
            else:
                plt.plot(s,h[i,j,:],color,lw=lw)
    plt.draw()