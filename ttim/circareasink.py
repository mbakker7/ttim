import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv, iv
import inspect # Used for storing the input
from .element import Element

class CircAreaSink(Element):
    """
    Create a circular area-sink with uniform infiltration rate in aquifer
    layer 0.
    Infiltration rate in length / time, positive for water entering
    the aquifer.
    
    Parameters
    ----------
    model : Model object
        model to which the element is added
    xc : float
        x-coordinate of center of area-sink
    yc : float
        y-coordinate of center of area-sink
    R : radius of area-sink
    tsandN : list of tuples
        tuples of starting time and infiltration rate after starting time
    label : string or None (default: None)
        label of the area-sink
            
    """
    
    def __init__(self, model, xc=0, yc=0, R=0.1, tsandN=[(0, 1)], \
                 name='CircAreaSink', label=None):
        self.storeinput(inspect.currentframe())
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=0, \
                         tsandbc=tsandN, type='g', name=name, label=label)
        self.xc = float(xc); self.yc = float(yc); self.R = float(R)
        self.model.addelement(self)
        
    def __repr__(self):
        return self.name + ' at ' + str((self.xc, self.yc))

    def initialize(self):
        self.aq = self.model.aq.find_aquifer_data(self.xc, self.yc)
        self.setbc()
        self.setflowcoef()
        self.an = self.aq.coef[0, :] * self.flowcoef  # Since recharge is in layer 1 (pylayer=0), and RHS is -N
        self.an.shape = (self.aq.Naq, self.model.Nin, self.model.Npin)
        self.termin  = self.aq.lab2 * self.R * self.an * kv(1, self.R/self.aq.lab2)
        self.termin2 = self.aq.lab2 ** 2 * self.an
        self.terminq = self.R * self.an * kv(1, self.R / self.aq.lab2)
        self.termout = self.aq.lab2 * self.R * self.an * iv(1, self.R / self.aq.lab2)
        self.termoutq= self.R * self.an * iv(1, self.R / self.aq.lab2)
        self.dischargeinf = self.aq.coef[0, :] * self.flowcoef
        self.dischargeinflayers = np.sum(self.dischargeinf * self.aq.eigvec[self.layers, :, :], 1)

    def setflowcoef(self):
        '''Separate function so that this can be overloaded for other types'''
        self.flowcoef = 1.0 / self.model.p  # Step function

    def potinf(self, x, y, aq=None):
        '''Can be called with only one x,y value'''
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.nparam, aq.Naq, self.model.Nin, self.model.Npin), 'D')
        if aq == self.aq:
            r = np.sqrt((x - self.xc) ** 2 + (y - self.yc) ** 2)
            pot = np.zeros(self.model.Npin, 'D')
            if r < self.R:
                for i in range(self.aq.Naq):
                    for j in range(self.model.Nin):
                        #if r / abs(self.aq.lab2[i,j,0]) < self.rzero:
                        rv[0, i, j, :] = -self.termin[i, j, :] * iv(0, r / self.aq.lab2[i, j, :]) + self.termin2[i, j, :]
            else:
                for i in range(self.aq.Naq):
                    for j in range(self.model.Nin):
                        if (r - self.R) / abs(self.aq.lab2[i, j, 0]) < self.rzero:
                            rv[0, i, j, :] = self.termout[i, j, :] * kv(0, r / self.aq.lab2[i, j, :])
        rv.shape = (self.nparam, aq.Naq, self.model.Np)
        return rv
    
    def disinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        qx = np.zeros((self.nparam, aq.Naq, self.model.Np), 'D')
        qy = np.zeros((self.nparam, aq.Naq, self.model.Np), 'D')
        if aq == self.aq:
            qr = np.zeros((self.nparam, aq.Naq, self.model.Nin, self.model.Npin), 'D')
            r = np.sqrt((x - self.xc) ** 2 + (y - self.yc) ** 2)
            if r < self.R:
                for i in range(self.aq.Naq):
                    for j in range(self.model.Nin):
                        #if r / abs(self.aq.lab2[i,j,0]) < self.rzero:
                        qr[0, i, j, :] = self.terminq[i, j, :] * iv(1, r / self.aq.lab2[i, j, :])
            else:
                for i in range(self.aq.Naq):
                    for j in range(self.model.Nin):
                        if (r - self.R) / abs(self.aq.lab2[i, j, 0]) < self.rzero:
                            qr[0, i, j, :] = self.termoutq[i, j, :] * kv(1, r / self.aq.lab2[i, j, :])                
            qr.shape = (self.nparam, aq.Naq, self.model.Np)
            qx[:] = qr * (x - self.xc) / r
            qy[:] = qr * (y - self.yc) / r
        return qx, qy
    
    def plot(self):
        plt.plot(self.xc + self.R * np.cos(np.linspace(0, 2 * np.pi, 100)), \
                 self.yc + self.R * np.sin(np.linspace(0, 2 * np.pi, 100)), 'k')