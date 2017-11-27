import numpy as np
import inspect # Used for storing the input
from .element import Element

class LineSinkHoBase(Element):
    '''Higher Order LineSink Base Class. All Higher Order Line Sink elements are derived from this class'''
    def __init__(self, model, x1=-1, y1=0, x2=1, y2=0, tsandbc=[(0.0,1.0)],
                 res=0.0, wh='H', order=0, layers=0, type='',
                 name='LineSinkBase', label=None, addtomodel=True):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers,
                         tsandbc=tsandbc, type=type, name=name, label=label)
        self.order = order
        self.Nparam = (self.order + 1) * len(self.pylayers)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.res = res
        self.wh = wh
        if addtomodel: self.model.addElement(self)
        #self.xa,self.ya,self.xb,self.yb,self.np = np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1,'i')  # needed to call bessel.circle_line_intersection

    def __repr__(self):
        return self.name + ' from ' + str((self.x1, self.y1)) + \
               ' to ' + str((self.x2, self.y2))

    def initialize(self):
        self.Ncp = self.order + 1
        self.z1 = self.x1 + 1j * self.y1
        self.z2 = self.x2 + 1j * self.y2
        self.L = np.abs(self.z1 - self.z2)
        #
        thetacp = np.arange(np.pi, 0, -np.pi / self.Ncp) - 0.5 * np.pi / self.Ncp
        Zcp = np.zeros(self.Ncp, 'D')
        Zcp.real = np.cos(thetacp)
        Zcp.imag = 1e-6  # control point just on positive site (this is handy later on)
        zcp = Zcp * (self.z2 - self.z1) / 2 + 0.5 * (self.z1 + self.z2)
        self.xc = zcp.real
        self.yc = zcp.imag
        #
        self.aq = self.model.aq.findAquiferData(self.xc[0], self.yc[0])
        self.setbc()
        coef = self.aq.coef[self.pylayers, :]
        self.setflowcoef()
        self.term = self.flowcoef * coef  # shape (self.Nlayers,self.aq.Naq,self.model.Np)
        self.term2 = self.term.reshape(self.Nlayers, self.aq.Naq, self.model.Nin, self.model.Npin)
        #self.term2 = np.empty((self.Nparam,self.aq.Naq,self.model.Nin,self.model.Npin),'D')
        #for i in range(self.Nlayers):
        #    self.term2[i*(self.order+1):(i+1)*(self.order+1),:,:,:] = self.term[i,:,:].reshape((1,self.aq.Naq,self.model.Nin,self.model.Npin))
        self.strengthinf = self.flowcoef * coef
        self.strengthinflayers = np.sum(self.strengthinf * self.aq.eigvec[self.pylayers, :, :], 1)
        if self.wh == 'H':
            self.wh = self.aq.Haq[self.pylayers]
        elif self.wh == '2H':
            self.wh = 2.0 * self.aq.Haq[self.pylayers]
        else:
            self.wh = np.atleast_1d(self.wh) * np.ones(self.Nlayers)
        self.resfach = self.res / (self.wh * self.L)  # Q = (h - hls) / resfach
        self.resfacp = self.resfach * self.aq.T[self.pylayers]  # Q = (Phi - Phils) / resfacp

    def setflowcoef(self):
        '''Separate function so that this can be overloaded for other types'''
        self.flowcoef = 1 / self.model.p  # Step function

    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.findAquiferData(x, y)
        rv = np.zeros((self.Nparam, aq.Naq, self.model.Nin, self.model.Npin), 'D')
        if aq == self.aq:
            pot = np.zeros((self.order + 1, self.model.Npin), 'D')
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if bessel.isinside(self.z1, self.z2, x + y * 1j, self.Rzero * self.aq.lababs[i, j]):
                        pot[:,:] = bessel.bessellsv2(x, y, self.z1, self.z2, self.aq.lab2[i, j, :], \
                                                     self.order, self.Rzero * self.aq.lababs[i, j]) / self.L  # Divide by L as the parameter is now total discharge
                        for k in range(self.Nlayers):
                            rv[k::self.Nlayers, i, j, :] = self.term2[k, i, j, :] * pot
        rv.shape = (self.Nparam, aq.Naq, self.model.Np)
        return rv

    def disinf(self, x, y, aq=None):
        '''Can be called with only one x,y value'''
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        rvx = np.zeros((self.Nparam, aq.Naq, self.model.Nin, self.model.Npin), 'D')
        rvy = np.zeros((self.Nparam, aq.Naq, self.model.Nin, self.model.Npin), 'D')
        if aq == self.aq:
            qxqy = np.zeros((2 * (self.order + 1), self.model.Npin), 'D')
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if bessel.isinside(self.z1, self.z2, x + y * 1j, self.Rzero * self.aq.lababs[i, j]):
                        qxqy[:, :] = bessel.bessellsqxqyv2(x, y, self.z1, self.z2, self.aq.lab2[i, j, :],\
                                                           self.order, self.Rzero * self.aq.lababs[i, j]) / self.L  # Divide by L as the parameter is now total discharge
                        for k in range(self.Nlayers):
                            rvx[k::self.Nlayers, i, j, :] = self.term2[k, i, j, :] * qxqy[:self.order + 1, :]
                            rvy[k::self.Nlayers, i, j, :] = self.term2[k, i, j, :] * qxqy[self.order + 1:, :]
        rvx.shape = (self.Nparam, aq.Naq, self.model.Np)
        rvy.shape = (self.Nparam, aq.Naq, self.model.Np)
        return rvx, rvy

    def headinside(self, t):
        return self.model.head(self.xc, self.yc, t)[self.pylayers] - self.resfach[:, np.newaxis] * self.strength(t)

    def layout(self):
        return 'line', [self.x1, self.x2], [self.y1, self.y2]
    
class HeadLineSinkHo(LineSinkHoBase, HeadEquationNores):
    '''HeadLineSink of which the head varies through time. May be screened in multiple layers but all with the same head'''
    def __init__(self, model, x1=-1, y1=0, x2=1, y2=0, tsandh=[(0.0,1.0)],\
                 order=0, layers=0, label=None, addtomodel=True):
        self.storeinput(inspect.currentframe())
        LineSinkHoBase.__init__(self, model, x1=x1, y1=y1, x2=x2, y2=y2, tsandbc=tsandh,  \
                                res=0.0, wh='H', order=order, layers=layers, type='v',  \
                                name='HeadLineSinkHo', label=label, addtomodel=addtomodel)
        self.Nunknowns = self.Nparam

    def initialize(self):
        LineSinkHoBase.initialize(self)
        self.parameters = np.zeros((self.model.Ngvbc, self.Nparam, self.model.Np), 'D')
        self.pc = np.empty(self.Nparam)
        for i, T in enumerate(self.aq.T[self.pylayers]):
            self.pc[i::self.Nlayers] =  T # Needed in solving; we solve for a unit head