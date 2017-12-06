import numpy as np
import matplotlib.pyplot as plt
import inspect # Used for storing the input
from .element import Element
from .bessel import *
from .equation import LeakyWallEquation

class LineDoubletHoBase(Element):
    '''Higher Order LineDoublet Base Class. All Higher Order Line Doublet elements are derived from this class'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandbc=[(0.0,0.0)],res='imp',order=0,layers=0,type='',name='LineDoubletHoBase',label=None,addtomodel=True):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers,tsandbc=tsandbc, type=type, name=name, label=label)
        self.order = order
        self.Nparam = (self.order+1) * len(self.pylayers)
        self.x1 = float(x1); self.y1 = float(y1); self.x2 = float(x2); self.y2 = float(y2)
        if res == 'imp':
            self.res = np.inf
        else:
            self.res = float(res)
        if addtomodel: self.model.addElement(self)
        #self.xa,self.ya,self.xb,self.yb,self.np = np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1,'i')  # needed to call bessel.circle_line_intersection

    def __repr__(self):
        return self.name + ' from ' + str((self.x1,self.y1)) +' to '+str((self.x2,self.y2))

    def initialize(self):
        self.Ncp = self.order + 1
        self.z1 = self.x1 + 1j*self.y1; self.z2 = self.x2 + 1j*self.y2
        self.L = np.abs(self.z1-self.z2)
        self.thetaNormOut = np.arctan2(self.y2-self.y1,self.x2-self.x1) - np.pi/2.0
        self.cosout = np.cos( self.thetaNormOut ) * np.ones(self.Ncp); self.sinout = np.sin( self.thetaNormOut ) * np.ones(self.Ncp)
        #
        thetacp = np.arange(np.pi,0,-np.pi/self.Ncp) - 0.5 * np.pi/self.Ncp
        Zcp = np.zeros( self.Ncp, 'D' )
        Zcp.real = np.cos(thetacp)
        Zcp.imag = 1e-6  # control point just on positive site (this is handy later on)
        zcp = Zcp * (self.z2 - self.z1) / 2.0 + 0.5 * (self.z1 + self.z2)
        self.xc = zcp.real; self.yc = zcp.imag
        Zcp.imag = -1e-6  # control point just on negative side (this is needed for building the system of equations)
        zcp = Zcp * (self.z2 - self.z1) / 2.0 + 0.5 * (self.z1 + self.z2)
        self.xcneg = zcp.real; self.ycneg = zcp.imag  # control points just on negative side     
        #
        self.aq = self.model.aq.findAquiferData(self.xc[0],self.yc[0])
        self.setbc()
        coef = self.aq.coef[self.pylayers,:]
        self.setflowcoef()
        self.term = self.flowcoef * coef  # shape (self.Nlayers,self.aq.Naq,self.model.Np)
        self.term2 = self.term.reshape(self.Nlayers,self.aq.Naq,self.model.Nin,self.model.Npin)
        self.resfac = self.aq.Haq[self.pylayers] / self.res
        # Still gotta change strengthinf
        self.strengthinf = self.flowcoef * coef
        self.strengthinflayers = np.sum(self.strengthinf * self.aq.eigvec[self.pylayers,:,:], 1)

    def setflowcoef(self):
        '''Separate function so that this can be overloaded for other types'''
        self.flowcoef = 1.0 / self.model.p  # Step function

    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.findAquiferData( x, y )
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            pot = np.zeros((self.order+1,self.model.Npin),'D')
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if bessel.isinside(self.z1,self.z2,x+y*1j,self.Rzero*self.aq.lababs[i,j]):
                        pot[:,:] = bessel.besselldv2(x,y,self.z1,self.z2,self.aq.lab2[i,j,:],self.order,self.Rzero*self.aq.lababs[i,j]) / self.L  # Divide by L as the parameter is now total discharge
                        for k in range(self.Nlayers):
                            rv[k::self.Nlayers,i,j,:] = self.term2[k,i,j,:] * pot
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv

    def disinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.findAquiferData( x, y )
        rvx,rvy = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D'), np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            qxqy = np.zeros((2*(self.order+1),self.model.Npin),'D')
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if bessel.isinside(self.z1,self.z2,x+y*1j,self.Rzero*self.aq.lababs[i,j]):
                        qxqy[:,:] = bessel.besselldqxqyv2(x,y,self.z1,self.z2,self.aq.lab2[i,j,:],self.order,self.Rzero*self.aq.lababs[i,j]) / self.L  # Divide by L as the parameter is now total discharge
                        for k in range(self.Nlayers):
                            rvx[k::self.Nlayers,i,j,:] = self.term2[k,i,j,:] * qxqy[:self.order+1,:]
                            rvy[k::self.Nlayers,i,j,:] = self.term2[k,i,j,:] * qxqy[self.order+1:,:]
        rvx.shape = (self.Nparam,aq.Naq,self.model.Np)
        rvy.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rvx,rvy

    def plot(self):
        plt.plot([self.x1, self.x2], [self.y1, self.y2], 'k')
    
class LeakyLineDoublet(LineDoubletHoBase, LeakyWallEquation):
    """
    Create a segment of a leaky wall, which is
    simulated with a line-doublet. The specific discharge through
    the wall is equal to the head difference across the wall
    divided by the resistance of the wall. 
    
    Parameters
    ----------
    
    model : Model object
        Model to which the element is added
    x1 : scalar
        x-coordinate of fist point of line-doublet
    y1 : scalar
        y-coordinate of fist point of line-doublet
    x2 : scalar
        x-coordinate of second point of line-doublet
    y2 : scalar
        y-coordinate of second point of line-doublet
    res : scalar or string
        if string: 'imp' for an impermeable wall (same as res = np.inf)
        if scalar: resistance of leaky wall
    order : int (default is 0)
        polynomial order of potential jump along line-doublet
        (head jump if transmissivity is equal on each side of wall)
    layers : scalar, list or array
        layer(s) in which element is placed
        if scalar: element is placed in this layer
        if list or array: element is placed in all these layers 
    label: str or None
        label of element
    
    See Also
    --------
    
    :class:`.LeakyLineDoubletString`
    
    """
    
    def __init__(self, model, x1=-1, y1=0, x2=1, y2=0, res='imp', order=0,
                 layers=0, label=None, addtomodel=True):
        self.storeinput(inspect.currentframe())
        LineDoubletHoBase.__init__(self, model, x1=x1, y1=y1, x2=x2, y2=y2, \
                                   tsandbc=[(0, 0)], res=res, order=order, \
                                   layers=layers, type='z', name='LeakyLineDoublet', \
                                   label=label, addtomodel=addtomodel)
        self.Nunknowns = self.Nparam

    def initialize(self):
        LineDoubletHoBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
class LeakyLineDoubletString(Element, LeakyWallEquation):
    """
    Create a string of leaky wall segements consisting
    of line-doublets
    
    Parameters
    ----------
    
    model : Model object
        Model to which the element is added
    xy : array or list
        list or array of (x,y) pairs of coordinates of end-points of
        the segements in the string
    res : scalar or string
        if string: 'imp' for an impermeable wall (same as res = np.inf)
        if scalar: resistance of leaky wall
    order : int (default is 0)
        polynomial order of potential jump along line-doublet
        (head jump if transmissivity is equal on each side of wall)
    layers : scalar, list or array
        layer(s) in which element is placed
        if scalar: element is placed in this layer
        if list or array: element is placed in all these layers
    label: str or None
        label of element
    
    See Also
    --------
    
    :class:`.LeakyLineDoublet`
    
    """
    
    def __init__(self, model, xy=[(-1, 0), (1, 0)], res='imp', order=0, \
                 layers=0, label=None):
        self.storeinput(inspect.currentframe())
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers, \
                         tsandbc=[(0, 0)], type='z', name='LeakyLineDoubletString', \
                         label=label)
        self.res = res
        self.order = order
        self.ldList = []
        xy = np.atleast_2d(xy).astype('d')
        self.x,self.y = xy[:,0], xy[:,1]
        self.Nld = len(self.x) - 1
        for i in range(self.Nld):
            self.ldList.append(LeakyLineDoublet(model, x1=self.x[i], y1=self.y[i], \
                                                x2=self.x[i + 1], y2=self.y[i + 1], \
                                                res=self.res, order=self.order, \
                                                layers=layers, label=label, \
                                                addtomodel=False))
        self.model.addElement(self)

    def __repr__(self):
        return self.name + ' with nodes ' + str(zip(self.x,self.y))

    def initialize(self):
        for ld in self.ldList:
            ld.initialize()
        self.Ncp = self.Nld * self.ldList[0].Ncp  # Same order for all elements in string
        self.Nparam = self.Nld * self.ldList[0].Nparam
        self.Nunknowns = self.Nparam
        self.xld,self.yld = np.empty((self.Nld,2)), np.empty((self.Nld,2))
        for i,ld in enumerate(self.ldList):
            self.xld[i,:] = [ld.x1,ld.x2]
            self.yld[i,:] = [ld.y1,ld.y2]
        self.xldlayout = np.hstack((self.xld[:,0],self.xld[-1,1])) # Only used for layout when it is a continuous string
        self.yldlayout = np.hstack((self.yld[:,0],self.yld[-1,1]))
        self.aq = self.model.aq.findAquiferData(self.ldList[0].xc,self.ldList[0].yc)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.setbc()
        # As parameters are only stored for the element not the list, we need to combine the following
        self.resfac = self.ldList[0].resfac  # same for all elements in the list
        self.xc, self.yc = np.zeros(self.Ncp), np.zeros(self.Ncp)
        self.xcneg, self.ycneg = np.zeros(self.Ncp), np.zeros(self.Ncp)
        self.cosout, self.sinout = np.zeros(self.Ncp), np.zeros(self.Ncp)
        for i,ld in enumerate(self.ldList):
            self.xc[i*ld.Ncp:(i+1)*ld.Ncp], self.yc[i*ld.Ncp:(i+1)*ld.Ncp] = ld.xc, ld.yc
            self.xcneg[i*ld.Ncp:(i+1)*ld.Ncp], self.ycneg[i*ld.Ncp:(i+1)*ld.Ncp] = ld.xcneg, ld.ycneg
            self.cosout[i*ld.Ncp:(i+1)*ld.Ncp], self.sinout[i*ld.Ncp:(i+1)*ld.Ncp] = ld.cosout, ld.sinout

    def potinf(self, x, y, aq=None):
        '''Returns array (Nunknowns,Nperiods)'''
        if aq is None: aq = self.model.aq.findAquiferData( x, y )
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        for i,ld in enumerate(self.ldList):
            rv[i*ld.Nparam:(i+1)*ld.Nparam,:] = ld.potinf(x,y,aq)
        return rv

    def disinf(self, x, y, aq=None):
        '''Returns array (Nunknowns,Nperiods)'''
        if aq is None: aq = self.model.aq.findAquiferData( x, y )
        rvx,rvy = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D'),np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        for i,ld in enumerate(self.ldList):
            qx,qy = ld.disinf(x,y,aq)
            rvx[i*ld.Nparam:(i+1)*ld.Nparam,:] = qx
            rvy[i*ld.Nparam:(i+1)*ld.Nparam,:] = qy
        return rvx,rvy
    
    def plot(self):
        plt.plot(self.xldlayout, self.yldlayout, 'k')
