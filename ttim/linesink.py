import numpy as np
import matplotlib.pyplot as plt
import inspect # Used for storing the input
from .element import Element
from .bessel import *
from .equation import HeadEquation, HeadEquationNores, MscreenEquation, MscreenDitchEquation

class LineSinkBase(Element):
    '''LineSink Base Class. All LineSink elements are derived from this class'''
    def __init__(self, model, x1=-1, y1=0, x2=1, y2=0, tsandbc=[(0, 1)], \
                 res=0, wh='H', layers=0, type='', name='LineSinkBase', \
                 label=None, addtomodel=True):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers,
                         tsandbc=tsandbc, type=type, name=name, label=label)
        self.Nparam = len(self.pylayers)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.res = np.atleast_1d(res).astype(float)
        self.wh = wh
        if addtomodel: self.model.addElement(self)
        self.xa,self.ya,self.xb,self.yb,self.np = np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1,'i')  # needed to call bessel.circle_line_intersection

    def __repr__(self):
        return self.name + ' from ' + str((self.x1,self.y1)) +' to '+str((self.x2,self.y2))

    def initialize(self):
        self.xc = np.array([0.5*(self.x1+self.x2)]); self.yc = np.array([0.5*(self.y1+self.y2)])
        self.Ncp = 1
        self.z1 = self.x1 + 1j*self.y1; self.z2 = self.x2 + 1j*self.y2
        self.L = np.abs(self.z1-self.z2)
        self.order = 0 # This is for univform discharge only
        self.aq = self.model.aq.findAquiferData(self.xc,self.yc)
        self.setbc()
        coef = self.aq.coef[self.pylayers,:]
        self.setflowcoef()
        self.term = self.flowcoef * coef  # shape (self.Nparam,self.aq.Naq,self.model.Np)
        self.term2 = self.term.reshape(self.Nparam,self.aq.Naq,self.model.Nin,self.model.Npin)
        self.dischargeinf = self.flowcoef * coef
        self.dischargeinflayers = np.sum(self.dischargeinf * self.aq.eigvec[self.pylayers,:,:], 1)
        if type(self.wh) is str:
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
        self.flowcoef = 1.0 / self.model.p  # Step function

    def potinf(self, x, y, aq=None):
        '''Can be called with only one x,y value'''
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        rv = np.zeros((self.Nparam, aq.Naq, self.model.Nin, self.model.Npin), 'D')
        if aq == self.aq:
            pot = np.zeros(self.model.Npin, 'D')
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    bessel.circle_line_intersection(self.z1,self.z2,x+y*1j,self.Rzero*abs(self.model.aq.lab2[i,j,0]),self.xa,self.ya,self.xb,self.yb,self.np)
                    if self.np > 0:
                        za = complex(self.xa,self.ya); zb = complex(self.xb,self.yb) # f2py has problem returning complex arrays -> fixed in new numpy
                        bessel.bessellsuniv(x,y,za,zb,self.aq.lab2[i,j,:],pot)
                        rv[:,i,j,:] = self.term2[:,i,j,:] * pot / self.L  # Divide by L as the parameter is now total discharge
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv

    def disinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        rvx = np.zeros((self.Nparam, aq.Naq, self.model.Nin, self.model.Npin), 'D')
        rvy = np.zeros((self.Nparam, aq.Naq, self.model.Nin, self.model.Npin), 'D')
        if aq == self.aq:
            qxqy = np.zeros((2,self.model.Npin),'D')
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if bessel.isinside(self.z1,self.z2,x+y*1j,self.Rzero*self.aq.lababs[i,j]):
                        qxqy[:,:] = bessel.bessellsqxqyv2(x,y,self.z1,self.z2,self.aq.lab2[i,j,:],self.order,self.Rzero*self.aq.lababs[i,j]) / self.L  # Divide by L as the parameter is now total discharge
                        rvx[:,i,j,:] = self.term2[:,i,j,:] * qxqy[0]
                        rvy[:,i,j,:] = self.term2[:,i,j,:] * qxqy[1]
        rvx.shape = (self.Nparam, aq.Naq, self.model.Np)
        rvy.shape = (self.Nparam, aq.Naq, self.model.Np)
        return rvx, rvy

    def headinside(self,t):
        """The head inside the line-sink
        
        Parameters
        ----------
        t : array or float
            time(s) for whih head is computed
        
        Returns
        -------
        array (length number of layers)
            Head inside the line-sink for each layer that the line-sink is screened in
            
        """
        
        return self.model.head(self.xc,self.yc,t)[self.pylayers] - self.resfach[:,np.newaxis] * self.discharge(t)

    def plot(self):
        plt.plot([self.x1, self.x2], [self.y1, self.y2], 'k')
        
class LineSink(LineSinkBase):
    '''LineSink with non-zero and potentially variable discharge through time
    really only used for testing'''
    def __init__(self, model, x1=-1, y1=0, x2=1, y2=0, tsandQ=[(0, 1)], \
                 res=0, wh='H', layers=0, label=None, addtomodel=True):
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self, model, x1=x1, y1=y1, x2=x2, y2=y2, tsandbc=tsandQ, \
                              res=res, wh=wh, layers=layers, type='g', name='LineSink', \
                              label=label,addtomodel=addtomodel)

    
#class ZeroHeadLineSink(LineSinkBase,HeadEquation):
    #'''HeadLineSink that remains zero and constant through time'''
    #def __init__(self, model, x1=-1, y1=0, x2=1, y2=0, res=0.0, wh='H', \
    #             layers=0, label=None, addtomodel=True):
    #    self.storeinput(inspect.currentframe())
    #    LineSinkBase.__init__(self, model, x1=x1, y1=y1, x2=x2, y2=y2, \
    #                          tsandbc=[(0, 0)], res=res, wh=wh, layers=layers, \
    #                          type='z', name='ZeroHeadLineSink', label=label, \
    #                          addtomodel=addtomodel)
    #    self.Nunknowns = self.Nparam
    #    
    #def initialize(self):
    #    LineSinkBase.initialize(self)
    #    self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
class HeadLineSink(LineSinkBase, HeadEquation):
    """
    Create a head-specified line-sink
    which may optionally have a width and resistance
    Inflow per unit length of line-sink is computed as
    
    .. math::
        \sigma = w(h_{aq} - h_{ls})/c
    
    where :math:`c` is the resistance of the bottom of the line-sink,
    :math:`w` is the width over which water enters the line-sink,
    :math:`h_{aq}` is the head in the aquifer at the center of the line-sink,
    :math:`h_{ls}` is the specified head inside the line-sink
    Note that all that matters is the conductance term :math:`w/c` but
    both are specified separately
    
    Parameters
    ----------
    
    model : Model object
        Model to which the element is added
    x1 : scalar
        x-coordinate of fist point of line-sink
    y1 : scalar
        y-coordinate of fist point of line-sink
    x2 : scalar
        x-coordinate of second point of line-sink
    y2 : scalar
        y-coordinate of second point of line-sink
    tsandh : list or 2D array of (time, head) values or string
        if list or 2D array: pairs of time and head after that time
        if 'fixed': head is fixed (no change in head) during entire simulation
    res : scalar (default is 0)
        resistance of line-sink
    wh : scalar or str
        distance over which water enters line-sink
        if 'H': (default) distance is equal to the thickness of the aquifer layer (when flow comes mainly from one side)
        if '2H': distance is twice the thickness of the aquifer layer (when flow comes from both sides)
        if scalar: the width of the stream that partially penetrates the aquifer layer
    layers : scalar, list or array
        layer(s) in which element is placed
        if scalar: element is placed in this layer
        if list or array: element is placed in all these layers 
    label: str or None
        label of element
    
    See Also
    --------
    
    :class:`.HeadLineSinkString`
    
    """
    
    def __init__(self, model, x1=-1, y1=0, x2=1, y2=0, tsandh=[(0, 1)], \
                 res=0, wh='H', layers=0, label=None, addtomodel=True):
        self.storeinput(inspect.currentframe())
        if tsandh == 'fixed':
            tsandh = [(0, 0)]
            etype = 'z'
        else:
            etype = 'v'
        LineSinkBase.__init__(self, model, x1=x1, y1=y1, x2=x2, y2=y2, \
                              tsandbc=tsandh, res=res, wh=wh, layers=layers, \
                              type=etype, name='HeadLineSink', label=label, \
                              addtomodel=addtomodel)
        self.Nunknowns = self.Nparam

    def initialize(self):
        LineSinkBase.initialize(self)
        self.parameters = np.zeros((self.model.Ngvbc, self.Nparam, self.model.Np), 'D')
        self.pc = self.aq.T[self.pylayers] # Needed in solving; We solve for a unit head
        
class LineSinkStringBase(Element):
    def __init__(self, model, tsandbc=[(0, 1)], layers=0, type='',
                 name='LineSinkStringBase', label=None):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers, \
                         tsandbc=tsandbc, type=type, name=name, label=label)
        self.lsList = []

    def __repr__(self):
        return self.name + ' with nodes ' + str(zip(self.x,self.y))

    def initialize(self):
        self.Ncp = self.Nls
        self.Nparam = self.Nlayers * self.Nls
        self.Nunknowns = self.Nparam
        self.xls,self.yls = np.empty((self.Nls,2)), np.empty((self.Nls,2))
        for i,ls in enumerate(self.lsList):
            ls.initialize()
            self.xls[i,:] = [ls.x1,ls.x2]
            self.yls[i,:] = [ls.y1,ls.y2]
        self.xlslayout = np.hstack((self.xls[:,0],self.xls[-1,1])) # Only used for layout when it is a continuous string
        self.ylslayout = np.hstack((self.yls[:,0],self.yls[-1,1]))
        self.aq = self.model.aq.findAquiferData(self.lsList[0].xc,self.lsList[0].yc)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.setbc()
        # As parameters are only stored for the element not the list, we need to combine the following
        self.resfach = []; self.resfacp = []
        for ls in self.lsList:
            ls.initialize()
            self.resfach.extend( ls.resfach.tolist() )  # Needed in solving
            self.resfacp.extend( ls.resfacp.tolist() )  # Needed in solving
        self.resfach = np.array(self.resfach); self.resfacp = np.array(self.resfacp)
        self.dischargeinf = np.zeros((self.Nparam,self.aq.Naq,self.model.Np),'D')
        self.dischargeinflayers = np.zeros((self.Nparam,self.model.Np),'D')
        self.xc, self.yc = np.zeros(self.Nls), np.zeros(self.Nls)
        for i in range(self.Nls):
            self.dischargeinf[i*self.Nlayers:(i+1)*self.Nlayers,:] = self.lsList[i].dischargeinf[:]
            self.dischargeinflayers[i*self.Nlayers:(i+1)*self.Nlayers,:] = self.lsList[i].dischargeinflayers
            self.xc[i], self.yc[i] = self.lsList[i].xc, self.lsList[i].yc

    def potinf(self,x,y,aq=None):
        '''Returns array (Nunknowns,Nperiods)'''
        if aq is None: aq = self.model.aq.findAquiferData( x, y )
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        for i in range(self.Nls):
            rv[i*self.Nlayers:(i+1)*self.Nlayers,:] = self.lsList[i].potinf(x,y,aq)
        return rv

    def disinf(self,x,y,aq=None):
        '''Returns array (Nunknowns,Nperiods)'''
        if aq is None: aq = self.model.aq.findAquiferData( x, y )
        rvx,rvy = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D'),np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        for i in range(self.Nls):
            qx,qy = self.lsList[i].disinf(x,y,aq)
            rvx[i*self.Nlayers:(i+1)*self.Nlayers,:] = qx
            rvy[i*self.Nlayers:(i+1)*self.Nlayers,:] = qy
        return rvx,rvy

    def headinside(self, t, derivative=0):
        """The head inside the line-sink string
        
        Parameters
        ----------
        t : array or float
            time(s) for whih head is computed
        
        Returns
        -------
        array size nline-sinks, nlayers, ntimes
            Head inside the line-sink for each line-sink, each layer that
            the line-sink is screened in, and each time
            
        """
        
        rv = np.zeros((self.Nls,self.Nlayers,np.size(t)))
        Q = self.discharge_list(t,derivative=derivative)
        for i in range(self.Nls):
            rv[i,:,:] = self.model.head(self.xc[i],self.yc[i],t,derivative=derivative)[self.pylayers] - self.resfach[i*self.Nlayers:(i+1)*self.Nlayers,np.newaxis] * Q[i]
        return rv
    
    def plot(self):
        plt.plot(self.xlslayout, self.ylslayout, 'k')

    def run_after_solve(self):
        for i in range(self.Nls):
            self.lsList[i].parameters[:] = self.parameters[:,i*self.Nlayers:(i+1)*self.Nlayers,:]

    def discharge_list(self,t,derivative=0):
        """The discharge of each line-sink in the string
        
        Parameters
        ----------
        t : array or float
            time(s) for whih discharge is computed
        
        Returns
        -------
        array size nline-sinks, nlayers, ntimes
            Discharge for each line-sink, each layer that
            the line-sink is screened in, and each time
            
        """
        rv = np.zeros((self.Nls,self.Nlayers,np.size(t)))
        for i in range(self.Nls):
            rv[i,:,:] = self.lsList[i].discharge(t,derivative=derivative)
        return rv
            
class HeadLineSinkString(LineSinkStringBase, HeadEquation):
    """
    Create string of head-specified line-sinks
    which may optionally have a width and resistance
    Inflow per unit length of line-sink is computed as
    
    .. math::
        \sigma = w(h_{aq} - h_{ls})/c
    
    where :math:`c` is the resistance of the bottom of the line-sink,
    :math:`w` is the width over which water enters the line-sink,
    :math:`h_{aq}` is the head in the aquifer at the center of the line-sink,
    :math:`h_{ls}` is the specified head inside the line-sink
    Note that all that matters is the conductance term :math:`w/c` but
    both are specified separately
    
    Parameters
    ----------
    
    model : Model object
        Model to which the element is added
    xy : array or list
        list or array of (x,y) pairs of coordinates of end-points of
        line-sinks in string
    tsandh : list or 2D array of (time, head) values or string
        if list or 2D array: pairs of time and head after that time
        if 'fixed': head is fixed (no change in head) during entire simulation
    res : scalar (default is 0)
        resistance of line-sink
    wh : scalar or str
        distance over which water enters line-sink
        if 'H': (default) distance is equal to the thickness of the aquifer layer (when flow comes mainly from one side)
        if '2H': distance is twice the thickness of the aquifer layer (when flow comes from both sides)
        if scalar: the width of the stream that partially penetrates the aquifer layer
    layers : scalar, list or array
        layer(s) in which element is placed
        if scalar: element is placed in this layer
        if list or array: element is placed in all these layers 
    label: str or None
        label of element
    
    See Also
    --------
    
    :class:`.HeadLineSink`
    
    """
    
    def __init__(self, model, xy=[(-1, 0), (1, 0)], tsandh=[(0, 1)], \
                 res=0, wh='H', layers=0, label=None):
        if tsandh == 'fixed':
            tsandh = [(0, 0)]
            etype = 'z'
        else:
            etype = 'v'
        LineSinkStringBase.__init__(self, model, tsandbc=tsandh, layers=layers, \
                                    type=etype, name='HeadLineSinkString', label=label)
        xy = np.atleast_2d(xy).astype('d')
        self.x = xy[:, 0]
        self.y = xy[:, 1]
        self.Nls = len(self.x) - 1
        for i in range(self.Nls):
            self.lsList.append(HeadLineSink(model, x1=self.x[i], y1=self.y[i], \
                                            x2=self.x[i + 1], y2=self.y[i + 1], \
                                            tsandh=tsandh, res=res, wh=wh, \
                                            layers=layers, label=None, \
                                            addtomodel=False) )
        self.model.addElement(self)

    def initialize(self):
        LineSinkStringBase.initialize(self)
        self.pc = np.zeros(self.Nls * self.Nlayers)
        for i in range(self.Nls):
            self.pc[i * self.Nlayers:(i + 1) * self.Nlayers] = self.lsList[i].pc
            
class MscreenLineSink(LineSinkBase,MscreenEquation):
    '''MscreenLineSink that varies through time. Must be screened in multiple layers but heads are same in all screened layers'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandQ=[(0.0,1.0)],res=0.0,wh='H',layers=[0,1],vres=0.0,wv=1.0,label=None,addtomodel=True):
        #assert len(layers) > 1, "TTim input error: number of layers for MscreenLineSink must be at least 2"
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=tsandQ,res=res,wh=wh,layers=layers,type='v',name='MscreenLineSink',label=label,addtomodel=addtomodel)
        self.Nunknowns = self.Nparam
        self.vres = np.atleast_1d(vres)  # Vertical resistance inside line-sink
        self.wv = wv
        if len(self.vres) == 1: self.vres = self.vres[0] * np.ones(self.Nlayers-1)
    def initialize(self):
        LineSinkBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.vresfac = self.vres / (self.wv * self.L)  # Qv = (hn - hn-1) / vresfac[n-1]
            
class LineSinkDitchString(LineSinkStringBase, MscreenDitchEquation):
    """
    Create ditch consisting of a string of line-sink.
    The total discharge for the string is specified and divided over the
    line-sinks such that the head at the center inside each line-sink is
    equal. A width and resistance may optionally be specified.
    Inflow per unit length of line-sink is computed as
    
    .. math::
        \sigma = w(h_{aq} - h_{ls})/c
    
    where :math:`c` is the resistance of the bottom of the line-sink,
    :math:`w` is the width over which water enters the line-sink,
    :math:`h_{aq}` is the head in the aquifer at the center of the line-sink,
    :math:`h_{ls}` is the specified head inside the line-sink
    Note that all that matters is the conductance term :math:`w/c` but
    both are specified separately
    
    Parameters
    ----------
    
    model : Model object
        Model to which the element is added
    xy : array or list
        list or array of (x,y) pairs of coordinates of end-points of
        line-sinks in string
    tsandQ : list or 2D array of (time, discharge) values
        if list or 2D array: pairs of time and discharge after that time
    res : scalar (default is 0)
        resistance of line-sink
    wh : scalar or str
        distance over which water enters line-sink
        if 'H': (default) distance is equal to the thickness of the aquifer layer (when flow comes mainly from one side)
        if '2H': distance is twice the thickness of the aquifer layer (when flow comes from both sides)
        if scalar: the width of the stream that partially penetrates the aquifer layer
    layers : scalar, list or array
        layer(s) in which element is placed
        if scalar: element is placed in this layer
        if list or array: element is placed in all these layers 
    label: str or None
        label of element
    
    """
    
    def __init__(self, model, xy=[(-1, 0), (1, 0)], tsandQ=[(0, 1)], res=0, \
                 wh='H', layers=0, Astorage=None, label=None):
        self.storeinput(inspect.currentframe())
        LineSinkStringBase.__init__(self, model, tsandbc=tsandQ, layers=layers, \
                                    type='v', name='LineSinkDitchString', \
                                    label=label)
        xy = np.atleast_2d(xy).astype('d')
        self.x,self.y = xy[:, 0], xy[:, 1]
        self.Nls = len(self.x) - 1
        for i in range(self.Nls):
            self.lsList.append(MscreenLineSink(model, x1=self.x[i], y1=self.y[i], \
                                               x2=self.x[i + 1], y2=self.y[i + 1], \
                                               tsandQ=tsandQ, res=res, wh=wh,
                                               layers=layers, label=None, addtomodel=False))
        self.Astorage = Astorage
        self.model.addElement(self)
    def initialize(self):
        LineSinkStringBase.initialize(self)
        self.vresfac = np.zeros_like(self.resfach)  # set to zero, as I don't quite know what it would mean if it is not zero

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
        self.dischargeinf = self.flowcoef * coef
        self.dischargeinflayers = np.sum(self.dischargeinf * self.aq.eigvec[self.pylayers, :, :], 1)
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
        """The head inside the line-sink
        
        Returns
        -------
        array (length number of screens)
            Head inside the well for each screen
            
        """
        
        return self.model.head(self.xc, self.yc, t)[self.pylayers] - self.resfach[:, np.newaxis] * self.discharge(t)

    def plot(self):
        plt.plot([self.x1, self.x2], [self.y1, self.y2], 'k')
    
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