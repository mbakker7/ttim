'''
Copyright (C), 2010-2015, Mark Bakker.
TTim is distributed under the MIT license
'''

import numpy as np
import matplotlib.pyplot as plt
from bessel import *
from invlap import *
from scipy.special import kv,iv # Needed for K1 in Well class, and in CircInhom
from cmath import tanh as cmath_tanh
import inspect # Used for storing the input
import os, sys
from mathieu_functions import mathieu

class TimModel:
    def __init__(self,kaq=[1,1],Haq=[1,1],c=[1e100,100],Saq=[0.3,0.003],Sll=[0],topboundary='imp',tmin=1,tmax=10,M=20):
        self.elementList = []
        self.elementDict = {}
        self.vbcList = []  # List with variable boundary condition 'v' elements
        self.zbcList = []  # List with zero and constant boundary condition 'z' elements
        self.gbcList = []  # List with given boundary condition 'g' elements; given bc elements don't have any unknowns
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.M = M
        self.aq = Aquifer(self,kaq,Haq,c,Saq,Sll,topboundary)
        self.compute_laplace_parameters()
        self.name = 'TimModel'
        self.modelname = 'ml' # Used for writing out input
        bessel.initialize()
    def __repr__(self):
        return 'Model'
    def initialize(self):
        self.gvbcList = self.gbcList + self.vbcList
        self.vzbcList = self.vbcList + self.zbcList
        self.elementList = self.gbcList + self.vbcList + self.zbcList  # Given elements are first in list
        self.Ngbc = len(self.gbcList)
        self.Nvbc = len(self.vbcList)
        self.Nzbc = len(self.zbcList)
        self.Ngvbc = self.Ngbc + self.Nvbc
        self.aq.initialize()
        for e in self.elementList:
            e.initialize()
    def addElement(self,e):
        if e.label is not None: self.elementDict[e.label] = e
        if e.type == 'g':
            self.gbcList.append(e)
        elif e.type == 'v':
            self.vbcList.append(e)
        elif e.type == 'z':
            self.zbcList.append(e)
    def removeElement(self,e):
        if e.label is not None: self.elementDict.pop(e.label)
        if e.type == 'g':
            self.gbcList.remove(e)
        elif e.type == 'v':
            self.vbcList.remove(e)
        elif e.type == 'z':
            self.zbcList.remove(e)
    def addInhom(self,inhom):
        self.aq.inhomList.append(inhom)
    def compute_laplace_parameters(self):
        '''
        Nin: Number of time intervals
        npint: Number of p values per interval
        npval: Total number of p values (Nin*npval)
        p[npval]: Array with p values
        '''
        itmin = np.floor(np.log10(self.tmin))
        itmax = np.ceil(np.log10(self.tmax))
        self.tintervals = 10.0**np.arange(itmin,itmax+1)
        # lower and upper limit are adjusted to prevent any problems from t exactly at the beginning and end of the interval
        # also, you cannot count on t >= 10**log10(t) for all possible t
        self.tintervals[0] = self.tintervals[0] * ( 1 - np.finfo(float).epsneg )
        self.tintervals[-1] = self.tintervals[-1] * ( 1 + np.finfo(float).eps )
        #alpha = 1.0
        alpha = 0.0  # I don't see why it shouldn't be 0.0
        tol = 1e-9
        self.Nin = len(self.tintervals)-1
        run = np.arange(2*self.M+1)  # so there are 2M+1 terms in Fourier series expansion
        self.p = []
        self.gamma = []
        for i in range(self.Nin):
            T = self.tintervals[i+1] * 2.0
            gamma = alpha - np.log(tol) / (T/2.0)
            p = gamma + 1j * np.pi * run / T
            self.p.extend( p.tolist() )
            self.gamma.append(gamma)
        self.p = np.array(self.p)
        self.gamma = np.array(self.gamma)
        self.Np = len(self.p)
        self.Npin = 2 * self.M + 1
        self.aq.initialize()
    def potential(self,x,y,t,pylayers=None,aq=None,derivative=0,returnphi=0):
        '''Returns pot[Naq,Ntimes] if layers=None, otherwise pot[len(layers,Ntimes)]
        t must be ordered '''
        if aq is None: aq = self.aq.findAquiferData(x,y)
        if pylayers is None: pylayers = range(aq.Naq)
        Nlayers = len(pylayers)
        time = np.atleast_1d(t).copy()
        pot = np.zeros((self.Ngvbc, aq.Naq, self.Np),'D')
        for i in range(self.Ngbc):
            pot[i,:] += self.gbcList[i].unitpotential(x,y,aq)
        for e in self.vzbcList:
            pot += e.potential(x,y,aq)
        if pylayers is None:
            pot = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec, 2 )
        else:
            pot = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec[pylayers,:], 2 )
        if derivative > 0: pot *= self.p**derivative
        if returnphi: return pot
        rv = np.zeros((Nlayers,len(time)))
        if (time[0] < self.tmin) or (time[-1] > self.tmax): print 'Warning, some of the times are smaller than tmin or larger than tmax; zeros are substituted'
        #
        for k in range(self.Ngvbc):
            e = self.gvbcList[k]
            for itime in range(e.Ntstart):
                t = time - e.tstart[itime]
                it = 0
                if t[-1] >= self.tmin:  # Otherwise all zero
                    if (t[0] < self.tmin): it = np.argmax( t >= self.tmin )  # clever call that should be replaced with find_first function when included in numpy
                    for n in range(self.Nin):
                        tp = t[ (t >= self.tintervals[n]) & (t < self.tintervals[n+1]) ]
                        ## I think these lines are not needed anymore as I modified tintervals[0] and tintervals[-1] by eps
                        #if n == self.Nin-1:
                        #    tp = t[ (t >= self.tintervals[n]) & (t <= self.tintervals[n+1]) ]
                        #else:
                        #    tp = t[ (t >= self.tintervals[n]) & (t < self.tintervals[n+1]) ]
                        Nt = len(tp)
                        if Nt > 0:  # if all values zero, don't do the inverse transform
                            for i in range(Nlayers):
                                # I used to check the first value only, but it seems that checking that nothing is zero is needed and should be sufficient
                                #if np.abs( pot[k,i,n*self.npint] ) > 1e-20:  # First value very small
                                if not np.any( pot[k,i,n*self.Npin:(n+1)*self.Npin] == 0.0) : # If there is a zero item, zero should be returned; funky enough this can be done with a straight equal comparison
                                    rv[i,it:it+Nt] += e.bc[itime] * invlaptrans.invlap( tp, self.tintervals[n], self.tintervals[n+1], pot[k,i,n*self.Npin:(n+1)*self.Npin], self.gamma[n], self.M, Nt )
                            it = it + Nt
        return rv
    def discharge(self,x,y,t,layers=None,aq=None,derivative=0):
        '''Returns qx[Naq,Ntimes],qy[Naq,Ntimes] if layers=None, otherwise qx[len(layers,Ntimes)],qy[len(layers,Ntimes)]
        t must be ordered '''
        if aq is None: aq = self.aq.findAquiferData(x,y)
        if layers is None:
            pylayers = range(aq.Naq)
        else:
            pylayers = np.atleast_1d(layers)  # corrected for base zero
        Nlayers = len(pylayers)
        time = np.atleast_1d(t).copy()
        disx,disy = np.zeros((self.Ngvbc, aq.Naq, self.Np),'D'), np.zeros((self.Ngvbc, aq.Naq, self.Np),'D')
        for i in range(self.Ngbc):
            qx,qy = self.gbcList[i].unitdischarge(x,y,aq)
            disx[i,:] += qx; disy[i,:] += qy
        for e in self.vzbcList:
            qx,qy = e.discharge(x,y,aq)
            disx += qx; disy += qy
        if pylayers is None:
            disx = np.sum( disx[:,np.newaxis,:,:] * aq.eigvec, 2 )
            disy = np.sum( disy[:,np.newaxis,:,:] * aq.eigvec, 2 )
        else:
            disx = np.sum( disx[:,np.newaxis,:,:] * aq.eigvec[pylayers,:], 2 )
            disy = np.sum( disy[:,np.newaxis,:,:] * aq.eigvec[pylayers,:], 2 )
        if derivative > 0:
            disx *= self.p**derivative
            disy *= self.p**derivative
        rvx,rvy = np.zeros((Nlayers,len(time))), np.zeros((Nlayers,len(time)))
        if (time[0] < self.tmin) or (time[-1] > self.tmax): print 'Warning, some of the times are smaller than tmin or larger than tmax; zeros are substituted'
        #
        for k in range(self.Ngvbc):
            e = self.gvbcList[k]
            for itime in range(e.Ntstart):
                t = time - e.tstart[itime]
                it = 0
                if t[-1] >= self.tmin:  # Otherwise all zero
                    if (t[0] < self.tmin): it = np.argmax( t >= self.tmin )  # clever call that should be replaced with find_first function when included in numpy
                    for n in range(self.Nin):
                        tp = t[ (t >= self.tintervals[n]) & (t < self.tintervals[n+1]) ]
                        Nt = len(tp)
                        if Nt > 0:  # if all values zero, don't do the inverse transform
                            for i in range(Nlayers):
                                if not np.any( disx[k,i,n*self.Npin:(n+1)*self.Npin] == 0.0) : # If there is a zero item, zero should be returned; funky enough this can be done with a straight equal comparison
                                    rvx[i,it:it+Nt] += e.bc[itime] * invlaptrans.invlap( tp, self.tintervals[n], self.tintervals[n+1], disx[k,i,n*self.Npin:(n+1)*self.Npin], self.gamma[n], self.M, Nt )
                                    rvy[i,it:it+Nt] += e.bc[itime] * invlaptrans.invlap( tp, self.tintervals[n], self.tintervals[n+1], disy[k,i,n*self.Npin:(n+1)*self.Npin], self.gamma[n], self.M, Nt )
                            it = it + Nt
        return rvx,rvy
    def head(self,x,y,t,layers=None,aq=None,derivative=0):
        if aq is None: aq = self.aq.findAquiferData(x,y)
        if layers is None:
            pylayers = range(aq.Naq)
        else:
            pylayers = np.atleast_1d(layers)  # corrected for base zero
        pot = self.potential(x,y,t,pylayers,aq,derivative)
        return aq.potentialToHead(pot,pylayers)
    def headinside(self,elabel,t):
        return self.elementDict[elabel].headinside(t)
    def strength(self,elabel,t):
        return self.elementDict[elabel].strength(t)
    def headalongline(self,x,y,t,layers=None):
        '''Returns head[nlayers,len(t),len(x)]
        Assumes same number of layers for each x and y
        layers may be None or list of layers for which head is computed'''
        xg,yg = np.atleast_1d(x),np.atleast_1d(y)
        if layers is None:
            Nlayers = self.aq.findAquiferData(xg[0],yg[0]).Naq
        else:
            Nlayers = len(np.atleast_1d(layers))
        nx = len(xg)
        if len(yg) == 1:
            yg = yg * np.ones(nx)
        t = np.atleast_1d(t)
        h = np.zeros( (Nlayers,len(t),nx) )
        for i in range(nx):
            h[:,:,i] = self.head(xg[i],yg[i],t,layers)
        return h
    def headgrid(self,x1,x2,nx,y1,y2,ny,t,layers=None,printrow=False):
        '''Returns h[nlayers,Ntimes,Ny,Nx]. If layers is None, all layers are returned'''
        xg,yg = np.linspace(x1,x2,nx), np.linspace(y1,y2,ny)
        if layers is None:
            Nlayers = self.aq.findAquiferData(xg[0],yg[0]).Naq
        else:
            Nlayers = len(np.atleast_1d(layers))
        t = np.atleast_1d(t)
        h = np.empty( (Nlayers,len(t),ny,nx) )
        for j in range(ny):
            if printrow: print str(j)+' '
            for i in range(nx):
                h[:,:,j,i] = self.head(xg[i],yg[j],t,layers)
        return h
    def headgrid2(self,xg,yg,t,layers=None,printrow=False):
        '''Returns h[nlayers,Ntimes,Ny,Nx]. If layers is None, all layers are returned'''
        nx,ny = len(xg), len(yg)
        if layers is None:
            Nlayers = self.aq.findAquiferData(xg[0],yg[0]).Naq
        else:
            Nlayers = len(np.atleast_1d(layers))
        t = np.atleast_1d(t)
        h = np.empty( (Nlayers,len(t),ny,nx) )
        for j in range(ny):
            if printrow: print str(j)+' '
            for i in range(nx):
                h[:,:,j,i] = self.head(xg[i],yg[j],t,layers)
        return h
    #def velocity(self, x, y, t, layers=None, aq=None):
    #    # implemented for Model3D
    #    if aq is None: aq = self.aq.findAquiferData(x,y)
    #    if layers is None:
    #        layers = range(aq.Naq)
    #    else:
    #        layers = np.atleast_1d(layers)  # corrected for base zero
    #    h = self.head(x, y, t, aq=aq)
    #    qx, qy = self.discharge(x, y, t, aq=aq)
    #
    #
    #    def velocity(self,x,y,z):
    #    head = self.headVector(x,y)
    #    [disx, disy] = self.dischargeCollection(x,y)
    #    aqdata = self.aq.findAquiferData(x,y)
    #    pyLayer = self.inWhichPyLayer(x,y,z,aqdata)
    #    assert pyLayer != -9999 and pyLayer != 9999, 'TimML error: (x,y,z) outside aquifer '+str((x,y,z))
    #    if pyLayer >= 0:  # In aquifer
    #        vx = disx[pyLayer] / ( aqdata.H[pyLayer] * aqdata.n[pyLayer] )
    #        vy = disy[pyLayer] / ( aqdata.H[pyLayer] * aqdata.n[pyLayer] )
    #        if pyLayer > 0:
    #            vztop = ( head[pyLayer] - head[pyLayer-1] ) / ( aqdata.c[pyLayer] * aqdata.n[pyLayer] )
    #        else:
    #            if aqdata.type == aqdata.conf:
    #                vztop = self.qzTop(x,y) / aqdata.n[pyLayer]
    #            elif aqdata.type == aqdata.semi:
    #                vztop = ( head[0] - aqdata.hstar ) / ( aqdata.c[0] * aqdata.n[0] )
    #        if pyLayer < aqdata.Naquifers-1:
    #            vzbot = ( head[pyLayer+1] - head[pyLayer] ) / ( aqdata.c[pyLayer+1] * aqdata.n[pyLayer] )
    #        else:
    #            vzbot = 0.0
    #        vz = (z - aqdata.zb[pyLayer]) * (vztop - vzbot) / aqdata.H[pyLayer] + vzbot
    #    else:  # In leaky layer
    #        vx = 0.0
    #        vy = 0.0
    #        vz = ( head[-pyLayer] - head[-pyLayer-1] ) / ( aqdata.c[-pyLayer] * aqdata.nll[-pyLayer] ) 
    #    return array([vx,vy,vz])
    def inverseLapTran(self,pot,t):
        '''returns array of potentials of len(t)
        t must be ordered and tmin <= t <= tmax'''
        t = np.atleast_1d(t)
        rv = np.zeros(len(t))
        it = 0
        if t[-1] >= self.tmin:  # Otherwise all zero
            if (t[0] < self.tmin): it = np.argmax( t >= self.tmin )  # clever call that should be replaced with find_first function when included in numpy
            for n in range(self.Nin):
                if n == self.Nin-1:
                    tp = t[ (t >= self.tintervals[n]) & (t <= self.tintervals[n+1]) ]
                else:
                    tp = t[ (t >= self.tintervals[n]) & (t < self.tintervals[n+1]) ]
                Nt = len(tp)
                if Nt > 0:  # if all values zero, don't do the inverse transform
                    # Not needed anymore: if np.abs( pot[n*self.npint] ) > 1e-20:
                    if not np.any( pot[n*self.Npin:(n+1)*self.Npin] == 0.0) : # If there is a zero item, zero should be returned; funky enough this can be done with a straight equal comparison
                        rv[it:it+Nt] = invlaptrans.invlap( tp, self.tintervals[n], self.tintervals[n+1], pot[n*self.Npin:(n+1)*self.Npin], self.gamma[n], self.M, Nt )
                    it = it + Nt
        return rv
    def solve(self,printmat = 0,sendback=0,silent=False):
        '''Compute solution'''
        # Initialize elements
        self.initialize()
        # Compute number of equations
        self.Neq = np.sum( [e.Nunknowns for e in self.elementList] )
        if silent is False:
            print 'self.neq ',self.Neq
        if self.Neq == 0:
            if silent is False: print 'No unknowns. Solution complete'
            return
        mat = np.empty( (self.Neq,self.Neq,self.Np), 'D' )
        rhs = np.empty( (self.Neq,self.Ngvbc,self.Np), 'D' )
        ieq = 0
        for e in self.elementList:
            if e.Nunknowns > 0:
                mat[ ieq:ieq+e.Nunknowns, :, : ], rhs[ ieq:ieq+e.Nunknowns, :, : ] = e.equation()
                ieq += e.Nunknowns
        if printmat:
            return mat,rhs
        for i in range( self.Np ):
            sol = np.linalg.solve( mat[:,:,i], rhs[:,:,i] )
            icount = 0
            for e in self.elementList:
                for j in range(e.Nunknowns):
                    e.parameters[:,j,i] = sol[icount,:]
                    icount += 1
                e.run_after_solve()
        if silent is False:
            print 'solution complete'
        elif (silent == 'dot') or (silent == '.'):
            print '.',
            sys.stdout.flush()  # Can be replaced with print with flush in Python 3.3
        if sendback:
            return sol
        return
    def storeinput(self,frame):
        self.inputargs, _, _, self.inputvalues = inspect.getargvalues(frame)
    def write(self):
        rv = self.modelname + ' = '+self.name+'(\n'
        for key in self.inputargs[1:]:  # The first argument (self) is ignored
            if isinstance(self.inputvalues[key],np.ndarray):
                rv += key + ' = ' + np.array2string(self.inputvalues[key],separator=',') + ',\n'
            elif isinstance(self.inputvalues[key],str):                
                rv += key + " = '" + self.inputvalues[key] + "',\n"
            else:
                rv += key + ' = ' + str(self.inputvalues[key]) + ',\n'
        rv += ')\n'
        return rv
    def writemodel(self,fname):
        self.initialize()  # So that the model can be written without solving first
        f = open(fname,'w')
        f.write('from ttim import *\n')
        f.write( self.write() )
        for e in self.elementList:
            f.write( e.write() )
        f.close()
        
        
def param_maq(kaq=[1],z=[1,0],c=[],Saq=[0.001],Sll=[0],topboundary='imp',phreatictop=False):
    # Computes the parameters for a TimModel from input for a maq model
    kaq = np.atleast_1d(kaq).astype('d')
    Naq = len(kaq)
    z = np.atleast_1d(z).astype('d')
    c = np.atleast_1d(c).astype('d')
    Saq = np.atleast_1d(Saq).astype('d')
    if len(Saq) == 1: Saq = Saq * np.ones(Naq)
    Sll = np.atleast_1d(Sll).astype('d')
    H = z[:-1] - z[1:]
    assert np.all(H >= 0), 'Error: Not all layers thicknesses are non-negative' + str(H) 
    if topboundary[:3] == 'imp':
        assert len(z) == 2*Naq, 'Error: Length of z needs to be ' + str(2*Naq)
        assert len(c) == Naq-1, 'Error: Length of c needs to be ' + str(Naq-1)
        assert len(Saq) == Naq, 'Error: Length of Saq needs to be ' + str(Naq)
        if len(Sll) == 1: Sll = Sll * np.ones(Naq-1)
        assert len(Sll) == Naq-1, 'Error: Length of Sll needs to be ' + str(Naq-1)
        Haq = H[::2]
        Saq = Saq * Haq
        if phreatictop: Saq[0] = Saq[0] / H[0]
        Sll = Sll * H[1::2]
        c = np.hstack((1e100,c))  # changed (nan,c) to (1e100,c) as I get an error
        Sll = np.hstack((1e-30,Sll)) # Was: Sll = np.hstack((np.nan,Sll)), but that gives error when c approaches inf
    else: # leaky layers on top
        assert len(z) == 2*Naq+1, 'Error: Length of z needs to be ' + str(2*Naq+1)
        assert len(c) == Naq, 'Error: Length of c needs to be ' + str(Naq)
        assert len(Saq) == Naq, 'Error: Length of Saq needs to be ' + str(Naq)
        if len(Sll) == 1: Sll = Sll * np.ones(Naq)
        assert len(Sll) == Naq, 'Error: Length of Sll needs to be ' + str(Naq)
        Haq = H[1::2]
        Saq = Saq * Haq
        Sll = Sll * H[::2]
        if phreatictop and (topboundary[:3]=='lea'): Sll[0] = Sll[0] / H[0]
    return kaq,Haq,c,Saq,Sll
        
class ModelMaq(TimModel):
    def __init__(self,kaq=[1],z=[1,0],c=[],Saq=[0.001],Sll=[0],topboundary='imp',phreatictop=False,tmin=1,tmax=10,M=20):
        self.storeinput(inspect.currentframe())
        kaq,Haq,c,Saq,Sll = param_maq(kaq,z,c,Saq,Sll,topboundary,phreatictop)
        TimModel.__init__(self,kaq,Haq,c,Saq,Sll,topboundary,tmin,tmax,M)
        self.name = 'ModelMaq'
        
def param_3d(kaq=[1],z=[1,0],Saq=[0.001],kzoverkh=1.0,phreatictop=False,semi=False):
    # Computes the parameters for a TimModel from input for a 3D model
    kaq = np.atleast_1d(kaq).astype('d')
    z = np.atleast_1d(z).astype('d')
    Naq = len(z) - 1
    if len(kaq) == 1: kaq = kaq * np.ones(Naq)
    Saq = np.atleast_1d(Saq).astype('d')
    if len(Saq) == 1: Saq = Saq * np.ones(Naq)
    kzoverkh = np.atleast_1d(kzoverkh).astype('d')
    if len(kzoverkh) == 1: kzoverkh = kzoverkh * np.ones(Naq)
    H = z[:-1] - z[1:]
    c = 0.5 * H[:-1] / ( kzoverkh[:-1] * kaq[:-1] ) + 0.5 * H[1:] / ( kzoverkh[1:] * kaq[1:] )
    Saq = Saq * H
    if phreatictop: Saq[0] = Saq[0] / H[0]
    c = np.hstack((1e100,c))
    if semi: c[0] = 0.5 * H[0] / (kzoverkh[0] * kaq[0])
    Sll = 1e-20 * np.ones(len(c))
    return kaq,H,c,Saq,Sll

class Model3D(TimModel):
    def __init__(self,kaq=1,z=[4,3,2,1],Saq=0.001,kzoverkh=0.1,phreatictop=True,semi=False,tmin=1,tmax=10,M=20):
        '''z must have the length of the number of layers + 1'''
        self.storeinput(inspect.currentframe())
        kaq,H,c,Saq,Sll = param_3d(kaq,z,Saq,kzoverkh,phreatictop,semi)
        if semi is False:
            TimModel.__init__(self,kaq,H,c,Saq,Sll,'imp',tmin,tmax,M)
        else:
            TimModel.__init__(self,kaq,H,c,Saq,Sll,'semi',tmin,tmax,M)
        self.name = 'Model3D'
        
class ModelXsec(TimModel):
    ''' Dummy aquifer data as there is no background aquifer. Naqlayers needs to be specified correctly'''
    def __init__(self, Naqlayers = 1, tmin = 1, tmax = 10, M = 20):
        self.storeinput(inspect.currentframe())
        a = np.ones(Naqlayers)
        TimModel.__init__(self, kaq = a, Haq = a, c = a[:-1], Saq = a, Sll = a[:-1], topboundary = 'imp', tmin = tmin, tmax = tmax, M = M)
        self.name = 'ModelXsec'
    
class AquiferData:
    def __init__(self,model,kaq,Haq,c,Saq,Sll,topboundary):
        self.model = model
        self.kaq = np.atleast_1d(kaq).astype('d')
        self.Naq = len(kaq)
        self.Haq = np.atleast_1d(Haq).astype('d')
        self.T = self.kaq * self.Haq
        self.Tcol = self.T.reshape(self.Naq,1)
        self.c = np.atleast_1d(c).astype('d')
        self.c[self.c > 1e100] = 1e100 
        self.Saq = np.atleast_1d(Saq).astype('d')
        self.Sll = np.atleast_1d(Sll).astype('d')
        self.Sll[self.Sll < 1e-20] = 1e-20 # Cannot be zero
        self.topboundary = topboundary[:3]
        self.D = self.T / self.Saq
        self.area = 1e200 # Smaller than default of ml.aq so that inhom is found
    def __repr__(self):
        return 'Inhom T: ' + str(self.T)
    def initialize(self):
        '''
        eigval[Naq,npval]: Array with eigenvalues
        lab[Naq,npval]: Array with lambda values
        lab2[Naq,Nin,npint]: Array with lambda values reorganized per interval
        eigvec[Naq,Naq,npval]: Array with eigenvector matrices
        coef[Naq,Naq,npval]: Array with coefficients;
        coef[ipylayers,:,np] are the coefficients if the element is in ipylayers belonging to Laplace parameter number np
        '''
        # Recompute T for when kaq is changed manually
        self.T = self.kaq * self.Haq
        self.Tcol = self.T.reshape(self.Naq,1)
        self.D = self.T / self.Saq
        #
        self.eigval = np.zeros((self.Naq,self.model.Np),'D')
        self.lab = np.zeros((self.Naq,self.model.Np),'D')
        self.eigvec = np.zeros((self.Naq,self.Naq,self.model.Np),'D')
        self.coef = np.zeros((self.Naq,self.Naq,self.model.Np),'D')
        b = np.diag(np.ones(self.Naq))
        for i in range(self.model.Np):
            w,v = self.compute_lab_eigvec(self.model.p[i]) # Eigenvectors are columns of v
            ## moved to compute_lab_eigvec routine
            #index = np.argsort( abs(w) )[::-1]
            #w = w[index]; v = v[:,index]
            self.eigval[:,i] = w; self.eigvec[:,:,i] = v
            self.coef[:,:,i] = np.linalg.solve( v, b ).T
        self.lab = 1.0 / np.sqrt(self.eigval)
        self.lab2 = self.lab.copy(); self.lab2.shape = (self.Naq,self.model.Nin,self.model.Npin)
        self.lababs = np.abs(self.lab2[:,:,0]) # used to check distances
    def compute_lab_eigvec(self, p, returnA = False, B = None):
        sqrtpSc = np.sqrt( p * self.Sll * self.c )
        a, b = np.zeros_like(sqrtpSc), np.zeros_like(sqrtpSc)
        small = np.abs(sqrtpSc) < 200
        a[small] = sqrtpSc[small] / np.tanh(sqrtpSc[small])
        b[small] = sqrtpSc[small] / np.sinh(sqrtpSc[small])
        a[~small] = sqrtpSc[~small] / ( (1.0 - np.exp(-2.0*sqrtpSc[~small])) / (1.0 + np.exp(-2.0*sqrtpSc[~small])) )
        b[~small] = sqrtpSc[~small] * 2.0 * np.exp(-sqrtpSc[~small]) / (1.0 - np.exp(-2.0*sqrtpSc[~small]))
        if (self.topboundary[:3] == 'sem') or (self.topboundary[:3] == 'lea'):
            if abs(sqrtpSc[0]) < 200:
                dzero = sqrtpSc[0] * np.tanh( sqrtpSc[0] )
            else:
                dzero = sqrtpSc[0] * cmath_tanh( sqrtpSc[0] )  # Bug in complex tanh in numpy
        d0 = p / self.D
        if B is not None:
            d0 = d0 * B  # B is vector of load efficiency paramters
        d0[:-1] += a[1:] / (self.c[1:] * self.T[:-1])
        d0[1:]  += a[1:] / (self.c[1:] * self.T[1:])
        if self.topboundary[:3] == 'lea':
            d0[0] += dzero / ( self.c[0] * self.T[0] )
        elif self.topboundary[:3] == 'sem':
            d0[0] += a[0] / ( self.c[0] * self.T[0] )
            
        dm1 = -b[1:] / (self.c[1:] * self.T[:-1])
        dp1 = -b[1:] / (self.c[1:] * self.T[1:])
        A = np.diag(dm1,-1) + np.diag(d0,0) + np.diag(dp1,1)
        if returnA: return A
        w,v = np.linalg.eig(A)
        # sorting moved here
        index = np.argsort( abs(w) )[::-1]
        w = w[index]; v = v[:,index]
        return w,v
    def headToPotential(self,h,pylayers):
        return h * self.Tcol[pylayers]
    def potentialToHead(self,pot,pylayers):
        return pot / self.Tcol[pylayers]
    def isInside(self,x,y):
        print 'Must overload AquiferData.isInside method'
        return True
    def inWhichPyLayer(self, z):
        '''Returns -9999 if above top of system, +9999 if below bottom of system, negative for in leaky layer.
        leaky layer -n is on top of aquifer n'''
        if z > self.zt[0]:
            return -9999
        for i in range(self.Naquifers-1):
            if z >= self.zb[i]:
                return i
            if z > self.zt[i+1]:
                return -i-1
        if z >= self.zb[self.Naquifers-1]:
            return self.Naquifers - 1
        return +9999
    
class Aquifer(AquiferData):
    def __init__(self,model,kaq,Haq,c,Saq,Sll,topboundary):
        AquiferData.__init__(self,model,kaq,Haq,c,Saq,Sll,topboundary)
        self.inhomList = []
        self.area = 1e300 # Needed to find smallest inhomogeneity
    def __repr__(self):
        return 'Background Aquifer T: ' + str(self.T)
    def initialize(self):
        AquiferData.initialize(self)
        for inhom in self.inhomList:
            inhom.initialize()
    def findAquiferData(self,x,y):
        rv = self
        for aq in self.inhomList:
            if aq.isInside(x,y):
                if aq.area < rv.area:
                    rv = aq
        return rv
    
class XsecAquiferData(AquiferData):
    def __init__(self, model, x1=0, x2=0, kaq=[1], Haq=[1], c=[1], Saq=[.1], Sll=[.1], topboundary='imp'):
        AquiferData.__init__(self, model, kaq, Haq, c, Saq, Sll, topboundary)
        self.x1, self.x2 = float(x1), float(x2)
        self.model.addinhom(self)
    def isInside(self,x,y):
        rv = False
        if (x > self.x1) & (x <= self.x2): rv = True
        return rv
    
class XsecAquiferDataMaq(XsecAquiferData):
    def __init__(self, model, x1=0, x2=0, kaq=[1], z=[1,0], c=[], Saq=[0.001], Sll=[0], topboundary='imp', phreatictop=False):
        kaq, Haq, c, Saq, Sll = param_maq(kaq, z, c, Saq, Sll, topboundary, phreatictop)
        XsecAquiferData.__init__(self, model, x1, x2, kaq, Haq, c, Saq, Sll, topboundary)
        
class XsecAquiferData3D(XsecAquiferData):
    def __init__(self, model, x1=0, x2=0, kaq=[1,1,1], z=[4,3,2,1], Saq=[0.3,0.001,0.001], kzoverkh=[.1,.1,.1], phreatictop=True, semi=False):
        kaq,Haq,c,Saq,Sll = param_3d(kaq,z,Saq,kzoverkh,phreatictop,semi)
        top = 'imp'
        if semi: top = 'semi'
        XsecAquiferData.__init__(self, model, x1, x2, kaq, Haq, c, Saq, Sll, top)
    
class CircInhomData(AquiferData):
    def __init__(self,model,x0=0,y0=0,R=1,kaq=[1],Haq=[1],c=[1],Saq=[.1],Sll=[.1],topboundary='imp'):
        AquiferData.__init__(self,model,kaq,Haq,c,Saq,Sll,topboundary)
        self.x0, self.y0, self.R = float(x0), float(y0), float(R)
        self.Rsq = self.R**2
        self.area = np.pi * self.Rsq
        self.model.addinhom(self)
    def isInside(self,x,y):
        rv = False
        if (x-self.x0)**2 + (y-self.y0)**2 < self.Rsq: rv = True
        return rv

class CircInhomDataMaq(CircInhomData):
    def __init__(self,model,x0=0,y0=0,R=1,kaq=[1],z=[1,0],c=[],Saq=[0.001],Sll=[0],topboundary='imp',phreatictop=False):
        kaq,Haq,c,Saq,Sll = param_maq(kaq,z,c,Saq,Sll,topboundary,phreatictop)
        CircInhomData.__init__(self,model,x0,y0,R,kaq,Haq,c,Saq,Sll,topboundary)

    
class CircInhomData3D(CircInhomData):
    def __init__(self,model,x0=0,y0=0,R=1,kaq=[1,1,1],z=[4,3,2,1],Saq=[0.3,0.001,0.001],kzoverkh=[.1,.1,.1],phreatictop=True):
        kaq,Haq,c,Saq,Sll = param_3d(kaq,z,Saq,kzoverkh,phreatictop)
        CircInhomData.__init__(self,model,x0,y0,R,kaq,Haq,c,Saq,Sll,'imp')
    
class EllipseInhomDataMaq(AquiferData):
    def __init__(self,model,x0=0,y0=0,along=2.0,bshort=1.0,angle=0.0,kaq=[1],z=[1,0],c=[],Saq=[0.001],Sll=[0],topboundary='imp',phreatictop=False):
        kaq,Haq,c,Saq,Sll = param_maq(kaq,z,c,Saq,Sll,topboundary,phreatictop)
        AquiferData.__init__(self,model,kaq,Haq,c,Saq,Sll,topboundary)
        self.x0, self.y0, self.along, self.bshort, self.angle = float(x0), float(y0), float(along), float(bshort), float(angle)
        assert self.along > self.bshort, "TTim Input error: Long axis of ellipse must be larger than short axis"
        self.cosal = np.cos(self.angle); self.sinal = np.sin(self.angle)
        self.dfoc = 2.0 * np.sqrt( self.along**2 - self.bshort**2 )  # focal length
        self.afoc = self.dfoc / 2.0  # half the focal length
        self.etastar = np.arccosh( self.along / self.afoc )
        self.z0 = self.x0 + self.y0*1j
        self.area = 1.0 # Needs to be implemented; Used in finding an inhomogeneity
        self.model.addinhom(self)
    def initialize(self):
        AquiferData.initialize(self)
        # q = -L^2 / (4^2 * lab^2) where L is focal length
        self.q =  self.dfoc**2 / (16.0 * self.lab**2)
    def isInside(self,x,y):
        eta,psi = self.xytoetapsi(x,y)
        return eta < self.etastar
    def xytoetapsi(self,x,y):
        Z = (x+y*1j - self.z0) * np.exp( -1j * self.angle )
        tau = np.log( Z / self.afoc + np.sqrt( Z/self.afoc - 1 ) * np.sqrt( Z/self.afoc + 1 ) )
        return tau.real,tau.imag
    def etapsitoxy(self,eta,psi):
        xloc = self.afoc * np.cosh(eta) * np.cos(psi)
        yloc = self.afoc * np.sinh(eta) * np.sin(psi)
        x = xloc * self.cosal - yloc * self.sinal + self.x0
        y = xloc * self.sinal + yloc * self.cosal + self.y0
        return x,y
    def outwardnormal(self,x,y):
        eta,psi = self.xytoetapsi(x,y)
        alpha = np.arctan2( np.cosh(eta)*np.sin(psi), np.sinh(eta)*np.cos(psi) ) + self.angle
        return np.cos(alpha), np.sin(alpha)
    def outwardnormalangle(self,x,y):
        eta,psi = self.xytoetapsi(x,y)
        alpha = np.arctan2( np.cosh(eta)*np.sin(psi), np.sinh(eta)*np.cos(psi) ) + self.angle
        return alpha
  
class Element:
    def __init__(self, model, Nparam=1, Nunknowns=0, layers=0, tsandbc=[(0.0,0.0)], type='z', name='', label=None):
        '''Types of elements
        'g': strength is given through time
        'v': boundary condition is variable through time
        'z': boundary condition is zero through time
        Definition of nlayers, Ncp, Npar, nunknowns:
        nlayers: Number of layers that the element is screened in, set in Element
        Ncp: Number of control points along the element
        nparam: Number of parameters, commonly nlayers * Ncp
        nunknowns: Number of unknown parameters, commonly zero or Npar
        '''
        self.model = model
        self.aq = None # Set in the initialization function
        self.Nparam = Nparam  # Number of parameters
        self.Nunknowns = Nunknowns
        self.layers = np.atleast_1d(layers)
        self.pylayers = self.layers  # corrected for base zero
        self.Nlayers = len(self.layers)
        #
        tsandbc = np.atleast_2d(tsandbc).astype('d')
        assert tsandbc.shape[1] == 2, "TTim input error: tsandQ or tsandh need to be 2D lists or arrays like [(0,1),(2,5),(8,0)] "
        self.tstart,self.bcin = tsandbc[:,0],tsandbc[:,1]
        if self.tstart[0] > 0:
            self.tstart = np.hstack((np.zeros(1),self.tstart))
            self.bcin = np.hstack((np.zeros(1),self.bcin))
        #
        self.type = type  # 'z' boundary condition through time or 'v' boundary condition through time
        self.name = name
        self.label = label
        if self.label is not None: assert self.label not in self.model.elementDict.keys(), "TTim error: label "+self.label+" already exists"
        self.Rzero = 30.0
    def setbc(self):
        if len(self.tstart) > 1:
            self.bc = np.zeros_like(self.bcin)
            self.bc[0] = self.bcin[0]
            self.bc[1:] = self.bcin[1:] - self.bcin[:-1]
        else:
            self.bc = self.bcin.copy()
        self.Ntstart = len(self.tstart)
    def initialize(self):
        '''Initialization of terms that cannot be initialized before other elements or the aquifer is defined.
        As we don't want to require a certain order of entering elements, these terms are initialized when Model.solve is called 
        The initialization class needs to be overloaded by all derived classes'''
        pass
    def potinf(self,x,y,aq=None):
        '''Returns complex array of size (nparam,Naq,npval)'''
        raise 'Must overload Element.potinf()'
    def potential(self,x,y,aq=None):
        '''Returns complex array of size (ngvbc,Naq,npval)'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        return np.sum( self.parameters[:,:,np.newaxis,:] * self.potinf(x,y,aq), 1 )
    def unitpotential(self,x,y,aq=None):
        '''Returns complex array of size (Naq,npval)
        Can be more efficient for given elements'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        return np.sum( self.potinf(x,y,aq), 0 )
    def disinf(self,x,y,aq=None):
        '''Returns 2 complex arrays of size (nparam,Naq,npval)'''
        raise 'Must overload Element.disinf()'
    def discharge(self,x,y,aq=None):
        '''Returns 2 complex arrays of size (ngvbc,Naq,npval)'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        qx,qy = self.disinf(x,y,aq)
        return np.sum( self.parameters[:,:,np.newaxis,:] * qx, 1 ), np.sum( self.parameters[:,:,np.newaxis,:] * qy, 1 )
    def unitdischarge(self,x,y,aq=None):
        '''Returns 2 complex arrays of size (Naq,npval)
        Can be more efficient for given elements'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        qx,qy = self.disinf(x,y,aq)
        return np.sum( qx, 0 ), np.sum( qy, 0 )
    # Functions used to build equations
    def potinflayers(self,x,y,pylayers=0,aq=None):
        '''layers can be scalar, list, or array. returns array of size (len(layers),nparam,npval)
        only used in building equations'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        pot = self.potinf(x,y,aq)
        rv = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec, 2 )
        rv = rv.swapaxes(0,1) # As the first axes needs to be the number of layers
        return rv[pylayers,:]
    def potentiallayers(self,x,y,pylayers=0,aq=None):
        '''Returns complex array of size (ngvbc,len(layers),npval)
        only used in building equations'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        pot = self.potential(x,y,aq)
        phi = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec, 2 )
        return phi[:,pylayers,:]
    def unitpotentiallayers(self,x,y,pylayers=0,aq=None):
        '''Returns complex array of size (len(layers),npval)
        only used in building equations'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        pot = self.unitpotential(x,y,aq)
        phi = np.sum( pot[np.newaxis,:,:] * aq.eigvec, 1 )
        return phi[pylayers,:]
    def disinflayers(self,x,y,pylayers=0,aq=None):
        '''layers can be scalar, list, or array. returns 2 arrays of size (len(layers),nparam,npval)
        only used in building equations'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        qx,qy = self.disinf(x,y,aq)
        rvx = np.sum( qx[:,np.newaxis,:,:] * aq.eigvec, 2 ); rvy = np.sum( qy[:,np.newaxis,:,:] * aq.eigvec, 2 )
        rvx = rvx.swapaxes(0,1); rvy = rvy.swapaxes(0,1) # As the first axes needs to be the number of layers
        return rvx[pylayers,:], rvy[pylayers,:]
    def dischargelayers(self,x,y,pylayers=0,aq=None):
        '''Returns 2 complex array of size (ngvbc,len(layers),npval)
        only used in building equations'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        qx,qy = self.discharge(x,y,aq)
        rvx = np.sum( qx[:,np.newaxis,:,:] * aq.eigvec, 2 ); rvy = np.sum( qy[:,np.newaxis,:,:] * aq.eigvec, 2 )
        return rvx[:,pylayers,:], rvy[:,pylayers,:]
    def unitdischargelayers(self,x,y,pylayers=0,aq=None):
        '''Returns complex array of size (len(layers),npval)
        only used in building equations'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        qx,qy = self.unitdischarge(x,y,aq)
        rvx = np.sum( qx[np.newaxis,:,:] * aq.eigvec, 1 ); rvy = np.sum( qy[np.newaxis,:,:] * aq.eigvec, 1 )
        return rvx[pylayers,:], rvy[pylayers,:]
    # Other functions
    def strength(self,t,derivative=0):
        '''returns array of strengths (nlayers,len(t)) t must be ordered and tmin <= t <= tmax'''
        # Could potentially be more efficient if s is pre-computed for all elements, but I don't know if that is worthwhile to store as it is quick now
        time = np.atleast_1d(t).copy()
        if (time[0] < self.model.tmin) or (time[-1] > self.model.tmax): print 'Warning, some of the times are smaller than tmin or larger than tmax; zeros are substituted'
        rv = np.zeros((self.Nlayers,np.size(time)))
        if self.type == 'g':
            s = self.strengthinflayers * self.model.p ** derivative
            for itime in range(self.Ntstart):
                time -=  self.tstart[itime]
                for i in range(self.Nlayers):
                    rv[i] += self.bc[itime] * self.model.inverseLapTran(s[i],time)
        else:
            s = np.sum( self.parameters[:,:,np.newaxis,:] * self.strengthinf, 1 )
            s = np.sum( s[:,np.newaxis,:,:] * self.aq.eigvec, 2 )
            s = s[:,self.pylayers,:] * self.model.p ** derivative
            for k in range(self.model.Ngvbc):
                e = self.model.gvbcList[k]
                for itime in range(e.Ntstart):
                    t = time - e.tstart[itime]
                    #print 'e,time ',e,t
                    if t[-1] >= self.model.tmin:  # Otherwise all zero
                        for i in range(self.Nlayers):
                            rv[i] += e.bc[itime] * self.model.inverseLapTran(s[k,i],t)
        return rv
        
    #def potential(self,x,y,t,layers=None,aq=None,derivative=0,returnphi=0):
    #    '''Returns pot[Naq,Ntimes] if layers=None, otherwise pot[len(layers,Ntimes)]
    #    t must be ordered '''
    #    if aq is None: aq = self.aq.findAquiferData(x,y)
    #    if layers is None: layers = range(aq.Naq)
    #    nlayers = len(layers)
    #    time = np.atleast_1d(t).copy()
    #    pot = np.zeros((self.ngvbc, aq.Naq, self.npval),'D')
    #    for i in range(self.ngbc):
    #        pot[i,:] += self.gbclist[i].unitpotential(x,y,aq)
    #    for e in self.vzbcList:
    #        pot += e.potential(x,y,aq)
    #    if layers is None:
    #        pot = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec, 2 )
    #    else:
    #        pot = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec[layers,:], 2 )
    #    if derivative > 0: pot *= self.p**derivative
    #    if returnphi: return pot
    #    rv = np.zeros((nlayers,len(time)))
    #    if (time[0] < self.tmin) or (time[-1] > self.tmax): print 'Warning, some of the times are smaller than tmin or larger than tmax; zeros are substituted'
    #    #
    #    for k in range(self.ngvbc):
    #        e = self.gvbcList[k]
    #        for itime in range(e.Ntstart):
    #            t = time - e.tstart[itime]
    #            it = 0
    #            if t[-1] >= self.tmin:  # Otherwise all zero
    #                if (t[0] < self.tmin): it = np.argmax( t >= self.tmin )  # clever call that should be replaced with find_first function when included in numpy
    #                for n in range(self.Nin):
    #                    tp = t[ (t >= self.tintervals[n]) & (t < self.tintervals[n+1]) ]
    #                    ## I think these lines are not needed anymore as I modified tintervals[0] and tintervals[-1] by eps
    #                    #if n == self.Nin-1:
    #                    #    tp = t[ (t >= self.tintervals[n]) & (t <= self.tintervals[n+1]) ]
    #                    #else:
    #                    #    tp = t[ (t >= self.tintervals[n]) & (t < self.tintervals[n+1]) ]
    #                    Nt = len(tp)
    #                    if Nt > 0:  # if all values zero, don't do the inverse transform
    #                        for i in range(nlayers):
    #                            # I used to check the first value only, but it seems that checking that nothing is zero is needed and should be sufficient
    #                            #if np.abs( pot[k,i,n*self.npint] ) > 1e-20:  # First value very small
    #                            if not np.any( pot[k,i,n*self.npint:(n+1)*self.npint] == 0.0) : # If there is a zero item, zero should be returned; funky enough this can be done with a straight equal comparison
    #                                rv[i,it:it+Nt] += e.bc[itime] * invlaptrans.invlap( tp, self.tintervals[n], self.tintervals[n+1], pot[k,i,n*self.npint:(n+1)*self.npint], self.gamma[n], self.M, Nt )
    #                        it = it + Nt
    #    return rv        
        
        
        
    def headinside(self,t):
        print "This function not implemented for this element"
        return
    def layout(self):
        return '','',''
    def storeinput(self,frame):
        self.inputargs, _, _, self.inputvalues = inspect.getargvalues(frame)
    def write(self):
        rv = self.name + '(' + self.model.modelname + ',\n'
        for key in self.inputargs[2:]:  # The first two are ignored
            if isinstance(self.inputvalues[key],np.ndarray):
                rv += key + ' = ' + np.array2string(self.inputvalues[key],separator=',') + ',\n'
            elif isinstance(self.inputvalues[key],str):                
                rv += key + " = '" + self.inputvalues[key] + "',\n"
            else:
                rv += key + ' = ' + str(self.inputvalues[key]) + ',\n'
        rv += ')\n'
        return rv
    def run_after_solve(self):
        '''function to run after a solution is completed.
        for most elements nothing needs to be done,
        but for strings of elements some arrays may need to be filled'''
        pass
    
class HeadEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for head-specified conditions. (really written as constant potential element)
        Works for nunknowns = 1
        Returns matrix part nunknowns,neq,npval, complex
        Returns rhs part nunknowns,nvbc,npval, complex
        Phi_out - c*T*q_s = Phi_in
        Well: q_s = Q / (2*pi*r_w*H)
        LineSink: q_s = sigma / H = Q / (L*H)
        '''
        mat = np.empty( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        for icp in range(self.Ncp):
            istart = icp*self.Nlayers
            ieq = 0  
            for e in self.model.elementList:
                if e.Nunknowns > 0:
                    mat[istart:istart+self.Nlayers,ieq:ieq+e.Nunknowns,:] = e.potinflayers(self.xc[icp],self.yc[icp],self.pylayers)
                    if e == self:
                        for i in range(self.Nlayers): mat[istart+i,ieq+istart+i,:] -= self.resfacp[istart+i] * e.strengthinflayers[istart+i]
                    ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                rhs[istart:istart+self.Nlayers,i,:] -= self.model.gbcList[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.pylayers)  # Pretty cool that this works, really
            if self.type == 'v':
                iself = self.model.vbcList.index(self)
                for i in range(self.Nlayers):
                    rhs[istart+i,self.model.Ngbc+iself,:] = self.pc[istart+i] / self.model.p
        return mat, rhs

class HeadEquationNores:
    def equation(self):
        '''Mix-in class that returns matrix rows for head-specified conditions. (really written as constant potential element)
        Returns matrix part nunknowns,neq,npval, complex
        Returns rhs part nunknowns,nvbc,npval, complex
        '''
        mat = np.empty( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        for icp in range(self.Ncp):
            istart = icp*self.Nlayers
            ieq = 0  
            for e in self.model.elementList:
                if e.Nunknowns > 0:
                    mat[istart:istart+self.Nlayers,ieq:ieq+e.Nunknowns,:] = e.potinflayers(self.xc[icp],self.yc[icp],self.pylayers)
                    ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                rhs[istart:istart+self.Nlayers,i,:] -= self.model.gbcList[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.pylayers)  # Pretty cool that this works, really
            if self.type == 'v':
                iself = self.model.vbcList.index(self)
                for i in range(self.Nlayers):
                    rhs[istart+i,self.model.Ngbc+iself,:] = self.pc[istart+i] / self.model.p
        return mat, rhs
    
class LeakyWallEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for leaky-wall condition
        Returns matrix part nunknowns,neq,npval, complex
        Returns rhs part nunknowns,nvbc,npval, complex
        '''
        mat = np.empty( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        for icp in range(self.Ncp):
            istart = icp*self.Nlayers
            ieq = 0  
            for e in self.model.elementList:
                if e.Nunknowns > 0:
                    qx,qy = e.disinflayers(self.xc[icp],self.yc[icp],self.pylayers)
                    mat[istart:istart+self.Nlayers,ieq:ieq+e.Nunknowns,:] = qx * self.cosout[icp] + qy * self.sinout[icp]
                    if e == self:
                        hmin = e.potinflayers(self.xcneg[icp],self.ycneg[icp],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis,np.newaxis]
                        hplus = e.potinflayers(self.xc[icp],self.yc[icp],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis,np.newaxis]
                        mat[istart:istart+self.Nlayers,ieq:ieq+e.Nunknowns,:] -= self.resfac[:,np.newaxis,np.newaxis] * (hplus-hmin)
                    ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                qx,qy = self.model.gbcList[i].unitdischargelayers(self.xc[icp],self.yc[icp],self.pylayers)
                rhs[istart:istart+self.Nlayers,i,:] -=  qx * self.cosout[icp] + qy * self.sinout[icp]
            #if self.type == 'v':
            #    iself = self.model.vbclist.index(self)
            #    for i in range(self.nlayers):
            #        rhs[istart+i,self.model.ngbc+iself,:] = self.pc[istart+i] / self.model.p
        return mat, rhs
    
class NoflowEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for no-flow condition
        Returns matrix part nunknowns,neq,npval, complex
        Returns rhs part nunknowns,nvbc,npval, complex
        '''
        mat = np.empty( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        for icp in range(self.Ncp):
            istart = icp*self.Nlayers
            ieq = 0  
            for e in self.model.elementList:
                if e.Nunknowns > 0:
                    qx,qy = e.disinflayers(self.xc[icp],self.yc[icp],self.pylayers)
                    mat[istart:istart+self.Nlayers,ieq:ieq+e.Nunknowns,:] = qx * self.cosout + qy * self.sinout
                    ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                qx,qy = self.model.gbcList[i].unitdischargelayers(self.xc[icp],self.yc[icp],self.pylayers)
                rhs[istart:istart+self.Nlayers,i,:] -=  qx * self.cosout + qy * self.sinout
            #if self.type == 'v':
            #    iself = self.model.vbclist.index(self)
            #    for i in range(self.nlayers):
            #        rhs[istart+i,self.model.ngbc+iself,:] = self.pc[istart+i] / self.model.p
        return mat, rhs
    
class HeadEquationNew:
    '''Variable Head BC'''
    def equation(self):
        '''Mix-in class that returns matrix rows for head-specified conditions. (really written as constant potential element)
        Works for nunknowns = 1
        Returns matrix part nunknowns,neq,npval, complex
        Returns rhs part nunknowns,nvbc,npval, complex
        Phi_out - c*T*q_s = Phi_in
        Well: q_s = Q / (2*pi*r_w*H)
        LineSink: q_s = sigma / H = Q / (L*H)
        '''
        mat = np.empty( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        for icp in range(self.Ncp):
            istart = icp*self.Nlayers
            ieq = 0  
            for e in self.model.elementList:
                if e.Nunknowns > 0:
                    mat[istart:istart+self.Nlayers,ieq:ieq+e.Nunknowns,:] = e.potinflayers(self.xc[icp],self.yc[icp],self.pylayers)
                    if e == self:
                        for i in range(self.Nlayers): mat[istart+i,ieq+istart+i,:] -= self.resfacp[istart+i] * e.strengthinflayers[istart+i]
                    ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                rhs[istart:istart+self.Nlayers,i,:] -= self.model.gbcList[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.pylayers)  # Pretty cool that this works, really
            if self.type == 'v':
                iself = self.model.vbcList.index(self)
                for i in range(self.Nlayers):
                    rhs[istart+i,self.model.Ngbc+iself,:] = self.pc[istart+i,:]
        return mat, rhs
    
class WellBoreStorageEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for multi-aquifer element with
        total given discharge, uniform but unknown head and InternalStorageEquation
        '''
        mat = np.zeros( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' ) # Important to set to zero for some of the equations
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        ieq = 0
        for e in self.model.elementList:
            if e.Nunknowns > 0:
                head = e.potinflayers(self.xc,self.yc,self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis,np.newaxis]
                mat[:-1,ieq:ieq+e.Nunknowns,:] = head[:-1,:] - head[1:,:]
                mat[-1,ieq:ieq+e.Nunknowns,:] -= np.pi * self.rc**2 * self.model.p * head[0,:]
                if e == self:
                    disterm = self.strengthinflayers * self.res / ( 2 * np.pi * self.rw * self.aq.Haq[self.pylayers][:,np.newaxis] )
                    if self.Nunknowns > 1:  # Multiple layers
                        for i in range(self.Nunknowns-1):
                            mat[i,ieq+i,:] -= disterm[i]
                            mat[i,ieq+i+1,:] += disterm[i+1]
                    mat[-1,ieq:ieq+self.Nunknowns,:] += self.strengthinflayers
                    mat[-1,ieq,:] += np.pi * self.rc**2 * self.model.p * disterm[0]
                ieq += e.Nunknowns
        for i in range(self.model.Ngbc):
            head = self.model.gbcList[i].unitpotentiallayers(self.xc,self.yc,self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis]
            rhs[:-1,i,:] -= head[:-1,:] - head[1:,:]
            rhs[-1,i,:] += np.pi * self.rc**2 * self.model.p * head[0,:]
        if self.type == 'v':
            iself = self.model.vbcList.index(self)
            rhs[-1,self.model.Ngbc+iself,:] += self.flowcoef
            if self.hdiff is not None:
                rhs[:-1,self.model.Ngbc+iself,:] += self.hdiff[:,np.newaxis] / self.model.p  # head[0] - head[1] = hdiff
        return mat, rhs

class MscreenEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for multi-screen conditions where total discharge is specified.
        Works for nunknowns = 1
        Returns matrix part nunknowns,neq,npval, complex
        Returns rhs part nunknowns,nvbc,npval, complex
        head_out - c*q_s = h_in
        Set h_i - h_(i+1) = 0 and Sum Q_i = Q'''
        mat = np.zeros( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )  # Needs to be zero for last equation, but I think setting the whole array is quicker
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        ieq = 0
        for icp in range(self.Ncp):
            istart = icp*self.Nlayers
            ieq = 0 
            for e in self.model.elementList:
                if e.Nunknowns > 0:
                    head = e.potinflayers(self.xc[icp],self.yc[icp],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis,np.newaxis]  # T[self.layers,np.newaxis,np.newaxis] is not allowed
                    mat[istart:istart+self.Nlayers-1,ieq:ieq+e.Nunknowns,:] = head[:-1,:] - head[1:,:]
                    if e == self:
                        for i in range(self.Nlayers-1):
                            mat[istart+i,ieq+istart+i,:] -= self.resfach[istart+i] * e.strengthinflayers[istart+i]
                            mat[istart+i,ieq+istart+i+1,:] += self.resfach[istart+i+1] * e.strengthinflayers[istart+i+1]
                            mat[istart+i,ieq+istart:ieq+istart+i+1,:] -= self.vresfac[istart+i] * e.strengthinflayers[istart+i]
                        mat[istart+self.Nlayers-1,ieq+istart:ieq+istart+self.Nlayers,:] = 1.0
                    ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                head = self.model.gbcList[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis]
                rhs[istart:istart+self.Nlayers-1,i,:] -= head[:-1,:] - head[1:,:]
            if self.type == 'v':
                iself = self.model.vbcList.index(self)
                rhs[istart+self.Nlayers-1,self.model.Ngbc+iself,:] = 1.0  # If self.type == 'z', it should sum to zero, which is the default value of rhs
        return mat, rhs
    
class MscreenDitchEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for multi-scren conditions where total discharge is specified.
        Returns matrix part nunknowns,neq,npval, complex
        Returns rhs part nunknowns,nvbc,npval, complex
        head_out - c*q_s = h_in
        Set h_i - h_(i+1) = 0 and Sum Q_i = Q
        I would say
        headin_i - headin_(i+1) = 0
        headout_i - c*qs_i - headout_(i+1) + c*qs_(i+1) = 0 
        In case of storage:
        Sum Q_i - A * p^2 * headin = Q
        '''
        mat = np.zeros( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )  # Needs to be zero for last equation, but I think setting the whole array is quicker
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        ieq = 0
        for icp in range(self.Ncp):
            istart = icp*self.Nlayers
            ieq = 0 
            for e in self.model.elementList:
                if e.Nunknowns > 0:
                    head = e.potinflayers(self.xc[icp],self.yc[icp],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis,np.newaxis]  # T[self.layers,np.newaxis,np.newaxis] is not allowed
                    if self.Nlayers > 1: mat[istart:istart+self.Nlayers-1,ieq:ieq+e.Nunknowns,:] = head[:-1,:] - head[1:,:]
                    mat[istart+self.Nlayers-1,ieq:ieq+e.Nunknowns,:] = head[0,:] # Store head in top layer in 2nd to last equation of this control point
                    if e == self:
                        # Correct head in top layer in second to last equation to make it head inside
                        mat[istart+self.Nlayers-1,ieq+istart,:] -= self.resfach[istart] * e.strengthinflayers[istart]
                        if icp == 0:
                            istartself = ieq  # Needed to build last equation
                        for i in range(self.Nlayers-1):
                            mat[istart+i,ieq+istart+i,:] -= self.resfach[istart+i] * e.strengthinflayers[istart+i]
                            mat[istart+i,ieq+istart+i+1,:] += self.resfach[istart+i+1] * e.strengthinflayers[istart+i+1]
                            #vresfac not yet used here; it is set to zero ad I don't quite now what is means yet
                            #mat[istart+i,ieq+istart:ieq+istart+i+1,:] -= self.vresfac[istart+i] * e.strengthinflayers[istart+i]
                    ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                head = self.model.gbcList[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis]
                if self.Nlayers > 1: rhs[istart:istart+self.Nlayers-1,i,:] -= head[:-1,:] - head[1:,:]
                rhs[istart+self.Nlayers-1,i,:] -= head[0,:] # Store minus the head in top layer in second to last equation for this control point
        # Modify last equations
        for icp in range(self.Ncp-1):
            ieq = (icp+1) * self.Nlayers - 1
            mat[ieq,:,:] -= mat[ieq+self.Nlayers,:,:]  # Head first layer control point icp - Head first layer control point icp + 1
            rhs[ieq,:,:] -= rhs[ieq+self.Nlayers,:,:]
        # Last equation setting the total discharge of the ditch
        # print 'istartself ',istartself
        mat[-1,:,:] = 0.0  
        mat[-1,istartself:istartself+self.Nparam,:] = 1.0
        if self.Astorage is not None:
            matlast = np.zeros( (self.model.Neq,  self.model.Np), 'D' )  # Used to store last equation in case of ditch storage
            rhslast = np.zeros( (self.model.Np), 'D' )  # Used to store last equation in case of ditch storage 
            ieq = 0
            for e in self.model.elementList:
                head = e.potinflayers(self.xc[0],self.yc[0],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis,np.newaxis]  # T[self.layers,np.newaxis,np.newaxis] is not allowed
                matlast[ieq:ieq+e.Nunknowns] -= self.Astorage * self.model.p**2 * head[0,:]
                if e == self:
                    # only need to correct first unknown 
                    matlast[ieq] += self.Astorage * self.model.p**2 * self.resfach[0] * e.strengthinflayers[0]
                ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                head = self.model.gbcList[i].unitpotentiallayers(self.xc[0],self.yc[0],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis]
                rhslast += self.Astorage * self.model.p**2 * head[0] 
            mat[-1] += matlast
        rhs[-1,:,:] = 0.0
        if self.type == 'v':
            iself = self.model.vbcList.index(self)
            rhs[-1,self.model.Ngbc+iself,:] = 1.0  # If self.type == 'z', it should sum to zero, which is the default value of rhs
            if self.Astorage is not None: rhs[-1,self.model.Ngbc+iself,:] += rhslast
        return mat, rhs
    
class InhomEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for inhomogeneity conditions'''
        mat = np.empty( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        for icp in range(self.Ncp):
            istart = icp*2*self.Nlayers
            ieq = 0  
            for e in self.model.elementList:
                if e.Nunknowns > 0:
                    mat[istart:istart+self.Nlayers,ieq:ieq+e.Nunknowns,:] = \
                    e.potinflayers(self.xc[icp],self.yc[icp],self.pylayers,self.aqin) / self.aqin.T[self.pylayers][:,np.newaxis,np.newaxis] - \
                    e.potinflayers(self.xc[icp],self.yc[icp],self.pylayers,self.aqout) / self.aqout.T[self.pylayers][:,np.newaxis,np.newaxis]
                    qxin,qyin = e.disinflayers(self.xc[icp],self.yc[icp],self.pylayers,self.aqin)
                    qxout,qyout = e.disinflayers(self.xc[icp],self.yc[icp],self.pylayers,self.aqout)
                    mat[istart+self.Nlayers:istart+2*self.Nlayers,ieq:ieq+e.Nunknowns,:] = \
                        (qxin-qxout) * np.cos(self.thetacp[icp]) + (qyin-qyout) * np.sin(self.thetacp[icp])
                    ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                rhs[istart:istart+self.Nlayers,i,:] -= \
                (self.model.gbcList[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.pylayers,self.aqin)  / self.aqin.T[self.pylayers][:,np.newaxis] - \
                 self.model.gbcList[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.pylayers,self.aqout) / self.aqout.T[self.pylayers][:,np.newaxis] )
                qxin,qyin = self.model.gbcList[i].unitdischargelayers(self.xc[icp],self.yc[icp],self.pylayers,self.aqin)
                qxout,qyout = self.model.gbcList[i].unitdischargelayers(self.xc[icp],self.yc[icp],self.pylayers,self.aqout)
                rhs[istart+self.Nlayers:istart+2*self.Nlayers,i,:] -= (qxin-qxout) * np.cos(self.thetacp[icp]) + (qyin-qyout) * np.sin(self.thetacp[icp])
        return mat, rhs
    
class BesselRatioApprox:
    # Never fully debugged
    def __init__(self,Norder,Nterms):
        self.Norder= Norder+1
        self.Nterms = Nterms+1
        self.krange = np.arange(self.Nterms)
        self.minonek = (-np.ones(self.Nterms)) ** self.krange
        self.hankeltot = np.ones( (self.Norder,2*self.Nterms), 'd' )
        self.muk = np.ones( (self.Norder,self.Nterms), 'd' )
        self.nuk = np.ones( (self.Norder,self.Nterms), 'd' )
        for n in range(self.Norder):
            mu = 4.0*n**2
            for k in range(1,self.Nterms):
                self.hankeltot[n,k] = self.hankeltot[n,k-1] * (mu - (2*k-1)**2) / ( 4.0 * k )
            for k in range(self.Nterms):
                self.muk[n,k] = ( 4.0 * n**2 + 16.0 * k**2 - 1.0 ) / ( 4.0 * n**2 - (4.0*k - 1.0)**2 )
                self.nuk[n,k] = ( 4.0 * n**2 + 4.0 * (2.0*k+1.0)**2 - 1.0 ) / ( 4.0 * n**2 - (4.0*k + 1.0)**2 )
        self.hankelnk = self.hankeltot[:,:self.Nterms]
        self.hankeln2k = self.hankeltot[:,::2]
        self.hankeln2kp1 = self.hankeltot[:,1::2]
    def ivratio( self, rho, R, lab):
        lab = np.atleast_1d(lab)
        rv = np.empty((self.Norder,len(lab)),'D')
        for k in range(len(lab)):
            top = np.sum( self.minonek * self.hankelnk / ( 2.0 * rho / lab[k] )**self.krange, 1 )
            bot = np.sum( self.minonek * self.hankelnk / ( 2.0 * R / lab[k] )**self.krange, 1 )
            rv[:,k] = top / bot * np.sqrt ( float(R) / rho ) * np.exp( (rho-R)/ lab[k] )
        return rv
    def kvratio( self, rho, R, lab ):
        lab = np.atleast_1d(lab)
        rv = np.empty((self.Norder,len(lab)),'D')
        for k in range(len(lab)):
            top = np.sum( self.hankelnk / ( 2.0 * rho / lab[k] )**self.krange, 1 )
            bot = np.sum( self.hankelnk / ( 2.0 * R / lab[k] )**self.krange, 1 )
            rv[:,k] = top / bot * np.sqrt ( float(R) / rho ) * np.exp( (R-rho)/ lab[k] )
        return rv    
    def ivratiop( self, rho, R, lab ):
        lab = np.atleast_1d(lab)
        rv = np.empty((self.Norder,len(lab)),'D')
        for k in range(len(lab)):
            top = np.sum( self.muk * self.hankeln2k / ( 2.0 * rho / lab[k] )**(2*self.krange), 1 ) - \
                  np.sum( self.nuk * self.hankeln2kp1 / ( 2.0 * rho / lab[k] )**(2*self.krange+1), 1 )
            bot = np.sum( self.minonek * self.hankelnk / ( 2.0 * R / lab[k] )**self.krange, 1 )
            rv[:,k] = top / bot * np.sqrt ( float(R) / rho ) * np.exp( (rho-R)/ lab[k] )
        return rv
    def kvratiop( self, rho, R, lab ):
        lab = np.atleast_1d(lab)
        rv = np.empty((self.Norder,len(lab)),'D')
        for k in range(len(lab)):
            top = np.sum( self.muk * self.hankeln2k / ( 2.0 * rho / lab[k] )**(2*self.krange), 1 ) + \
                  np.sum( self.nuk * self.hankeln2kp1 / ( 2.0 * rho / lab[k] )**(2*self.krange+1), 1 )
            bot = np.sum( self.hankelnk / ( 2.0 * R / lab[k] )**self.krange, 1 )
            rv[:,k] = -top / bot * np.sqrt ( float(R) / rho ) * np.exp( (R-rho)/ lab[k] )
        return rv
    
class CircInhomRadial(Element,InhomEquation):
    def __init__(self,model,x0=0,y0=0,R=1.0,label=None):
        Element.__init__(self, model, Nparam=2*model.aq.Naq, Nunknowns=2*model.aq.Naq, layers=range(1,model.aq.Naq+1), type='z', name='CircInhom', label=label)
        self.x0 = float(x0); self.y0 = float(y0); self.R = float(R)
        self.model.addelement(self)
        self.approx = BesselRatioApprox(0,2)
    def __repr__(self):
        return self.name + ' at ' + str((self.x0,self.y0))
    def initialize(self):
        self.xc = np.array([self.x0 + self.R]); self.yc = np.array([self.y0])
        self.thetacp = np.zeros(1)
        self.Ncp = 1
        self.aqin = self.model.aq.find_aquifer_data(self.x0 + (1.0 - 1e-8) * self.R, self.y0)
        assert self.aqin.R == self.R, 'TTim Input Error: Radius of CircInhom and CircInhomData must be equal'
        self.aqout = self.model.aq.find_aquifer_data(self.x0 + (1.0 + 1e-8) * self.R, self.y0)
        self.setbc()
        self.facin = np.ones_like(self.aqin.lab2)
        self.facout = np.ones_like(self.aqout.lab2)
        self.circ_in_small = np.ones((self.aqin.Naq,self.model.Nin),dtype='i') # To keep track which circles are small
        self.circ_out_small = np.ones((self.aqout.Naq,self.model.Nin),dtype='i')
        self.Rbig = 700
        #for i in range(self.aqin.Naq):
        #    for j in range(self.model.Nin):
        #        assert self.R / abs(self.aqin.lab2[i,j,0]) < self.Rbig, 'TTim input error, Radius too big'
        #        assert self.R / abs(self.aqout.lab2[i,j,0]) < self.Rbig, 'TTim input error, Radius too big'
        #        if self.R / abs(self.aqin.lab2[i,j,0]) < self.Rbig:
        #            self.circ_in_small[i,j] = 1
        #            self.facin[i,j,:] = 1.0 / iv(0, self.R / self.aqin.lab2[i,j,:])
        #        if self.R / abs(self.aqout.lab2[i,j,0]) < self.Rbig:
        #            self.circ_out_small[i,j] = 1
        #            self.facout[i,j,:] = 1.0 / kv(0, self.R / self.aqout.lab2[i,j,:])
        #for i in range(self.aqin.Naq):
        #    for j in range(self.model.Nin):
        #        assert self.R / abs(self.aqin.lab2[i,j,0]) < 900, 'radius too large compared to aqin lab2[i,j,0] '+str((i,j))
        #        assert self.R / abs(self.aqout.lab2[i,j,0]) < 900, 'radius too large compared to aqin lab2[i,j,0] '+str((i,j))
        #self.facin = 1.0 / iv(0, self.R / self.aqin.lab2)
        #self.facout = 1.0 / kv(0, self.R / self.aqout.lab2)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aqin:
            r = np.sqrt( (x-self.x0)**2 + (y-self.y0)**2 )
            for i in range(self.aqin.Naq):
                for j in range(self.model.Nin):
                    if abs(r-self.R) / abs(self.aqin.lab2[i,j,0]) < self.Rzero:
                        if self.circ_in_small[i,j]:
                            rv[i,i,j,:] = self.facin[i,j,:] * iv( 0, r / self.aqin.lab2[i,j,:] )
                        else:
                            print 'using approx'
                            rv[i,i,j,:] = self.approx.ivratio(r,self.R,self.aqin.lab2[i,j,:])
        if aq == self.aqout:
            r = np.sqrt( (x-self.x0)**2 + (y-self.y0)**2 )
            for i in range(self.aqout.Naq):
                for j in range(self.model.Nin):
                    if abs(r-self.R) / abs(self.aqout.lab2[i,j,0]) < self.Rzero:
                        if self.circ_out_small[i,j]:
                            rv[self.aqin.Naq+i,i,j,:] = self.facin[i,j,:] * kv( 0, r / self.aqout.lab2[i,j,:] )
                        else:
                            print 'using approx'
                            rv[self.aqin.Naq+i,i,j,:] = self.approx.kvratio(r,self.R,self.aqout.lab2[i,j,:])
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    def disinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        qx,qy = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D'), np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        if aq == self.aqin:
            qr = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
            r = np.sqrt( (x-self.x0)**2 + (y-self.y0)**2 )
            if r < 1e-20: r = 1e-20  # As we divide by that on the return
            for i in range(self.aqin.Naq):
                for j in range(self.model.Nin):
                    if abs(r-self.R) / abs(self.aqin.lab2[i,j,0]) < self.Rzero:
                        if self.circ_in_small[i,j]:
                            qr[i,i,j,:] = -self.facin[i,j,:] * iv( 1, r / self.aqin.lab2[i,j,:] ) / self.aqin.lab2[i,j,:]
                        else:
                            qr[i,i,j,:] = -self.approx.ivratiop(r,self.R,self.aqin.lab2[i,j,:]) / self.aqin.lab2[i,j,:]
            qr.shape = (self.Nparam,aq.Naq,self.model.Np)
            qx[:] = qr * (x-self.x0) / r; qy[:] = qr * (y-self.y0) / r
        if aq == self.aqout:
            qr = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
            r = np.sqrt( (x-self.x0)**2 + (y-self.y0)**2 )
            for i in range(self.aqout.Naq):
                for j in range(self.model.Nin):
                    if abs(r-self.R) / abs(self.aqout.lab2[i,j,0]) < self.Rzero:
                        if self.circ_out_small[i,j]:
                            qr[self.aqin.Naq+i,i,j,:] = self.facin[i,j,:] * kv( 1, r / self.aqout.lab2[i,j,:] ) / self.aqout.lab2[i,j,:]
                        else:
                            qr[self.aqin.Naq+i,i,j,:] = self.approx.kvratiop(r,self.R,self.aqout.lab2[i,j,:]) / self.aqout.lab2[i,j,:]
            qr.shape = (self.Nparam,aq.Naq,self.model.Np)
            qx[:] = qr * (x-self.x0) / r; qy[:] = qr * (y-self.y0) / r
        return qx,qy
    def layout(self):
        return 'line', self.x0 + self.R * np.cos(np.linspace(0,2*np.pi,100)), self.y0 + self.R * np.sin(np.linspace(0,2*np.pi,100))
                
class CircInhom(Element,InhomEquation):
    def __init__(self,model,x0=0,y0=0,R=1.0,order=0,label=None,test=False):
        Element.__init__(self, model, Nparam=2*model.aq.Naq*(2*order+1), Nunknowns=2*model.aq.Naq*(2*order+1), layers=range(model.aq.Naq), type='z', name='CircInhom', label=label)
        self.x0 = float(x0); self.y0 = float(y0); self.R = float(R)
        self.order = order
        self.approx = BesselRatioApprox(0,3)
        self.test=test
        self.model.addelement(self)
    def __repr__(self):
        return self.name + ' at ' + str((self.x0,self.y0))
    def initialize(self):
        self.Ncp = 2*self.order + 1
        self.thetacp = np.arange(0,2*np.pi,(2*np.pi)/self.Ncp)
        self.xc = self.x0 + self.R * np.cos( self.thetacp )
        self.yc = self.y0 + self.R * np.sin( self.thetacp )
        self.aqin = self.model.aq.find_aquifer_data(self.x0 + (1 - 1e-10) * self.R, self.y0)
        self.aqout = self.model.aq.find_aquifer_data(self.x0 + (1.0 + 1e-8) * self.R, self.y0)
        assert self.aqin.Naq == self.aqout.Naq, 'TTim input error: Number of layers needs to be the same inside and outside circular inhomogeneity'
        # Now that aqin is known, check that radii of circles are the same
        assert self.aqin.R == self.R, 'TTim Input Error: Radius of CircInhom and CircInhomData must be equal'
        self.setbc()
        self.facin = np.zeros((self.order+1,self.aqin.Naq,self.model.Nin,self.model.Npin),dtype='D')
        self.facout = np.zeros((self.order+1,self.aqin.Naq,self.model.Nin,self.model.Npin),dtype='D')
        self.circ_in_small = np.zeros((self.aqin.Naq,self.model.Nin),dtype='i') # To keep track which circles are small
        self.circ_out_small = np.zeros((self.aqout.Naq,self.model.Nin),dtype='i')
        self.besapprox = BesselRatioApprox(self.order,2) # Nterms = 2 is probably enough
        self.Rbig = 200
        for i in range(self.aqin.Naq):
            for j in range(self.model.Nin):
                # When the circle is too big, an assertion is thrown. In the future, the approximation of the ratio of bessel functions needs to be completed
                # For now, the logic is there, but not used
                if self.test:
                    print 'inside  relative radius: ',self.R / abs(self.aqin.lab2[i,j,0])
                    print 'outside relative radius: ',self.R / abs(self.aqout.lab2[i,j,0])
                #assert self.R / abs(self.aqin.lab2[i,j,0]) < self.Rbig, 'TTim input error, Radius too big'
                #assert self.R / abs(self.aqout.lab2[i,j,0]) < self.Rbig, 'TTim input error, Radius too big'
                if self.R / abs(self.aqin.lab2[i,j,0]) < self.Rbig:
                    self.circ_in_small[i,j] = 1
                    for n in range(self.order+1):
                        self.facin[n,i,j,:] = 1.0 / iv(n, self.R / self.aqin.lab2[i,j,:])
                if self.R / abs(self.aqout.lab2[i,j,0]) < self.Rbig:
                    self.circ_out_small[i,j] = 1
                    for n in range(self.order+1):
                        self.facout[n,i,j,:] = 1.0 / kv(n, self.R / self.aqout.lab2[i,j,:])
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((2*aq.Naq,1+2*self.order,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aqin:
            r = np.sqrt( (x-self.x0)**2 + (y-self.y0)**2 )
            alpha = np.arctan2(y-self.y0, x-self.x0)
            for i in range(self.aqin.Naq):
                for j in range(self.model.Nin):
                    if abs(r-self.R) / abs(self.aqin.lab2[i,j,0]) < self.Rzero:
                        if self.circ_in_small[i,j]:
                            pot = np.zeros((self.model.Npin),'D')
                            rv[i,0,i,j,:] = iv( 0, r / self.aqin.lab2[i,j,:] ) * self.facin[0,i,j,:]
                            for n in range(1,self.order+1):
                                pot[:] = iv( n, r / self.aqin.lab2[i,j,:] ) * self.facin[n,i,j,:]
                                rv[i,2*n-1,i,j,:] = pot * np.cos(n*alpha)
                                rv[i,2*n  ,i,j,:] = pot * np.sin(n*alpha)
                        else:
                            pot = self.besapprox.ivratio(r,self.R,self.aqin.lab2[i,j,:])
                            rv[i,0,i,j,:] = pot[0]
                            for n in range(1,self.order+1):
                                rv[i,2*n-1,i,j,:] = pot[n] * np.cos(n*alpha)
                                rv[i,2*n  ,i,j,:] = pot[n] * np.sin(n*alpha)
        if aq == self.aqout:
            r = np.sqrt( (x-self.x0)**2 + (y-self.y0)**2 )
            alpha = np.arctan2(y-self.y0, x-self.x0)
            for i in range(self.aqout.Naq):
                for j in range(self.model.Nin):
                    if abs(r-self.R) / abs(self.aqout.lab2[i,j,0]) < self.Rzero:
                        if self.circ_out_small[i,j]:
                            pot = np.zeros((self.model.Npin),'D')
                            rv[aq.Naq+i,0,i,j,:] = kv( 0, r / self.aqout.lab2[i,j,:] ) * self.facout[0,i,j,:]
                            for n in range(1,self.order+1):
                                pot[:] = kv( n, r / self.aqout.lab2[i,j,:] ) * self.facout[n,i,j,:]
                                rv[aq.Naq+i,2*n-1,i,j,:] = pot * np.cos(n*alpha)
                                rv[aq.Naq+i,2*n  ,i,j,:] = pot * np.sin(n*alpha)
                        else:
                            pot = self.besapprox.kvratio(r,self.R,self.aqout.lab2[i,j,:])
                            rv[aq.Naq+i,0,i,j,:] = pot[0]
                            for n in range(1,self.order+1):
                                rv[aq.Naq+i,2*n-1,i,j,:] = pot[n] * np.cos(n*alpha)
                                rv[aq.Naq+i,2*n  ,i,j,:] = pot[n] * np.sin(n*alpha)
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    def disinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        qx = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        qy = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        if aq == self.aqin:
            r = np.sqrt( (x-self.x0)**2 + (y-self.y0)**2 )
            alpha = np.arctan2(y-self.y0, x-self.x0)
            qr = np.zeros((aq.Naq,1+2*self.order,aq.Naq,self.model.Nin,self.model.Npin),'D')
            qt = np.zeros((aq.Naq,1+2*self.order,aq.Naq,self.model.Nin,self.model.Npin),'D')
            if r < 1e-20: r = 1e-20  # As we divide by that on the return
            for i in range(self.aqin.Naq):
                for j in range(self.model.Nin):
                    if abs(r-self.R) / abs(self.aqin.lab2[i,j,0]) < self.Rzero:
                        if self.circ_in_small[i,j]:
                            pot = np.zeros((self.order+2,self.model.Npin),'D')
                            for n in range(self.order+2):
                                pot[n] = iv( n, r / self.aqin.lab2[i,j,:] )
                            qr[i,0,i,j,:] = -pot[1] / self.aqin.lab2[i,j,:] * self.facin[0,i,j,:]
                            for n in range(1,self.order+1):
                                qr[i,2*n-1,i,j,:] = -(pot[n-1] + pot[n+1]) / 2 / self.aqin.lab2[i,j,:] * np.cos(n*alpha) * self.facin[n,i,j,:]
                                qr[i,2*n  ,i,j,:] = -(pot[n-1] + pot[n+1]) / 2 / self.aqin.lab2[i,j,:] * np.sin(n*alpha) * self.facin[n,i,j,:] 
                                qt[i,2*n-1,i,j,:] =   pot[n] * np.sin(n*alpha) * n / r * self.facin[n,i,j,:]
                                qt[i,2*n  ,i,j,:] =  -pot[n] * np.cos(n*alpha) * n / r * self.facin[n,i,j,:]
                        else:
                            pot  = self.besapprox.ivratio(r,self.R,self.aqin.lab2[i,j,:])
                            potp = self.besapprox.ivratiop(r,self.R,self.aqin.lab2[i,j,:])
                            qr[i,0,i,j,:] = -potp[0] / self.aqin.lab2[i,j,:]
                            for n in range(1,self.order+1):
                                qr[i,2*n-1,i,j,:] = -potp[n] / self.aqin.lab2[i,j,:] * np.cos(n*alpha)
                                qr[i,2*n  ,i,j,:] = -potp[n] / 2 / self.aqin.lab2[i,j,:] * np.sin(n*alpha)
                                qt[i,2*n-1,i,j,:] =  pot[n] * np.sin(n*alpha) * n / r
                                qt[i,2*n  ,i,j,:] = -pot[n] * np.cos(n*alpha) * n / r
            qr.shape = (self.Nparam/2,aq.Naq,self.model.Np)
            qt.shape = (self.Nparam/2,aq.Naq,self.model.Np)
            qx[:self.Nparam/2,:,:] = qr * np.cos(alpha) - qt * np.sin(alpha);
            qy[:self.Nparam/2,:,:] = qr * np.sin(alpha) + qt * np.cos(alpha);
        if aq == self.aqout:
            r = np.sqrt( (x-self.x0)**2 + (y-self.y0)**2 )
            alpha = np.arctan2(y-self.y0, x-self.x0)
            qr = np.zeros((aq.Naq,1+2*self.order,aq.Naq,self.model.Nin,self.model.Npin),'D')
            qt = np.zeros((aq.Naq,1+2*self.order,aq.Naq,self.model.Nin,self.model.Npin),'D')
            if r < 1e-20: r = 1e-20  # As we divide by that on the return
            for i in range(self.aqout.Naq):
                for j in range(self.model.Nin):
                    if abs(r-self.R) / abs(self.aqout.lab2[i,j,0]) < self.Rzero:
                        if self.circ_out_small[i,j]:
                            pot = np.zeros((self.order+2,self.model.Npin),'D')
                            for n in range(self.order+2):
                                pot[n] = kv( n, r / self.aqout.lab2[i,j,:] )
                            qr[i,0,i,j,:] = pot[1] / self.aqout.lab2[i,j,:] * self.facout[0,i,j,:]
                            for n in range(1,self.order+1):
                                qr[i,2*n-1,i,j,:] = (pot[n-1] + pot[n+1]) / 2 / self.aqout.lab2[i,j,:] * np.cos(n*alpha) * self.facout[n,i,j,:]
                                qr[i,2*n  ,i,j,:] = (pot[n-1] + pot[n+1]) / 2 / self.aqout.lab2[i,j,:] * np.sin(n*alpha) * self.facout[n,i,j,:]
                                qt[i,2*n-1,i,j,:] =   pot[n] * np.sin(n*alpha) * n / r * self.facout[n,i,j,:]
                                qt[i,2*n  ,i,j,:] =  -pot[n] * np.cos(n*alpha) * n / r * self.facout[n,i,j,:]
                        else:
                            pot  = self.besapprox.kvratio(r,self.R,self.aqout.lab2[i,j,:])
                            potp = self.besapprox.kvratiop(r,self.R,self.aqout.lab2[i,j,:])
                            qr[i,0,i,j,:] = -potp[0] / self.aqout.lab2[i,j,:]
                            for n in range(1,self.order+1):
                                qr[i,2*n-1,i,j,:] = -potp[n] / self.aqout.lab2[i,j,:] * np.cos(n*alpha)
                                qr[i,2*n  ,i,j,:] = -potp[n] / self.aqout.lab2[i,j,:] * np.sin(n*alpha)
                                qt[i,2*n-1,i,j,:] =  pot[n] * np.sin(n*alpha) * n / r
                                qt[i,2*n  ,i,j,:] = -pot[n] * np.cos(n*alpha) * n / r
            qr.shape = (self.Nparam/2,aq.Naq,self.model.Np)
            qt.shape = (self.Nparam/2,aq.Naq,self.model.Np)
            qx[self.Nparam/2:,:,:] = qr * np.cos(alpha) - qt * np.sin(alpha);
            qy[self.Nparam/2:,:,:] = qr * np.sin(alpha) + qt * np.cos(alpha);            
        return qx,qy
    def layout(self):
        return 'line', self.x0 + self.R * np.cos(np.linspace(0,2*np.pi,100)), self.y0 + self.R * np.sin(np.linspace(0,2*np.pi,100))

def CircInhomMaq(model,x0=0,y0=0,R=1,order=1,kaq=[1],z=[1,0],c=[],Saq=[0.001],Sll=[0],topboundary='imp',phreatictop=False,label=None,test=False):
    CircInhomDataMaq(model,x0,y0,R,kaq,z,c,Saq,Sll,topboundary,phreatictop)
    return CircInhom(model,x0,y0,R,order,label,test)
    
def CircInhom3D(model,x0=0,y0=0,R=1,order=1,kaq=[1,1,1],z=[4,3,2,1],Saq=[0.3,0.001,0.001],kzoverkh=[.1,.1,.1],phreatictop=True,label=None):
    CircInhomData3D(model,x0,y0,R,kaq,z,Saq,kzoverkh,phreatictop)       
    return CircInhom(model,x0,y0,R,order,label)

def EllipseInhomMaq(model,x0=0,y0=0,along=2.0,bshort=1.0,angle=0.0,order=1,kaq=[1],z=[1,0],c=[],Saq=[0.001],Sll=[0],topboundary='imp',phreatictop=False,label=None):
    EllipseInhomDataMaq(model,x0,y0,along,bshort,angle,kaq,z,c,Saq,Sll,topboundary,phreatictop)
    return EllipseInhom(model,x0,y0,along,bshort,angle,order,label)
    
def EllipseInhom3D(self,model,x0=0,y0=0,along=2.0,bshort=1.0,angle=0.0,order=1,kaq=[1,1,1],z=[4,3,2,1],Saq=[0.3,0.001,0.001],kzoverkh=[.1,.1,.1],phreatictop=True,label=None):
    EllipseInhomData3D(model,x0,y0,along,bshort,angle,kaq,z,Saq,kzoverkh,phreatictop)       
    return EllipseInhom(model,x0,y0,along,bshort,angle,order,label)

class EllipseInhom(Element,InhomEquation):
    def __init__(self,model,x0=0,y0=0,along=2.0,bshort=1.0,angle=0.0,order=0,label=None):
        Element.__init__(self, model, Nparam=2*model.aq.Naq*(2*order+1), Nunknowns=2*model.aq.Naq*(2*order+1), layers=range(0,model.aq.Naq), \
                         type='z', name='EllipseInhom', label=label)
        self.x0, self.y0, self.along, self.bshort, self.angle = float(x0), float(y0), float(along), float(bshort), float(angle)
        self.order = order
        self.model.addelement(self)
    def __repr__(self):
        return self.name + ' at ' + str((self.x0,self.y0))
    def initialize(self):
        self.aqin = self.model.aq.find_aquifer_data(self.x0, self.y0)
        self.aqout = self.model.aq.find_aquifer_data(self.x0 + (1.0 + 1e-8) * self.along, self.y0)
        self.qin =   self.aqin.dfoc**2 / (16.0 * self.aqin.lab**2)
        self.qout =  self.aqin.dfoc**2 / (16.0 * self.aqout.lab**2)
        self.xytoetapsi = self.aqin.xytoetapsi
        self.etapsitoxy = self.aqin.etapsitoxy
        self.etastar = self.aqin.etastar
        self.afoc = self.aqin.afoc;
        self.Ncp = 2*self.order + 1
        psicp = np.arange(0,2*np.pi-1e-8,2*np.pi/self.Ncp)
        self.xc, self.yc = self.etapsitoxy(self.etastar, psicp)
        self.thetacp = self.aqin.outwardnormalangle(self.xc,self.yc)
        self.setbc()
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.mfin = [] # list with mathieu function objects
        self.mfout = []
        for i in range(self.aqin.Naq):
            rowin = []
            rowout = []
            for j in range(self.model.Np):
                rowin.append(mathieu(self.qin[i,j]))
                rowout.append(mathieu(self.qout[i,j]))
            self.mfin.append(rowin)
            self.mfout.append(rowout)
        self.neven = [0] + range(1,2*self.order,2)
        self.nodd  = range(2,2*self.order+1,2)
        self.norder = range(self.order+1)
    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((2*aq.Naq,1+2*self.order,aq.Naq,self.model.Np),'D')
        if aq == self.aqin:
            eta,psi = self.xytoetapsi(x,y)
            for i in range(self.aqin.Naq):
                for j in range(self.model.Np):
                    if True:  # Need a condition here when we set the function to zero
                        #Faster if all orders computed at the same time
                        #rv[i,0,i,j] = self.mfin[i][j].ce(0,psi) * self.mfin[i][j].Ie(0,eta)
                        #for n in range(1,self.order+1):
                        #    rv[i,2*n-1,i,j] = self.mfin[i][j].ce(n,psi) * self.mfin[i][j].Ie(n,eta)
                        #    rv[i,2*n  ,i,j] = self.mfin[i][j].se(n,psi) * self.mfin[i][j].Io(n,eta)
                        rv[i,self.neven,i,j] = self.mfin[i][j].ce(self.norder,psi)     * self.mfin[i][j].Ie(self.norder,eta)
                        rv[i,self.nodd ,i,j] = self.mfin[i][j].se(self.norder[1:],psi) * self.mfin[i][j].Io(self.norder[1:],eta)                            
        if aq == self.aqout:
            eta,psi = self.xytoetapsi(x,y)
            for i in range(self.aqout.Naq):
                for j in range(self.model.Np):
                    if True: # Need a condition here when we set the function to zero
                        rv[aq.Naq+i,self.neven,i,j] = self.mfout[i][j].ce(self.norder,psi)     * self.mfout[i][j].Ke(self.norder,eta)
                        rv[aq.Naq+i,self.nodd ,i,j] = self.mfout[i][j].se(self.norder[1:],psi) * self.mfout[i][j].Ko(self.norder[1:],eta)
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    def disinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        qx = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        qy = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        if aq == self.aqin:
            eta,psi = self.xytoetapsi(x,y)
            qeta = np.zeros((aq.Naq,1+2*self.order,aq.Naq,self.model.Np),'D')
            qpsi = np.zeros((aq.Naq,1+2*self.order,aq.Naq,self.model.Np),'D')
            for i in range(self.aqin.Naq):
                for j in range(self.model.Np):
                    if True:  # Need a condition here when we set the function to zero
                        ## Faster if all orders pre-computed? ce = mf[i,j].ce(range(self.order+1))
                        #qeta[i,0,i,j] = self.mfin[i][j].ce(0,psi) * self.mfin[i][j].dIe(0,eta)
                        #qpsi[i,0,i,j] = self.mfin[i][j].dce(0,psi) * self.mfin[i][j].Ie(0,eta)
                        #for n in range(1,self.order+1):
                        #    qeta[i,2*n-1,i,j] = self.mfin[i][j].ce(n,psi) * self.mfin[i][j].dIe(n,eta)
                        #    qeta[i,2*n  ,i,j] = self.mfin[i][j].se(n,psi) * self.mfin[i][j].dIo(n,eta)
                        #    qpsi[i,2*n-1,i,j] = self.mfin[i][j].dce(n,psi) * self.mfin[i][j].Ie(n,eta)
                        #    qpsi[i,2*n  ,i,j] = self.mfin[i][j].dse(n,psi) * self.mfin[i][j].Io(n,eta)
                        # Faster if all orders pre-computed? ce = mf[i,j].ce(range(self.order+1))
                        qeta[i,self.neven,i,j] = self.mfin[i][j].ce(self.norder,psi)     * self.mfin[i][j].dIe(self.norder,eta)
                        qeta[i,self.nodd ,i,j] = self.mfin[i][j].se(self.norder[1:],psi) * self.mfin[i][j].dIo(self.norder[1:],eta)
                        qpsi[i,self.neven,i,j] = self.mfin[i][j].dce(self.norder,psi)     * self.mfin[i][j].Ie(self.norder,eta)
                        qpsi[i,self.nodd ,i,j] = self.mfin[i][j].dse(self.norder[1:],psi) * self.mfin[i][j].Io(self.norder[1:],eta)
            qeta.shape = (self.Nparam/2,aq.Naq,self.model.Np)
            qpsi.shape = (self.Nparam/2,aq.Naq,self.model.Np)
            factor = -1.0 / ( self.afoc * np.sqrt( np.cosh(eta)**2 - np.cos(psi)**2 ) )
            cosangle,sinangle = self.aqin.outwardnormal(x,y)
            qx[:self.Nparam/2,:,:] = factor * ( qeta * cosangle - qpsi * sinangle )
            qy[:self.Nparam/2,:,:] = factor * ( qeta * sinangle + qpsi * cosangle )
        if aq == self.aqout:
            eta,psi = self.xytoetapsi(x,y)
            qeta = np.zeros((aq.Naq,1+2*self.order,aq.Naq,self.model.Np),'D')
            qpsi = np.zeros((aq.Naq,1+2*self.order,aq.Naq,self.model.Np),'D')
            for i in range(self.aqin.Naq):
                for j in range(self.model.Np):
                    if True:  # Need a condition here when we set the function to zero
                            qeta[i,self.neven,i,j] = self.mfout[i][j].ce(self.norder,psi)     * self.mfout[i][j].dKe(self.norder,eta)
                            qeta[i,self.nodd ,i,j] = self.mfout[i][j].se(self.norder[1:],psi) * self.mfout[i][j].dKo(self.norder[1:],eta)
                            qpsi[i,self.neven,i,j] = self.mfout[i][j].dce(self.norder,psi)     * self.mfout[i][j].Ke(self.norder,eta)
                            qpsi[i,self.nodd ,i,j] = self.mfout[i][j].dse(self.norder[1:],psi) * self.mfout[i][j].Ko(self.norder[1:],eta)
            qeta.shape = (self.Nparam/2,aq.Naq,self.model.Np)
            qpsi.shape = (self.Nparam/2,aq.Naq,self.model.Np)
            factor = -1.0 / ( self.afoc * np.sqrt( np.cosh(eta)**2 - np.cos(psi)**2 ) )
            cosangle,sinangle = self.aqin.outwardnormal(x,y)
            qx[self.Nparam/2:,:,:] = factor * ( qeta * cosangle - qpsi * sinangle )
            qy[self.Nparam/2:,:,:] = factor * ( qeta * sinangle + qpsi * cosangle )        
        return qx,qy
    def layout(self):
        theta = arange(0,2*pi+0.001,pi/50)
        return [ list( self.etapsitoxy(self.etastar,theta)[0] ), list( self.etapsitoxy(self.etastar,theta)[1] ) ]
    def layout(self):
        psi = np.linspace(0,2*np.pi,100)
        x,y = self.etapsitoxy(self.etastar,psi)
        return 'line', x, y

class WellBase(Element):
    '''Well Base Class. All Well elements are derived from this class'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandbc=[(0.0,1.0)],res=0.0,layers=0,type='',name='WellBase',label=None):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers, tsandbc=tsandbc, type=type, name=name, label=label)
        self.Nparam = len(self.pylayers)  # Defined here and not in Element as other elements can have multiple parameters per layers
        self.xw = float(xw); self.yw = float(yw); self.rw = float(rw); self.res = res
        self.model.addelement(self)
    def __repr__(self):
        return self.name + ' at ' + str((self.xw,self.yw))
    def initialize(self):
        self.xc = np.array([self.xw + self.rw]); self.yc = np.array([self.yw]) # Control point to make sure the point is always the same for all elements
        self.Ncp = 1
        self.aq = self.model.aq.find_aquifer_data(self.xw, self.yw)
        self.setbc()
        coef = self.aq.coef[self.pylayers,:]
        laboverrwk1 = self.aq.lab / (self.rw * kv(1,self.rw/self.aq.lab))
        self.setflowcoef()
        self.term = -1.0 / (2*np.pi) * laboverrwk1 * self.flowcoef * coef  # shape (self.nparam,self.aq.Naq,self.model.npval)
        self.term2 = self.term.reshape(self.Nparam,self.aq.Naq,self.model.Nin,self.model.Npin)
        self.strengthinf = self.flowcoef * coef
        self.strengthinflayers = np.sum(self.strengthinf * self.aq.eigvec[self.pylayers,:,:], 1) 
        self.resfach = self.res / ( 2*np.pi*self.rw*self.aq.Haq[self.pylayers] )  # Q = (h - hw) / resfach
        self.resfacp = self.resfach * self.aq.T[self.pylayers]  # Q = (Phi - Phiw) / resfacp
    def setflowcoef(self):
        '''Separate function so that this can be overloaded for other types'''
        self.flowcoef = 1.0 / self.model.p  # Step function
    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            r = np.sqrt( (x-self.xw)**2 + (y-self.yw)**2 )
            pot = np.zeros(self.model.Npin,'D')
            if r < self.rw: r = self.rw  # If at well, set to at radius
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if r / abs(self.aq.lab2[i,j,0]) < self.Rzero:
                        bessel.k0besselv( r / self.aq.lab2[i,j,:], pot )
                        rv[:,i,j,:] = self.term2[:,i,j,:] * pot
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    def disinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        qx,qy = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D'), np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        if aq == self.aq:
            qr = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
            r = np.sqrt( (x-self.xw)**2 + (y-self.yw)**2 )
            pot = np.zeros(self.model.Npin,'D')
            if r < self.rw: r = self.rw  # If at well, set to at radius
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if r / abs(self.aq.lab2[i,j,0]) < self.Rzero:
                        qr[:,i,j,:] = self.term2[:,i,j,:] * kv(1, r / self.aq.lab2[i,j,:]) / self.aq.lab2[i,j,:]
            qr.shape = (self.Nparam,aq.Naq,self.model.Np)
            qx[:] = qr * (x-self.xw) / r; qy[:] = qr * (y-self.yw) / r
        return qx,qy
    def headinside(self,t,derivative=0):
        '''Returns head inside the well for the layers that the well is screened in'''
        return self.model.head(self.xc,self.yc,t,derivative=derivative)[self.pylayers] - self.resfach[:,np.newaxis] * self.strength(t,derivative=derivative)
    def layout(self):
        return 'point',self.xw,self.yw

class LineSinkBase(Element):
    '''LineSink Base Class. All LineSink elements are derived from this class'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandbc=[(0.0,1.0)],res=0.0,wh='H',layers=0,type='',name='LineSinkBase',label=None,addtomodel=True):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers, tsandbc=tsandbc, type=type, name=name, label=label)
        self.Nparam = len(self.pylayers)
        self.x1 = float(x1); self.y1 = float(y1); self.x2 = float(x2); self.y2 = float(y2); self.res = res; self.wh = wh
        if addtomodel: self.model.addelement(self)
        self.xa,self.ya,self.xb,self.yb,self.np = np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1,'i')  # needed to call bessel.circle_line_intersection
    def __repr__(self):
        return self.name + ' from ' + str((self.x1,self.y1)) +' to '+str((self.x2,self.y2))
    def initialize(self):
        self.xc = np.array([0.5*(self.x1+self.x2)]); self.yc = np.array([0.5*(self.y1+self.y2)])
        self.Ncp = 1
        self.z1 = self.x1 + 1j*self.y1; self.z2 = self.x2 + 1j*self.y2
        self.L = np.abs(self.z1-self.z2)
        self.order = 0 # This is for univform strength only
        self.aq = self.model.aq.find_aquifer_data(self.xc, self.yc)
        self.setbc()
        coef = self.aq.coef[self.pylayers,:]
        self.setflowcoef()
        self.term = self.flowcoef * coef  # shape (self.nparam,self.aq.Naq,self.model.npval)
        self.term2 = self.term.reshape(self.Nparam,self.aq.Naq,self.model.Nin,self.model.Npin)
        self.strengthinf = self.flowcoef * coef
        self.strengthinflayers = np.sum(self.strengthinf * self.aq.eigvec[self.pylayers,:,:], 1)
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
    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            pot = np.zeros(self.model.Npin,'D')
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
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rvx,rvy = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D'), np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            qxqy = np.zeros((2,self.model.Npin),'D')
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if bessel.isinside(self.z1,self.z2,x+y*1j,self.Rzero*self.aq.lababs[i,j]):
                        qxqy[:,:] = bessel.bessellsqxqyv2(x,y,self.z1,self.z2,self.aq.lab2[i,j,:],self.order,self.Rzero*self.aq.lababs[i,j]) / self.L  # Divide by L as the parameter is now total discharge
                        rvx[:,i,j,:] = self.term2[:,i,j,:] * qxqy[0]
                        rvy[:,i,j,:] = self.term2[:,i,j,:] * qxqy[1]
        rvx.shape = (self.Nparam,aq.Naq,self.model.Np)
        rvy.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rvx,rvy
    def headinside(self,t):
        return self.model.head(self.xc,self.yc,t)[self.pylayers] - self.resfach[:,np.newaxis] * self.strength(t)
    def layout(self):
        return 'line', [self.x1,self.x2], [self.y1,self.y2]
        

class CircAreaSink(Element):
    '''Circular Area Sink'''
    def __init__(self,model,xc=0,yc=0,R=0.1,tsandbc=[(0.0,1.0)],name='CircAreaSink',label=None):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=0, tsandbc=tsandbc, type='g', name=name, label=label)
        self.xc = float(xc); self.yc = float(yc); self.R = float(R)
        self.model.addelement(self)
    def __repr__(self):
        return self.name + ' at ' + str((self.xc,self.yc))
    def initialize(self):
        self.aq = self.model.aq.find_aquifer_data(self.xc, self.yc)
        self.setbc()
        self.setflowcoef()
        self.an = self.aq.coef[0,:] * self.flowcoef  # Since recharge is in layer 1 (pylayer=0), and RHS is -N
        self.an.shape = (self.aq.Naq,self.model.Nin,self.model.Npin)
        self.termin  = self.aq.lab2 * self.R * self.an * kv(1,self.R/self.aq.lab2)
        self.termin2 = self.aq.lab2**2 * self.an
        self.terminq = self.R * self.an * kv(1,self.R/self.aq.lab2)
        self.termout = self.aq.lab2 * self.R * self.an * iv(1,self.R/self.aq.lab2)
        self.termoutq= self.R * self.an * iv(1,self.R/self.aq.lab2)

        self.strengthinf = self.aq.coef[0,:] * self.flowcoef
        self.strengthinflayers = np.sum(self.strengthinf * self.aq.eigvec[self.pylayers,:,:], 1)

    def setflowcoef(self):
        '''Separate function so that this can be overloaded for other types'''
        self.flowcoef = 1.0 / self.model.p  # Step function
    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            r = np.sqrt( (x-self.xc)**2 + (y-self.yc)**2 )
            pot = np.zeros(self.model.Npin,'D')
            if r < self.R:
                for i in range(self.aq.Naq):
                    for j in range(self.model.Nin):
                        #if r / abs(self.aq.lab2[i,j,0]) < self.rzero:
                        rv[0,i,j,:] = -self.termin[i,j,:] * iv(0,r/self.aq.lab2[i,j,:]) + self.termin2[i,j,:]
            else:
                for i in range(self.aq.Naq):
                    for j in range(self.model.Nin):
                        if (r-self.R) / abs(self.aq.lab2[i,j,0]) < self.Rzero:
                            rv[0,i,j,:] = self.termout[i,j,:] * kv(0,r/self.aq.lab2[i,j,:])
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    def disinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        qx,qy = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D'), np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        if aq == self.aq:
            qr = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
            r = np.sqrt( (x-self.xc)**2 + (y-self.yc)**2 )
            if r < self.R:
                for i in range(self.aq.Naq):
                    for j in range(self.model.Nin):
                        #if r / abs(self.aq.lab2[i,j,0]) < self.rzero:
                        qr[0,i,j,:] = self.terminq[i,j,:] * iv(1,r/self.aq.lab2[i,j,:])
            else:
                for i in range(self.aq.Naq):
                    for j in range(self.model.Nin):
                        if (r-self.R) / abs(self.aq.lab2[i,j,0]) < self.Rzero:
                            qr[0,i,j,:] = self.termoutq[i,j,:] * kv(1,r/self.aq.lab2[i,j,:])                
            qr.shape = (self.Nparam,aq.Naq,self.model.Np)
            qx[:] = qr * (x-self.xc) / r; qy[:] = qr * (y-self.yc) / r
        return qx,qy
    def layout(self):
        return 'line', self.xc + self.R*np.cos(np.linspace(0,2*np.pi,100)), self.xc + self.R*np.sin(np.linspace(0,2*np.pi,100))
        
class LineSinkHoBase(Element):
    '''Higher Order LineSink Base Class. All Higher Order Line Sink elements are derived from this class'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandbc=[(0.0,1.0)],res=0.0,wh='H',order=0,layers=0,type='',name='LineSinkBase',label=None,addtomodel=True):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers, tsandbc=tsandbc, type=type, name=name, label=label)
        self.order = order
        self.Nparam = (self.order+1) * len(self.pylayers)
        self.x1 = float(x1); self.y1 = float(y1); self.x2 = float(x2); self.y2 = float(y2); self.res = res; self.wh = wh
        if addtomodel: self.model.addelement(self)
        #self.xa,self.ya,self.xb,self.yb,self.np = np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1,'i')  # needed to call bessel.circle_line_intersection
    def __repr__(self):
        return self.name + ' from ' + str((self.x1,self.y1)) +' to '+str((self.x2,self.y2))
    def initialize(self):
        self.Ncp = self.order + 1
        self.z1 = self.x1 + 1j*self.y1; self.z2 = self.x2 + 1j*self.y2
        self.L = np.abs(self.z1-self.z2)
        #
        thetacp = np.arange(np.pi,0,-np.pi/self.Ncp) - 0.5 * np.pi/self.Ncp
        Zcp = np.zeros( self.Ncp, 'D' )
        Zcp.real = np.cos(thetacp)
        Zcp.imag = 1e-6  # control point just on positive site (this is handy later on)
        zcp = Zcp * (self.z2 - self.z1) / 2.0 + 0.5 * (self.z1 + self.z2)
        self.xc = zcp.real; self.yc = zcp.imag
        #
        self.aq = self.model.aq.find_aquifer_data(self.xc[0], self.yc[0])
        self.setbc()
        coef = self.aq.coef[self.pylayers,:]
        self.setflowcoef()
        self.term = self.flowcoef * coef  # shape (self.nlayers,self.aq.Naq,self.model.npval)
        self.term2 = self.term.reshape(self.Nlayers,self.aq.Naq,self.model.Nin,self.model.Npin)
        #self.term2 = np.empty((self.nparam,self.aq.Naq,self.model.Nin,self.model.npint),'D')
        #for i in range(self.nlayers):
        #    self.term2[i*(self.order+1):(i+1)*(self.order+1),:,:,:] = self.term[i,:,:].reshape((1,self.aq.Naq,self.model.Nin,self.model.npint))
        self.strengthinf = self.flowcoef * coef
        self.strengthinflayers = np.sum(self.strengthinf * self.aq.eigvec[self.pylayers,:,:], 1)
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
    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            pot = np.zeros((self.order+1,self.model.Npin),'D')
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if bessel.isinside(self.z1,self.z2,x+y*1j,self.Rzero*self.aq.lababs[i,j]):
                        pot[:,:] = bessel.bessellsv2(x,y,self.z1,self.z2,self.aq.lab2[i,j,:],self.order,self.Rzero*self.aq.lababs[i,j]) / self.L  # Divide by L as the parameter is now total discharge
                        for k in range(self.Nlayers):
                            rv[k::self.Nlayers,i,j,:] = self.term2[k,i,j,:] * pot
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    def disinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rvx,rvy = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D'), np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            qxqy = np.zeros((2*(self.order+1),self.model.Npin),'D')
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if bessel.isinside(self.z1,self.z2,x+y*1j,self.Rzero*self.aq.lababs[i,j]):
                        qxqy[:,:] = bessel.bessellsqxqyv2(x,y,self.z1,self.z2,self.aq.lab2[i,j,:],self.order,self.Rzero*self.aq.lababs[i,j]) / self.L  # Divide by L as the parameter is now total discharge
                        for k in range(self.Nlayers):
                            rvx[k::self.Nlayers,i,j,:] = self.term2[k,i,j,:] * qxqy[:self.order+1,:]
                            rvy[k::self.Nlayers,i,j,:] = self.term2[k,i,j,:] * qxqy[self.order+1:,:]
        rvx.shape = (self.Nparam,aq.Naq,self.model.Np)
        rvy.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rvx,rvy
    def headinside(self,t):
        return self.model.head(self.xc,self.yc,t)[self.pylayers] - self.resfach[:,np.newaxis] * self.strength(t)
    def layout(self):
        return 'line', [self.x1,self.x2], [self.y1,self.y2]
        
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
        if addtomodel: self.model.addelement(self)
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
        self.aq = self.model.aq.find_aquifer_data(self.xc[0], self.yc[0])
        self.setbc()
        coef = self.aq.coef[self.pylayers,:]
        self.setflowcoef()
        self.term = self.flowcoef * coef  # shape (self.nlayers,self.aq.Naq,self.model.npval)
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
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
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
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
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
    def layout(self):
        return 'line', [self.x1,self.x2], [self.y1,self.y2]
    
class DischargeWell(WellBase):
    '''Well with non-zero and potentially variable discharge through time'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandQ=[(0.0,1.0)],res=0.0,layers=0,label=None):
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=tsandQ,res=res,layers=layers,type='g',name='DischargeWell',label=label)
    
class Well(WellBase,WellBoreStorageEquation):
    '''One or multi-screen well with wellbore storage'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandQ=[(0.0,1.0)],res=0.0,layers=0,rc=None,wbstype='pumping',label=None):
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=tsandQ,res=res,layers=layers,type='v',name='MscreenWell',label=label)
        if (rc is None) or (rc <= 0.0):
            self.rc = 0.0
        else:
            self.rc = rc
        # hdiff is not used right now, but may be used in the future
        self.hdiff = None
        #if hdiff is not None:
        #    self.hdiff = np.atleast_1d(hdiff)
        #    assert len(self.hdiff) == self.nlayers - 1, 'hdiff needs to have length len(layers) -1'
        #else:
        #    self.hdiff = hdiff
        self.Nunknowns = self.Nparam
        self.wbstype = wbstype
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
    def setflowcoef(self):
        '''Separate function so that this can be overloaded for other types'''
        if self.wbstype == 'pumping':
            self.flowcoef = 1.0 / self.model.p  # Step function
        elif self.wbstype == 'slug':
            self.flowcoef = 1.0  # Delta function

        
class LineSink(LineSinkBase):
    '''LineSink with non-zero and potentially variable discharge through time'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandQ=[(0.0,1.0)],res=0.0,wh='H',layers=0,label=None,addtomodel=True):
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=tsandQ,res=res,wh=wh,layers=layers,type='g',name='LineSink',label=label,addtomodel=addtomodel)

class ZeroMscreenWell(WellBase,MscreenEquation):
    '''MscreenWell with zero discharge. Needs to be screened in multiple layers; Head is same in all screened layers'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,res=0.0,layers=[0,1],vres=0.0,label=None):
        assert len(layers) > 1, "TTim input error: number of layers for ZeroMscreenWell must be at least 2"
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=[(0.0,0.0)],res=res,layers=layers,type='z',name='ZeroMscreenWell',label=label)
        self.Nunknowns = self.Nparam
        self.vres = np.atleast_1d(vres)  # Vertical resistance inside well
        if len(self.vres) == 1: self.vres = self.vres[0] * np.ones(self.Nlayers-1)
        self.vresfac = self.vres / (np.pi * self.rw**2)  # Qv = (hn - hn-1) / vresfac[n-1]
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
class ZeroMscreenLineSink(LineSinkBase,MscreenEquation):
    '''MscreenLineSink with zero discharge. Needs to be screened in multiple layers; Head is same in all screened layers'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,res=0.0,wh='H',layers=[0,1],vres=0.0,wv=1.0,label=None,addtomodel=True):
        assert len(layers) > 1, "TTim input error: number of layers for ZeroMscreenLineSink must be at least 2"
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=[(0.0,0.0)],res=res,wh=wh,layers=layers,type='z',name='ZeroMscreenLineSink',label=label,addtomodel=addtomodel)
        self.Nunknowns = self.Nparam
        self.vres = np.atleast_1d(vres)  # Vertical resistance inside line-sink
        self.wv = wv
        if len(self.vres) == 1: self.vres = self.vres[0] * np.ones(self.Nlayers-1)
    def initialize(self):
        LineSinkBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.vresfac = self.vres / (self.wv * self.L)  # Qv = (hn - hn-1) / vresfac[n-1]
        
class MscreenWellOld(WellBase,MscreenEquation):
    '''MscreenWell that varies through time. May be screened in multiple layers but heads are same in all screened layers'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandQ=[(0.0,1.0)],res=0.0,layers=[0,1],label=None):
        assert len(layers) > 1, "TTim input error: number of layers for MscreenWell must be at least 2"
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=tsandQ,res=res,layers=layers,type='v',name='MscreenWell',label=label)
        self.Nunknowns = self.Nparam
        self.vresfac = np.zeros(self.Nlayers-1)  # Vertical resistance inside well, defined but not used; only used for ZeroMscreenWell
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
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
        
class ZeroHeadWell(WellBase,HeadEquation):
    '''HeadWell that remains zero and constant through time'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,res=0.0,layers=0,label=None):
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=[(0.0,0.0)],res=res,layers=layers,type='z',name='ZeroHeadWell',label=label)
        self.Nunknowns = self.Nparam
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
class ZeroHeadLineSink(LineSinkBase,HeadEquation):
    '''HeadLineSink that remains zero and constant through time'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,res=0.0,wh='H',layers=0,label=None,addtomodel=True):
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=[(0.0,0.0)],res=res,wh=wh,layers=layers,type='z',name='ZeroHeadLineSink',label=label,addtomodel=addtomodel)
        self.Nunknowns = self.Nparam
    def initialize(self):
        LineSinkBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
    
class HeadWell(WellBase,HeadEquation):
    '''HeadWell of which the head varies through time. May be screened in multiple layers but all with the same head'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandh=[(0.0,1.0)],res=0.0,layers=0,label=None):
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=tsandh,res=res,layers=layers,type='v',name='HeadWell',label=label)
        self.Nunknowns = self.Nparam
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.pc = self.aq.T[self.pylayers] # Needed in solving; We solve for a unit head
        

from scipy.special import erf
def fbar(p,t0=100.0,a=20.0):
    return np.sqrt(np.pi) / 2.0 * a * np.exp( -p*t0 + a**2*p**2/4.0 ) * ( 1.0 - erf( -t0/a + a*p/2.0 ) )
class HeadWellNew(WellBase,HeadEquationNew):
    '''HeadWell of which the head varies through time. May be screened in multiple layers but all with the same head'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandh=[(0.0,1.0)],res=0.0,layers=0,label=None):
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=tsandh,res=res,layers=layers,type='v',name='HeadWell',label=label)
        self.Nunknowns = self.Nparam
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.pc = np.empty((1,self.model.Np),'D')
        self.pc[0] = self.aq.T[self.pylayers] * fbar(self.model.p,t0=100,a=20.0) # Needed in solving
        
class HeadLineSink(LineSinkBase,HeadEquation):
    '''HeadLineSink of which the head varies through time. May be screened in multiple layers but all with the same head'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandh=[(0.0,1.0)],res=0.0,wh='H',layers=0,label=None,addtomodel=True):
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=tsandh,res=res,wh=wh,layers=layers,type='v',name='HeadLineSink',label=label,addtomodel=addtomodel)
        self.Nunknowns = self.Nparam
    def initialize(self):
        LineSinkBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.pc = self.aq.T[self.pylayers] # Needed in solving; We solve for a unit head
        
class HeadLineSinkHo(LineSinkHoBase,HeadEquationNores):
    '''HeadLineSink of which the head varies through time. May be screened in multiple layers but all with the same head'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandh=[(0.0,1.0)],order=0,layers=0,label=None,addtomodel=True):
        self.storeinput(inspect.currentframe())
        LineSinkHoBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=tsandh,res=0.0,wh='H',order=order,layers=layers,type='v',name='HeadLineSinkHo',label=label,addtomodel=addtomodel)
        self.Nunknowns = self.Nparam
    def initialize(self):
        LineSinkHoBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.pc = np.empty(self.Nparam)
        for i,T in enumerate(self.aq.T[self.pylayers]):
            self.pc[i::self.Nlayers] =  T # Needed in solving; we solve for a unit head
            
class LeakyLineDoublet(LineDoubletHoBase,LeakyWallEquation):
    '''Leaky LineDoublet'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,res='imp',order=0,layers=0,label=None,addtomodel=True):
        self.storeinput(inspect.currentframe())
        LineDoubletHoBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=[(0.0,0.0)],res=res,order=order,layers=layers,type='z',name='LeakyLineDoublet',label=label,addtomodel=addtomodel)
        self.Nunknowns = self.Nparam
    def initialize(self):
        LineDoubletHoBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )

class LineSinkStringBase(Element):
    def __init__(self,model,tsandbc=[(0.0,1.0)],layers=0,type='',name='LineSinkStringBase',label=None):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers, tsandbc=tsandbc, type=type, name=name, label=label)
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
        self.aq = self.model.aq.find_aquifer_data(self.lsList[0].xc, self.lsList[0].yc)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.setbc()
        # As parameters are only stored for the element not the list, we need to combine the following
        self.resfach = []; self.resfacp = []
        for ls in self.lsList:
            ls.initialize()
            self.resfach.extend( ls.resfach.tolist() )  # Needed in solving
            self.resfacp.extend( ls.resfacp.tolist() )  # Needed in solving
        self.resfach = np.array(self.resfach); self.resfacp = np.array(self.resfacp)
        self.strengthinf = np.zeros((self.Nparam,self.aq.Naq,self.model.Np),'D')
        self.strengthinflayers = np.zeros((self.Nparam,self.model.Np),'D')
        self.xc, self.yc = np.zeros(self.Nls), np.zeros(self.Nls)
        for i in range(self.Nls):
            self.strengthinf[i*self.Nlayers:(i+1)*self.Nlayers,:] = self.lsList[i].strengthinf[:]
            self.strengthinflayers[i*self.Nlayers:(i+1)*self.Nlayers,:] = self.lsList[i].strengthinflayers
            self.xc[i], self.yc[i] = self.lsList[i].xc, self.lsList[i].yc
    def potinf(self,x,y,aq=None):
        '''Returns array (nunknowns,Nperiods)'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        for i in range(self.Nls):
            rv[i*self.Nlayers:(i+1)*self.Nlayers,:] = self.lsList[i].potinf(x,y,aq)
        return rv
    def disinf(self,x,y,aq=None):
        '''Returns array (nunknowns,Nperiods)'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rvx,rvy = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D'),np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        for i in range(self.Nls):
            qx,qy = self.lsList[i].disinf(x,y,aq)
            rvx[i*self.Nlayers:(i+1)*self.Nlayers,:] = qx
            rvy[i*self.Nlayers:(i+1)*self.Nlayers,:] = qy
        return rvx,rvy
    def headinside(self,t,derivative=0):
        rv = np.zeros((self.Nls,self.Nlayers,np.size(t)))
        Q = self.strength_list(t,derivative=derivative)
        for i in range(self.Nls):
            rv[i,:,:] = self.model.head(self.xc[i],self.yc[i],t,derivative=derivative)[self.pylayers] - self.resfach[i*self.Nlayers:(i+1)*self.Nlayers,np.newaxis] * Q[i]
        return rv
    def layout(self):
        return 'line', self.xlslayout, self.ylslayout
    def run_after_solve(self):
        for i in range(self.Nls):
            self.lsList[i].parameters[:] = self.parameters[:,i*self.Nlayers:(i+1)*self.Nlayers,:]
    def strength_list(self,t,derivative=0):
        # conveniently using the strength functions of the individual line-sinks
        rv = np.zeros((self.Nls,self.Nlayers,np.size(t)))
        for i in range(self.Nls):
            rv[i,:,:] = self.lsList[i].strength(t,derivative=derivative)
        return rv
    
#class LineSinkString(LineSinkStringBase):
#    def __init__(self,model,xy=[(-1,0),(1,0)],tsandQ=[(0.0,1.0)],res=0.0,layers=1,label=None):
#        LineSinkStringBase.__init__(self,model,xy=xy,tsandbc=tsandQ,res=res,layers=layers,type='g',name='LineSinkString',label=label)
#        for i in range(self.Nls):
#            self.lsList.append( LineSink(model,x1=self.x[i],y1=self.y[i],x2=self.x[i+1],y2=self.y[i+1],tsandQ=tsandQ,res=res,layers=layers,label=None,addtomodel=False) )
#        self.model.addElement(self)
    
class ZeroMscreenLineSinkString(LineSinkStringBase,MscreenEquation):
    def __init__(self,model,xy=[(-1,0),(1,0)],res=0.0,wh='H',layers=[0,1],vres=0.0,wv=1.0,label=None):
        LineSinkStringBase.__init__(self,model,tsandbc=[(0.0,0.0)],layers=layers,type='z',name='ZeroMscreenLineSinkString',label=label)
        xy = np.atleast_2d(xy).astype('d')
        self.x,self.y = xy[:,0], xy[:,1]
        self.Nls = len(self.x) - 1
        for i in range(self.Nls):
            self.lsList.append( ZeroMscreenLineSink(model,x1=self.x[i],y1=self.y[i],x2=self.x[i+1],y2=self.y[i+1],res=res,wh=wh,layers=layers,vres=vres,wv=wv,label=None,addtomodel=False) )
        self.model.addelement(self)
    def initialize(self):
        LineSinkStringBase.initialize(self)
        self.vresfac = np.zeros_like( self.resfach )
        for i in range(self.Nls):
            self.vresfac[i*self.Nlayers:(i+1)*self.Nlayers-1] = self.lsList[i].vresfac[:]
    
class MscreenLineSinkString(LineSinkStringBase,MscreenEquation):
    def __init__(self,model,xy=[(-1,0),(1,0)],tsandQ=[(0.0,1.0)],res=0.0,wh='H',layers=[0,1],label=None):
        LineSinkStringBase.__init__(self,model,tsandbc=tsandQ,layers=layers,type='v',name='MscreenLineSinkString',label=label)
        xy = np.atleast_2d(xy).astype('d')
        self.x,self.y = xy[:,0], xy[:,1]
        self.Nls = len(self.x) - 1
        for i in range(self.Nls):
            self.lsList.append( MscreenLineSink(model,x1=self.x[i],y1=self.y[i],x2=self.x[i+1],y2=self.y[i+1],tsandQ=tsandQ,res=res,wh=wh,layers=layers,label=None,addtomodel=False) )
        self.model.addelement(self)
        
class MscreenLineSinkDitchString(LineSinkStringBase,MscreenDitchEquation):
    def __init__(self,model,xy=[(-1,0),(1,0)],tsandQ=[(0.0,1.0)],res=0.0,wh='H',layers=[0,1],Astorage=None,label=None):
        self.storeinput(inspect.currentframe())
        LineSinkStringBase.__init__(self,model,tsandbc=tsandQ,layers=layers,type='v',name='MscreenLineSinkDitchString',label=label)
        xy = np.atleast_2d(xy).astype('d')
        self.x,self.y = xy[:,0], xy[:,1]
        self.Nls = len(self.x) - 1
        for i in range(self.Nls):
            self.lsList.append( MscreenLineSink(model,x1=self.x[i],y1=self.y[i],x2=self.x[i+1],y2=self.y[i+1],tsandQ=tsandQ,res=res,wh=wh,layers=layers,label=None,addtomodel=False) )
        self.Astorage = Astorage
        self.model.addelement(self)
    def initialize(self):
        LineSinkStringBase.initialize(self)
        self.vresfac = np.zeros_like( self.resfach )  # set to zero, as I don't quite know what it would mean if it is not zero
        
class MscreenLineSinkDitchString2(LineSinkStringBase,MscreenDitchEquation):
    def __init__(self,model,xylist=[[(-1,0),(1,0)],[(2,0),(4,0)]],tsandQ=[(0.0,1.0)],res=0.0,wh='H',layers=[0,1],label=None):
        LineSinkStringBase.__init__(self,model,tsandbc=tsandQ,layers=layers,type='v',name='MscreenLineSinkStringDitch',label=label)
        for xy in xylist:
            xy = np.atleast_2d(xy).astype('d')
            x,y = xy[:,0], xy[:,1]
            for i in range(len(x) - 1):
                self.lsList.append( MscreenLineSink(model,x1=x[i],y1=y[i],x2=x[i+1],y2=y[i+1],tsandQ=tsandQ,res=res,wh=wh,layers=layers,label=None,addtomodel=False) )
        self.Nls = len(self.lsList)
        self.model.addelement(self)
    def initialize(self):
        LineSinkStringBase.initialize(self)
        self.vresfac = np.zeros_like( self.resfach )  # set to zero, as I don't quite know what it would mean if it is not zero
    def layout(self):
        return 'string', self.xls, self.yls
        
class ZeroHeadLineSinkString(LineSinkStringBase,HeadEquation):
    def __init__(self,model,xy=[(-1,0),(1,0)],res=0.0,wh='H',layers=0,label=None):
        LineSinkStringBase.__init__(self,model,tsandbc=[(0.0,0.0)],layers=layers,type='z',name='ZeroHeadLineSinkString',label=label)
        xy = np.atleast_2d(xy).astype('d')
        self.x,self.y = xy[:,0], xy[:,1]
        self.Nls = len(self.x) - 1
        for i in range(self.Nls):
            self.lsList.append( ZeroHeadLineSink(model,x1=self.x[i],y1=self.y[i],x2=self.x[i+1],y2=self.y[i+1],res=res,wh=wh,layers=layers,label=None,addtomodel=False) )
        self.model.addelement(self)

class HeadLineSinkString(LineSinkStringBase,HeadEquation):
    def __init__(self,model,xy=[(-1,0),(1,0)],tsandh=[(0.0,1.0)],res=0.0,wh='H',layers=0,label=None):
        LineSinkStringBase.__init__(self,model,tsandbc=tsandh,layers=layers,type='v',name='HeadLineSinkString',label=label)
        xy = np.atleast_2d(xy).astype('d')
        self.x,self.y = xy[:,0], xy[:,1]
        self.Nls = len(self.x) - 1
        for i in range(self.Nls):
            self.lsList.append( HeadLineSink(model,x1=self.x[i],y1=self.y[i],x2=self.x[i+1],y2=self.y[i+1],tsandh=tsandh,res=res,wh=wh,layers=layers,label=None,addtomodel=False) )
        self.model.addelement(self)
    def initialize(self):
        LineSinkStringBase.initialize(self)
        self.pc = np.zeros(self.Nls*self.Nlayers)
        for i in range(self.Nls): self.pc[i*self.Nlayers:(i+1)*self.Nlayers] = self.lsList[i].pc
        
class LeakyLineDoubletString(Element,LeakyWallEquation):
    def __init__(self,model,xy=[(-1,0),(1,0)],res='imp',order=0,layers=0,label=None):
        self.storeinput(inspect.currentframe())
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers, tsandbc=[(0.0,0.0)], type='z', name='LeakyLineDoubletString', label=label)
        self.res = res
        self.order = order
        self.ldList = []
        xy = np.atleast_2d(xy).astype('d')
        self.x,self.y = xy[:,0], xy[:,1]
        self.Nld = len(self.x) - 1
        for i in range(self.Nld):
            self.ldList.append( LeakyLineDoublet(model,x1=self.x[i],y1=self.y[i],x2=self.x[i+1],y2=self.y[i+1],res=self.res,order=self.order,layers=layers,label=label,addtomodel=False))
        self.model.addelement(self)
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
        self.aq = self.model.aq.find_aquifer_data(self.ldList[0].xc, self.ldList[0].yc)
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
    def potinf(self,x,y,aq=None):
        '''Returns array (nunknowns,Nperiods)'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        for i,ld in enumerate(self.ldList):
            rv[i*ld.Nparam:(i+1)*ld.Nparam,:] = ld.potinf(x,y,aq)
        return rv
    def disinf(self,x,y,aq=None):
        '''Returns array (nunknowns,Nperiods)'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rvx,rvy = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D'),np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        for i,ld in enumerate(self.ldList):
            qx,qy = ld.disinf(x,y,aq)
            rvx[i*ld.Nparam:(i+1)*ld.Nparam,:] = qx
            rvy[i*ld.Nparam:(i+1)*ld.Nparam,:] = qy
        return rvx,rvy
    def layout(self):
        return 'line', self.xldlayout, self.yldlayout
    
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
                
def timcontour( ml, xmin, xmax, nx, ymin, ymax, ny, levels = 10, t=0.0, layers = 0,\
               color = 'k', lw = 0.5, style = 'solid',layout = True, newfig = True,\
               labels = False, labelfmt = '%1.2f'):
    '''Contour heads with pylab'''
    print 'grid of '+str((nx,ny))+'. gridding in progress. hit ctrl-c to abort'
    h = ml.headgrid(xmin,xmax,nx,ymin,ymax,ny,t,layers)  # h[nlayers,Ntimes,Ny,Nx]
    xg, yg = np.linspace(xmin,xmax,nx), np.linspace(ymin,ymax,ny)
    Nlayers, Ntimes = h.shape[0:2]
    # Contour
    if type(levels) is list: levels = np.arange( levels[0],levels[1],levels[2] )
    # Colors
    if color is not None: color = [color]   
    if newfig:
        fig = plt.figure( figsize=(8,8) )
        ax = fig.add_subplot(111)
    else:
        #fig = plt.gcf()
        ax = plt.gca()
    ax.set_aspect('equal','box')
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    ax.set_autoscale_on(False)
    if layout: timlayout(ml,ax)
    # Contour
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    if color is None:
        a = ax.contour( xg, yg, h[0,0], levels, linewidths = lw, linestyles = style )
    else:
        a = ax.contour( xg, yg, h[0,0], levels, colors = color[0], linewidths = lw, linestyles = style )
    if labels:
        ax.clabel(a,fmt=labelfmt)
    plt.draw()
    
def surfgrid(ml,xmin,xmax,nx,ymin,ymax,ny,t,layer=0,filename='/temp/dump'):
    '''Give filename without extension'''
    h = ml.headgrid(xmin,xmax,nx,ymin,ymax,ny,t,layer)[0,0]
    zmin = h.min(); zmax = h.max()
    out = open(filename+'.grd','w')
    out.write('DSAA\n')
    out.write(str(nx)+' '+str(ny)+'\n')
    out.write(str(xmin)+' '+str(xmax)+'\n')
    out.write(str(ymin)+' '+str(ymax)+'\n')
    out.write(str(zmin)+' '+str(zmax)+'\n')
    for i in range(ny):
        for j in range(nx):
            out.write(str(h[i,j])+' ')
        out.write('\n')
    out.close
    
def pyvertcontour( ml, xmin, xmax, ymin, ymax, nx, zg, levels = 10, t=0.0,\
               color = 'k', width = 0.5, style = 'solid',layout = True, newfig = True, \
               labels = False, labelfmt = '%1.2f', fill=False, sendback = False):
    '''Contours head with pylab'''
    plt.rcParams['contour.negative_linestyle']='solid'
    # Compute grid
    xg = np.linspace(xmin,xmax,nx)
    yg = np.linspace(ymin,ymax,nx)
    sg = np.sqrt((xg-xg[0])**2 + (yg-yg[0])**2)
    print 'gridding in progress. hit ctrl-c to abort'
    pot = np.zeros( ( ml.aq.Naq, nx ), 'd' )
    t = np.atleast_1d(t)
    for ip in range(nx):
        pot[:,ip] = ml.head(xg[ip], yg[ip], t)[:,0]
    # Contour
    if type(levels) is list:
        levels = np.arange( levels[0],levels[1],levels[2] )
    elif levels == 'ask':
        print ' min,max: ',pot.min(),', ',pot.max(),'. Enter: hmin hmax step '
        input = raw_input().split()
        levels = np.arange(float(input[0]),float(input[1])+1e-8,float(input[2]))
    print 'Levels are ',levels
    # Colors
    if color is not None:
        color = [color]   
    if newfig:
        fig = plt.figure( figsize=(8,8) )
        ax = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax = plt.gca()
    ax.set_aspect('equal','box')
    ax.set_xlim(sg.min(),sg.max()); ax.set_ylim(zg.min(),zg.max())
    ax.set_autoscale_on(False)
    if fill:
        a = ax.contourf( sg, zg, pot, levels )
    else:
        if color is None:
            a = ax.contour( sg, zg, pot, levels, linewidths = width, linestyles = style )
        else:
            a = ax.contour( sg, zg, pot, levels, colors = color[0], linewidths = width, linestyles = style )
    if labels and not fill:
        ax.clabel(a,fmt=labelfmt)
    fig.canvas.draw()
    if sendback == 1: return a
    if sendback == 2: return sg,zg,pot
                
def timlayout( ml, ax = None, color = 'k', lw = 0.5, style = '-' ):
    show = False
    if ax is None:
        fig = plt.figure( figsize=(8,8) )
        ax = fig.add_subplot(111)
        show = True
    for e in ml.elementList:
        t,x,y = e.layout()
        if t == 'point':
            ax.plot( [x], [y], color+'o', markersize=3 ) 
        if t == 'line':
            ax.plot( x, y, color=color, ls = style, lw = lw )
        if t == 'string':
            N = np.shape(x)[0]
            for i in range(N):
                ax.plot( x[i], y[i], color=color, ls = style, lw = lw )
        if t == 'area':
            col = 0.7 + 0.2*np.random.rand()
            ax.fill( x, y, facecolor = [col,col,col], edgecolor = [col,col,col] )
    if show:
        ax.set_aspect('equal','box')
        plt.show()
        
### Test for OneD
class OneDBase(Element):
    def __init__(self, model, tsandbc=[(0.0,1.0)], layers=1, xleft = 0, xright = 1, type = '', name='OneDBase', label = None):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers, tsandbc=tsandbc, type=type, name=name, label=label)
        self.Nparam = len(self.pylayers)
        self.xleft = xleft
        self.xright = xright
        self.rightside = rightside
        self.model.addelement(self)
    def __repr__(self):
        return self.name + 'from xleft= ' + str(self.xleft) + ' to ' + str(self.xright)
    def initialize(self):
        self.aq = self.model.aq.find_aquifer_data(0.5 * (self.xleft + self.xright), 0)
        self.setbc()
        # Probably need to set control points xc here
        self.setflowcoef()
        self.term = self.flowcoef * self.aq.coef[self.pylayers,:]  # shape (self.nparam,self.aq.Naq,self.model.npval)
        self.term2 = self.term.reshape(self.Nparam,self.aq.Naq,self.model.Nin,self.model.Npin) # shape (self.nparam,self.aq.Naq,self.model.Nin, npint)
        self.strengthinf = self.flowcoef * self.aq.coef[self.pylayers,:]
        self.strengthinflayers = np.sum(self.strengthinf * self.aq.eigvec[self.pylayers,:,:], 1)
        if (self.xleft != -np.inf) & (self.xright != np.inf):
            self.L = self.xright - self.xleft
        if self.L is not np.inf:
            self.A = self.aq.lab2 / ( 1.0 - np.exp(-2.0*self.L/self.aq.lab2) )
            self.B = np.exp(-self.L/self.aq.lab2) * self.A
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
    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if x / abs(self.aq.lab2[i,j,0]) < 20.0:
                        l = j*self.model.Npin
                        if self.rightside == 'inf':
                            rv[:,i,j,:] = self.aq.lab2[i,j,:] * self.term2[:,i,j,:] * np.exp( -x / self.aq.lab2[i,j,:] )
                            #rv[:,i,j,:] = self.aq.lab2[i,j,:] / self.model.p[l:l+self.model.npint] * self.coef[:,i,j,:] * np.exp( -x / self.aq.lab2[i,j,:] )
                        elif self.rightside == 'imp':
                            rv[:,i,j,:] = self.model.p[l:l+self.model.Npin] * self.coef[:,i,j,:] * \
                                          ( self.A[i,j,:] * np.exp(-x/self.aq.lab2[i,j,:]) + self.B[i,j,:] * np.exp( (x-self.L)/self.aq.lab2[i,j,:] ) )
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    #def dischargeinf(self):
    #    rv = np.zeros((self.nparam,self.aq.Naq,self.model.npval),'D')
    #    for i in range(self.aq.Naq):
    #        rv[:,i,:] = 1.0 / self.model.p * self.coef2[:,i,:]
    #    return rv


class OneD(Element):
    def __init__(self, model, tsandbc=[(0.0,1.0)], layers=1, rightside='inf', L=np.inf, res = 0.0, type = 'g', label = None):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers, tsandbc=tsandbc, type=type, name='OneD', label=label)
        self.Nparam = len(self.pylayers)
        self.rightside = rightside
        self.L = L
        self.res = res
        self.wh = 'H' # Outflow over full aquifer thickness
        self.model.addelement(self)
    def __repr__(self):
        return self.name
    def initialize(self):
        self.aq = self.model.aq.find_aquifer_data(0, 0)
        self.setbc()
        #self.coef = self.aq.coef[self.layers,:]
        #self.coef = self.coef.reshape((self.nparam, self.aq.Naq, self.model.Nin, self.model.npint))
        coef = self.aq.coef[self.pylayers,:]
        self.setflowcoef()
        self.term = self.flowcoef * coef  # shape (self.nparam,self.aq.Naq,self.model.npval)
        self.term2 = self.term.reshape(self.Nparam,self.aq.Naq,self.model.Nin,self.model.Npin)
        self.strengthinf = self.flowcoef * coef
        self.strengthinflayers = np.sum(self.strengthinf * self.aq.eigvec[self.pylayers,:,:], 1)
        if self.L is not np.inf:
            self.A = self.aq.lab2 / ( 1.0 - np.exp(-2.0*self.L/self.aq.lab2) )
            self.B = np.exp(-self.L/self.aq.lab2) * self.A
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
    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if x / abs(self.aq.lab2[i,j,0]) < 20.0:
                        l = j*self.model.Npin
                        if self.rightside == 'inf':
                            rv[:,i,j,:] = self.aq.lab2[i,j,:] * self.term2[:,i,j,:] * np.exp( -x / self.aq.lab2[i,j,:] )
                            #rv[:,i,j,:] = self.aq.lab2[i,j,:] / self.model.p[l:l+self.model.npint] * self.coef[:,i,j,:] * np.exp( -x / self.aq.lab2[i,j,:] )
                        elif self.rightside == 'imp':
                            rv[:,i,j,:] = self.model.p[l:l+self.model.Npin] * self.coef[:,i,j,:] * \
                                          ( self.A[i,j,:] * np.exp(-x/self.aq.lab2[i,j,:]) + self.B[i,j,:] * np.exp( (x-self.L)/self.aq.lab2[i,j,:] ) )
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    def disinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        qx = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        qy = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if x / abs(self.aq.lab2[i,j,0]) < 20.0:
                        l = j*self.model.Npin
                        if self.rightside == 'inf':
                            qx[:,i,j,:] = self.term2[:,i,j,:] * np.exp( -x / self.aq.lab2[i,j,:] )
                            #rv[:,i,j,:] = self.aq.lab2[i,j,:] / self.model.p[l:l+self.model.npint] * self.coef[:,i,j,:] * np.exp( -x / self.aq.lab2[i,j,:] )
        qx.shape = (self.Nparam,aq.Naq,self.model.Np)
        qy.shape = (self.Nparam,aq.Naq,self.model.Np)
        return qx, qy
    #def dischargeinf(self):
    #    rv = np.zeros((self.nparam,self.aq.Naq,self.model.npval),'D')
    #    for i in range(self.aq.Naq):
    #        rv[:,i,:] = 1.0 / self.model.p * self.coef2[:,i,:]
    #    return rv
    
class HeadOneD(OneD,HeadEquation):
    def __init__(self, model, tsandh = [(0.0,1.0)], layers = 1, rightside = 'inf', L = np.inf, label = None):
        OneD.__init__(self, model, tsandbc = tsandh, layers=layers, rightside=rightside, L=L, type = 'v', label = label)
        self.Nunknowns = self.Nparam
        self.name = 'HeadOneD'
    def initialize(self):
        OneD.initialize(self)
        self.xc, self.yc = np.array([0]), np.array([0])
        self.Ncp = 1
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.pc = self.aq.T[self.pylayers] # Needed in solving; We solve for a unit head
    def check(self,full_output=False):
        maxerror = np.amax( np.abs( self.pc[:,np.newaxis] / self.model.p - self.model.phi(self.xc,self.yc)[self.pylayer,:] ) )
        if full_output:
            print self.name+' with control point at '+str((self.xc,self.yc))+' max error ',maxerror
        return maxerror
    
class XsecLake(Element):
    def __init__(self, model, x1 = 0.0, x2 = 1.0, tsandh=[(0.0,1.0)], B = 1.0, label = None):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=range(model.aq.Naq), tsandbc=tsandh, type='g', name='XsecLake', label=label)
        self.x1 = x1
        self.x2 = x2
        self.B = B
        self.model.addelement(self)
    def __repr__(self):
        return self.name + ' from x1 = ' + str(self.x1) + ' to ' + str(self.x2)
    def initialize(self):
        self.xm = 0.5*(self.x1+self.x2)
        if (self.xm == np.inf) | (self.xm == -np.inf):  # to correct for infinity
            self.xm = np.sign(self.xm) * 1e100
        self.aq = self.model.aq.find_aquifer_data(self.xm, 0)
        self.setbc()
        self.Phistar = self.aq.Tcol / self.model.p  # Step function
        self.Phibar = np.empty((self.aq.Naq, self.model.Np), 'D')
        for i in range(self.model.Np):
            A = self.aq.compute_lab_eigvec(self.model.p[i], returnA = True)
            AB = self.aq.compute_lab_eigvec(self.model.p[i], returnA = True, B = self.B)
            self.Phibar[:,i] = np.dot(np.dot(np.linalg.inv(A), AB), self.Phistar[:,i])
            self.Phibar[:,i] = np.dot(np.linalg.inv(self.aq.eigvec[:,:,i]), self.Phibar[:,i])  # As Phi gets multiplied with V later on
    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        if aq == self.aq:
            rv[0] = self.Phibar
        return rv
    def disinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        return np.zeros((self.Nparam,aq.Naq,self.model.Np),'D'), np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
    
class XsecInhom(Element, InhomEquation):
    def __init__(self, model, xinhom = 0, label = None):
        Element.__init__(self, model, Nparam=2*model.aq.Naq, Nunknowns=2*model.aq.Naq, layers=range(model.aq.Naq), type='z', name='XsecInhom', label=label)
        self.xinhom = xinhom
        self.model.addelement(self)
    def __repr__(self):
        return self.name + 'xinhom= ' + str(self.xinhom)
    def initialize(self):
        self.Ncp = 1
        self.thetacp = np.zeros(1)
        self.xc = np.array([self.xinhom])
        self.yc = np.zeros(1)
        self.aqin = self.model.aq.find_aquifer_data(self.xinhom * (1 - 1e-10) - 1 - 1e-10, self.yc)
        self.aqout = self.model.aq.find_aquifer_data(self.xinhom * (1 + 1e-10) + 1 - 1e-10, self.yc)
        assert self.aqin.Naq == self.aqout.Naq, 'TTim input error: Number of layers needs to be the same inside and outside circular inhomogeneity'
        self.setbc()
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )        
    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.Nparam, self.model.aq.Naq, self.model.Nin, self.model.Npin), 'D')
        if aq == self.aqin:
            for i in range(aq.Naq):
                for j in range(self.model.Nin):
                    if abs(x-self.xinhom) / abs(aq.lab2[i,j,0]) < 20.0:
                        rv[i,i,j,:] = np.exp( (x - self.xinhom) / aq.lab2[i,j,:] )
        elif aq == self.aqout:
            for i in range(aq.Naq):
                for j in range(self.model.Nin):
                    if abs(x-self.xinhom) / abs(aq.lab2[i,j,0]) < 20.0:
                        rv[aq.Naq+i,i,j,:] = np.exp( -(x - self.xinhom) / aq.lab2[i,j,:] )
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    def disinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.find_aquifer_data(x, y)
        qx = np.zeros((self.Nparam, self.model.aq.Naq, self.model.Nin, self.model.Npin), 'D')
        qy = np.zeros((self.Nparam, self.model.aq.Naq, self.model.Nin, self.model.Npin), 'D')
        if aq == self.aqin:
            for i in range(aq.Naq):
                for j in range(self.model.Nin):
                    if abs(x-self.xinhom) / abs(aq.lab2[i,j,0]) < 20.0:
                        qx[i,i,j,:] = -np.exp( (x - self.xinhom) / aq.lab2[i,j,:] ) / aq.lab2[i,j,:]
        elif aq == self.aqout:
            for i in range(aq.Naq):
                for j in range(self.model.Nin):
                    if abs(x-self.xinhom) / abs(aq.lab2[i,j,0]) < 20.0:
                        qx[aq.Naq+i,i,j,:] = np.exp( -(x - self.xinhom) / aq.lab2[i,j,:] ) / aq.lab2[i,j,:]
        qx.shape = (self.Nparam,aq.Naq,self.model.Np)
        qy.shape = (self.Nparam,aq.Naq,self.model.Np)
        return qx, qy
    
#ml = ModelMaq(kaq=[10,10,10], z=[12,8,4,0], c=[100,100], Saq=[0.001,0.001,0.001], Sll=[0.001,0.001], tmin=0.01, tmax=10, M=20)
#aq1 = XsecAquiferDataMaq(ml, x1=-np.inf, x2=0.0, kaq=[1,2,3], z=[6,5,4,3,2,1,0], c=[100,100,100], Saq=[0.001,0.001,0.001], Sll=[0.001,0.001,0.001],topboundary='semi')
#aq2 = XsecAquiferDataMaq(ml, x1=0.0,  x2=np.inf, kaq=[1,2,3], z=[5,4,3,2,1,0], c=[100,100], Saq=[0.001,0.001,0.001], Sll=[0.001,0.001])
#xinhom1 = XsecInhom(ml, xinhom = 0.0)
#xlake = XsecLake(ml, x1 = -np.inf, x2 = 0.0)
#ml.solve()

#ml = Model3D(kaq = 10, z = [12,8,4,0], Saq=[1e-3,1e-3,1e-3],kzoverkh=[1,1,1], phreatictop=False, tmin=0.1, tmax=10, M=20)
#aq1 = XsecAquiferDataMaq(ml, x1=-np.inf, x2=0.0, kaq=[10,10,10], z = [12,12,8,8,4,4,0], c=[100,0.4,0.4], Saq=np.array([0.001,0.001,0.001])/4.0, Sll=0, topboundary='semi')
#aq2 = XsecAquiferDataMaq(ml, x1=0, x2=np.inf, kaq=[10,10,10], z = [12,12,8,8,4,4,0], c=[100,0.4,0.4], Saq=np.array([0.001,0.001,0.001])/4.0, Sll=0, topboundary='semi')
#xinhom1 = XsecInhom(ml, xinhom = 0.0)
#xlake = XsecLake(ml, x1 = -np.inf, x2 = 0.0)
#ml.solve()

#ml2 = Model3D(kaq = 10, z = [12,0], Saq=[1e-3],kzoverkh=[1], phreatictop=False, tmin=0.1, tmax=10, M=20)
#aq1 = XsecAquiferDataMaq(ml2, x1=-np.inf, x2=0.0, kaq=[10], z = [12,12,0], c=[100], Saq=np.array([0.001])/12.0, Sll=0, topboundary='semi')
#aq2 = XsecAquiferDataMaq(ml2, x1=0, x2=np.inf, kaq=[10], z = [12,12,0], c=[100], Saq=np.array([0.001])/12.0, Sll=0, topboundary='semi')
#xinhom1 = XsecInhom(ml2, xinhom = 0.0)
#xlake = XsecLake(ml2, x1 = -np.inf, x2 = 0.0)
#ml2.solve()

#ml = ModelMaq(kaq=[1,2,3], z=[5,4,3,2,1,0], c=[100,100], Saq=[0.001,0.001,0.001], Sll=[0.001,0.001], tmin=0.01, tmax=10, M=0)
#aq1 = XsecAquiferDataMaq(ml, x1=-np.inf, x2=0.0, kaq=[1,2,3], z=[6,5,4,3,2,1,0], c=[100,100,100], Saq=[0.001,0.001,0.001], Sll=[0.001,0.001,0.001],topboundary='semi')
#aq2 = XsecAquiferDataMaq(ml, x1=0.0,  x2=100.0, kaq=[1,2,3], z=[5,4,3,2,1,0], c=[100,100], Saq=[0.001,0.001,0.001], Sll=[0.001,0.001])
#aq3 = XsecAquiferDataMaq(ml, x1=100.0,  x2=np.inf, kaq=[10,20,30], z=[5,4,3,2,1,0], c=[100,100], Saq=[0.001,0.001,0.001], Sll=[0.001,0.001])
#xinhom1 = XsecInhom(ml, xinhom = 0.0)
#xinhom2 = XsecInhom(ml, xinhom = 100.0)
#xlake = XsecLake(ml, x1 = -np.inf, x2 = 0.0)
#ml.solve()

#ml = ModelMaq(kaq=[1], z=[1,0], Saq=[0.001], tmin=0.1, tmax=1, M=20)
#aq1 = XsecAquiferDataMaq(ml, x1=-50.0, x2=50.0, kaq=[1], z=[2,1,0], c=[100], Saq=[0.001], topboundary='semi')
#aq2 = XsecAquiferDataMaq(ml, x1=-np.inf,  x2=-50.0, kaq=[1], z=[1,0], Saq=[0.001])
#aq3 = XsecAquiferDataMaq(ml, x1=50.0,  x2=np.inf, kaq=[1], z=[1,0], Saq=[0.001])
#xinhom1 = XsecInhom(ml, xinhom = -50.0)
#rxinhom2 = XsecInhom(ml, xinhom = 50.0)
#xlake = XsecLake(ml, x1 = -50.0, x2 = 50.0)
#ml.solve()


#ml = ModelMaq(kaq=[1,2,3], z=[5,4,3,2,1,0], c=[100,100], Saq=[0.001,0.001,0.001], Sll=[0.001,0.001], tmin=0.01, tmax=10, M=20)
#aq1 = XsecAquiferDataMaq(ml, x1=-np.inf, x2=-50.0, kaq=[1,2,3], z=[5,4,3,2,1,0], c=[100,100], Saq=[0.001,0.001,0.001], Sll=[0.001,0.001])
#aq2 = XsecAquiferDataMaq(ml, x1=-50.0,  x2=50.0, kaq=[1,2,3], z=[6,5,4,3,2,1,0], c=[100,100,100], Saq=[0.001,0.001,0.001], Sll=[0.001,0.001,0.001],topboundary='semi')
#aq3 = XsecAquiferDataMaq(ml, x1=50.0,  x2=np.inf, kaq=[1,2,3], z=[5,4,3,2,1,0], c=[100,100], Saq=[0.001,0.001,0.001], Sll=[0.001,0.001])
#xinhom1 = XsecInhom(ml, xinhom = -50.0)
#xinhom2 = XsecInhom(ml, xinhom =  50.0)
#xlake = XsecLake(ml, x1 = -50.0, x2 = 50.0)
#ml.solve()


##########################################

#S1 = 0.1
#S2 = 1e-3
#c = 100.0
#ml = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[c],Saq=[S1,S2],Sll=[1e-12],phreatictop=True,tmin=1,tmax=10,M=20)
##ml = ModelMaq(kaq=[4],z=[4,0],Saq=[S1],phreatictop=True,tmin=1,tmax=10,M=20)
#
#
#ca = CircAreaSink(ml,0,0,100,[0,1e-3])
#
#ml.initialize()
#ca.initialize()
#
#ml.solve()
#
##x = 12.0
##y = 3.0
##d = 1e-4
##p0 = ca.potinf(x,y)
##p1 = ca.potinf(x+d,y)
##p2 = ca.potinf(x,y+d)
##p3 = ca.potinf(x-d,y)
##p4 = ca.potinf(x,y-d)
##numlap = (p1+p2+p3+p4-4.0*p0)/d**2 - p0 / ml.aq.lab**2
##print 'numlap'
##print numlap
##print 'exact'
##print -ca.an
##qxnum = (p3-p1)/(2*d)
##qynum = (p4-p2)/(2*d)
##qx,qy = ca.disinf(x,y)
##print 'qxnum'
##print qxnum
##print 'qx'
##print qx
##print 'qynum'
##print qynum
##print 'qy'
##print qy
#x = 90.0
#y = 3.0
#d = 1e-1
#t = 2.0
#dt = 1e-2
#h0 = ml.head(x,y,t)[:,0]
#h1 = ml.head(x+d,y,t)[:,0]
#h2 = ml.head(x,y+d,t)[:,0]
#h3 = ml.head(x-d,y,t)[:,0]
#h4 = ml.head(x,y-d,t)[:,0]
#hmin = ml.head(x,y,t-dt)[:,0]
#hplus = ml.head(x,y,t+dt)[:,0]
#numlap = (h1+h2+h3+h4-4.0*h0)/(d**2)
#dhdt = (hplus-hmin)/(2*dt)
##qv = (h0[0]-h0[1]) / c


#ml1 = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=20)
#ml1 = ModelMaq(kaq=[4],z=[1,0],Saq=[1e-3],tmin=1,tmax=10,M=20)
#ls1 = LineSinkBase(ml1,type='g')
#ml1.solve()
#ml2 = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-1,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=20)
##ml2 = ModelMaq(kaq=[4],z=[1,0],Saq=[1e-3],tmin=1,tmax=10,M=20)
#ld1 = LineDoubletHoBase(ml2,order=2,type='g')
##ls2 = HeadLineSinkHo(ml2,order=5,layers=[1,2])
#ml2.solve()

#ml1 = ModelMaq(kaq=[3,4,5],z=[10,6,4,2,1,0],c=[200,100],Saq=[1e-1,1e-3,1e-4],Sll=[1e-5,1e-6],tmin=0.1,tmax=10,M=20)
#w = DischargeWell(ml1,0,20,.1,layers=[1])
#ld1 = LeakyLineDoublet(ml1,x1=-20,y1=0,x2=20,y2=0,res=4,order=2,layers=[1,2])  # Along x-axis
#ld2 = LeakyLineDoublet(ml1,x1=20,y1=0,x2=40,y2=20,res=4,order=2,layers=[1,2])  # 45 degree angle
#ml1.solve()
#
#ml2 = ModelMaq(kaq=[3,4,5],z=[10,6,4,2,1,0],c=[200,100],Saq=[1e-1,1e-3,1e-4],Sll=[1e-5,1e-6],tmin=0.1,tmax=10,M=20)
#w = DischargeWell(ml2,0,20,.1,layers=[1])
#lds = LeakyLineDoubletString(ml2,xy=[(-20,0),(20,0),(40,20)],res=4,order=2,layers=[1,2])
#ml2.solve()

#

#ml = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=20)
##ls = MscreenLineSinkDitchString(ml,[(-1,0),(0,0),(1,0)],tsandQ=[(0.0,1.0)],layers=[2])
#e1a = EllipseInhomDataMaq(ml,0,0,along=2.0,bshort=1.0,angle=0.0,kaq=[10,2],z=[4,2,1,0],c=[200],Saq=[2e-3,2e-4],Sll=[1e-5])
#e1 = EllipseInhom(ml,0,0,along=2.0,bshort=1.0,angle=0.0,order=5)
#e1 = EllipseInhomMaq(ml,0,0,along=2.0,bshort=1.0,angle=0.0,order=5,kaq=[10,2],z=[4,2,1,0],c=[200],Saq=[2e-3,2e-4],Sll=[1e-5])
## Same inside and outside
#c1 = CircInhomMaq(ml,0,0,2.0,order=5,kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6])
#c1 = CircInhomMaq(ml,0,0,2.0,order=5,kaq=[10,.1],z=[4,2,1,0],c=[200],Saq=[2e-3,2e-4],Sll=[1e-5])
##c2 = CircInhomMaq(ml,0,0,5000.0,order=1,kaq=[10,2],z=[4,2,1,0],c=[200],Saq=[2e-3,2e-4],Sll=[1e-5])
##ml.initialize()
##c2.circ_in_small[:] = 0
##c2.circ_out_small[:] = 0
#w = DischargeWell(ml,xw=.5,yw=0,rw=.1,tsandQ=[0,5.0],layers=1)
#ml.solve()

#ml.solve()       
#h1,h2 = np.zeros((2,e1.Ncp)), np.zeros((2,e1.Ncp))
#qn1,qn2 = np.zeros((2,e1.Ncp)), np.zeros((2,e1.Ncp))
#for i in range(e1.Ncp):
#    h1[:,i] = ml.head(e1.xc[i],e1.yc[i],2,aq=e1.aqin)[:,0]
#    h2[:,i] = ml.head(e1.xc[i],e1.yc[i],2,aq=e1.aqout)[:,0]
#    qx1,qy1 = ml.discharge(e1.xc[i],e1.yc[i],2,aq=e1.aqin)
#    qx2,qy2 = ml.discharge(e1.xc[i],e1.yc[i],2,aq=e1.aqout)
#    a = e1a.outwardnormalangle(e1.xc[i],e1.yc[i])
#    qn1[:,i] = qx1[:,0]*np.cos(a) + qy1[:,0]*np.sin(a)
#    qn2[:,i] = qx2[:,0]*np.cos(a) + qy2[:,0]*np.sin(a)



#ml = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.1,tmax=10)
#w1 = Well(ml,0,2,.1,tsandQ=[(0,10)],layers=[1])
#ls2 = ZeroHeadLineSinkString(ml,xy=[(-10,-2),(0,-4),(4,0)],layers=[1])
#ls1 = MscreenLineSinkDitchString(ml,xy=[(-10,0),(0,0),(10,10)],tsandQ=[(0.0,7.0)],res=0.0,wh='H',layers=[2],label=None)
#ml.solve()

#ml = ModelMaq([1,20,2],[25,20,18,10,8,0],c=[1000,2000],Saq=[0.1,1e-4,1e-4],Sll=[0,0],phreatictop=True,tmin=1e-6,tmax=10,M=30)
#w1 = Well(ml,0,0,.1,tsandQ=[(0,1000)],layers=[2])
#ls1 = ZeroMscreenLineSink(ml,10,-5,10,5,layers=[1,2,3],res=0.5,wh=1,vres=3,wv=1)
#w2 = ZeroMscreenWell(ml,10,0,res=1.0,layers=[1,2,3],vres=1.0)
#w3 = Well(ml,0,-10,.1,tsandQ=[(0,700)],layers=[2])
#ml.solve()
##ml1 = ModelMaq([1,20,2],[25,20,18,10,8,0],c=[1000,2000],Saq=[1e-4,1e-4,1e-4],Sll=[0,0],tmin=0.1,tmax=10000,M=30)
##w1 = Well(ml1,0,0,.1,tsandQ=[(0,1000)],layers=[2],res=0.1)
##ml1.solve()
#t = np.logspace(-1,3,100)
#h0 = ml.head(50,0,t)
##h1 = ml1.head(50,0,t)
##w = MscreenWell(ml,0,0,.1,tsandQ=[(0,1000),(100,0),(365,1000),(465,0)],layers=[2,3])
##w2 = HeadWell(ml,50,0,.2,tsandh=[(0,1)],layers=[2])
##y = [-500,-300,-200,-100,-50,0,50,100,200,300,500]
##x = 50 * np.ones(len(y))
##ls = ZeroHeadLineSinkString(ml,xy=zip(x,y),layers=[1])
##w = Well(ml,0,0,.1,tsandQ=[(0,1000),(100,0)],layers=[2])
##ml.solve()


#ml = Model3D( kaq=[2,1,5,10,4], z=[10,8,6,4,2,0], Saq=[.1,.0001,.0002,.0002,.0001], phreatictop=True, kzoverkh=0.1, tmin=1e-3, tmax=1e3 )
#w = MscreenWell(ml,0,-25,rw=.3,tsandQ=[(0,100),(100,50)],layers=[2,3])
#ml.solve()
    
##ml = Model3D(kaq=2.0,z=[10,5,0],Saq=[.002,.001],kzoverkh=0.2,phreatictop=False,tmin=.1,tmax=10,M=15)
#ml = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=100,tmax=300,M=50)
#w = HeadWellNew(ml,0,0,.1,tsandh=[(0.0,1.0)],layers=1)
#ml.solve()
##L1 = np.sqrt(10**2+5**2)
##ls1 = LineSink(ml,-10,-10,0,-5,tsandQ=[(0,.05*L1),(1,.02*L1)],res=1.0,layers=[1,2],label='mark1')
#w = MscreenWell(ml,-5,-5,.1,[0,5],layers=[1,2])
#L2 = np.sqrt(10**2+15**2)
#ls2 = LineSink(ml,0,-5,10,10,tsandQ=[(0,.03*L2),(2,.07*L2)],layers=[1],label='mark2')
##ls3a = ZeroHeadLineSink(ml,-10,5,-5,5,res=1.0,layers=[1,2])
##ls3b = ZeroHeadLineSink(ml,-5,5,0,5,res=1.0,layers=[1,2])
##ls3c = ZeroHeadLineSink(ml,0,5,5,5,res=1.0,layers=[1,2])
##lss = HeadLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
#lss = ZeroHeadLineSinkString(ml,[(-10,5),(-5,5),(0,5),(5,5)],res=1.0,layers=[1,2])
##lss = MscreenLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandQ=[(0,0.2),(3,0.1)],res=1.0,layers=[1,2])
##lss = ZeroMscreenLineSinkString(ml,[(-10,5),(-5,5),(0,5)],res=1.0,layers=[1,2])
##ml.initialize()
#ml.solve()
#print ml.potential(50,50,[0.5,5])

#ml2 = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.1,tmax=10,M=15)
#L1 = np.sqrt(10**2+5**2)
#ls1b = LineSink(ml2,-10,-10,0,-5,tsandQ=[(0,.05*L1),(1,.02*L1)],res=1.0,layers=[1,2],label='mark1')
#L2 = np.sqrt(10**2+15**2)
#ls2b = LineSink(ml2,0,-5,10,10,tsandQ=[(0,.03*L2),(2,.07*L2)],layers=[1],label='mark2')
##ls3a = HeadLineSink(ml2,-10,5,-5,5,tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
##ls3b = HeadLineSink(ml2,-5,5,0,5,tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
##ls3a = ZeroHeadLineSink(ml2,-10,5,-5,5,res=1.0,layers=[1,2])
##ls3b = ZeroHeadLineSink(ml2,-5,5,0,5,res=1.0,layers=[1,2])
##ls3a = MscreenLineSink(ml2,-10,5,-5,5,tsandQ=[(0,0.2),(3,0.1)],res=1.0,layers=[1,2])
##ls3b = MscreenLineSink(ml2,-5,5,0,5,tsandQ=[(0,0.2),(3,0.1)],res=1.0,layers=[1,2])
#ls3a = ZeroMscreenLineSink(ml2,-10,5,-5,5,res=1.0,layers=[1,2])
#ls3b = ZeroMscreenLineSink(ml2,-5,5,0,5,res=1.0,layers=[1,2])
##lssb = HeadLineSinkStringOld(ml2,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=0.0,layers=[1,2])
#ml2.solve()
#print ml2.potential(50,50,[0.5,5])

#lss = HeadLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
#lss = MscreenLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandQ=[(0,.03*5),(2,.07*5)],res=0.5,layers=[1,2])
#ls3a = MscreenLineSink(ml,-10,5,-5,5,tsandQ=[(0,.03*5),(2,.07*5)],res=0.5,layers=[1,2])
#ls3b = MscreenLineSink(ml,-5,5,0,5,tsandQ=[(0,.03*5),(2,.07*5)],res=0.5,layers=[1,2])
#
#ml2 = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.1,tmax=10,M=15)
#L1 = np.sqrt(10**2+5**2)
#ls1a = LineSink(ml2,-10,-10,0,-5,tsandQ=[(0,.05*L1),(1,.02*L1)],res=1.0,layers=[1,2],label='mark1')
#L2 = np.sqrt(10**2+15**2)
#ls2a = LineSink(ml2,0,-5,10,10,tsandQ=[(0,.03*L2),(2,.07*L2)],layers=[1],label='mark2')
#ls3a = HeadLineSink(ml2,-10,5,-5,5,tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
#ls3b = HeadLineSink(ml2,-5,5,0,5,tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])


##lss = HeadLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=0.0,layers=[1,2])
##ls3 = ZeroMscreenLineSink(ml,-10,5,0,5,res=1.0,layers=[1,2])
#ml = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.1,tmax=10,M=15)
#w1 = Well(ml,0,0,.1,tsandQ=[(0,5),(1,2)],res=1.0,layers=[1,2])
#w2 = Well(ml,100,0,.1,tsandQ=[(0,3),(2,7)],layers=[1])
##w3 = MscreenWell(ml,0,100,.1,tsandQ=[(0,2),(3,1)],res=2.0,layers=[1,2])
#w3 = ZeroMscreenWell(ml,0,100,.1,res=2.0,layers=[1,2])
##w3 = ZeroHeadWell(ml,0,100,.1,res=1.0,layers=[1,2])
##w3 = HeadWell(ml,0,100,.1,tsandh=[(0,2),(3,1)],res=1.0,layers=[1,2])
#ml.solve()
###print ml.potential(2,3,[.5,5])
#print ml.potential(50,50,[0.5,5])
#ml2.solve()
#print ml2.potential(50,50,[.5,5])
#print lss.strength([.5,5])
#
#ml2 = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=0.1,tmax=10,M=15)
#ls1a = LineSink(ml2,-10,-10,0,-5,tsandsig=[(0,.05),(1,.02)],res=1.0,layers=[1,2],label='mark1')
#ls2a = LineSink(ml2,0,-5,10,10,tsandsig=[(0,.03),(2,.07)],layers=[1],label='mark2')
#ls3a = HeadLineSinkStringOld(ml2,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=0.0,layers=[1,2])
#ml2.solve()
#print ml2.potential(50,50,[0.5,5])

#print 'Q from strength:  ',w3.strength(.5)
#print 'Q from head diff: ',(ml.head(w3.xc,w3.yc,.5)-w3.headinside(.5))/w3.res*2*np.pi*w3.rw*ml.aq.Haq[:,np.newaxis]
#print 'Q from head diff: ',(ml.head(w3.xc,w3.yc,.5)-2.0)/w3.res*2*np.pi*w3.rw*ml.aq.Haq[:,np.newaxis]
#print w3.strength([.5,5])
#print ls3.strength([.5,5])
#print sum(ls3.strength([.5,5]),0)
#Q = w3.strength([.5,5])
#print sum(Q,0)
#print ml.potential(w3.xc,w3.yc,[.5,5])