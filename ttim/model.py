import numpy as np
import matplotlib.pyplot as plt
from .invlap import *
import inspect # Used for storing the input
import sys
from .aquifer_parameters import param_3d, param_maq
from .aquifer import Aquifer

class TimModel:
    def __init__(self, kaq=[1, 1], Haq=[1, 1], c=[1e100, 100], Saq=[0.3, 0.003], \
                 Sll=[0], topboundary='conf', phreatictop=False, tmin=1, tmax=10, M=20):
        self.elementList = []
        self.elementDict = {}
        self.vbcList = []  # List with variable boundary condition 'v' elements
        self.zbcList = []  # List with zero and constant boundary condition 'z' elements
        self.gbcList = []  # List with given boundary condition 'g' elements; given bc elements don't have any unknowns
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.M = M
        self.aq = Aquifer(self, kaq, Haq, c, Saq, Sll, topboundary, phreatictop)
        self.compute_laplace_parameters()
        self.name = 'TimModel'
        self.modelname = 'ml' # Used for writing out input
        #bessel.initialize()
        
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
        Npin: Number of p values per interval
        Np: Total number of p values (Nin*Np)
        p[Np]: Array with p values
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
        
    def potential(self, x, y, t, pylayers=None, aq=None, derivative=0, returnphi=0):
        '''Returns pot[Naq,Ntimes] if layers=None, otherwise pot[len(pylayers,Ntimes)]
        t must be ordered '''
        if aq is None: aq = self.aq.findAquiferData(x, y)
        if pylayers is None:
            pylayers = range(aq.Naq)
        Nlayers = len(pylayers)
        time = np.atleast_1d(t).copy()
        pot = np.zeros((self.Ngvbc, aq.Naq, self.Np), 'D')
        for i in range(self.Ngbc):
            pot[i, :] += self.gbcList[i].unitpotential(x, y, aq)
        for e in self.vzbcList:
            pot += e.potential(x, y, aq)
        if pylayers is None:
            pot = np.sum(pot[:, np.newaxis, :, :] * aq.eigvec, 2)
        else:
            pot = np.sum(pot[:, np.newaxis, :, :] * aq.eigvec[pylayers, :], 2 )
        if derivative > 0: pot *= self.p ** derivative
        if returnphi:
            return pot
        rv = np.zeros((Nlayers, len(time)))
        if (time[0] < self.tmin) or (time[-1] > self.tmax):
            print('Warning, some of the times are smaller than tmin or larger than tmax; zeros are substituted')
        #
        for k in range(self.Ngvbc):
            e = self.gvbcList[k]
            for itime in range(e.Ntstart):
                t = time - e.tstart[itime]
                it = 0
                if t[-1] >= self.tmin:  # Otherwise all zero
                    if (t[0] < self.tmin):
                        it = np.argmax(t >= self.tmin)  # clever call that should be replaced with find_first function when included in numpy
                    for n in range(self.Nin):
                        tp = t[(t >= self.tintervals[n]) & (t < self.tintervals[n+1])]
                        ## I think these lines are not needed anymore as I modified tintervals[0] and tintervals[-1] by eps
                        #if n == self.Nin-1:
                        #    tp = t[ (t >= self.tintervals[n]) & (t <= self.tintervals[n+1]) ]
                        #else:
                        #    tp = t[ (t >= self.tintervals[n]) & (t < self.tintervals[n+1]) ]
                        Nt = len(tp)
                        if Nt > 0:  # if all values zero, don't do the inverse transform
                            for i in range(Nlayers):
                                # I used to check the first value only, but it seems that checking that nothing is zero is needed and should be sufficient
                                #if np.abs( pot[k,i,n*self.Npin] ) > 1e-20:  # First value very small
                                if not np.any(pot[k, i, n * self.Npin: (n + 1) * self.Npin] == 0) : # If there is a zero item, zero should be returned; funky enough this can be done with a straight equal comparison
                                    rv[i, it:it + Nt] += e.bc[itime] * \
                                    invlaptrans.invlap(tp, self.tintervals[n],
                                                       self.tintervals[n + 1],
                                                       pot[k, i , n * self.Npin:(n + 1) * self.Npin],
                                                       self.gamma[n], self.M, Nt)
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
        if (time[0] < self.tmin) or (time[-1] > self.tmax):
            print('Warning, some of the times are smaller than tmin or larger than tmax; zeros are substituted')
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
        '''Returns head[Nlayers,len(t),len(x)]
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
        '''Returns h[Nlayers,Ntimes,Ny,Nx]. If layers is None, all layers are returned'''
        xg,yg = np.linspace(x1,x2,nx), np.linspace(y1,y2,ny)
        if layers is None:
            Nlayers = self.aq.findAquiferData(xg[0],yg[0]).Naq
        else:
            Nlayers = len(np.atleast_1d(layers))
        t = np.atleast_1d(t)
        h = np.empty( (Nlayers,len(t),ny,nx) )
        for j in range(ny):
            if printrow:
                print(str(j)+' ')
            for i in range(nx):
                h[:,:,j,i] = self.head(xg[i],yg[j],t,layers)
        return h
    
    def headgrid2(self,xg,yg,t,layers=None,printrow=False):
        '''Returns h[Nlayers,Ntimes,Ny,Nx]. If layers is None, all layers are returned'''
        nx,ny = len(xg), len(yg)
        if layers is None:
            Nlayers = self.aq.findAquiferData(xg[0],yg[0]).Naq
        else:
            Nlayers = len(np.atleast_1d(layers))
        t = np.atleast_1d(t)
        h = np.empty( (Nlayers,len(t),ny,nx) )
        for j in range(ny):
            if printrow:
                print(str(j)+' ')
            for i in range(nx):
                h[:,:,j,i] = self.head(xg[i],yg[j],t,layers)
        return h
    
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
                    # Not needed anymore: if np.abs( pot[n*self.Npin] ) > 1e-20:
                    if not np.any( pot[n*self.Npin:(n+1)*self.Npin] == 0.0) : # If there is a zero item, zero should be returned; funky enough this can be done with a straight equal comparison
                        rv[it:it+Nt] = invlaptrans.invlap( tp, self.tintervals[n], self.tintervals[n+1], pot[n*self.Npin:(n+1)*self.Npin], self.gamma[n], self.M, Nt )
                    it = it + Nt
        return rv
    
    def solve(self,printmat=0, sendback=0, silent=False):
        '''Compute solution'''
        # Initialize elements
        self.initialize()
        # Compute number of equations
        self.Neq = np.sum([e.Nunknowns for e in self.elementList])
        if silent is False:
            print('self.Neq ', self.Neq)
        if self.Neq == 0:
            if silent is False:
                print('No unknowns. Solution complete')
            return
        mat = np.empty((self.Neq, self.Neq, self.Np), 'D')
        rhs = np.empty((self.Neq, self.Ngvbc, self.Np), 'D')
        ieq = 0
        for e in self.elementList:
            if e.Nunknowns > 0:
                mat[ieq:ieq+e.Nunknowns, :, :], rhs[ieq:ieq+e.Nunknowns, :, :] = e.equation()
                ieq += e.Nunknowns
        if printmat:
            return mat, rhs
        for i in range(self.Np):
            sol = np.linalg.solve(mat[:, :, i], rhs[:, :, i])
            icount = 0
            for e in self.elementList:
                for j in range(e.Nunknowns):
                    e.parameters[:, j, i] = sol[icount, :]
                    icount += 1
                e.run_after_solve()
        if silent is False:
            print('solution complete')
        elif (silent == 'dot') or (silent == '.'):
            print('.', end='', flush=True)
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
        
class ModelMaq(TimModel):
    """
    Create a Model object by specifying a mult-aquifer sequence of
    aquifer-leakylayer-aquifer-leakylayer-aquifer etc
    
    Parameters
    ----------
    kaq : float, array or list
        hydraulic conductivity of each aquifer from the top down
        if float, hydraulic conductivity is the same in all aquifers
    z : array or list
        elevation tops and bottoms of the aquifers from the top down
        leaky layers may have zero thickness
        if top='conf': length is 2 * number of aquifers
        if top='semi': length is 2 * number of aquifers + 1 as top
        of leaky layer on top of systems needs to be specified
    c : float, array or list
        resistance of leaky layers from the top down
        if float, resistance is the same for all leaky layers
        if top='conf': length is number of aquifers - 1
        if top='semi': length is number of aquifers
    Saq : float, array or list
        specific storage of all aquifers
        if float, sepcific storage is same in all aquifers
        if phreatictop is True and topboundary is 'conf', Saq of top
        aquifer is phreatic storage coefficient (and not multiplied
        with the layer thickness)
    Sll : float, array or list
        specific storage of all leaky layers
        if float, sepcific storage is same in all leaky layers
        if phreatictop is True and topboundary is 'semi', Sll of top
        leaky layer is phreatic storage coefficient (and not multiplied
        with the layer thickness)
    topboundary : string, 'conf' or 'semi' (default is 'conf')
        indicating whether the top is confined ('conf') or
        semi-confined ('semi')
    phreatictop : boolean
        the storage coefficient of the top model layer is treated as
        phreatic storage (and not multiplied with the aquifer thickness)
    tmin : scalar
        the minimum time for which heads can be computed after any change
        in boundary condition.
    tmax : scalar
        the maximum time for which heads can be computed.
    M : integer
        the number of terms to be used in the numerical inversion algorithm.
        20 is usually sufficient. If drawdown curves appear to oscillate,
        more terms may be needed, but this seldom happens. 
    
    """
    
    def __init__(self, kaq=[1], z=[1,0], c=[], Saq=[0.001], Sll=[0], \
                 topboundary='conf', phreatictop=False, \
                 tmin=1, tmax=10, M=20):
        self.storeinput(inspect.currentframe())
        kaq, Haq, c, Saq, Sll = param_maq(kaq, z, c, Saq, Sll, topboundary, phreatictop)
        TimModel.__init__(self, kaq, Haq, c, Saq, Sll, topboundary, phreatictop, tmin, tmax, M)
        self.name = 'ModelMaq'
        
class Model3D(TimModel):
    def __init__(self, kaq=1, z=[4, 3, 2, 1], Saq=0.001, kzoverkh=0.1, \
                 topboundar='conf', phreatictop=True, tmin=1, tmax=10, M=20):
        '''z must have the length of the number of layers + 1'''
        self.storeinput(inspect.currentframe())
        kaq, Haq, c, Saq, Sll = param_3d(kaq, z, Saq, kzoverkh, phreatictop, topboundary)
        TimModel.__init__(self, kaq, Haq, c, Saq, Sll, topboundary, phreatictop, tmin, tmax, M)
        self.name = 'Model3D'