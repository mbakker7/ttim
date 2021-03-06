import numpy as np
import matplotlib.pyplot as plt
#from .invlap import *
import inspect # Used for storing the input
import sys
from .aquifer_parameters import param_3d, param_maq
from .aquifer import Aquifer
#from .bessel import *
from .invlapnumba import compute_laplace_parameters_numba, invlap, invlapcomp
from .util import PlotTtim

class TimModel(PlotTtim):
    def __init__(self, kaq=[1, 1], Haq=[1, 1], Hll=[0], c=[1e100, 100], 
                 Saq=[1e-4, 1e-4], Sll=[0], topboundary='conf', 
                 phreatictop=False, tmin=1, tmax=10, tstart=0, M=10, 
                 kzoverkh=None, model3d=False):
        self.elementlist = []
        self.elementdict = {}
        self.vbclist = []  # variable boundary condition 'v' elements
        self.zbclist = []  # zero and constant boundary condition 'z' elements
        self.gbclist = []  # given boundary condition 'g' elements
        # note: given bc elements don't have any unknowns
        self.tmin = tmin
        self.tmax = tmax
        self.tstart = tstart
        self.M = M
        self.aq = Aquifer(self, kaq, Haq, Hll, c, Saq, Sll, topboundary, 
                          phreatictop, kzoverkh, model3d)
        self.compute_laplace_parameters()
        self.name = 'TimModel'
        self.modelname = 'ml' # Used for writing out input
        
    def __repr__(self):
        return 'Model'
    
    def initialize(self):
        self.gvbclist = self.gbclist + self.vbclist
        self.vzbclist = self.vbclist + self.zbclist
        # Given elements are first in list
        self.elementlist = self.gbclist + self.vbclist + self.zbclist  
        self.ngbc = len(self.gbclist)
        self.nvbc = len(self.vbclist)
        self.nzbc = len(self.zbclist)
        self.ngvbc = self.ngbc + self.nvbc
        self.aq.initialize()
        for e in self.elementlist:
            e.initialize()
        # lists used for inverse transform
        enumber = []
        etstart = []
        ebc = []
        for k in range(self.ngvbc):
            e = self.gvbclist[k]
            enumber.extend(len(e.tstart) * [k])
            etstart.extend(list(e.tstart))
            ebc.extend(list(e.bc))
        self.enumber = np.array(enumber)
        self.etstart = np.array(etstart)
        self.ebc = np.array(ebc)
            
    def addelement(self, e):
        if e.label is not None: self.elementdict[e.label] = e
        if e.type == 'g':
            self.gbclist.append(e)
        elif e.type == 'v':
            self.vbclist.append(e)
        elif e.type == 'z':
            self.zbclist.append(e)
            
    def removeelement(self, e):
        if e.label is not None: self.elementdict.pop(e.label)
        if e.type == 'g':
            self.gbclist.remove(e)
        elif e.type == 'v':
            self.vbclist.remove(e)
        elif e.type == 'z':
            self.zbclist.remove(e)
            
    def addinhom(self, inhom):
        self.aq.inhomlist.append(inhom)
        
    def compute_laplace_parameters(self):
        '''
        nint: Number of time intervals
        npint: Number of p values per interval
        npval: Total number of p values (nint * npint)
        p[npval]: Array with p values
        '''
        itmin = np.floor(np.log10(self.tmin))
        itmax = np.ceil(np.log10(self.tmax))
        self.tintervals = 10.0 ** np.arange(itmin, itmax+1)
        # lower and upper limit are adjusted to prevent any problems from t 
        # exactly at the beginning and end of the interval
        # also, you cannot count on t >= 10 ** log10(t) for all possible t
        self.tintervals[0] = self.tintervals[0] * (1 - 1e-12)
        self.tintervals[-1] = self.tintervals[-1] * (1 + 1e-12)
        self.nint = len(self.tintervals) - 1 # number of p-intervals
        self.npint = 2 * self.M + 1 # number of p values in an interval
        self.npval = self.nint * self.npint
        # numba 
        self.p = np.zeros((self.nint, self.npint), dtype=np.complex128)
        for i in range(self.nint):
            self.p[i] = compute_laplace_parameters_numba(
                                        self.tintervals[i + 1], self.M)
        #TODO: make self.p a 2D array
        self.p = np.ravel(self.p)
        self.aq.initialize()
    
    def potential(self, x, y, t, layers=None, aq=None, derivative=0, 
                  returnphi=0):
        '''Returns pot[naq, ntimes] if layers=None, 
        otherwise pot[len(layers,Ntimes)]
        t must be ordered '''
        if aq is None: aq = self.aq.find_aquifer_data(x, y)
        if layers is None:
            layers = range(aq.naq)
        nlayers = len(layers)
        time = np.atleast_1d(t) - self.tstart # used to be ).copy()
        pot = np.zeros((self.ngvbc, aq.naq, self.npval), 'D')
        for i in range(self.ngbc):
            pot[i, :] += self.gbclist[i].unitpotential(x, y, aq)
        for e in self.vzbclist:
            pot += e.potential(x, y, aq)
        if layers is None:
            pot = np.sum(pot[:, np.newaxis, :, :] * aq.eigvec, 2)
        else:
            pot = np.sum(pot[:, np.newaxis, :, :] * aq.eigvec[layers, :], 2)
        if derivative > 0: 
            pot *= self.p ** derivative
        if returnphi:
            return pot
        rv = invlapcomp(time, pot, self.npint, self.M, self.tintervals, 
                        self.enumber, self.etstart, self.ebc, nlayers)
        return rv
    
    def disvec(self, x, y, t, layers=None, aq=None, derivative=0):
        '''Returns qx[naq, ntimes], qy[naq, ntimes] if layers=None, otherwise
        qx[len(layers,Ntimes)],qy[len(layers, ntimes)]
        t must be ordered '''
        if aq is None: aq = self.aq.find_aquifer_data(x, y)
        if layers is None:
            layers = range(aq.naq)
        nlayers = len(layers)
        time = np.atleast_1d(t) - self.tstart
        disx = np.zeros((self.ngvbc, aq.naq, self.npval), 'D')
        disy = np.zeros((self.ngvbc, aq.naq, self.npval), 'D')
        for i in range(self.ngbc):
            qx, qy = self.gbclist[i].unitdisvec(x, y, aq)
            disx[i, :] += qx
            disy[i, :] += qy
        for e in self.vzbclist:
            qx, qy = e.disvec(x, y, aq)
            disx += qx
            disy += qy
        if layers is None:
            disx = np.sum(disx[:, np.newaxis, :, :] * aq.eigvec, 2)
            disy = np.sum(disy[:, np.newaxis, :, :] * aq.eigvec, 2)
        else:
            disx = np.sum(disx[:, np.newaxis, :, :] * aq.eigvec[layers, :], 2)
            disy = np.sum(disy[:, np.newaxis, :, :] * aq.eigvec[layers, :], 2)
        if derivative > 0:
            disx *= self.p ** derivative
            disy *= self.p ** derivative
        rvx = invlapcomp(time, disx, self.npint, self.M, self.tintervals, 
                         self.enumber, self.etstart, self.ebc, nlayers)
        rvy = invlapcomp(time, disy, self.npint, self.M, self.tintervals, 
                         self.enumber, self.etstart, self.ebc, nlayers)
        return rvx, rvy
    
#     def disvecold(self, x, y, t, layers=None, aq=None, derivative=0):
#         '''Returns qx[naq, ntimes], qy[naq, ntimes] if layers=None, otherwise
#         qx[len(layers,Ntimes)],qy[len(layers,Ntimes)]
#         t must be ordered '''
#         if aq is None: aq = self.aq.find_aquifer_data(x, y)
#         if layers is None:
#             layers = range(aq.naq)
#         else:
#             layers = np.atleast_1d(layers)  # corrected for base zero
#         Nlayers = len(layers)
#         time = np.atleast_1d(t - self.tstart).copy()
#         disx = np.zeros((self.ngvbc, aq.naq, self.npval), 'D')
#         disy = np.zeros((self.ngvbc, aq.naq, self.npval), 'D')
#         for i in range(self.ngbc):
#             qx,qy = self.gbclist[i].unitdisvec(x, y, aq)
#             disx[i,:] += qx; disy[i,:] += qy
#         for e in self.vzbclist:
#             qx,qy = e.disvec(x, y, aq)
#             disx += qx; disy += qy
#         if layers is None:
#             disx = np.sum(disx[:, np.newaxis, :, :] * aq.eigvec, 2)
#             disy = np.sum(disy[:, np.newaxis, :, :] * aq.eigvec, 2)
#         else:
#             disx = np.sum(disx[:, np.newaxis, :, :] * aq.eigvec[layers, :], 2)
#             disy = np.sum(disy[:, np.newaxis, :, :] * aq.eigvec[layers, :], 2)
#         if derivative > 0:
#             disx *= self.p ** derivative
#             disy *= self.p ** derivative
#         rvx = np.zeros((Nlayers, len(time)))
#         rvy = np.zeros((Nlayers, len(time)))
#         if (time[0] < self.tintervals[0]) or (time[-1] > self.tintervals[-1]):
#             print('Warning, some of the times are smaller than tmin or' + 
#                   'larger than tmax; zeros are substituted')
#         #
#         for k in range(self.ngvbc):
#             e = self.gvbclist[k]
#             for itime in range(e.ntstart):
#                 t = time - e.tstart[itime]
#                 it = 0
#                 if t[-1] >= self.tintervals[0]:  # Otherwise all zero
#                     if (t[0] < self.tintervals[0]):
#                         # clever call that should be replaced with find_first 
#                         # function when included in numpy
#                         it = np.argmax( t >= self.tintervals[0])  
#                     for n in range(self.nint):
#                         tp = t[(t >= self.tintervals[n]) & 
#                                (t < self.tintervals[n+1])]
#                         Nt = len(tp)
#                         if Nt > 0:  # if all values zero, no inverse transform
#                             for i in range(Nlayers):
#                                 if not np.any(disx[k, i, n * self.npint:
#                                               (n + 1) * self.npint] == 0.0) :
#                                     rvx[i, it: it + Nt] += e.bc[itime] * \
#                                         invlap(tp, self.tintervals[n + 1], 
#                                         disx[k, i , n * self.npint:
#                                              (n + 1) * self.npint],
#                                         self.M)
#                                 if not np.any(disy[k, i, n * self.npint:
#                                               (n + 1) * self.npint] == 0.0) :
#                                     rvy[i, it: it + Nt] += e.bc[itime] * \
#                                         invlap(tp, self.tintervals[n + 1], 
#                                         disy[k, i , n * self.npint:
#                                              (n + 1) * self.npint],
#                                         self.M)
#                             it = it + Nt
#         return rvx, rvy
    
    def head(self, x, y, t, layers=None, aq=None, derivative=0):
        """Head at x, y, t where t can be multiple times
        
        Parameters
        ----------
        x : float
        y : float
        t : list or array
            times for which grid is returned
        layers : integer, list or array, optional
            layers for which grid is returned
            if None: all layers are returned
        
        Returns
        -------
        h : array size `nlayers, ntimes`

        """
        
        if aq is None: aq = self.aq.find_aquifer_data(x, y)
        if layers is None:
            layers = range(aq.naq)
        else:
            layers = np.atleast_1d(layers)  # corrected for base zero
        pot = self.potential(x, y, t, layers, aq, derivative)
        return aq.potential_to_head(pot, layers)
    
    def headinside(self, elabel, t):
        return self.elementdict[elabel].headinside(t - self.tstart)
    
    def strength(self,elabel,t):
        return self.elementdict[elabel].strength(t - self.tstart)
    
    def headalongline(self, x, y, t, layers=None):
        """Head along line or curve
        
        Parameters
        ----------
        x : 1D array or list
            x values of line
        y : 1D array or list
            y values of line
        t : float or 1D array or list
            times for which grid is returned
        layers : integer, list or array, optional
            layers for which grid is returned
        
        Returns
        -------
        h : array size `nlayers, ntimes, nx`

        """
        
        xg = np.atleast_1d(x)
        yg = np.atleast_1d(y)
        if layers is None:
            Nlayers = self.aq.find_aquifer_data(xg[0], yg[0]).naq
        else:
            Nlayers = len(np.atleast_1d(layers))
        nx = len(xg)
        if len(yg) == 1:
            yg = yg * np.ones(nx)
        t = np.atleast_1d(t)
        h = np.zeros( (Nlayers,len(t),nx) )
        for i in range(nx):
            h[:,:,i] = self.head(xg[i], yg[i], t, layers)
        return h
    
    def headgrid(self, xg, yg, t, layers=None, printrow=False):
        """Grid of heads
        
        Parameters
        ----------
        xg : array
            x values of grid
        yg : array
            y values of grid
        t : list or array
            times for which grid is returned
        layers : integer, list or array, optional
            layers for which grid is returned
        printrow : boolean, optional
            prints dot to screen for each row of grid if set to `True`
        
        Returns
        -------
        h : array size `nlayers, ntimes, ny, nx`
        
        See also
        --------
        
        :func:`~ttim.model.Model.headgrid2`

        """
        
        nx = len(xg)
        ny = len(yg)
        if layers is None:
            nlayers = self.aq.find_aquifer_data(xg[0], yg[0]).naq
        else:
            nlayers = len(np.atleast_1d(layers))
        t = np.atleast_1d(t)
        h = np.empty( (nlayers,len(t),ny,nx) )
        for j in range(ny):
            if printrow:
                print('.', end='', flush=True)
            for i in range(nx):
                h[:, :, j, i] = self.head(xg[i], yg[j], t, layers)
        return h
    
    def headgrid2(self, x1, x2, nx, y1, y2, ny, t, layers=None, printrow=False):
        """Grid of heads
        
        Parameters
        ----------
        xg : array
            x values are generated as linspace(x1, x2, nx)
        yg : array
            y values are generated as linspace(y1, y2, ny)
        t : list or array
            times for which grid is returned
        layers : integer, list or array, optional
            layers for which grid is returned
        printrow : boolean, optional
            prints dot to screen for each row of grid if set to `True`
        
        Returns
        -------
        h : array size `nlayers, ntimes, ny, nx`
        
        See also
        --------
        
        :func:`~ttim.model.Model.headgrid`

        """
        
        xg = np.linspace(x1, x2, nx)
        yg = np.linspace(y1, y2, ny)
        return self.headgrid(xg, yg, t, layers, printrow)
    
    def inverseLapTran(self, pot, t):
        '''returns array of potentials of len(t)
        t must be ordered and tmin <= t <= tmax'''
        t = np.atleast_1d(t)
        rv = np.zeros(len(t))
        it = 0
        if t[-1] >= self.tmin:  # Otherwise all zero
            if (t[0] < self.tmin): it = np.argmax( t >= self.tmin )  
            for n in range(self.nint):
                if n == self.nint-1:
                    tp = t[(t >= self.tintervals[n]) & 
                           (t <= self.tintervals[n+1])]
                else:
                    tp = t[(t >= self.tintervals[n]) & 
                           (t < self.tintervals[n+1])]
                nt = len(tp)
                if nt > 0:  # if all values zero, don't do inverse transform
                    # Not needed anymore: if np.abs(pot[n*self.npint]) > 1e-20:
                    # If there is a zero item, zero should be returned; 
                    # funky enough this can be done with a 
                    # straight equal comparison
                    if not np.any(pot[n*self.npint:(n+1)*self.npint] == 0.0):
                        rv[it : it + nt] = invlap(tp, 
                            self.tintervals[n + 1], 
                            pot[n * self.npint: (n + 1) * self.npint],
                            self.M)
                    it = it + nt
        return rv
    
    def solve(self, printmat=0, sendback=0, silent=False):
        """Compute solution
        
        """
        
        # Initialize elements
        self.initialize()
        # Compute number of equations
        self.neq = np.sum([e.nunknowns for e in self.elementlist])
        if silent is False:
            print('self.neq ', self.neq)
        if self.neq == 0:
            if silent is False:
                print('No unknowns. Solution complete')
            return
        mat = np.empty((self.neq, self.neq, self.npval), 'D')
        rhs = np.empty((self.neq, self.ngvbc, self.npval), 'D')
        ieq = 0
        for e in self.elementlist:
            if e.nunknowns > 0:
                mat[ieq: ieq + e.nunknowns, :, :], \
                rhs[ieq: ieq + e.nunknowns, :, :] = e.equation()
                ieq += e.nunknowns
        if printmat:
            return mat, rhs
        for i in range(self.npval):
            sol = np.linalg.solve(mat[:, :, i], rhs[:, :, i])
            icount = 0
            for e in self.elementlist:
                for j in range(e.nunknowns):
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
                rv += key + ' = ' + np.array2string(self.inputvalues[key],
                                                    separator=',') + ',\n'
            elif isinstance(self.inputvalues[key],str):                
                rv += key + " = '" + self.inputvalues[key] + "',\n"
            else:
                rv += key + ' = ' + str(self.inputvalues[key]) + ',\n'
        rv += ')\n'
        return rv
    
    def writemodel(self,fname):
        self.initialize()  # So that model can be written without solving first
        f = open(fname,'w')
        f.write('from ttim import *\n')
        f.write( self.write() )
        for e in self.elementlist:
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
        the maximum time for which heads can be computed
    tstart : scalar
        time at start of simulation (default 0)
    M : integer
        the number of terms to be used in the numerical inversion algorithm.
        10 is usually sufficient. If drawdown curves appear to oscillate,
        more terms may be needed, but this seldom happens. 
    
    """
    
    def __init__(self, kaq=[1], z=[1,0], c=[], Saq=[0.001], Sll=[0], \
                 topboundary='conf', phreatictop=False, \
                 tmin=1, tmax=10, tstart=0, M=10):
        self.storeinput(inspect.currentframe())
        kaq, Haq, Hll, c, Saq, Sll = param_maq(kaq, z, c, Saq, Sll, topboundary,
                                               phreatictop)
        TimModel.__init__(self, kaq, Haq, Hll, c, Saq, Sll, topboundary, 
                          phreatictop, tmin, tmax, tstart, M)
        self.name = 'ModelMaq'
        
class Model3D(TimModel):
    """
    Create a multi-layer model object consisting of
    many aquifer layers. The resistance between the layers is computed
    from the vertical hydraulic conductivity of the layers.
    
    Parameters
    ----------
    kaq : float, array or list
        hydraulic conductivity of each layer from the top down
        if float, hydraulic conductivity is the same in all aquifers
    z : array or list
        elevation of top of system followed by bottoms of all layers
        from the top down
        bottom of layer is automatically equal to top of layer below it
        if topboundary='conf': length is number of layers + 1
        if topboundary='semi': length is number of layers + 2 as top
        of leaky layer on top of systems needs to be specified
    Saq : float, array or list
        specific storage of all aquifers layers
        if float, sepcific storage is same in all aquifers layers
        if phreatictop is True and topboundary is 'conf', Saq of top
        aquifer is phreatic storage coefficient (and not multiplied
        with the layer thickness)
    kzoverkh : float
        vertical anisotropy ratio vertical k divided by horizontal k
        if float, value is the same for all layers
        length is number of layers
    topboundary : string, 'conf' or 'semi' (default is 'conf')
        indicating whether the top is confined ('conf') or
        semi-confined ('semi').
        currently only implemented for 'conf'
    topres : float
        resistance of top semi-confining layer, only read if topboundary='semi'
    topthick: float
        thickness of top semi-confining layer, only read if topboundary='semi'
    phreatictop : boolean
        the storage coefficient of the top aquifer layer is treated as
        phreatic storage (and not multiplied with the aquifer thickness)
    tmin : scalar
        the minimum time for which heads can be computed after any change
        in boundary condition.
    tmax : scalar
        the maximum time for which heads can be computed.
    tstart : scalar
        time at start of simulation (default 0)
    M : integer (default 10)
        the number of terms to be used in the numerical inversion algorithm.
        10 is usually sufficient. If drawdown curves appear to oscillate,
        more terms may be needed, but this seldom happens. 
        
    """
    
    def __init__(self, kaq=1, z=[4, 3, 2, 1], Saq=0.001, kzoverkh=0.1, \
                 topboundary='conf', phreatictop=True, topres=0, topthick=0, 
                 topSll=0, tmin=1, tmax=10, tstart=0, M=10):
        '''z must have the length of the number of layers + 1'''
        self.storeinput(inspect.currentframe())
        kaq, Haq, Hll, c, Saq, Sll = param_3d(kaq, z, Saq, kzoverkh, 
                                              phreatictop, topboundary, topres, 
                                              topthick, topSll)
        TimModel.__init__(self, kaq, Haq, Hll, c, Saq, Sll, topboundary, 
                          phreatictop, tmin, tmax, tstart, M, 
                          kzoverkh, model3d=True)
        self.name = 'Model3D'