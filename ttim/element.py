import numpy as np
import inspect # Used for storing the input

class Element:
    def __init__(self, model, Nparam=1, Nunknowns=0, layers=0, \
                 tsandbc=[(0, 0)], type='z', name='', label=None):
        '''Types of elements
        'g': strength is given through time
        'v': boundary condition is variable through time
        'z': boundary condition is zero through time
        Definition of Nlayers, Ncp, Npar, Nunknowns:
        Nlayers: Number of layers that the element is screened in, set in Element
        Ncp: Number of control points along the element
        Nparam: Number of parameters, commonly Nlayers * Ncp
        Nunknowns: Number of unknown parameters, commonly zero or Npar
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
        self.tstart,self.bcin = tsandbc[:,0], tsandbc[:,1]
        if self.tstart[0] > 0:
            self.tstart = np.hstack((np.zeros(1), self.tstart))
            self.bcin = np.hstack((np.zeros(1), self.bcin))
        #
        self.type = type  # 'z' boundary condition through time or 'v' boundary condition through time
        self.name = name
        self.label = label
        if self.label is not None:
            assert self.label not in self.model.elementDict.keys(), \
                   "TTim error: label " + self.label + " already exists"
        self.Rzero = 30
        
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
    
    def potinf(self, x, y, aq=None):
        '''Returns complex array of size (Nparam, Naq, Np)'''
        raise 'Must overload Element.potinf()'
    
    def potential(self, x, y, aq=None):
        '''Returns complex array of size (Ngvbc, Naq, Np)'''
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        return np.sum(self.parameters[:, :, np.newaxis, :] * \
                      self.potinf(x, y, aq), 1)
    
    def unitpotential(self, x, y, aq=None):
        '''Returns complex array of size (Naq, Np)
        Can be more efficient for given elements'''
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        return np.sum(self.potinf(x, y, aq), 0)
    
    def disinf(self, x, y, aq=None):
        '''Returns 2 complex arrays of size (Nparam, Naq, Np)'''
        raise 'Must overload Element.disinf()'
    
    def discharge(self, x, y, aq=None):
        '''Returns 2 complex arrays of size (Ngvbc, Naq, Np)'''
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        qx, qy = self.disinf(x, y, aq)
        return np.sum(self.parameters[:, :, np.newaxis, :] * qx, 1), \
               np.sum( elf.parameters[:, :, np.newaxis, :] * qy, 1 )
    
    def unitdischarge(self, x, y, aq=None):
        '''Returns 2 complex arrays of size (Naq, Np)
        Can be more efficient for given elements'''
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        qx, qy = self.disinf(x, y, aq)
        return np.sum(qx, 0), np.sum(qy, 0)
    
    # Functions used to build equations
    def potinflayers(self,x,y,pylayers=0,aq=None):
        '''pylayers can be scalar, list, or array. returns array of size (len(pylayers),Nparam,Np)
        only used in building equations'''
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        pot = self.potinf(x, y, aq)
        rv = np.sum(pot[:, np.newaxis, :, :] * aq.eigvec, 2)
        rv = rv.swapaxes(0, 1) # As the first axes needs to be the number of layers
        return rv[pylayers, :]
    
    def potentiallayers(self, x, y, pylayers=0, aq=None):
        '''Returns complex array of size (Ngvbc, len(pylayers),Np)
        only used in building equations'''
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        pot = self.potential(x, y, aq)
        phi = np.sum(pot[:, np.newaxis, :, :] * aq.eigvec, 2)
        return phi[:, pylayers, :]
    
    def unitpotentiallayers(self, x, y, pylayers=0, aq=None):
        '''Returns complex array of size (len(pylayers), Np)
        only used in building equations'''
        if aq is None:
            aq = self.model.aq.findAquiferData(x,y)
        pot = self.unitpotential(x, y, aq)
        phi = np.sum(pot[np.newaxis, :, :] * aq.eigvec, 1)
        return phi[pylayers, :]
    
    def disinflayers(self, x, y, pylayers=0, aq=None):
        '''pylayers can be scalar, list, or array. returns 2 arrays of size (len(pylayers),Nparam,Np)
        only used in building equations'''
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        qx, qy = self.disinf(x, y, aq)
        rvx = np.sum(qx[:, np.newaxis, :, :] * aq.eigvec, 2)
        rvy = np.sum(qy[:, np.newaxis, :, :] * aq.eigvec, 2)
        rvx = rvx.swapaxes(0, 1)
        rvy = rvy.swapaxes(0, 1) # As the first axes needs to be the number of layers
        return rvx[pylayers, :], rvy[pylayers, :]
    
    def dischargelayers(self, x, y, pylayers=0, aq=None):
        '''Returns 2 complex array of size (Ngvbc, len(pylayers), Np)
        only used in building equations'''
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        qx, qy = self.discharge(x, y, aq)
        rvx = np.sum(qx[:, np.newaxis, :, :] * aq.eigvec, 2)
        rvy = np.sum(qy[:, np.newaxis, :, :] * aq.eigvec, 2)
        return rvx[:, pylayers, :], rvy[:, pylayers, :]
    
    def unitdischargelayers(self, x, y, pylayers=0, aq=None):
        '''Returns complex array of size (len(pylayers), Np)
        only used in building equations'''
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        qx, qy = self.unitdischarge(x, y, aq)
        rvx = np.sum(qx[np.newaxis, :, :] * aq.eigvec, 1)
        rvy = np.sum(qy[np.newaxis, :, :] * aq.eigvec, 1)
        return rvx[pylayers, :], rvy[pylayers, :]
    
    # Other functions
    def discharge(self, t, derivative=0):
        '''returns array of discharges (Nlayers,len(t)) t must be ordered and tmin <= t <= tmax'''
        # Could potentially be more efficient if s is pre-computed for all elements, but I don't know if that is worthwhile to store as it is quick now
        time = np.atleast_1d(t).astype('d')
        if (time[0] < self.model.tmin) or (time[-1] > self.model.tmax):
            print('Warning, some of the times are smaller than tmin or larger than tmax; zeros are substituted')
        rv = np.zeros((self.Nlayers, np.size(time)))
        if self.type == 'g':
            s = self.dischargeinflayers * self.model.p ** derivative
            for itime in range(self.Ntstart):
                time -=  self.tstart[itime]
                for i in range(self.Nlayers):
                    rv[i] += self.bc[itime] * self.model.inverseLapTran(s[i], time)
        else:
            s = np.sum(self.parameters[:, :, np.newaxis, :] * self.dischargeinf, 1)
            s = np.sum(s[:, np.newaxis, :, :] * self.aq.eigvec, 2)
            s = s[:, self.pylayers, :] * self.model.p ** derivative
            for k in range(self.model.Ngvbc):
                e = self.model.gvbcList[k]
                for itime in range(e.Ntstart):
                    t = time - e.tstart[itime]
                    if t[-1] >= self.model.tmin:  # Otherwise all zero
                        for i in range(self.Nlayers):
                            rv[i] += e.bc[itime] * self.model.inverseLapTran(s[k, i], t)
        return rv
        
    def headinside(self, t):
        print("This function not implemented for this element")
        return
    
    def storeinput(self, frame):
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
    
    def plot(self):
        pass
        