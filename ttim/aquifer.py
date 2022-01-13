import numpy as np
import matplotlib.pyplot as plt
import inspect # Used for storing the input

class AquiferData:
    def __init__(self, model, kaq, z, Haq, Hll, c, Saq, Sll, poraq, porll, 
                 ltype, topboundary, phreatictop, kzoverkh=None, model3d=False):
        '''kzoverkh and model3d only need to be specified when model
        is model3d'''
        self.model = model
        self.kaq = np.atleast_1d(kaq).astype('d')
        self.z = np.atleast_1d(z).astype('d')
        self.naq = len(self.kaq)
        self.nlayers = len(self.z) - 1
        self.Haq = np.atleast_1d(Haq).astype('d')
        self.Hll = np.atleast_1d(Hll).astype('d')
        self.T = self.kaq * self.Haq
        self.Tcol = self.T.reshape(self.naq, 1)
        self.c = np.atleast_1d(c).astype('d')
        self.c[self.c > 1e100] = 1e100 
        self.Saq = np.atleast_1d(Saq).astype('d')
        self.Sll = np.atleast_1d(Sll).astype('d')
        self.Sll[self.Sll < 1e-20] = 1e-20 # Cannot be zero
        self.poraq = np.atleast_1d(poraq).astype('d')
        self.porll = np.atleast_1d(porll).astype('d')
        self.ltype = np.atleast_1d(ltype)
        self.zaqtop = self.z[:-1][self.ltype == 'a']
        self.zaqbot = self.z[1:][self.ltype == 'a']
        self.layernumber = np.zeros(self.nlayers, dtype='int')
        self.layernumber[self.ltype == 'a'] = np.arange(self.naq)
        self.layernumber[self.ltype == 'l'] = np.arange(self.nlayers - self.naq)
        if self.ltype[0] == 'a':
            self.layernumber[self.ltype == 'l'] += 1  # first leaky layer below first aquifer layer
        self.topboundary = topboundary[:3]
        self.phreatictop = phreatictop
        self.kzoverkh = kzoverkh
        if self.kzoverkh is not None:
            self.kzoverkh = np.atleast_1d(self.kzoverkh).astype('d')
            if len(self.kzoverkh) == 1: 
                self.kzoverkh = self.kzoverkh * np.ones(self.naq)
        self.model3d = model3d
        if self.model3d:
            assert self.kzoverkh is not None, \
                "model3d specified without kzoverkh"
        #self.D = self.T / self.Saq
        self.area = 1e200 # Smaller than default of ml.aq so that inhom is found
    
    def __repr__(self):
        return 'Inhom T: ' + str(self.T)
    
    def initialize(self):
        '''
        eigval[naq, npval]: Array with eigenvalues
        lab[naq, npval]: Array with lambda values
        lab2[naq, nint, npint]: Array with lambda values reorganized per
        interval
        eigvec[naq, naq, npval]: Array with eigenvector matrices
        coef[naq ,naq, npval]: Array with coefficients;
        coef[ilayers, :, np] are the coefficients if the element is in
        ilayers belonging to Laplace parameter number np
        '''
        # Recompute T for when kaq is changed
        self.T = self.kaq * self.Haq
        self.Tcol = self.T.reshape(self.naq, 1)
        # Compute Saq and Sll
        self.Scoefaq = self.Saq * self.Haq
        self.Scoefll = self.Sll * self.Hll
        if (self.topboundary == 'con') and self.phreatictop:
            self.Scoefaq[0] = self.Scoefaq[0] / self.Haq[0]
        elif (self.topboundary == 'lea') and self.phreatictop:
            self.Scoefll[0] = self.Scoefll[0] / self.Hll[0]
        self.D = self.T / self.Scoefaq
        # Compute c if model3d for when kaq is changed
        if self.model3d:
            self.c[1:] = \
                0.5 * self.Haq[:-1] / (self.kzoverkh[:-1] * self.kaq[:-1]) + \
                0.5 * self.Haq[1:] /  (self.kzoverkh[1:] * self.kaq[1:])
        #
        self.eigval = np.zeros((self.naq, self.model.npval), 'D')
        self.lab = np.zeros((self.naq, self.model.npval), 'D')
        self.eigvec = np.zeros((self.naq, self.naq, self.model.npval), 'D')
        self.coef = np.zeros((self.naq, self.naq, self.model.npval), 'D')
        b = np.diag(np.ones(self.naq))
        for i in range(self.model.npval):
            w, v = self.compute_lab_eigvec(self.model.p[i]) 
            # Eigenvectors are columns of v
            self.eigval[:, i] = w; self.eigvec[:, :, i] = v
            self.coef[:, :, i] = np.linalg.solve(v, b).T
        self.lab = 1.0 / np.sqrt(self.eigval)
        self.lab2 = self.lab.copy() 
        self.lab2.shape = (self.naq, self.model.nint, self.model.npint)
        self.lababs = np.abs(self.lab2[:, :, 0]) # used to check distances
        self.eigvec2 = self.eigvec.copy()
        self.eigvec2.shape = (self.naq, self.naq, 
                              self.model.nint, self.model.npint)
    
    def compute_lab_eigvec(self, p, returnA = False, B = None):
        sqrtpSc = np.sqrt( p * self.Scoefll * self.c )
        a, b = np.zeros_like(sqrtpSc), np.zeros_like(sqrtpSc)
        small = np.abs(sqrtpSc) < 200
        a[small] = sqrtpSc[small] / np.tanh(sqrtpSc[small])
        b[small] = sqrtpSc[small] / np.sinh(sqrtpSc[small])
        a[~small] = sqrtpSc[~small] / ((1.0 - np.exp(-2.0*sqrtpSc[~small])) / 
                                       (1.0 + np.exp(-2.0*sqrtpSc[~small])))
        b[~small] = sqrtpSc[~small] * 2.0 * np.exp(-sqrtpSc[~small]) / \
                    (1.0 - np.exp(-2.0*sqrtpSc[~small]))
        if (self.topboundary[:3] == 'sem') or (self.topboundary[:3] == 'lea'):
            dzero = sqrtpSc[0] * np.tanh(sqrtpSc[0])

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
        w, v = np.linalg.eig(A)
        # sorting moved here
        index = np.argsort(abs(w))[::-1]
        w = w[index]
        v = v[:, index]
        return w, v
    
    def head_to_potential(self, h, layers):
        return h * self.Tcol[layers]
    
    def potential_to_head(self, pot, layers):
        return pot / self.Tcol[layers]
    
    def isInside(self,x,y):
        print('Must overload AquiferData.isInside method')
        return True
    
    def inWhichLayer(self, z):
        '''Returns -9999 if above top of system, 
        +9999 if below bottom of system, 
        negative for in leaky layer.
        leaky layer -n is on top of aquifer n'''
        if z > self.zt[0]:
            return -9999
        for i in range(self.naq-1):
            if z >= self.zb[i]:
                return i
            if z > self.zt[i+1]:
                return -i-1
        if z >= self.zb[self.naq-1]:
            return self.naq - 1
        return +9999
    
    def findlayer(self, z):
        '''
        Returns layer-number, layer-type and model-layer-number'''
        if z > self.z[0]:
            modellayer = -1
            ltype = 'above'
            layernumber = None
        elif z < self.z[-1]:
            modellayer = len(self.layernumber)
            ltype = 'below'
            layernumber = None
        else:
            modellayer = np.argwhere((z <= self.z[:-1]) & 
                                     (z >= self.z[1:]))[0, 0]
            layernumber = self.layernumber[modellayer]
            ltype = self.ltype[modellayer] 
        return layernumber, ltype, modellayer
    
class Aquifer(AquiferData):
    def __init__(self, model, kaq, z, Haq, Hll, c, Saq, Sll, poraq, porll, 
                 ltype, topboundary, phreatictop, kzoverkh=None, model3d=False):
        AquiferData.__init__(self, model, kaq, z, Haq, Hll, c, Saq, Sll, 
            poraq, porll, ltype, topboundary, phreatictop, kzoverkh, model3d)
        self.inhomlist = []
        self.area = 1e300 # Needed to find smallest inhomogeneity
    
    def __repr__(self):
        return 'Background Aquifer T: ' + str(self.T)
    
    
    def initialize(self):
        AquiferData.initialize(self)
        for inhom in self.inhomlist:
            inhom.initialize()
    
    def find_aquifer_data(self, x, y):
        rv = self
        for aq in self.inhomlist:
            if aq.isInside(x, y):
                if aq.area < rv.area:
                    rv = aq
        return rv