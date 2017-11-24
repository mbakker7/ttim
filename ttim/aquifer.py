import numpy as np
import matplotlib.pyplot as plt
import inspect # Used for storing the input

class AquiferData:
    def __init__(self, model, kaq, Haq, c, Saq, Sll, topboundary, phreatictop):
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
        self.phreatictop = phreatictop  # used in calibration
        #self.D = self.T / self.Saq
        self.area = 1e200 # Smaller than default of ml.aq so that inhom is found
    
    def __repr__(self):
        return 'Inhom T: ' + str(self.T)
    
    def initialize(self):
        '''
        eigval[Naq,Np]: Array with eigenvalues
        lab[Naq,Np]: Array with lambda values
        lab2[Naq,Nin,Npin]: Array with lambda values reorganized per interval
        eigvec[Naq,Naq,Np]: Array with eigenvector matrices
        coef[Naq,Naq,Np]: Array with coefficients;
        coef[ilayers,:,np] are the coefficients if the element is in ilayers belonging to Laplace parameter number np
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
    
    def headToPotential(self,h,layers):
        return h * self.Tcol[layers]
    
    def potentialToHead(self,pot,layers):
        return pot / self.Tcol[layers]
    
    def isInside(self,x,y):
        print('Must overload AquiferData.isInside method')
        return True
    
    def inWhichLayer(self, z):
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
    
    def set_kaq(self, value, layer):
        self.kaq[layer] = value
        
    def set_Saq(self, value, layer):
        self.Saq[layer] = value
    
class Aquifer(AquiferData):
    def __init__(self, model, kaq, Haq, c, Saq, Sll, topboundary, phreatictop):
        AquiferData.__init__(self, model, kaq, Haq, c, Saq, Sll, \
                             topboundary, phreatictop)
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