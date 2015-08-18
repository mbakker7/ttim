import numpy as np

# you are allowed to freely use, modify, and redistribute 
# this Python library, as long as you include the following acknowledgement
#
# this Python library developed by Kris Kuhlman: klkuhlm <at> sandia <dot> gov
# from September 2009 - February 2010.  Although the library is tested, it is not
# guaranteed to be correct or effective.  please let me know if you find
# deficiencies or bugs.  If you would like to see my testing results, I will 
# send them to you upon request.
#
# this Python library is for computing modified Mathieu
# functions of complex Mathieu parameter.  All functions require
# numpy version 1.3 or greater, while the radial MF require
# scipy (0.7 or better has fixed many bugs with the Bessel functions)
#
# as an example the library can be used in the following manner
# to plot the cosine-elliptic angular MF for complex q
# over the range 0 < psi < pi/2 and the orders 0 <= ord <= 7
#
# >> import numpy as np
# >> import matplotlib.pyplot as plt
# >> import mathieu_functions as mf
# >> m = mf.mathieu(2.0 + 5.0j)
# >> ord = np.arange(8)
# >> psi = np.linspace(0,np.pi/2,200)
# >> ce = m.ce(ord,psi)
# >> for n in ord:
# >>     plt.plot(ce[n,:].real,'-')
# >>     plt.plot(ce[n,:].imag,':')
# >> plt.xlabel('$\\psi$')
# >> plt.ylabel('ce$_{n}(\\psi,q=2+5i)$')

# $Id: mathieu_functions.py,v 1.1 2010/03/12 21:44:41 klkuhlm Exp klkuhlm $

# numpy major/minor version number
npvn = [int(x) for x in np.__version__.split('.')[0:2]]
if npvn[0] == 1 and npvn[1] < 3:
    raise ImportError, 'Mathieu funciton library requires numpy version 1.3 or greater' 

class mathieu(object):
    """Class containing all things related to modified Mathieu functions 
    (i.e.,  Mathieu functions of complex Mathieu parameter, -q).

    Create an instance of the class to automatically compute the requisite
    eigenvectors of Mathieu coefficients for computing all Mathieu 
    functions (for the -q specified at construction).

    Mathieu functions are then methods of the mathieu class corresponding
    to the value of -q used to initialize it."""

    class MathieuError(ValueError):
        pass

    def __init__(self, q, M=20, norm=2, cutoff=1.0E-12):
        """Determine the eigenvectors (A and B) necessary to compute 
        Mathieu functions for the specified value of the Mathieu parameter (q), \
        given a specified norm convention (eigenvectors define direction only,
        length is arbitrary).  The mcn is computed and saved, but not typically
        needed outside of testing/plotting purposes (it is not needed to compute
        Mathieu functions).
        
        required:    q :: negative scalar Mathieu parameter, generally complex

        optional:    M :: size of infinite matrix (default = 20)
                cutoff :: relative level of significance for computing 
                          maximum possible order of MF, given M and q

        M should in general be a function of |q|; here is implemented as a 
        function of 1) the maximum order of Mathieu function desired, and 
        2) the accuracy desired.  The routines make basic checks 
        regarding points 1 and 2, but there is no known functional relationship 
        that captures the variability in the size of the Mathieu parameter."""

        # the size of the "infinite" matrix used in eigenvalue calcs
        self.M = M
        self.cutoff = cutoff
        self.norm = 2

        # the matricies of eigenvectors which are used to compute 
        # modified Mathieu functions later
        self.A = np.empty((M,M,2),dtype=complex)
        self.B = np.empty((M,M,2),dtype=complex)

        q_test = np.asarray(q)
        if not q_test.shape == ():
            error = 'only scalar values of q accepted, q.shape=' + str(q_test.shape)
            raise self.MathieuError, error

        # the complex Mathieu parameter (negative sign removed)
        self.q = q
                
        coeff = np.zeros((M,M),dtype=complex)
        self.mcn = np.empty((M,4),dtype=complex)

        # A/B axis=0: subscript, index related to infinite sum
        # A/B axis=1: superscript, n in order=2n+1
        # A/B axis=2: 0(even) or 1(odd) 
        
        ord = np.arange(0,M+3)
        sgn = np.where(ord%2==0,1,-1)

        self.ord = ord
        self.sgnord = sgn

        # even coefficents (a) of even order (De_{2n} in eq 3.12
        # J.J. Stamnes & B. Spjelkavik (Pure & Applied Optics, 1995))
        ##################################################
    
        voff = np.ones((M-1,),dtype=complex)*q
        voffm =  np.diag(voff,k=-1) +  np.diag(voff,k=+1)
        coeff += np.diag(np.array((2.0*ord[0:M])**2,dtype=complex),k=0)
        coeff += voffm
        coeff[1,0] *= 2.0

        # compute eigenvalues (Mathieu characteristic numbers) and 
        # eigenvectors (Mathieu coefficients) 
        self.mcn[:,0],self.A[:,:,0] = np.linalg.eig(coeff)

        # normalize so |ce_2n(psi=0)| = +1
        self.A[:,:,0] /= np.sum(self.A[:,:,0]*sgn[0:M,None],axis=0)[None,:]

        # even coefficents (a) of odd order (De_{2n+1} in eq3.14 St&Sp)
        ##################################################

        coeff = np.zeros((M,M),dtype=complex)
        coeff += np.diag(np.array((2.0*ord[0:M] + 1.0)**2,dtype=complex),k=0)
        coeff += voffm
        coeff[0,0] += q

        self.mcn[:,1],self.A[:,:,1] = np.linalg.eig(coeff)

        # normalize so |se'_2n+1(psi=0)| = +1
        self.A[:,:,1] /=  np.sum((2.0*ord[0:M,None]+1)*sgn[0:M,None]*self.A[:,:,1],axis=0)[None,:]

        # odd coefficents (b) of even order (Do_{2n+2} in eq3.16 St&Sp)
        ##################################################

        coeff = np.zeros((M,M),dtype=complex)  
        coeff += np.diag(np.array((2.0*ord[1:M+1])**2,dtype=complex),k=0)
        coeff += voffm
        
        self.mcn[:,2],self.B[:,:,0] = np.linalg.eig(coeff)

        # normalize so |se'_2n+2(psi=0)| = +1            
        self.B[:,:,0] /= np.sum((2.0*ord[0:M,None]+2)*sgn[0:M,None]*self.B[:,:,0],axis=0)[None,:]

        # odd coefficents (b) of odd order (Do_{2n+1} in eq3.18 St&Sp)
        ##################################################

        coeff = np.zeros((M,M),dtype=complex)  
        coeff += np.diag(np.array((2.0*ord[0:M]+1.0)**2,dtype=complex),k=0)
        coeff += voffm
        coeff[0,0] -= q
       
        self.mcn[:,3],self.B[:,:,1] = np.linalg.eig(coeff)
        
        # normalize so |ce_2n+1(psi=0)| = +1
        self.B[:,:,1] /= np.sum(self.B[:,:,1]*sgn[0:M,None],axis=0)[None,:]

        ##################################################
        # compute maximum accurate order given current M and q
        # since the 4 cases {odd,even}{A,B} seem to give roughly the
        # same answer, just pick A even arbitrarily as one to check.

        # average value on diagonal
        d = np.sum(np.abs(np.diagonal(self.A[:,:,0])))/self.M

        # compute average size of off-diagonal terms (-n for below main diag), 
        # scaled by main diagonal size
        for n in range(1,self.M):
            if np.sum(np.abs(np.diagonal(self.A[:,:,0],-n)))/(d*(self.M-n)) < cutoff:
                buff = n
                break
        try:
            buff
        except NameError:
            error = ("infinite matrix size too small for given " +
            "q=(%.1f,%.1f), cannot meet cutoff (%.1e) with matrix size (%i)" %
                     (q.real,q.imag,cutoff,self.M))
            raise self.MathieuError,error
        else:
            self.buffer = buff

        # add 4th dimension for vectorizing with respect to argument now, 
        # rather than over and over in the code below
        self.A.shape = (M,M,2,1)
        self.B.shape = (M,M,2,1)
            
    ##################################################
    ##################################################
    # utility functions used internally in this class only (leading single underscore)

    def _error_check(self,n):
        if np.max(n)+self.buffer > self.M:
            err = ("max Mathieu function order (%i) too high," +
                   "increase 'inf' matrix size M=%i," +
                   "given q=(%.1f,%.1fj) and buffer=%i" % 
                   (np.max(n),self.M,self.q.real,self.q.imag,self.buffer))
            raise self.MathieuError, err

    def _AngFuncSetup(self,n,z):
        self._error_check(n)
        vv = self.ord[0:self.M]
        nn = np.atleast_1d(n)
        return (nn,np.atleast_1d(z),vv,np.where(vv%2==0,1,-1)[:,None],nn%2==0,nn%2==1)

    def _RadFuncSetup(self,n,z):
        self._error_check(n)
        sqrtq = np.sqrt(self.q)
        nn = np.atleast_1d(n)
        zz = np.atleast_1d(z)
        return (nn,zz,sqrtq,sqrtq*np.exp(-zz),sqrtq*np.exp(zz),self.M,nn%2==0,nn%2==1)

    def _RadDerivFuncSetup(self,n,z):
        self._error_check(n)
        sqrtq = np.sqrt(self.q)
        nn = np.atleast_1d(n)
        zz = np.atleast_1d(z)
        enz = np.exp(-zz)
        epz = np.exp(zz)
        return (nn,zz,sqrtq,enz,epz,sqrtq*enz,sqrtq*epz,self.M,nn%2==0,nn%2==1)

    def _deriv(self,z,W,t):
        """Compute derivatives of I & K Bessel functions using recurrence
        without recomputing the functions again, assuming the derivatives 
        of order 0:n are needed from the functions of order 0:n
        
        z : argument vector
        W : Bessel function matrix (ord,arg)
        t : type of bessel function 0=I, 1=K """
    
        assert t==0 or t==1
        s = np.array([1.0, -1.0])
        
        WD = np.empty_like(W)
        # n is highest order of Bessel fcn needed (starting at zero)
        n = W.shape[0] - 1  
    
        # three different recurrence relations used
        # 1) low end
        WD[0,:] = W[1,:]*s[t]
    
        # 2) middle
        WD[1:n-1,:] = 0.5*(W[0:n-2,:] + W[2:n,:])*s[t]
    
        # 3) high end
        WD[n,:] = W[n-1,:]*s[t] - n/z[None,:]*W[n,:]       
        return WD

    ##################################################
    ##################################################

    def ce(self,n,z):
        """even first-kind angular mathieu function (ce) 
        for scalar or vector orders or argument

        called Se(-q) by Blanch or Qe(q) by Alhargan
        These use identities in 7.02 of Blanch's AMS#59 pub

        n is modified Mathieu function order (scalar or vector)
        z is angular argument to modified Mathieu function (periodic in
        at least 2*pi, so this is the logical range)"""

        n,z,v,vi,EV,OD = self._AngFuncSetup(n,z)
        y = np.empty((n.shape[0],z.shape[0]),dtype=complex)
        # j vector used in "fancy indexing" of axis 1 below
        j = n[:]//2     # <- int(np.floor(n/2.0))

        y[EV,:] = np.sum(self.A[:,j[EV],0,:]*np.cos(np.outer(2*v,np.pi/2-z))[:,None,:],axis=0)
        y[OD,:] = np.sum(self.B[:,j[OD],1,:]*np.sin(np.outer(2*v+1,np.pi/2-z))[:,None,:],axis=0)
        return np.squeeze(y)

    def se(self,n,z):
        """odd first-kind angular mathieu function (se)
        for scalar or vector orders or argument
        
        called So(-q) by Blanch or Qo(q) by Alhargan

        n: modified Mathieu function order (scalar or vector)
        z: angular argument to modified Mathieu function (periodic in
        at least 2*pi, so this is the logical range"""

        n,z,v,vi,EV,OD = self._AngFuncSetup(n,z)
        y = np.empty((n.shape[0],z.shape[0]),dtype=complex)
        j = (n[:]-1)//2  # se_0() invalid (set NaN below)

        y[EV,:] = np.sum(self.B[:,j[EV],0,:]*np.sin(np.outer(2*v+2,np.pi/2-z))[:,None,:],axis=0)
        y[OD,:] = np.sum(self.A[:,j[OD],1,:]*np.cos(np.outer(2*v+1,np.pi/2-z))[:,None,:],axis=0)
        y[n==0,:] = np.NaN
        return np.squeeze(y)

    ##################################################
    ##################################################

    def dce(self,n,z):
        """even first-kind angular mathieu function derivative (Dce) 
        for scalar or vector orders or argument

        n is modified Mathieu function order (scalar or vector)
        z is angular argument to modified Mathieu function (periodic in
        at least 2*pi, so this is the logical range)"""

        n,z,v,vi,EV,OD = self._AngFuncSetup(n,z)

        y = np.empty((n.shape[0],z.shape[0]),dtype=complex)
        j = n[:]//2 
        jsgn = np.where(j%2==0,1,-1)[:,None]

        y[EV,:] = np.sum(2*v[:,None,None]*self.A[:,j[EV],0,:]*
                           np.sin(np.outer(2*v,np.pi/2-z))[:,None,:],axis=0)
        y[OD,:] = -np.sum((2*v[:,None,None]+1)*self.B[:,j[OD],1,:]*
                           np.cos(np.outer(2*v+1,np.pi/2-z))[:,None,:],axis=0)
        return np.squeeze(y)

    def dse(self,n,z):
        """odd first-kind angular mathieu function derivative (Dse) 
        for scalar or vector orders or argument

        n: modified Mathieu function order (scalar or vector)
        z: angular argument to modified Mathieu function (periodic in
        at least 2*pi, so this is the logical range"""

        n,z,v,vi,EV,OD = self._AngFuncSetup(n,z)
        y = np.empty((n.shape[0],z.shape[0]),dtype=complex)
        j = (n[:]-1)//2  # se_0() invalid

        y[EV,:] = -np.sum((2*v[:,None,None]+2)*self.B[:,j[EV],0,:]*
                           np.cos(np.outer(2*v+2,np.pi/2-z))[:,None,:],axis=0)
        y[OD,:] = np.sum((2*v[:,None,None]+1)*self.A[:,j[OD],1,:]*
                          np.sin(np.outer(2*v+1,np.pi/2-z))[:,None,:],axis=0)
        y[n==0,:] = np.NaN
        return np.squeeze(y)


    ##################################################
    ##################################################

    def Ie(self,n,z):
        """even first-kind radial Mathieu function (Ie) 
        analogous to I Bessel functions, called Ce in McLachlan p248 (eq 1&2), 
        for scalar or vector orders or argument

        n: order (scalar or vector)
        z: radial argument (scalar or vector)"""

        from scipy.special import ive
        n,z,sqrtq,v1,v2,M,EV,OD = self._RadFuncSetup(n,z)
        y = np.empty((n.shape[0],z.shape[0]),dtype=complex)

        ord = self.ord[0:M+1]
        sgn = self.sgnord[0:M,None,None]
        I1 = ive(ord[0:M+1,None],v1[None,:])[:,None,:]
        I2 = ive(ord[0:M+1,None],v2[None,:])[:,None,:]
        j = n[:]//2

        y[EV,:] = (np.sum(sgn*self.A[0:M,j[EV],0,:]*I1[0:M,:,:]*I2[0:M,:,:],axis=0)/
                   self.A[0,j[EV],0,:])
        
        y[OD,:] = (np.sum(sgn*self.B[0:M,j[OD],1,:] * 
                     (I1[0:M,:,:]*I2[1:M+1,:,:] + I1[1:M+1,:,:]*I2[0:M,:,:]),axis=0)/
                   self.B[0,j[OD],1,:])

	# scaling factor to un-scale products of Bessel functions from ive()
        y *= np.exp(np.abs(v1.real) + np.abs(v2.real))[None,:]
        return np.squeeze(y)

    def Io(self,n,z):
        """odd first-kind radial Mathieu function (Io) 
        analogous to I Bessel functions, called Se in McLachlan p248 (eq 3&4), 
        for scalar or vector orders or argument

        n: order (scalar or vector)
        z: radial argument (scalar or vector)"""
        
        from scipy.special import ive
        n,z,sqrtq,v1,v2,M,EV,OD = self._RadFuncSetup(n,z)
        y = np.empty((n.shape[0],z.shape[0]),dtype=complex)

        ord = self.ord[0:M+2]
        sgn = self.sgnord[0:M,None,None]
        I1 = ive(ord[0:M+2,None],v1[None,:])[:,None,:]
        I2 = ive(ord[0:M+2,None],v2[None,:])[:,None,:]        
        j = (n[:]-1)//2  # Io_0() invalid

        y[EV,:] = (np.sum(sgn*self.B[0:M,j[EV],0,:] *
                     (I1[0:M,:,:]*I2[2:M+2,:,:] - I1[2:M+2,:,:]*I2[0:M,:,:]),axis=0)/
                    self.B[0,j[EV],0,:])

        y[OD,:] = (np.sum(sgn*self.A[0:M,j[OD],1,:] *
                     (I1[0:M,:,:]*I2[1:M+1,:,:] - I1[1:M+1,:,:]*I2[0:M,:,:]),axis=0)/
                    self.A[0,j[OD],1,:])

        y *= np.exp(np.abs(v1.real) + np.abs(v2.real))[None,:]
        y[n==0,:] = np.NaN
        return np.squeeze(y)

    def Ke(self,n,z):
        """even second-kind radial Mathieu function (Ke) 
        analogous to K Bessel functions, called Fek in McLachlan p248 (eq 5&6), 
        for scalar or vector orders or argument

        n: order (scalar or vector)
        z: radial argument (scalar or vector)"""

        from scipy.special import kve,ive
        n,z,sqrtq,v1,v2,M,EV,OD = self._RadFuncSetup(n,z)
        y = np.empty((n.shape[0],z.shape[0]),dtype=complex)

        ord = self.ord[0:M+1]
        I = ive(ord[0:M+1,None],v1[None,:])[:,None,:]
        K = kve(ord[0:M+1,None],v2[None,:])[:,None,:]
        j = n[:]//2

        y[EV,:] = np.sum(self.A[0:M,j[EV],0,:]*I[0:M,:,:]*K[0:M,:,:],axis=0)/self.A[0,j[EV],0,:]
              
        y[OD,:] = (np.sum(self.B[0:M,j[OD],1,:] *
                          (I[0:M,:,:]*K[1:M+1,:,:] - I[1:M+1,:,:]*K[0:M,:,:]),axis=0)/
                   self.B[0,j[OD],1,:])

        y *= np.exp(np.abs(v1.real) - v2)[None,:]
        return np.squeeze(y)

    def Ko(self,n,z):
        """odd second-kind radial Mathieu function (Ko) 
        (analogous to K Bessel functions, called Gek in McLachlan p248), 
        for scalar or vector orders or argument

        n: order (scalar or vector)
        z: radial argument (scalar or vector)"""
        
        from scipy.special import kve,ive
        n,z,sqrtq,v1,v2,M,EV,OD = self._RadFuncSetup(n,z)
        y = np.empty((n.shape[0],z.shape[0]),dtype=complex)

        ord = self.ord[0:M+2]
        I = ive(ord[0:M+2,None],v1[None,:])[:,None,:]
        K = kve(ord[0:M+2,None],v2[None,:])[:,None,:]
        j = (n[:]-1)//2  

        y[EV,:] = (np.sum(self.B[0:M,j[EV],0,:] *
                     (I[0:M,:,:]*K[2:M+2,:,:] - I[2:M+2,:,:]*K[0:M,:,:]),axis=0)/
                   self.B[0,j[EV],0,:])

        y[OD,:] = (np.sum(self.A[0:M,j[OD],1,:] *
                     (I[0:M,:,:]*K[1:M+1,:,:] + I[1:M+1,:,:]*K[0:M,:,:]),axis=0)/
                   self.A[0,j[OD],1,:])

        y *= np.exp(np.abs(v1.real) - v2)[None,:]
        y[n==0,:] = np.NaN   # Ko_0() invalid
        return np.squeeze(y)

    ##################################################
    ##################################################

    def dIe(self,n,z):
        """even first-kind radial Mathieu function derivative (DIe) 
        (analogous to I Bessel functions, called Ce in McLachlan p248), 
        for scalar or vector orders or argument

        n: order (scalar or vector)
        z: radial argument (scalar or vector)"""
        
        from scipy.special import ive
        n,z,sqrtq,enz,epz,v1,v2,M,EV,OD = self._RadDerivFuncSetup(n,z)
        dy = np.empty((n.shape[0],z.shape[0]),dtype=complex)

        ord = self.ord[0:M+1]
        sgn = self.sgnord[0:M,None,None]

        I1 = ive(ord[0:M+1,None],v1[None,:])[:,None,:]
        I2 = ive(ord[0:M+1,None],v2[None,:])[:,None,:]        
        dI1 = self._deriv(v1,I1[:,0,:],0)[:,None,:]
        dI2 = self._deriv(v2,I2[:,0,:],0)[:,None,:]
        j = n[:]//2

        dy[EV,:] = (sqrtq/self.A[0,j[EV],0,:]*np.sum(sgn*self.A[0:M,j[EV],0,:] *
                     (epz*I1[0:M,:,:]*dI2[0:M,:,:] - enz*dI1[0:M,:,:]*I2[0:M,:,:]),axis=0))

        dy[OD,:] = (sqrtq/self.B[0,j[OD],1,:]*np.sum(sgn*self.B[0:M,j[OD],1,:] *
                     (epz*I1[0:M,:,:]*dI2[1:M+1,:,:] - enz*dI1[0:M,:,:]*I2[1:M+1,:,:] +
                      epz*I1[1:M+1,:,:]*dI2[0:M,:,:] - enz*dI1[1:M+1,:,:]*I2[0:M,:,:]),axis=0))

        dy *= np.exp(np.abs(v1.real) + np.abs(v2.real))[None,:]
        return np.squeeze(dy)

    def dIo(self,n,z):
        """odd first-kind radial Mathieu function derivative (DIo) 
        (analogous to I Bessel functions, called Se in McLachlan p248), 
        for scalar or vector orders or argument

        n: order (scalar or vector)
        z: radial argument (scalar or vector)"""
        
        from scipy.special import ive
        n,z,sqrtq,enz,epz,v1,v2,M,EV,OD = self._RadDerivFuncSetup(n,z)
        dy = np.empty((n.shape[0],z.shape[0]),dtype=complex)

        ord = self.ord[0:M+2]
        sgn = self.sgnord[0:M,None,None]
        I1 = ive(ord[0:M+2,None],v1[None,:])[:,None,:]
        I2 = ive(ord[0:M+2,None],v2[None,:])[:,None,:]
        dI1 = self._deriv(v1,I1[0:M+2,0,:],0)[:,None,:]
        dI2 = self._deriv(v2,I2[0:M+2,0,:],0)[:,None,:]
        j = (n[:]-1)//2

        dy[EV,:] = (sqrtq/self.B[0,j[EV],0,:]*np.sum(sgn*self.B[0:M,j[EV],0,:] *
                     (epz*I1[0:M,:,:]*dI2[2:M+2,:,:] - enz*dI1[0:M,:,:]*I2[2:M+2,:,:] - 
                     (epz*I1[2:M+2,:,:]*dI2[0:M,:,:] - enz*dI1[2:M+2,:,:]*I2[0:M,:,:])),axis=0))

        dy[OD,:] = (sqrtq/self.A[0,j[OD],1,:]*np.sum(sgn*self.A[0:M,j[OD],1,:] * 
                     (epz*I1[0:M,:,:]*dI2[1:M+1,:,:] - enz*dI1[0:M,:,:]*I2[1:M+1,:,:] - 
                     (epz*I1[1:M+1,:,:]*dI2[0:M,:,:] - enz*dI1[1:M+1,:,:]*I2[0:M,:,:])),axis=0))
        
        dy *= np.exp(np.abs(v1.real) + np.abs(v2.real))[None,:]
        dy[n==0,:] = np.NaN # dIo_0() invalid
        return np.squeeze(dy)

    def dKe(self,n,z):
        """even second-kind radial Mathieu function derivative (DKe) 
        (analogous to K Bessel functions, called Fek in McLachlan p248), 
        for scalar or vector orders or argument

        n: order (scalar or vector)
        z: radial argument (scalar or vector)"""

        from scipy.special import kve,ive
        n,z,sqrtq,enz,epz,v1,v2,M,EV,OD = self._RadDerivFuncSetup(n,z)
        dy = np.empty((n.shape[0],z.shape[0]),dtype=complex)

        ord = self.ord[0:M+1]
        I = ive(ord[0:M+1,None],v1[None,:])[:,None,:]
        K = kve(ord[0:M+1,None],v2[None,:])[:,None,:]
        dI = self._deriv(v1,I[0:M+1,0,:],0)[:,None,:]
        dK = self._deriv(v2,K[0:M+1,0,:],1)[:,None,:]
        j = n[:]//2

        dy[EV,:] = (sqrtq/self.A[0,j[EV],0,:]*np.sum(self.A[0:M,j[EV],0,:] *
                     (epz*I[0:M,:,:]*dK[0:M,:,:] - enz*dI[0:M,:,:]*K[0:M,:,:]),axis=0))

        dy[OD,:] = (sqrtq/self.B[0,j[OD],1,:]*np.sum(self.B[0:M,j[OD],1,:] *
                     (epz*I[0:M,:,:]*dK[1:M+1,:,:] - enz*dI[0:M,:,:]*K[1:M+1,:,:] - 
                     (epz*I[1:M+1,:,:]*dK[0:M,:,:] - enz*dI[1:M+1,:,:]*K[0:M,:,:])),axis=0))

        dy *= np.exp(np.abs(v1.real) - v2)[None,:]
        return np.squeeze(dy)

    def dKo(self,n,z):
        """odd second-kind radial Mathieu function derivative (DKo) 
        (analogous to K Bessel functions, called Gek in McLachlan p248), 
        for scalar or vector orders or argument

        n: order (scalar or vector)
        z: radial argument (scalar or vector)"""
        
        from scipy.special import kve,ive
        n,z,sqrtq,enz,epz,v1,v2,M,EV,OD = self._RadDerivFuncSetup(n,z)
        dy = np.empty((n.shape[0],z.shape[0]),dtype=complex)

        ord = self.ord[0:M+2]
        I = ive(ord[0:M+2,None],v1[None,:])[:,None,:]
        K = kve(ord[0:M+2,None],v2[None,:])[:,None,:]
        dI = self._deriv(v1,I[0:M+2,0,:],0)[:,None,:]
        dK = self._deriv(v2,K[0:M+2,0,:],1)[:,None,:]
        j = (n[:]-1)//2 

        dy[EV,:] = (sqrtq/self.B[0,j[EV],0,:]*np.sum(self.B[0:M,j[EV],0,:] *
                     (epz*I[0:M,:,:]*dK[2:M+2,:,:] - enz*dI[0:M,:,:]*K[2:M+2,:,:] -
                     (epz*I[2:M+2,:,:]*dK[0:M,:,:] - enz*dI[2:M+2,:,:]*K[0:M,:,:])),axis=0))

        dy[OD,:] = (sqrtq/self.A[0,j[OD],1,:]*np.sum(self.A[0:M,j[OD],1,:] *
                     (epz*I[0:M,:,:]*dK[1:M+1,:,:] - enz*dI[0:M,:,:]*K[1:M+1,:,:] + 
                      epz*I[1:M+1,:,:]*dK[0:M,:,:] - enz*dI[1:M+1,:,:]*K[0:M,:,:]),axis=0))

        dy *= np.exp(np.abs(v1.real) - v2)[None,:]
        dy[n==0,:] = np.NaN  # dKo_0() invalid
        return np.squeeze(dy)


