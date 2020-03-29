import numpy as np

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
        mat = np.empty( (self.nunknowns,self.model.neq,self.model.npval), 'D' )
        rhs = np.zeros( (self.nunknowns,self.model.ngvbc,self.model.npval), 'D' )  # Needs to be initialized to zero
        for icp in range(self.ncp):
            istart = icp*self.nlayers
            ieq = 0  
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    mat[istart:istart+self.nlayers,ieq:ieq+e.nunknowns,:] = e.potinflayers(self.xc[icp],self.yc[icp],self.layers)
                    if e == self:
                        for i in range(self.nlayers): mat[istart+i,ieq+istart+i,:] -= self.resfacp[istart+i] * e.dischargeinflayers[istart+i]
                    ieq += e.nunknowns
            for i in range(self.model.ngbc):
                rhs[istart:istart+self.nlayers,i,:] -= self.model.gbclist[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.layers)  # Pretty cool that this works, really
            if self.type == 'v':
                iself = self.model.vbclist.index(self)
                for i in range(self.nlayers):
                    rhs[istart+i,self.model.ngbc+iself,:] = self.pc[istart+i] / self.model.p
        return mat, rhs
    
class WellBoreStorageEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for multi-aquifer element with
        total given discharge, uniform but unknown head and InternalStorageEquation
        '''
        mat = np.zeros( (self.nunknowns,self.model.neq,self.model.npval), 'D' ) # Important to set to zero for some of the equations
        rhs = np.zeros( (self.nunknowns,self.model.ngvbc,self.model.npval), 'D' )  # Needs to be initialized to zero
        ieq = 0
        for e in self.model.elementlist:
            if e.nunknowns > 0:
                head = e.potinflayers(self.xc[0], self.yc[0], self.layers) \
                       / self.aq.T[self.layers][:,np.newaxis,np.newaxis]
                mat[:-1,ieq:ieq+e.nunknowns,:] = head[:-1,:] - head[1:,:]
                mat[-1,ieq:ieq+e.nunknowns,:] -= np.pi * self.rc**2 * self.model.p * head[0,:]
                if e == self:
                    disterm = self.dischargeinflayers * self.res / (2 * np.pi 
                        * self.rw * self.aq.Haq[self.layers][:,np.newaxis])
                    if self.nunknowns > 1:  # Multiple layers
                        for i in range(self.nunknowns-1):
                            mat[i,ieq+i,:] -= disterm[i]
                            mat[i,ieq+i+1,:] += disterm[i+1]
                    mat[-1,ieq:ieq+self.nunknowns,:] += self.dischargeinflayers
                    mat[-1,ieq,:] += np.pi * self.rc**2 * self.model.p * disterm[0]
                ieq += e.nunknowns
        for i in range(self.model.ngbc):
            head = self.model.gbclist[i].unitpotentiallayers(
                self.xc[0], self.yc[0], self.layers) \
                / self.aq.T[self.layers][:, np.newaxis]
            rhs[:-1, i, :] -= head[:-1, :] - head[1:, :]
            rhs[-1, i, :] += np.pi * self.rc ** 2 * self.model.p * head[0, :]
        if self.type == 'v':
            iself = self.model.vbclist.index(self)
            rhs[-1,self.model.ngbc+iself,:] += self.flowcoef
            if self.hdiff is not None:
                # head[0] - head[1] = hdiff
                rhs[:-1,self.model.ngbc+iself,:] += \
                    self.hdiff[:,np.newaxis] / self.model.p  
        return mat, rhs
    
class HeadEquationNores:
    def equation(self):
        '''Mix-in class that returns matrix rows for head-specified conditions. (really written as constant potential element)
        Returns matrix part nunknowns,neq,npval, complex
        Returns rhs part nunknowns,nvbc,npval, complex
        '''
        mat = np.empty((self.nunknowns, self.model.neq, self.model.npval), 'D')
        rhs = np.zeros((self.nunknowns, self.model.ngvbc, self.model.npval), 'D')  # Needs to be initialized to zero
        for icp in range(self.ncp):
            istart = icp * self.nlayers
            ieq = 0  
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    mat[istart:istart+self.nlayers, ieq:ieq+e.nunknowns,:] = e.potinflayers(self.xc[icp], self.yc[icp], self.layers)
                    ieq += e.nunknowns
            for i in range(self.model.ngbc):
                rhs[istart:istart+self.nlayers,i,:] -= self.model.gbclist[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.layers)  # Pretty cool that this works, really
            if self.type == 'v':
                iself = self.model.vbclist.index(self)
                for i in range(self.nlayers):
                    rhs[istart+i,self.model.ngbc+iself,:] = self.pc[istart+i] / self.model.p
        return mat, rhs
    
class LeakyWallEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for leaky-wall condition
        Returns matrix part nunknowns,neq,npval, complex
        Returns rhs part nunknowns,nvbc,npval, complex
        '''
        mat = np.empty( (self.nunknowns,self.model.neq,self.model.npval), 'D' )
        rhs = np.zeros( (self.nunknowns,self.model.ngvbc,self.model.npval), 'D' )  # Needs to be initialized to zero
        for icp in range(self.ncp):
            istart = icp*self.nlayers
            ieq = 0  
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    qx,qy = e.disvecinflayers(self.xc[icp],self.yc[icp],self.layers)
                    mat[istart:istart+self.nlayers,ieq:ieq+e.nunknowns,:] = qx * self.cosout[icp] + qy * self.sinout[icp]
                    if e == self:
                        hmin = e.potinflayers(self.xcneg[icp],self.ycneg[icp],self.layers) / self.aq.T[self.layers][:,np.newaxis,np.newaxis]
                        hplus = e.potinflayers(self.xc[icp],self.yc[icp],self.layers) / self.aq.T[self.layers][:,np.newaxis,np.newaxis]
                        mat[istart:istart+self.nlayers,ieq:ieq+e.nunknowns,:] -= self.resfac[:,np.newaxis,np.newaxis] * (hplus-hmin)
                    ieq += e.nunknowns
            for i in range(self.model.ngbc):
                qx,qy = self.model.gbclist[i].unitdisveclayers(self.xc[icp],self.yc[icp],self.layers)
                rhs[istart:istart+self.nlayers,i,:] -=  qx * self.cosout[icp] + qy * self.sinout[icp]
            #if self.type == 'v':
            #    iself = self.model.vbclist.index(self)
            #    for i in range(self.nlayers):
            #        rhs[istart+i,self.model.ngbc+iself,:] = self.pc[istart+i] / self.model.p
        return mat, rhs
    
class MscreenEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for multi-screen conditions where total discharge is specified.
        Works for nunknowns = 1
        Returns matrix part nunknowns,neq,npval, complex
        Returns rhs part nunknowns,nvbc,npval, complex
        head_out - c*q_s = h_in
        Set h_i - h_(i+1) = 0 and Sum Q_i = Q'''
        mat = np.zeros( (self.nunknowns,self.model.neq,self.model.npval), 'D' )  # Needs to be zero for last equation, but I think setting the whole array is quicker
        rhs = np.zeros( (self.nunknowns,self.model.ngvbc,self.model.npval), 'D' )  # Needs to be initialized to zero
        ieq = 0
        for icp in range(self.ncp):
            istart = icp*self.nlayers
            ieq = 0 
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    head = e.potinflayers(self.xc[icp],self.yc[icp],self.layers) / self.aq.T[self.layers][:,np.newaxis,np.newaxis]  # T[self.layers,np.newaxis,np.newaxis] is not allowed
                    mat[istart:istart+self.nlayers-1,ieq:ieq+e.nunknowns,:] = head[:-1,:] - head[1:,:]
                    if e == self:
                        for i in range(self.nlayers-1):
                            mat[istart+i,ieq+istart+i,:] -= self.resfach[istart+i] * e.dischargeinflayers[istart+i]
                            mat[istart+i,ieq+istart+i+1,:] += self.resfach[istart+i+1] * e.dischargeinflayers[istart+i+1]
                            mat[istart+i,ieq+istart:ieq+istart+i+1,:] -= self.vresfac[istart+i] * e.dischargeinflayers[istart+i]
                        mat[istart+self.nlayers-1,ieq+istart:ieq+istart+self.nlayers,:] = 1.0
                    ieq += e.nunknowns
            for i in range(self.model.ngbc):
                head = self.model.gbclist[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.layers) / self.aq.T[self.layers][:,np.newaxis]
                rhs[istart:istart+self.nlayers-1,i,:] -= head[:-1,:] - head[1:,:]
            if self.type == 'v':
                iself = self.model.vbclist.index(self)
                rhs[istart+self.nlayers-1,self.model.ngbc+iself,:] = 1.0  # If self.type == 'z', it should sum to zero, which is the default value of rhs
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
        mat = np.zeros( (self.nunknowns,self.model.neq,self.model.npval), 'D' )  # Needs to be zero for last equation, but I think setting the whole array is quicker
        rhs = np.zeros( (self.nunknowns,self.model.ngvbc,self.model.npval), 'D' )  # Needs to be initialized to zero
        ieq = 0
        for icp in range(self.ncp):
            istart = icp*self.nlayers
            ieq = 0 
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    head = e.potinflayers(self.xc[icp],self.yc[icp],self.layers) / self.aq.T[self.layers][:,np.newaxis,np.newaxis]  # T[self.layers,np.newaxis,np.newaxis] is not allowed
                    if self.nlayers > 1: mat[istart:istart+self.nlayers-1,ieq:ieq+e.nunknowns,:] = head[:-1,:] - head[1:,:]
                    mat[istart+self.nlayers-1,ieq:ieq+e.nunknowns,:] = head[0,:] # Store head in top layer in 2nd to last equation of this control point
                    if e == self:
                        # Correct head in top layer in second to last equation to make it head inside
                        mat[istart+self.nlayers-1,ieq+istart,:] -= self.resfach[istart] * e.dischargeinflayers[istart]
                        if icp == 0:
                            istartself = ieq  # Needed to build last equation
                        for i in range(self.nlayers-1):
                            mat[istart+i,ieq+istart+i,:] -= self.resfach[istart+i] * e.dischargeinflayers[istart+i]
                            mat[istart+i,ieq+istart+i+1,:] += self.resfach[istart+i+1] * e.dischargeinflayers[istart+i+1]
                            #vresfac not yet used here; it is set to zero ad I don't quite now what is means yet
                            #mat[istart+i,ieq+istart:ieq+istart+i+1,:] -= self.vresfac[istart+i] * e.dischargeinflayers[istart+i]
                    ieq += e.nunknowns
            for i in range(self.model.ngbc):
                head = self.model.gbclist[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.layers) / self.aq.T[self.layers][:,np.newaxis]
                if self.nlayers > 1: rhs[istart:istart+self.nlayers-1,i,:] -= head[:-1,:] - head[1:,:]
                rhs[istart+self.nlayers-1,i,:] -= head[0,:] # Store minus the head in top layer in second to last equation for this control point
        # Modify last equations
        for icp in range(self.ncp-1):
            ieq = (icp+1) * self.nlayers - 1
            mat[ieq,:,:] -= mat[ieq+self.nlayers,:,:]  # Head first layer control point icp - Head first layer control point icp + 1
            rhs[ieq,:,:] -= rhs[ieq+self.nlayers,:,:]
        # Last equation setting the total discharge of the ditch
        # print 'istartself ',istartself
        mat[-1,:,:] = 0.0  
        mat[-1,istartself:istartself+self.nparam,:] = 1.0
        if self.Astorage is not None:
            matlast = np.zeros( (self.model.neq,  self.model.np), 'D' )  # Used to store last equation in case of ditch storage
            rhslast = np.zeros( (self.model.np), 'D' )  # Used to store last equation in case of ditch storage 
            ieq = 0
            for e in self.model.elementlist:
                head = e.potinflayers(self.xc[0],self.yc[0],self.layers) / self.aq.T[self.layers][:,np.newaxis,np.newaxis]  # T[self.layers,np.newaxis,np.newaxis] is not allowed
                matlast[ieq:ieq+e.nunknowns] -= self.Astorage * self.model.p**2 * head[0,:]
                if e == self:
                    # only need to correct first unknown 
                    matlast[ieq] += self.Astorage * self.model.p**2 * self.resfach[0] * e.dischargeinflayers[0]
                ieq += e.nunknowns
            for i in range(self.model.ngbc):
                head = self.model.gbclist[i].unitpotentiallayers(self.xc[0],self.yc[0],self.layers) / self.aq.T[self.layers][:,np.newaxis]
                rhslast += self.Astorage * self.model.p**2 * head[0] 
            mat[-1] += matlast
        rhs[-1,:,:] = 0.0
        if self.type == 'v':
            iself = self.model.vbclist.index(self)
            rhs[-1,self.model.ngbc+iself,:] = 1.0  # If self.type == 'z', it should sum to zero, which is the default value of rhs
            if self.Astorage is not None: rhs[-1,self.model.ngbc+iself,:] += rhslast
        return mat, rhs