import numpy as np

class HeadEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for head-specified conditions. (really written as constant potential element)
        Works for Nunknowns = 1
        Returns matrix part Nunknowns,Neq,Np, complex
        Returns rhs part Nunknowns,Nvbc,Np, complex
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
                        for i in range(self.Nlayers): mat[istart+i,ieq+istart+i,:] -= self.resfacp[istart+i] * e.dischargeinflayers[istart+i]
                    ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                rhs[istart:istart+self.Nlayers,i,:] -= self.model.gbcList[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.pylayers)  # Pretty cool that this works, really
            if self.type == 'v':
                iself = self.model.vbcList.index(self)
                for i in range(self.Nlayers):
                    rhs[istart+i,self.model.Ngbc+iself,:] = self.pc[istart+i] / self.model.p
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
                    disterm = self.dischargeinflayers * self.res / ( 2 * np.pi * self.rw * self.aq.Haq[self.pylayers][:,np.newaxis] )
                    if self.Nunknowns > 1:  # Multiple layers
                        for i in range(self.Nunknowns-1):
                            mat[i,ieq+i,:] -= disterm[i]
                            mat[i,ieq+i+1,:] += disterm[i+1]
                    mat[-1,ieq:ieq+self.Nunknowns,:] += self.dischargeinflayers
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
    
class HeadEquationNores:
    def equation(self):
        '''Mix-in class that returns matrix rows for head-specified conditions. (really written as constant potential element)
        Returns matrix part Nunknowns,Neq,Np, complex
        Returns rhs part Nunknowns,Nvbc,Np, complex
        '''
        mat = np.empty((self.Nunknowns, self.model.Neq, self.model.Np), 'D')
        rhs = np.zeros((self.Nunknowns, self.model.Ngvbc, self.model.Np), 'D')  # Needs to be initialized to zero
        for icp in range(self.Ncp):
            istart = icp * self.Nlayers
            ieq = 0  
            for e in self.model.elementList:
                if e.Nunknowns > 0:
                    mat[istart:istart+self.Nlayers, ieq:ieq+e.Nunknowns,:] = e.potinflayers(self.xc[icp], self.yc[icp], self.pylayers)
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
        Returns matrix part Nunknowns,Neq,Np, complex
        Returns rhs part Nunknowns,Nvbc,Np, complex
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
            #    iself = self.model.vbcList.index(self)
            #    for i in range(self.Nlayers):
            #        rhs[istart+i,self.model.Ngbc+iself,:] = self.pc[istart+i] / self.model.p
        return mat, rhs
    
class MscreenEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for multi-screen conditions where total discharge is specified.
        Works for Nunknowns = 1
        Returns matrix part Nunknowns,Neq,Np, complex
        Returns rhs part Nunknowns,Nvbc,Np, complex
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
                    head = e.potinflayers(self.xc[icp],self.yc[icp],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis,np.newaxis]  # T[self.pylayers,np.newaxis,np.newaxis] is not allowed
                    mat[istart:istart+self.Nlayers-1,ieq:ieq+e.Nunknowns,:] = head[:-1,:] - head[1:,:]
                    if e == self:
                        for i in range(self.Nlayers-1):
                            mat[istart+i,ieq+istart+i,:] -= self.resfach[istart+i] * e.dischargeinflayers[istart+i]
                            mat[istart+i,ieq+istart+i+1,:] += self.resfach[istart+i+1] * e.dischargeinflayers[istart+i+1]
                            mat[istart+i,ieq+istart:ieq+istart+i+1,:] -= self.vresfac[istart+i] * e.dischargeinflayers[istart+i]
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
        Returns matrix part Nunknowns,Neq,Np, complex
        Returns rhs part Nunknowns,Nvbc,Np, complex
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
                    head = e.potinflayers(self.xc[icp],self.yc[icp],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis,np.newaxis]  # T[self.pylayers,np.newaxis,np.newaxis] is not allowed
                    if self.Nlayers > 1: mat[istart:istart+self.Nlayers-1,ieq:ieq+e.Nunknowns,:] = head[:-1,:] - head[1:,:]
                    mat[istart+self.Nlayers-1,ieq:ieq+e.Nunknowns,:] = head[0,:] # Store head in top layer in 2nd to last equation of this control point
                    if e == self:
                        # Correct head in top layer in second to last equation to make it head inside
                        mat[istart+self.Nlayers-1,ieq+istart,:] -= self.resfach[istart] * e.dischargeinflayers[istart]
                        if icp == 0:
                            istartself = ieq  # Needed to build last equation
                        for i in range(self.Nlayers-1):
                            mat[istart+i,ieq+istart+i,:] -= self.resfach[istart+i] * e.dischargeinflayers[istart+i]
                            mat[istart+i,ieq+istart+i+1,:] += self.resfach[istart+i+1] * e.dischargeinflayers[istart+i+1]
                            #vresfac not yet used here; it is set to zero ad I don't quite now what is means yet
                            #mat[istart+i,ieq+istart:ieq+istart+i+1,:] -= self.vresfac[istart+i] * e.dischargeinflayers[istart+i]
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
                head = e.potinflayers(self.xc[0],self.yc[0],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis,np.newaxis]  # T[self.pylayers,np.newaxis,np.newaxis] is not allowed
                matlast[ieq:ieq+e.Nunknowns] -= self.Astorage * self.model.p**2 * head[0,:]
                if e == self:
                    # only need to correct first unknown 
                    matlast[ieq] += self.Astorage * self.model.p**2 * self.resfach[0] * e.dischargeinflayers[0]
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