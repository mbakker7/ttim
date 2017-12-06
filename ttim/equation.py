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