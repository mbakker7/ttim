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
                        for i in range(self.Nlayers): mat[istart+i,ieq+istart+i,:] -= self.resfacp[istart+i] * e.strengthinflayers[istart+i]
                    ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                rhs[istart:istart+self.Nlayers,i,:] -= self.model.gbcList[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.pylayers)  # Pretty cool that this works, really
            if self.type == 'v':
                iself = self.model.vbcList.index(self)
                for i in range(self.Nlayers):
                    rhs[istart+i,self.model.Ngbc+iself,:] = self.pc[istart+i] / self.model.p
        return mat, rhs