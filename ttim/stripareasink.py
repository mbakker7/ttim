import numpy as np

from ttim.element import Element


class StripAreaSinkInhom(Element):
    def __init__(
        self,
        model,
        x1,
        x2,
        tsandN=[(0.0, 1.0)],
        layers=0,
        name="StripAreaSinkInhom",
        label=None,
    ):
        Element.__init__(
            self,
            model,
            nparam=1,
            nunknowns=0,
            layers=layers,
            tsandbc=tsandN,
            type="g",
            name=name,
            label=label,
        )
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.model.addelement(self)

    def __repr__(self):
        return f"{self.__class__.__name__}: " + str([self.x1, self.x2])

    def initialize(self):
        self.xc = (self.x1 + self.x2) / 2.0
        self.L = np.abs(self.x2 - self.x1)
        self.aq = self.model.aq.find_aquifer_data(self.xc, 0.0)
        self.setbc()
        self.setflowcoef()
        self.term = self.flowcoef * self.aq.lab**2 * self.aq.coef[self.layers]
        self.dischargeinf = self.aq.coef[0, :] * self.flowcoef
        self.dischargeinflayers = np.sum(
            self.dischargeinf * self.aq.eigvec[self.layers, :, :], 1
        )

    def setflowcoef(self):
        """Separate function so that this can be overloaded for other types."""
        self.flowcoef = 1.0 / self.model.p  # Step function

    def potinf(self, x, _, aq=None):
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, 0.0)
        rv = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        if aq == self.aq:
            if (x > self.x1) and (x < self.x2):
                rv[:] = self.term
        return rv

    def disvecinf(self, x, _, aq=None):
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, 0.0)
        qx = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        qy = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        return qx, qy

    def plot(self, ax):
        pass