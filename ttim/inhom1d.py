import numpy as np

from ttim.aquifer import AquiferData
from ttim.aquifer_parameters import param_3d, param_maq
from ttim.linesink1d import FluxDiffLineSink1D, HeadDiffLineSink1D
from ttim.stripareasink import StripAreaSinkInhom


class StripInhom(AquiferData):
    tiny = 1e-12

    def __init__(
        self,
        model,
        x1,
        x2,
        kaq,
        z,
        Haq,
        Hll,
        c,
        Saq,
        Sll,
        poraq,
        porll,
        ltype,
        topboundary,
        phreatictop,
        tsandhstar,
        tsandN,
        kzoverkh=None,
        model3d=False,
    ):
        super().__init__(
            model,
            kaq,
            z,
            Haq,
            Hll,
            c,
            Saq,
            Sll,
            poraq,
            porll,
            ltype,
            topboundary,
            phreatictop,
            kzoverkh=kzoverkh,
            model3d=model3d,
        )
        self.x1 = x1
        self.x2 = x2
        self.tsandhstar = tsandhstar
        self.tsandN = tsandN
        self.inhom_number = self.model.aq.add_inhom(self)
        self.addlinesinks = True  # Set to False not to add line-sinks

    def __repr__(self):
        return f"{self.__class__.__name__}: " + str([self.x1, self.x2])

    def is_inside(self, x, _):
        return (x >= self.x1) and (x < self.x2)

    def initialize(self):
        super().initialize()
        self.create_elements()

    def create_elements(self):
        # HeadDiff on right side, FluxDiff on left side
        if self.x1 == -np.inf:
            xin = self.x2 - self.tiny
            # xoutright = self.x2 + self.tiny
            aqin = self.model.aq.find_aquifer_data(xin, 0)
            # aqoutright = self.model.aq.find_aquifer_data(xoutright, 0)
            if self.addlinesinks:
                HeadDiffLineSink1D(
                    self.model,
                    self.x2,
                    layers=range(self.naq),
                    aq=aqin,
                    label=None,
                )
        elif self.x2 == np.inf:
            xin = self.x1 + self.tiny
            # xoutleft = self.x1 - self.tiny
            aqin = self.model.aq.find_aquifer_data(xin, 0)
            # aqoutleft = self.model.aq.find_aquifer_data(xoutleft, 0)
            if self.addlinesinks:
                FluxDiffLineSink1D(
                    self.model, self.x1, range(self.naq), aq=aqin, label=None
                )
        else:
            xin = 0.5 * (self.x1 + self.x2)
            # xoutleft = self.x1 - self.tiny
            # xoutright = self.x2 + self.tiny
            aqin = self.model.aq.find_aquifer_data(xin, 0)
            # aqleft = self.model.aq.find_aquifer_data(xoutleft, 0)
            # aqright = self.model.aq.find_aquifer_data(xoutright, 0)
            if self.addlinesinks:
                HeadDiffLineSink1D(
                    self.model, self.x2, range(self.naq), label=None, aq=aqin
                )
                FluxDiffLineSink1D(
                    self.model, self.x1, range(self.naq), label=None, aq=aqin
                )
        if self.tsandN is not None:
            assert self.topboundary == "con", Exception(
                "Infiltration can only be applied to a confined aquifer."
            )
            StripAreaSinkInhom(self.model, self.x1, self.x2, tsandN=self.tsandN)
        if self.tsandhstar is not None:
            assert self.topboundary == "sem", Exception(
                "hstar can only be implemented on top of a semi-confined aquifer."
            )
            StripHstarInhom(self.model, self.x1, self.x2, tsandhstar=self.tsandhstar)


class StripInhomMaq(StripInhom):
    def __init__(
        self,
        model,
        x1,
        x2,
        kaq=[1],
        z=[1, 0],
        c=[],
        Saq=[0.001],
        Sll=[0],
        poraq=[0.3],
        porll=[0.3],
        topboundary="conf",
        phreatictop=False,
        tsandhstar=None,
        tsandN=None,
    ):
        kaq, Haq, Hll, c, Saq, Sll, poraq, porll, ltype = param_maq(
            kaq, z, c, Saq, Sll, poraq, porll, topboundary, phreatictop
        )
        super().__init__(
            model,
            x1,
            x2,
            kaq,
            z,
            Haq,
            Hll,
            c,
            Saq,
            Sll,
            poraq,
            porll,
            ltype,
            topboundary,
            phreatictop,
            tsandhstar,
            tsandN,
        )


class StripInhom3D(StripInhom):
    def __init__(
        self,
        model,
        x1,
        x2,
        kaq=1,
        z=[4, 3, 2, 1],
        Saq=0.001,
        kzoverkh=0.1,
        poraq=0.3,
        topboundary="conf",
        phreatictop=False,
        topres=0,
        topthick=0,
        topSll=0,
        toppor=0.3,
        tsandhstar=None,
        tsandN=None,
    ):
        kaq, Haq, Hll, c, Saq, Sll, poraq, porll, ltype, z = param_3d(
            kaq,
            z,
            Saq,
            kzoverkh,
            poraq,
            phreatictop,
            topboundary,
            topres,
            topthick,
            topSll,
            toppor,
        )
        super().__init__(
            model,
            x1,
            x2,
            kaq,
            z,
            Haq,
            Hll,
            c,
            Saq,
            Sll,
            poraq,
            porll,
            ltype,
            topboundary,
            phreatictop,
            tsandhstar,
            tsandN,
        )
