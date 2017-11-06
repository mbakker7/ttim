from ttim import *

ml = ModelMaq(kaq = [1.0, 5.0],
                   z = [3,2, 1,0],
                   c = [10.],
                   Saq = [0.3, 0.01],
                   Sll = [0.001],
                   tmin = 1e-3,
                   tmax = 1e6,
                   M = 15)
w1 = DischargeWell(ml, xw = 0, yw = 0, rw = 1e-5, tsandQ = [(0,1)], layers = 0)
ml.solve()

