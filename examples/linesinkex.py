import numpy as np
import matplotlib.pyplot as plt
from ttim import *

ml = ModelMaq(kaq=[1, 20, 2], z=[25, 20, 18, 10, 8, 0], c=[100, 200],
              Saq=[0.1, 1e-4, 1e-4], Sll=[0, 0], phreatictop=True,
              tmin=0.1, tmax=1000, M=20, f2py=False)
yls = [-100, 0, 100]
xls = 50 * np.ones(len(yls))
ls1 = HeadLineSinkString(ml, list(zip(xls, yls)), tsandh=[(0, 2)], layers=0, label='river')
ml.solve()
x = np.linspace(-100, 100, 101)
h = ml.headalongline(x, 50, t=100)
for i in range(3):
    plt.plot(x, h[i, 0])
plt.grid()
plt.show()