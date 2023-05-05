import numpy as np


def test_float():
    tmin = 0.1
    t = np.logspace(-1, 3, 100)
    assert (t[0] - tmin) == 0.0, "floats not equal"

def test_float2():
    tmin = 0.1
    t = np.atleast_1d(np.logspace(-1, 3, 100))
    assert (t[0] - tmin) == 0.0, "floats not equal"