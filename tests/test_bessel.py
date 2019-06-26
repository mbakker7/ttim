from ttim import besselnumba
from ttim.bessel import bessel
import numpy as np

bessel.initialize()


def test_variables():
    list_of_vars = ['a',
                    'a1',
                    'afar',
                    'b',
                    'b1',
                    'gam',
                    'nrange',
                    'tiny',
                    'wg',
                    'xg']

    result = []
    for var in list_of_vars:
        numba_var = getattr(besselnumba, var)
        frtrn_var = getattr(bessel, var)
        # print(var, np.allclose(numba_var, frtrn_var))
        result.append(np.allclose(numba_var, frtrn_var))
    assert np.all(result) == True, "Not all variables equal!"
    return result


def test_besselk0far():
    a = bessel.besselk0far(1+1j, 11)
    b = besselnumba.besselk0far(1+1j, 11)
    assert a == b, "not equal"
    return a, b


def test_besselk0near():
    a = bessel.besselk0near(1+1j, 17)
    b = besselnumba.besselk0near(1+1j, 17)
    assert a == b, "not equal"
    return a, b


def test_besselk1near():
    a = bessel.besselk1near(1+1j, 20)
    b = besselnumba.besselk1near(1+1j, 20)
    assert a == b, "not equal"
    return a, b


def test_besselk0cheb():
    a = bessel.besselk0cheb(1+1j, 6)
    b = besselnumba.besselk0cheb(1+1j, 6)
    assert a == b, "not equal"
    return a, b


def test_besselk1cheb():
    a = bessel.besselk1cheb(1+1j, 6)
    b = besselnumba.besselk1cheb(1+1j, 6)
    assert a == b, "not equal"
    return a, b


def test_besselk0():
    a = bessel.besselk0(10., 10., 100.)
    b = besselnumba.besselk0(10., 10., 100.)
    assert a == b, "not equal"
    return a, b


def test_besselk1():
    a = bessel.besselk1(10., 10., 100.)
    b = besselnumba.besselk1(10., 10., 100.)
    assert a == b, "not equal"
    return a, b


def test_k0bessel():
    a = bessel.k0bessel(1. + 1.j)
    b = besselnumba.k0bessel(1. + 1.j)
    assert a == b, "not equal"
    return a, b


def test_besselk0v():
    omega = np.zeros(1, dtype=np.complex)
    bessel.besselk0v(10., 10., np.array([100.]), omega=omega)
    b = besselnumba.besselk0v(10., 10., np.array(
        [100.]), 1, np.zeros(1, dtype=np.complex))
    assert omega == b, "not equal"
    return omega, b


def test_k0besselv():
    omega = np.zeros(1, dtype=np.complex)
    bessel.k0besselv(np.array([1+1j]), omega=omega)
    b = besselnumba.k0besselv(
        np.array([1+1j]), 1, np.zeros(1, dtype=np.complex))
    assert omega == b, "not equal"
    return omega, b


def test_besselcheb():
    a = bessel.besselcheb(1. + 1.j, 6)
    b = besselnumba.besselcheb(1. + 1.j, 6)
    assert a == b, "not equal"
    return a, b


def test_ucheb():
    a = bessel.ucheb(1, 1, 1+1j, 1)
    b = besselnumba.ucheb(1, 1, 1+1j, 1)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselk0complex():
    a = bessel.besselk0complex(10., 10.)
    b = besselnumba.besselk0complex(10., 10.)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_lapls_int_ho():
    x = 10.
    y = 10.
    z1 = 1. + 1.j
    z2 = 2. + 2.j
    order = 1
    a = bessel.lapls_int_ho(x, y, z1, z2, order)
    b = besselnumba.lapls_int_ho(x, y, z1, z2, order)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessellsreal():
    x = 5.
    y = 5.
    x1 = 0.
    y1 = 0.
    x2 = 10.
    y2 = 10.
    lab = 100.
    a = bessel.bessellsreal(x, y, x1, y1, x2, y2, lab)
    b = besselnumba.bessellsreal(x, y, x1, y1, x2, y2, lab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessellsrealho():
    x = 5.
    y = 5.
    x1 = 0.
    y1 = 0.
    x2 = 10.
    y2 = 10.
    lab = 100.
    order = 1
    a = bessel.bessellsrealho(x, y, x1, y1, x2, y2, lab, order)
    b = besselnumba.bessellsrealho(x, y, x1, y1, x2, y2, lab, order)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells_int():
    x = 5.
    y = 5.
    lab = 100.
    z1 = 1. + 1.j
    z2 = 5. + 5.j
    a = bessel.bessells_int(x, y, z1, z2, lab)
    b = besselnumba.bessells_int(x, y, z1, z2, lab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells_int_ho():
    x = 5.
    y = 5.
    z1 = 1. + 1.j
    z2 = 5. + 5.j
    lab = 100.
    order = 1
    d1 = 1.
    d2 = -1.
    a = bessel.bessells_int_ho(x, y, z1, z2, lab, order, d1, d2)
    b = besselnumba.bessells_int_ho(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells_int_ho_qxqy():
    x = 5.
    y = 5.
    z1 = 1. + 1.j
    z2 = 5. + 5.j
    lab = 100.
    order = 1
    d1 = 1.
    d2 = -1.
    a = bessel.bessells_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
    b = besselnumba.bessells_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells_gauss():
    x = 5.
    y = 5.
    z1 = 1. + 1.j
    z2 = 5. + 5.j
    lab = 100.
    a = bessel.bessells_gauss(x, y, z1, z2, lab)
    b = besselnumba.bessells_gauss(x, y, z1, z2, lab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_lapld_int_ho():
    x = 5.
    y = 5.
    z1 = 1. + 1.j
    z2 = 5. + 5.j
    order = 1
    a = bessel.lapld_int_ho(x, y, z1, z2, order)
    b = besselnumba.lapld_int_ho(x, y, z1, z2, order)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_lapld_int_ho_d1d2():
    x = 5.
    y = 5.
    z1 = 1. + 1.j
    z2 = 5. + 5.j
    order = 1
    d1 = 1.
    d2 = -1.
    a = bessel.lapld_int_ho_d1d2(x, y, z1, z2, order, d1, d2)
    b = besselnumba.lapld_int_ho_d1d2(x, y, z1, z2, order, d1, d2)
    assert np.allclose(a, b), "not equal"
    return a, b


if __name__ == "__main__":
    t0 = test_variables()
    t1 = test_besselk0far()
    t2 = test_besselk0near()
    t3 = test_besselk1near()
    t4 = test_besselk0cheb()
    t5 = test_besselk1cheb()
    t6 = test_besselk0()
    t7 = test_besselk1()
    t8 = test_k0bessel()
    t9 = test_besselk0v()
    t10 = test_k0besselv()
    t11 = test_besselcheb()
    t12 = test_ucheb()
    t13 = test_besselk0complex()
    t14 = test_lapls_int_ho()
    t15 = test_bessellsreal()
    t16 = test_bessellsrealho()
    t17 = test_bessells_int()
    t18 = test_bessells_int_ho()
    t19 = test_bessells_int_ho_qxqy()
    t20 = test_bessells_gauss()

    txx = test_lapld_int_ho()
    tyy = test_lapld_int_ho_d1d2()
