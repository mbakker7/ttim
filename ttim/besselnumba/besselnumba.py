import numpy as np
import numba

"""
real(kind=8) :: pi, tiny
real(kind=8), dimension(0:20) :: a, b, afar, a1, b1
real(kind=8), dimension(0:20) :: nrange
real(kind=8), dimension(0:20,0:20) :: gam
real(kind=8), dimension(8) :: xg, wg

initialize
----------
implicit none
real(kind=8) :: c, fac, twologhalf
real(kind=8), dimension(0:20) :: bot
real(kind=8), dimension(1:21) :: psi
integer :: n,m
"""

tiny = 1e-10
c = np.log(0.5) + 0.577215664901532860

fac = 1.0

nrange = np.arange(21, dtype=np.float_)

a = np.zeros(21, dtype=np.float_)
a[0] = 1.0
b = np.zeros(21, dtype=np.float_)

for n in range(1, 21):
    fac = n*fac
    a[n] = 1.0 / (4.0**nrange[n] * fac**2)
    b[n] = b[n-1] + 1 / nrange[n]

b = (b - c) * a
a = -a / 2.0

gam = np.zeros((21, 21), dtype=np.float_)
for n in range(21):
    for m in range(n+1):
        gam[n, m] = np.prod(nrange[m+1:n+1]) / np.prod(nrange[1:n-m+1])

afar = np.zeros(21, dtype=np.float_)
afar[0] = np.sqrt(np.pi / 2.)

for n in range(1, 21):
    afar[n] = -(2. * n - 1.)**2 / (n * 8) * afar[n-1]

fac = 1.0
bot = np.zeros(21, dtype=np.float_)
bot[0] = 4.0
for n in range(1, 21):
    fac = n * fac
    bot[n] = fac * (n+1)*fac * 4.0**(n+1)

psi = np.zeros(21, dtype=np.float_)
for n in range(2, 22):
    psi[n-1] = psi[n-2] + 1 / (n-1)
psi = psi - 0.577215664901532860

a1 = np.empty(21, dtype=np.float_)
b1 = np.empty(21, dtype=np.float_)
twologhalf = 2 * np.log(0.5)
for n in range(21):
    a1[n] = 1 / bot[n]
    b1[n] = (twologhalf - (2.0 * psi[n] + 1 / (n+1))) / bot[n]


wg = np.zeros(8, dtype=np.float_)
xg = np.zeros(8, dtype=np.float_)

wg[0] = 0.101228536290378
wg[1] = 0.22238103445338
wg[2] = 0.31370664587789
wg[3] = 0.36268378337836
wg[4] = 0.36268378337836
wg[5] = 0.313706645877890
wg[6] = 0.22238103445338
wg[7] = 0.10122853629038

xg[0] = -0.960289856497536
xg[1] = -0.796666477413626
xg[2] = -0.525532409916329
xg[3] = -0.183434642495650
xg[4] = 0.183434642495650
xg[5] = 0.525532409916329
xg[6] = 0.796666477413626
xg[7] = 0.960289856497536


@numba.njit(nogil=True)
def besselk0near(z, Nt):
    """
    implicit none
    complex(kind=8), intent(in) :: z
    integer, intent(in) :: Nt
    complex(kind=8) :: omega
    complex(kind=8) :: rsq, log1, term
    integer :: n
    """
    rsq = z ** 2
    term = 1.0 + 0.0j
    log1 = np.log(rsq)
    omega = a[0] * log1 + b[0]

    for n in range(1, Nt+1):
        term = term * rsq
        omega = omega + (a[n] * log1 + b[n]) * term

    return omega


@numba.njit(nogil=True)
def besselk0cheb(z, Nt):
    """
    implicit none
    complex(kind=8), intent(in) :: z
    integer, intent(in) :: Nt
    complex(kind=8) :: omega
    integer :: n, n2, ts
    real(kind=8) :: a, b, c, A3, u
    complex(kind=8) :: A1, A2, cn, cnp1, cnp2, cnp3
    complex(kind=8) :: z1, z2, S, T

    """
    cnp1 = np.complex(1., 0.)
    cnp2 = np.complex(0., 0.)
    cnp3 = np.complex(0., 0.)
    a = 0.5
    c = 1.
    b = 1. + a - c

    z1 = 2. * z
    z2 = 2. * z1
    ts = (-1)**(Nt+1)
    S = ts
    T = 1.

    for n in range(Nt, -1, -1):
        u = (n+a) * (n+b)
        n2 = 2 * n
        A1 = 1. - (z2 + (n2+3.)*(n+a+1.)*(n+b+1.) / (n2+4.)) / u
        A2 = 1. - (n2+2.)*(n2+3.-z2) / u
        A3 = -(n+1.)*(n+3.-a)*(n+3.-b) / (u*(n+2.))
        cn = (2.*n+2.) * A1 * cnp1 + A2 * cnp2 + A3 * cnp3
        ts = -ts
        S = S + ts * cn
        T = T + cn
        cnp3 = cnp2
        cnp2 = cnp1
        cnp1 = cn
    cn = cn / 2.
    S = S - cn
    T = T - cn
    omega = 1. / np.sqrt(z1) * T / S
    omega = np.sqrt(np.pi) * np.exp(-z) * omega

    return omega


@numba.njit(nogil=True)
def besselk0(x, y, lab):
    """
    implicit none
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: lab
    complex(kind=8) :: z, omega
    real(kind=8) :: cond
    """
    z = np.sqrt(x**2 + y**2) / lab
    cond = np.abs(z)

    if (cond < 6):
        omega = besselk0near(z, 17)
    else:
        omega = besselk0cheb(z, 6)

    return omega

@numba.njit(nogil=True)
def bessells_int_test1(x, y, z1, z2, lab):
    """
    implicit none
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2,lab
    real(kind=8) :: biglab, biga, L, ang, tol
    complex(kind=8) :: zeta, zetabar, omega, log1, log2, term1, term2, d1minzeta, d2minzeta
    complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
    complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
    complex(kind=8), dimension(0:40) :: alpha, beta, alpha2
    integer :: n
    """
    zminzbar = np.zeros(21, dtype=np.complex_)

    L = np.abs(z2-z1)
    biga = np.abs(lab)
    ang = np.arctan2(lab.imag, lab.real)
    biglab = 2 * biga / L

    tol = 1e-12

    exprange = np.exp(-np.complex(0, 2) * ang * nrange)
    anew = a * exprange
    bnew = (b - a * np.complex(0, 2) * ang) * exprange

#zminzbar = np.zeros(21, dtype=np.complex_)
exprange = np.zeros(21, dtype=np.complex_)
anew = np.zeros(21, dtype=np.complex_)
bnew = np.zeros(21, dtype=np.complex_)

@numba.njit(nogil=True)
def bessells_int_test2(x, y, z1, z2, lab, exprange=exprange, anew=anew, bnew=bnew):
    """
    implicit none
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2,lab
    real(kind=8) :: biglab, biga, L, ang, tol
    complex(kind=8) :: zeta, zetabar, omega, log1, log2, term1, term2, d1minzeta, d2minzeta
    complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
    complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
    complex(kind=8), dimension(0:40) :: alpha, beta, alpha2
    integer :: n
    """

    L = np.abs(z2-z1)
    biga = np.abs(lab)
    ang = np.arctan2(lab.imag, lab.real)
    biglab = 2 * biga / L

    tol = 1e-12
    for i in np.arange(21):
        exprange[i] = np.exp(-2j * ang * i)
        anew[i] = a[i] * exprange[i]
        bnew[i] = (b[i] - a[i] * 2j * ang) * exprange[i]



@numba.njit(nogil=True)
def bessells_int(x, y, z1, z2, lab):
    """
    implicit none
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2,lab
    real(kind=8) :: biglab, biga, L, ang, tol
    complex(kind=8) :: zeta, zetabar, omega, log1, log2, term1, term2, d1minzeta, d2minzeta
    complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
    complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
    complex(kind=8), dimension(0:40) :: alpha, beta, alpha2
    integer :: n
    """
    zminzbar = np.zeros(21, dtype=np.complex_)

    L = np.abs(z2-z1)
    biga = np.abs(lab)
    ang = np.arctan2(lab.imag, lab.real)
    biglab = 2 * biga / L

    tol = 1e-12

    exprange = np.exp(-np.complex(0, 2) * ang * nrange)
    anew = a * exprange
    bnew = (b - a * np.complex(0, 2) * ang) * exprange

    zeta = (2 * np.complex(x, y) - (z1+z2)) / (z2-z1) / biglab
    zetabar = np.conj(zeta)
    # #for n in range(21):
    # #    zminzbar[n] = (zeta-zetabar)**(20-n)  # Ordered from high power to low power
    zminzbar[20] = 1

    for n in range(1, 21):
        # Ordered from high power to low power
        zminzbar[20-n] = zminzbar[21-n] * (zeta-zetabar)

    gamnew = np.zeros((21, 21), dtype=np.complex_)
    gam2 = np.zeros((21, 21), dtype=np.complex_)
    for n in range(21):
        gamnew[n, 0:n+1] = gam[n, 0:n+1] * zminzbar[20-n:20+1]
        gam2[n, 0:n+1] = np.conj(gamnew[n, 0:n+1])

    alpha = np.zeros(41, dtype=np.complex_)
    beta = np.zeros(41, dtype=np.complex_)
    alpha2 = np.zeros(41, dtype=np.complex_)

    alpha[0] = anew[0]
    beta[0] = bnew[0]
    alpha2[0] = anew[0]

    for n in range(1, 21):
        alpha[n:2*n+1] = alpha[n:2*n+1] + anew[n] * gamnew[n, 0:n+1]
        beta[n:2*n+1] = beta[n:2*n+1] + bnew[n] * gamnew[n, 0:n+1]
        alpha2[n:2*n+1] = alpha2[n:2*n+1] + anew[n] * gam2[n, 0:n+1]

    omega = 0
    d1minzeta = -1/biglab - zeta
    d2minzeta = 1/biglab - zeta

    if (np.abs(d1minzeta) < tol):
        d1minzeta = d1minzeta + np.complex(tol, 0)
    if (np.abs(d2minzeta) < tol):
        d2minzeta = d2minzeta + np.complex(tol, 0)

    log1 = np.log(d1minzeta)
    log2 = np.log(d2minzeta)
    term1 = 1
    term2 = 1

    # I tried to serialize this, but it didn't speed things up
    for n in range(41):
        term1 = term1 * d1minzeta
        term2 = term2 * d2minzeta
        omega = omega + (alpha[n] * log2 - alpha[n] /
                         (n + 1) + beta[n]) * term2 / (n+1)
        omega = omega - (alpha[n] * log1 - alpha[n] /
                         (n+1) + beta[n]) * term1 / (n+1)
        omega = omega + (alpha2[n] * np.conj(log2) -
                         alpha2[n] / (n+1)) * np.conj(term2) / (n+1)
        omega = omega - (alpha2[n] * np.conj(log1) -
                         alpha2[n] / (n+1)) * np.conj(term1) / (n+1)

    omega = -biga / (2*np.pi) * omega

    return omega
    
@numba.njit(nogil=True)
def bessells_gauss(x, y, z1, z2, lab):
    """
    implicit none
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), intent(in) :: lab
    complex(kind=8) :: omega
    integer :: n
    real(kind=8) :: L, x0
    complex(kind=8) :: bigz, biglab
    """
    L = np.abs(z2-z1)
    biglab = 2 * lab / L
    bigz = (2 * np.complex(x, y) - (z1+z2)) / (z2-z1)
    omega = np.complex(0, 0)
    for n in range(1, 9):
        x0 = bigz.real - xg[n-1]
        omega = omega + wg[n-1] * besselk0(x0, bigz.imag, biglab)

    omega = -L/(4*np.pi) * omega
    return omega


@numba.njit(nogil=True)
def bessellsuni(x, y, z1, z2, lab):
    """
    # Uniform strength
    implicit none
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), intent(in) :: lab
    complex(kind=8) :: omega

    integer :: Nls, n
    real(kind=8) :: Lnear, L
    complex(kind=8) :: z, delz, za, zb
    """

    Lnear = 3.
    z = np.complex(x, y)
    omega = np.complex(0., 0.)
    L = np.abs(z2-z1)
    if (L < Lnear * np.abs(lab)):  # No need to break integral up
        if (np.abs(z - 0.5 * (z1 + z2)) < 0.5 * Lnear * L):  # Do integration
            omega = bessells_int(x, y, z1, z2, lab)
        else:
            omega = bessells_gauss(x, y, z1, z2, lab)
    else:  # Break integral up in parts
        Nls = np.ceil(L / (Lnear*np.abs(lab)))
        delz = (z2 - z1) / Nls
        L = np.abs(delz)
        for n in range(1, Nls + 1):
            za = z1 + (n - 1) * delz
            zb = z1 + n * delz
            if (np.abs(z - 0.5 * (za + zb)) < 0.5 * Lnear * L):  # integration
                omega = omega + bessells_int(x, y, za, zb, lab)
            else:
                omega = omega + bessells_gauss(x, y, za, zb, lab)
    return omega


@numba.njit(nogil=True)
def bessellsuniv(x, y, z1, z2, lab, rzero):
    """
    # Uniform strength
    implicit none
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2
    integer, intent(in) :: nlab
    complex(kind=8), dimension(nlab), intent(in) :: lab
    complex(kind=8), dimension(nlab), intent(inout) :: omega
    integer :: n
    """
    nlab = len(lab)
    omega = np.zeros(nlab, dtype=np.complex_)
    za, zb, N = circle_line_intersection(z1, z2, x + y * 1j, rzero * abs(lab[0]))
    if N > 0:
        for n in range(nlab):
            omega[n] = bessellsuni(x, y, z1, z2, lab[n])
    return omega

@numba.njit(nogil=True)
def circle_line_intersection(z1, z2, zc, R):
    """
    implicit none
    complex(kind=8), intent(in) :: z1, z2, zc
    real(kind=8), intent(in) :: R
    real(kind=8), intent(inout) :: xouta, youta, xoutb, youtb
    integer, intent(inout) :: N
    real(kind=8) :: Lover2, d, xa, xb
    complex(kind=8) :: bigz, za, zb
    """
    N = 0
    za = np.complex(0, 0)
    zb = np.complex(0, 0)
    Lover2 = np.abs(z2-z1) / 2
    bigz = (2*zc - (z1+z2)) * Lover2 / (z2-z1)
    if (abs(bigz.imag) < R):
        d = np.sqrt(R ** 2 - bigz.imag ** 2)
        xa = bigz.real - d
        xb = bigz.real + d
        if ((xa < Lover2) and (xb > -Lover2)):
            N = 2
            if (xa < -Lover2):
                za = z1
            else:
                za = (xa * (z2-z1) / Lover2 + (z1+z2)) / 2.0
            if (xb > Lover2):
                zb = z2
            else:
                zb = (xb * (z2-z1) / Lover2 + (z1+z2)) / 2.0
    return za, zb, N

@numba.njit(nogil=True)
def bessellsv2(x, y, z1, z2, lab, order, R):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,R
    complex(kind=8), intent(in) :: z1,z2
    integer, intent(in) :: nlab
    real(kind=8) :: d1, d2
    complex(kind=8), dimension(nlab), intent(in) :: lab
    complex(kind=8), dimension(order+1,nlab) :: omega
    integer :: n, nterms
    """
    nlab = len(lab)
    nterms = order+1
    omega = np.zeros((order+1, nlab), dtype=np.complex_)
    # Check if endpoints need to be adjusted using the largest lambda (the first one)
    d1, d2 = find_d1d2(z1, z2, np.complex(x, y), R*np.abs(lab[0]))
    for n in range(nlab):
        omega[:nterms+1, n] = bessells(x, y, z1, z2, lab[n], order, d1, d2)
    return omega

@numba.njit(nogil=True)
def find_d1d2(z1, z2, zc, R):
    """
    implicit none
    complex(kind=8), intent(in) :: z1, z2, zc
    real(kind=8), intent(in) :: R
    real(kind=8), intent(inout) :: d1, d2
    real(kind=8) :: Lover2, d, xa, xb
    complex(kind=8) :: bigz
    """
    d1 = -1.
    d2 = 1.
    Lover2 = np.abs(z2-z1) / 2
    bigz = (2*zc - (z1+z2)) * Lover2 / (z2-z1)
    if (np.abs((bigz.imag)) < R):
        d = np.sqrt(R**2 - bigz.imag**2)
        xa = bigz.real - d
        xb = bigz.real + d
        if ((xa < Lover2) and (xb > -Lover2)):
            if (xa < -Lover2):
                d1 = -1.
            else:
                d1 = xa / Lover2
            if (xb > Lover2):
                d2 = 1.
            else:
                d2 = xb / Lover2
    return d1, d2

@numba.njit(nogil=True)
def bessells(x, y, z1, z2, lab, order, d1in, d2in):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1in,d2in
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), intent(in) :: lab
    complex(kind=8), dimension(0:order) :: omega

    integer :: Nls, n
    real(kind=8) :: Lnear, L, d1, d2, delta
    complex(kind=8) :: z, delz, za, zb
    """
    omega = np.zeros(order + 1, dtype=np.complex_)
    Lnear = 3
    z = np.complex(x, y)
    L = np.abs(z2-z1)
    if (L < Lnear*np.abs(lab)):  # No need to break integral up
        if (np.abs(z - 0.5*(z1+z2)) < 0.5 * Lnear * L):  # Do integration
            omega = bessells_int_ho(x, y, z1, z2, lab, order, d1in, d2in)
        else:
            omega = bessells_gauss_ho_d1d2(
                x, y, z1, z2, lab, order, d1in, d2in)
    else:  # Break integral up in parts
        Nls = np.ceil(L / (Lnear*np.abs(lab)))
        delta = 2 / Nls
        delz = (z2-z1)/Nls
        L = np.abs(delz)
        for n in range(1, Nls+1):
            d1 = -1 + (n-1) * delta
            d2 = -1 + n * delta
            if ((d2 < d1in) or (d1 > d2in)):
                continue
            d1 = np.max(np.array([d1, d1in]))
            d2 = np.min(np.array([d2, d2in]))
            za = z1 + (n-1) * delz
            zb = z1 + n * delz
            if (np.abs(z - 0.5*(za+zb)) < 0.5 * Lnear * L):  # Do integration
                omega = omega + \
                    bessells_int_ho(x, y, z1, z2, lab, order, d1, d2)
            else:
                omega = omega + \
                    bessells_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2)
    return omega

@numba.njit(nogil=True)
def bessells_int_ho(x, y, z1, z2, lab, order, d1, d2):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1,d2
    complex(kind=8), intent(in) :: z1,z2,lab
    complex(kind=8), dimension(0:order) :: omega
    real(kind=8) :: biglab, biga, L, ang, tol
    complex(kind=8) :: zeta, zetabar, log1, log2, term1, term2, d1minzeta, d2minzeta, cm
    complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
    complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
    complex(kind=8), dimension(0:40) :: alpha, beta, alpha2
    complex(kind=8), dimension(0:50) :: alphanew, betanew, alphanew2 ! Order fixed to 10
    integer :: m, n, p
    """
    L = np.abs(z2-z1)
    biga = np.abs(lab)
    ang = np.arctan2(lab.imag, lab.real)
    biglab = 2 * biga / L

    tol = 1e-12

    exprange = np.exp(-np.complex(0, 2) * ang * nrange)
    anew = a * exprange
    bnew = (b - a * np.complex(0, 2) * ang) * exprange

    zeta = (2 * np.complex(x, y) - (z1+z2)) / (z2-z1) / biglab
    zetabar = np.conj(zeta)

    # #for n in range(21):
    # #    zminzbar[n] = (zeta-zetabar)**(20-n)  # Ordered from high power to low power
    #
    zminzbar = np.zeros(21, dtype=np.complex_)
    zminzbar[20] = 1

    for n in range(1, 21):
        # Ordered from high power to low power
        zminzbar[20-n] = zminzbar[21-n] * (zeta-zetabar)

    gamnew = np.zeros((21, 21), dtype=np.complex_)
    gam2 = np.zeros((21, 21), dtype=np.complex_)
    for n in range(21):
        gamnew[n, 0:n+1] = gam[n, 0:n+1] * zminzbar[20-n:20+1]
        gam2[n, 0:n+1] = np.conj(gamnew[n, 0:n+1])

    alpha = np.zeros(41, dtype=np.complex_)
    beta = np.zeros(41, dtype=np.complex_)
    alpha2 = np.zeros(41, dtype=np.complex_)
    alpha[0] = anew[0]
    beta[0] = bnew[0]
    alpha2[0] = anew[0]
    for n in range(1, 21):
        alpha[n:2*n+1] = alpha[n:2*n+1] + anew[n] * gamnew[n, 0:n+1]
        beta[n:2*n+1] = beta[n:2*n+1] + bnew[n] * gamnew[n, 0:n+1]
        alpha2[n:2*n+1] = alpha2[n:2*n+1] + anew[n] * gam2[n, 0:n+1]

    d1minzeta = d1/biglab - zeta
    d2minzeta = d2/biglab - zeta
    # #d1minzeta = -1/biglab - zeta
    # #d2minzeta = 1/biglab - zeta
    if (np.abs(d1minzeta) < tol):
        d1minzeta = d1minzeta + np.complex(tol, 0)
    if (np.abs(d2minzeta) < tol):
        d2minzeta = d2minzeta + np.complex(tol, 0)
    log1 = np.log(d1minzeta)
    log2 = np.log(d2minzeta)

    alphanew = np.zeros(51, dtype=np.complex_)
    alphanew2 = np.zeros(51, dtype=np.complex_)
    betanew = np.zeros(51, dtype=np.complex_)

    omega = np.zeros(order+1, dtype=np.complex_)

    for p in range(order+1):
        alphanew[0:40+p+1] = 0
        betanew[0:40+p+1] = 0
        alphanew2[0:40+p+1] = 0

        for m in range(0, p+1):
            cm = biglab**p * gam[p, m] * zeta**(p-m)
            alphanew[m:40+m+1] = alphanew[m:40+m+1] + cm * alpha[0:40+1]
            betanew[m:40+m+1] = betanew[m:40+m+1] + cm * beta[0:40+1]
            cm = biglab**p * gam[p, m] * zetabar**(p-m)
            alphanew2[m:40+m+1] = alphanew2[m:40+m+1] + cm * alpha2[0:40+1]

        omega[p] = 0
        term1 = 1
        term2 = 1
        for n in range(41):
            term1 = term1 * d1minzeta
            term2 = term2 * d2minzeta
            omega[p] = omega[p] + (alphanew[n] * log2 -
                                   alphanew[n] / (n+1) + betanew[n]) * term2 / (n+1)
            omega[p] = omega[p] - (alphanew[n] * log1 -
                                   alphanew[n] / (n+1) + betanew[n]) * term1 / (n+1)
            omega[p] = omega[p] + (alphanew2[n] * np.conj(log2) -
                                   alphanew2[n] / (n+1)) * np.conj(term2) / (n+1)
            omega[p] = omega[p] - (alphanew2[n] * np.conj(log1) -
                                   alphanew2[n] / (n+1)) * np.conj(term1) / (n+1)

    omega = -biga / (2*np.pi) * omega
    return omega

@numba.njit(nogil=True)
def bessells_gauss_ho(x, y, z1, z2, lab, order):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), intent(in) :: lab
    complex(kind=8), dimension(0:order) :: omega
    integer :: n, p
    real(kind=8) :: L, x0
    complex(kind=8) :: bigz, biglab
    complex(kind=8), dimension(8) :: k0
    """
    L = np.abs(z2-z1)
    biglab = 2 * lab / L
    bigz = (2 * np.complex(x, y) - (z1+z2)) / (z2-z1)

    k0 = np.zeros(8, dtype=np.complex_)
    for n in range(8):
        x0 = bigz.real - xg[n]
        k0[n] = besselk0(x0, bigz.imag, biglab)

    omega = np.zeros(order+1, dtype=np.complex_)
    for p in range(order+1):
        omega[p] = np.complex(0, 0)
        for n in range(8):
            omega[p] = omega[p] + wg[n] * xg[n]**p * k0[n]
        omega[p] = -L/(4*np.pi) * omega[p]

    return omega

@numba.njit(nogil=True)
def bessells_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2):
    """
    Returns integral from d1 to d2 along real axis while strength is still Delta^order from -1 to +1

    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1,d2
    complex(kind=8), intent(in) :: z1,z2,lab
    complex(kind=8), dimension(0:order) :: omega, omegac
    integer :: n, m
    real(kind=8) :: xp, yp, dc, fac
    complex(kind=8) :: z1p,z2p,bigz1,bigz2
    """
    omega = np.zeros(order+1, dtype=np.complex_)
    bigz1 = np.complex(d1, 0)
    bigz2 = np.complex(d2, 0)
    z1p = 0.5 * (z2-z1) * bigz1 + 0.5 * (z1+z2)
    z2p = 0.5 * (z2-z1) * bigz2 + 0.5 * (z1+z2)
    omegac = bessells_gauss_ho(x, y, z1p, z2p, lab, order)
    dc = (d1+d2) / (d2-d1)
    for n in range(order+1):
        for m in range(n+1):
            omega[n] = omega[n] + gam[n, m] * dc**(n-m) * omegac[m]
        omega[n] = (0.5 * (d2-d1))**n * omega[n]
    return omega

@numba.njit(nogil=True)
def isinside(z1, z2, zc, R):
    """ Checks whether point zc is within oval with 'radius' R from line element
    implicit none
    complex(kind=8), intent(in) :: z1, z2, zc
    real(kind=8), intent(in) :: R
    integer :: irv
    real(kind=8) :: Lover2, d, xa, xb
    complex(kind=8) :: bigz
    """
    irv = 0
    Lover2 = np.abs(z2 - z1) / 2
    bigz = (2 * zc - (z1 + z2)) * np.abs(z2 - z1) / (2 * (z2 - z1))
    if (np.abs(bigz.imag) < R):
        d = np.sqrt(R ** 2 - bigz.imag ** 2)
        xa = bigz.real - d
        xb = bigz.real + d
        if ((xa < Lover2) and (xb > - Lover2)):
            irv = 1
    return irv



