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
def besselk0far(z, Nt):
    """
    implicit none
    complex(kind=8), intent(in) :: z
    integer, intent(in) :: Nt
    complex(kind=8) :: omega, term
    integer :: n

    """

    term = 1.0
    omega = afar[0]

    for n in range(1, Nt+1):
        term = term / z
        omega = omega + afar[n] * term

    omega = np.exp(-z) / np.sqrt(z) * omega

    return omega


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
    rsq = z**2
    term = 1.0 + 0.0j
    log1 = np.log(rsq)
    omega = a[0] * log1 + b[0]

    for n in range(1, Nt+1):
        term = term * rsq
        omega = omega + (a[n] * log1 + b[n]) * term

    return omega


@numba.njit(nogil=True)
def besselk1near(z, Nt):
    """
    implicit none
    complex(kind=8), intent(in) :: z
    integer, intent(in) :: Nt
    complex(kind=8) :: omega
    complex(kind=8) :: zsq, log1, term
    integer :: n
    """
    zsq = z**2
    term = z
    log1 = np.log(zsq)
    omega = 1. / z + (a1[0] * log1 + b1[0]) * z

    for n in range(1, Nt+1):
        term = term * zsq
        omega = omega + (a1[n] * log1 + b1[n]) * term

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
def besselk1cheb(z, Nt):
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
    cnp1 = 1.0 + 0.0j
    cnp2 = 0.0 + 0.0j
    cnp3 = 0.0 + 0.0j

    a = 1.5
    c = 3.0
    b = 1.0 + a - c

    z1 = 2 * z
    z2 = 2 * z1
    ts = (-1)**(Nt+1)
    S = ts
    T = 1.0

    for n in range(Nt, -1, -1):
        u = (n + a) * (n + b)
        n2 = 2 * n
        A1 = 1 - (z2 + (n2 + 3) * (n + a + 1) * (n + b + 1) / (n2 + 4)) / u
        A2 = 1 - (n2 + 2) * (n2 + 3 - z2) / u
        A3 = -(n + 1) * (n + 3 - a) * (n + 3 - b) / (u * (n + 2))
        cn = (2 * n + 2) * A1 * cnp1 + A2 * cnp2 + A3 * cnp3
        ts = -ts
        S = S + ts * cn
        T = T + cn
        cnp3 = cnp2
        cnp2 = cnp1
        cnp1 = cn

    cn = cn / 2
    S = S - cn
    T = T - cn
    omega = 1 / (np.sqrt(z1) * z1) * T / S
    omega = 2 * z * np.sqrt(np.pi) * np.exp(-z) * omega

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
def besselk1(x, y, lab):
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
        omega = besselk1near(z, 20)
    else:
        omega = besselk1cheb(z, 6)

    return omega


@numba.njit(nogil=True)
def k0bessel(z):
    """
    implicit none
    complex(kind=8), intent(in) :: z
    complex(kind=8) :: omega
    real(kind=8) :: cond
    """
    cond = np.abs(z)

    if (cond < 6):
        omega = besselk0near(z, 17)
    else:
        omega = besselk0cheb(z, 6)

    return omega


@numba.njit(nogil=True)
def besselk0v(x, y, lab, nlab, omega):
    """
    implicit none
    real(kind=8), intent(in) :: x,y
    integer, intent(in) :: nlab
    complex(kind=8), dimension(nlab), intent(in) :: lab
    complex(kind=8), dimension(nlab), intent(inout) :: omega
    integer :: n

    """
    for n in range(nlab):
        omega[n] = besselk0(x, y, lab[n])
    return omega


@numba.njit(nogil=True)
def k0besselv(z, nlab, omega):
    """
    implicit none
    integer, intent(in) :: nlab
    complex(kind=8), dimension(nlab), intent(in) :: z
    complex(kind=8), dimension(nlab), intent(inout) :: omega
    integer :: n
    """
    for n in range(nlab):
        omega[n] = k0bessel(z[n])
    return omega


@numba.njit(nogil=True)
def besselk0OLD(x, y, lab):
    """
    implicit none
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: lab
    complex(kind=8) :: z, omega
    real(kind=8) :: cond
    """
    z = np.sqrt(x**2 + y**2) / lab
    cond = np.abs(z)

    if (cond < 4):
        omega = besselk0near(z, 12)  # Was 10
    elif (cond < 8):
        omega = besselk0near(z, 18)
    elif (cond < 12):
        omega = besselk0far(z, 11)  # was 6
    else:
        omega = besselk0far(z, 8)  # was 4

    return omega


# @numba.njit(nogil=True)
# def besselcheb(z, Nt):
#     """
#     implicit none
#     integer, intent(in) :: Nt
#     complex(kind=8), intent(in) :: z
#     complex(kind=8) :: omega
#     complex(kind=8) :: z2
#     """
#     z2 = 2.0 * z
#     omega = np.sqrt(np.pi) * np.exp(-z) * ucheb(0.5, 1, z2, Nt)
#     return omega
# 
# 
# @numba.njit(nogil=True)
# def ucheb(a, c, z, n0):
#     """
#     implicit none
#     integer, intent(in) :: c, n0
#     real(kind=8), intent(in) :: a
#     complex(kind=8), intent(in) :: z
#     complex(kind=8) :: ufunc
# 
#     integer :: n, n2, ts
#     real(kind=8) :: A3, u, b
#     complex(kind=8) :: A1, A2, cn,cnp1,cnp2,cnp3
#     complex(kind=8) :: z2, S, T
# 
#     """
# 
#     cnp1 = np.complex(1.)
#     cnp2 = np.complex(0.)
#     cnp3 = np.complex(0.)
#     ts = (-1)**(n0+1)
#     S = ts
#     T = 1.
#     z2 = 2. * z
#     b = 1. + a - c
# 
#     for n in range(n0, -1, -1):
#         u = (n+a) * (n+b)
#         n2 = 2 * n
#         A1 = 1. - (z2 + (n2+3)*(n+a+1)*(n+b+1.) / (n2+4.)) / u
#         A2 = 1. - (n2+2.)*(n2+3.-z2) / u
#         A3 = -(n+1)*(n+3-a)*(n+3-b) / (u*(n+2))
#         cn = (2*n+2) * A1 * cnp1 + A2 * cnp2 + A3 * cnp3
#         ts = -ts
#         S = S + ts * cn
#         T = T + cn
#         cnp3 = cnp2
#         cnp2 = cnp1
#         cnp1 = cn
#     cn = cn / 2.
#     S = S - cn
#     T = T - cn
#     ufunc = z**(-a) * T / S
#     return ufunc


# @numba.njit(nogil=True)
# def besselk0complex(x, y):
#     """
#     implicit none
#     real(kind=8), intent(in) :: x,y
#     real(kind=8) :: phi
#     real(kind=8) :: d
#     complex(kind=8) :: zeta, zetabar, omega, logdminzdminzbar, dminzeta, term
#     complex(kind=8), dimension(0:20) :: zminzbar
#     complex(kind=8), dimension(0:20,0:20) :: gamnew
#     complex(kind=8), dimension(0:40) :: alpha, beta
#     """
#     d = 0.
#     zeta = np.complex(x, y)
#     zetabar = np.conj(zeta)
#     zminzbar = np.zeros(21, dtype=np.complex_)
#     for n in range(21):
#         # Ordered from high power to low power
#         zminzbar[n] = (zeta-zetabar)**(20-n)
# 
#     gamnew = np.asarray(gam, dtype=np.complex_)
#     for n in range(21):
#         gamnew[n, 0:n+1] = gamnew[n, 0:n+1] * zminzbar[20-n:20+1]
# 
#     alpha = np.zeros(41, dtype=np.complex_)
#     beta = np.zeros(41, dtype=np.complex_)
#     alpha[0] = a[0]
#     beta[0] = b[0]
#     for n in range(1, 21):
#         alpha[n:2*n+1] = alpha[n:2*n+1] + a[n] * gamnew[n, :n+1]
#         beta[n:2*n+1] = beta[n:2*n+1] + b[n] * gamnew[n, :n+1]
# 
#     omega = np.complex(0., 0.)
#     logdminzdminzbar = np.log((d-zeta) * (d-zetabar))
#     dminzeta = d - zeta
#     term = 1.
# 
#     for n in range(41):
#         omega = omega + (alpha[n] * logdminzdminzbar + beta[n]) * term
#         term = term * dminzeta
# 
#     phi = np.real(omega)
# 
#     return phi


@numba.njit(nogil=True)
def lapls_int_ho(x, y, z1, z2, order):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), dimension(1:order+1) :: omega
    complex(kind=8), dimension(1:order+2) :: qm
    integer :: m, i
    real(kind=8) :: L
    complex(kind=8) :: z, zplus1, zmin1, log1, log2, log3, zpower
    """
    omega = np.empty((order+1,), dtype=np.complex_)
    qm = np.empty((order+2,), dtype=np.complex_)

    L = np.abs(z2-z1)
    z = (2 * np.complex(x, y) - (z1+z2)) / (z2-z1)
    zplus1 = z + 1
    zmin1 = z - 1

    # Not sure if this gives correct answer at corner point(z also appears in qm)
    # should really be caught in code that calls this def
    if (np.abs(zplus1) < tiny):
        zplus1 = tiny
    if (np.abs(zmin1) < tiny):
        zmin1 = tiny

    qm[0] = 0
    qm[1] = 2
    for m in range(2, order+1, 2):
        qm[m+1] = qm[m-1] * z * z + 2 / m

    for m in range(1, order+1, 2):
        qm[m+1] = qm[m] * z

    log1 = np.log((zmin1) / (zplus1))
    log2 = np.log(zmin1)
    log3 = np.log(zplus1)

    zpower = 1
    for i in range(1, order+2):
        zpower = zpower * z
        omega[i-1] = -L/(4*np.pi*i) * (zpower * log1 +
                                       qm[i] - log2 + (-1)**i * log3)

    return omega


@numba.njit(nogil=True)
def bessellsreal(x, y, x1, y1, x2, y2, lab):
    """
    implicit none
    real(kind=8), intent(in) :: x,y,x1,y1,x2,y2,lab
    real(kind=8) :: phi, biglab, biga, L
    complex(kind=8) :: z1, z2, zeta, zetabar, omega, log1, log2, term1, term2, d1minzeta, d2minzeta
    complex(kind=8), dimension(0:20) :: zminzbar
    complex(kind=8), dimension(0:20,0:20) :: gamnew
    complex(kind=8), dimension(0:40) :: alpha, beta

    """
    zminzbar = np.zeros(21, dtype=np.complex_)

    z1 = np.complex(x1, y1)
    z2 = np.complex(x2, y2)
    L = np.abs(z2-z1)
    biga = np.abs(lab)
    biglab = 2 * biga / L

    zeta = (2 * np.complex(x, y) - (z1+z2)) / (z2-z1) / biglab
    zetabar = np.conj(zeta)
    for n in range(21):
        # Ordered from high power to low power
        zminzbar[n] = (zeta-zetabar)**(20-n)
    gamnew = np.zeros((21, 21), dtype=np.complex_)
    for n in range(21):
        gamnew[n, 0:n+1] = gam[n, 0:n+1] * zminzbar[20-n:20+1]

    alpha = np.zeros(41, dtype=np.complex_)
    beta = np.zeros(41, dtype=np.complex_)
    alpha[0] = a[0]
    beta[0] = b[0]
    for n in range(1, 21):
        alpha[n:2*n+1] = alpha[n:2*n+1] + a[n] * gamnew[n, 0:n+1]
        beta[n:2*n+1] = beta[n:2*n+1] + b[n] * gamnew[n, 0:n+1]

    omega = 0
    d1minzeta = -1/biglab - zeta
    d2minzeta = 1/biglab - zeta
    log1 = np.log(d1minzeta)
    log2 = np.log(d2minzeta)
    term1 = 1
    term2 = 1

    # I tried to serialize this, but it didn't speed things up
    for n in range(41):
        term1 = term1 * d1minzeta
        term2 = term2 * d2minzeta
        omega = omega + (2 * alpha[n] * log2 - 2 *
                         alpha[n] / (n+1) + beta[n]) * term2 / (n+1)
        omega = omega - (2 * alpha[n] * log1 - 2 *
                         alpha[n] / (n+1) + beta[n]) * term1 / (n+1)

    phi = -biga / (2*np.pi) * np.real(omega)

    return phi


@numba.njit(nogil=True)
def bessellsrealho(x, y, x1, y1, x2, y2, lab, order):
    """
    implicit none
    real(kind=8), intent(in) :: x,y,x1,y1,x2,y2,lab
    integer, intent(in) :: order
    real(kind=8), dimension(0:order) :: phi
    real(kind=8) :: biglab, biga, L
    complex(kind=8) :: z1, z2, zeta, zetabar, omega, log1, log2, term1, term2, d1minzeta, d2minzeta
    complex(kind=8), dimension(0:20) :: zminzbar
    complex(kind=8), dimension(0:20,0:20) :: gamnew
    complex(kind=8), dimension(0:40) :: alpha, beta
    complex(kind=8) :: cm
    complex(kind=8), dimension(0:50) :: alphanew, betanew ! Maximum programmed order is 10
    integer :: n, m, p
    """
    phi = np.zeros(order+1, dtype=np.float_)
    zminzbar = np.zeros(21, dtype=np.complex_)

    z1 = np.complex(x1, y1)
    z2 = np.complex(x2, y2)
    L = np.abs(z2-z1)
    biga = np.abs(lab)
    biglab = 2 * biga / L

    zeta = (2 * np.complex(x, y) - (z1+z2)) / (z2-z1) / biglab
    zetabar = np.conj(zeta)
    for n in range(21):
        # Ordered from high power to low power
        zminzbar[n] = (zeta-zetabar)**(20-n)
        
    gamnew = np.zeros((21, 21), dtype=np.complex_)
    for n in range(21):
        gamnew[n, 0:n+1] = gam[n, 0:n+1] * zminzbar[20-n:20+1]

    alpha = np.zeros(41, dtype=np.complex_)
    beta = np.zeros(41, dtype=np.complex_)
    alpha[0] = a[0]
    beta[0] = b[0]
    for n in range(1, 21):
        alpha[n:2*n+1] = alpha[n:2*n+1] + a[n] * gamnew[n, 0:n+1]
        beta[n:2*n+1] = beta[n:2*n+1] + b[n] * gamnew[n, 0:n+1]

    d1minzeta = -1/biglab - zeta
    d2minzeta = 1/biglab - zeta
    log1 = np.log(d1minzeta)
    log2 = np.log(d2minzeta)

    alphanew = np.zeros(51, dtype=np.complex_)
    betanew = np.zeros(51, dtype=np.complex_)

    for p in range(order+1):
        alphanew[0:40+p] = 0
        betanew[0:40+p] = 0
        for m in range(p+1):
            cm = biglab**p * gam[p, m] * zeta**(p-m)
            alphanew[m:40+m+1] = alphanew[m:40+m+1] + cm * alpha[0:40+1]
            betanew[m:40+m+1] = betanew[m:40+m+1] + cm * beta[0:40+1]

        omega = 0
        term1 = 1
        term2 = 1
        # I tried to serialize this, but it didn't speed things up
        for n in range(40 + p + 1):
            term1 = term1 * d1minzeta
            term2 = term2 * d2minzeta
            omega = omega + \
                (2 * alphanew[n] * log2 - 2 * alphanew[n] /
                 (n+1) + betanew[n]) * term2 / (n+1)
            omega = omega - \
                (2 * alphanew[n] * log1 - 2 * alphanew[n] /
                 (n+1) + betanew[n]) * term1 / (n+1)

        phi[p] = -biga / (2*np.pi) * np.real(omega)

    return phi


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
                         (n+1) + beta[n]) * term2 / (n+1)
        omega = omega - (alpha[n] * log1 - alpha[n] /
                         (n+1) + beta[n]) * term1 / (n+1)
        omega = omega + (alpha2[n] * np.conj(log2) -
                         alpha2[n] / (n+1)) * np.conj(term2) / (n+1)
        omega = omega - (alpha2[n] * np.conj(log1) -
                         alpha2[n] / (n+1)) * np.conj(term1) / (n+1)

    omega = -biga / (2*np.pi) * omega

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
def bessells_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1,d2
    complex(kind=8), intent(in) :: z1,z2,lab
    complex(kind=8), dimension(0:2*order+1) :: qxqy
    complex(kind=8), dimension(0:order) :: qx, qy
    real(kind=8) :: biglab, biga, L, ang, angz, tol, bigx, bigy
    complex(kind=8) :: zeta, zetabar, log1, log2, term1, term2, d1minzeta, d2minzeta, bigz
    complex(kind=8) :: cm, biglabcomplex
    complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
    complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
    complex(kind=8), dimension(0:40) :: alpha, beta, alpha2
    complex(kind=8), dimension(0:51) :: alphanew, betanew, alphanew2 ! Order fixed to 10
    complex(kind=8), dimension(0:order+1) :: omega ! To store intermediate result
    complex(kind=8), dimension(0:order) :: omegalap ! To store intermediate result

    integer :: m, n, p
    """
    zminzbar = np.zeros(21, dtype=np.complex_)

    L = np.abs(z2-z1)
    bigz = (2 * np.complex(x, y) - (z1+z2)) / (z2-z1)
    bigx = bigz.real
    bigy = bigz.imag
    biga = np.abs(lab)
    ang = np.arctan2(lab.imag, lab.real)
    angz = np.arctan2((z2-z1).imag, (z2-z1).real)
    biglab = 2 * biga / L
    biglabcomplex = 2.0 * lab / L

    tol = 1e-12

    exprange = np.exp(-np.complex(0, 2) * ang * nrange)
    anew = a1 * exprange
    bnew = (b1 - a1 * np.complex(0, 2) * ang) * exprange

    zeta = (2 * np.complex(x, y) - (z1+z2)) / (z2-z1) / biglab
    zetabar = np.conj(zeta)
    zminzbar[20] = 1

    for n in range(1, 21):
        # Ordered from high power to low po
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

    d1minzeta = d1 / biglab - zeta
    d2minzeta = d2 / biglab - zeta

    if (np.abs(d1minzeta) < tol):
        d1minzeta = d1minzeta + np.complex(tol, 0)
    if (np.abs(d2minzeta) < tol):
        d2minzeta = d2minzeta + np.complex(tol, 0)
    log1 = np.log(d1minzeta)
    log2 = np.log(d2minzeta)

    alphanew = np.zeros(52, dtype=np.complex_)
    alphanew2 = np.zeros(52, dtype=np.complex_)
    betanew = np.zeros(52, dtype=np.complex_)

    omega = np.zeros(order+2, dtype=np.complex_)
    qxqy = np.zeros(2*order+2, dtype=np.complex_)

    for p in range(0, order+2):

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
        for n in range(40 + p + 1):
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

    omega = biglab / (2*np.pi*biglabcomplex**2) * omega
    omegalap = lapld_int_ho_d1d2(x, y, z1, z2, order, d1, d2)

    # multiplication with 2/L inherently included
    qx = -(bigx * omega[0:order+1] - omega[1:order+2] + omegalap.imag)
    qy = -(bigy * omega[0:order+1] + omegalap.real)

    qxqy[0:order+1] = qx * np.cos(angz) - qy * np.sin(angz)
    qxqy[order+1:2*order+2] = qx * np.sin(angz) + qy * np.cos(angz)

    return qxqy


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
def bessells_gauss_ho_qxqy(x, y, z1, z2, lab, order):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), intent(in) :: lab
    complex(kind=8), dimension(0:2*order+1) :: qxqy
    integer :: n, p
    real(kind=8) :: L, bigy, angz
    complex(kind=8) :: bigz, biglab
    real(kind=8), dimension(8) :: r, xmind
    complex(kind=8), dimension(8) :: k1
    complex(kind=8), dimension(0:order) :: qx,qy
    """
    qxqy = np.zeros(2*order+2, dtype=np.complex_)
    xmind = np.zeros(8, dtype=np.complex_)
    k1 = np.zeros(8, dtype=np.complex_)
    r = np.zeros(8, dtype=np.complex_)

    L = np.abs(z2-z1)
    biglab = 2 * lab / L
    bigz = (2 * np.complex(x, y) - (z1+z2)) / (z2-z1)
    bigy = bigz.imag
    for n in range(8):
        xmind[n] = bigz.real - xg[n]
        r[n] = np.sqrt(xmind[n]**2 + bigz.imag**2)
        k1[n] = besselk1(xmind[n], bigz.imag, biglab)

    qx = np.zeros(order+1, dtype=np.complex_)
    qy = np.zeros(order+1, dtype=np.complex_)
    for p in range(order+1):
        for n in range(8):
            qx[p] = qx[p] + wg[n] * xg[n]**p * xmind[n] * k1[n] / r[n]
            qy[p] = qy[p] + wg[n] * xg[n]**p * bigy * k1[n] / r[n]

    qx = -qx * L / (4*np.pi*biglab) * 2/L
    qy = -qy * L / (4*np.pi*biglab) * 2/L

    angz = np.arctan2((z2-z1).imag, (z2-z1).real)
    qxqy[0:order+1] = qx * np.cos(angz) - qy * np.sin(angz)
    qxqy[order+1:2*order+2] = qx * np.sin(angz) + qy * np.cos(angz)

    return qxqy


@numba.njit(nogil=True)
def bessells_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2):
    """
    Returns integral from d1 to d2 along real axis while strength is still Delta^order from -1 to +1

    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1,d2
    complex(kind=8), intent(in) :: z1,z2,lab
    complex(kind=8), dimension(0:2*order+1) :: qxqy, qxqyc
    integer :: n, m
    real(kind=8) :: xp, yp, dc, fac
    complex(kind=8) :: z1p,z2p,bigz1,bigz2
    """
    qxqy = np.zeros(2*order+2, dtype=np.complex_)

    bigz1 = np.complex(d1, 0.)
    bigz2 = np.complex(d2, 0.)
    z1p = 0.5 * (z2-z1) * bigz1 + 0.5 * (z1+z2)
    z2p = 0.5 * (z2-z1) * bigz2 + 0.5 * (z1+z2)
    qxqyc = bessells_gauss_ho_qxqy(x, y, z1p, z2p, lab, order)
    dc = (d1+d2) / (d2-d1)
    for n in range(order+1):
        for m in range(n+1):
            qxqy[n] = qxqy[n] + gam[n, m] * dc**(n-m) * qxqyc[m]
            qxqy[n+order+1] = qxqy[n+order+1] + \
                gam[n, m] * dc**(n-m) * qxqyc[m+order+1]
        qxqy[n] = (0.5*(d2-d1))**n * qxqy[n]
        qxqy[n+order+1] = (0.5*(d2-d1))**n * qxqy[n+order+1]

    return qxqy


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
    omega = np.zeros(order+1, dtype=np.complex_)

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
def bessellsv(x, y, z1, z2, lab, order, R, nlab):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,R
    complex(kind=8), intent(in) :: z1,z2
    integer, intent(in) :: nlab
    real(kind=8) :: d1, d2
    complex(kind=8), dimension(nlab), intent(in) :: lab
    complex(kind=8), dimension(nlab*(order+1)) :: omega
    integer :: n, nterms
    """
    nterms = order+1
    omega = np.zeros(nlab*(order+1), dtype=np.complex_)
    # Check if endpoints need to be adjusted using
    # the largest lambda (the first one)
    d1, d2 = find_d1d2(z1, z2, np.complex(x, y), R*np.abs(lab[0]))
    for n in range(0, nlab):
        omega[n*nterms:(n+1)*nterms] = bessells(x, y, z1, z2, lab[n],
                                                  order, d1, d2)
    return omega


@numba.njit(nogil=True)
def bessellsv2(x, y, z1, z2, lab, order, R, nlab):
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
    nterms = order+1
    omega = np.zeros((order+1, nlab), dtype=np.complex_)
    # Check if endpoints need to be adjusted using the largest lambda (the first one)
    d1, d2 = find_d1d2(z1, z2, np.complex(x, y), R*np.abs(lab[0]))
    for n in range(nlab):
        omega[:nterms+1, n] = bessells(x, y, z1, z2, lab[n], order, d1, d2)
    return omega


@numba.njit(nogil=True)
def bessellsqxqy(x, y, z1, z2, lab, order, d1in, d2in):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1in,d2in
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), intent(in) :: lab
    complex(kind=8), dimension(0:2*order+1) :: qxqy

    integer :: Nls, n
    real(kind=8) :: Lnear, L, d1, d2, delta
    complex(kind=8) :: z, delz, za, zb
    """
    Lnear = 3.
    z = np.complex(x, y)
    qxqy = np.zeros(2*order+2, dtype=np.complex_)
    L = np.abs(z2-z1)
    # print *,'Lnear*np.abs(lab) ',Lnear*np.abs(lab)
    if (L < Lnear*np.abs(lab)):  # No need to break integral up
        if (np.abs(z - 0.5*(z1+z2)) < 0.5 * Lnear * L):  # Do integration
            qxqy = bessells_int_ho_qxqy(x, y, z1, z2, lab, order, d1in, d2in)
        else:
            qxqy = bessells_gauss_ho_qxqy_d1d2(
                x, y, z1, z2, lab, order, d1in, d2in)

    else:  # Break integral up in parts
        Nls = np.ceil(L / (Lnear*np.abs(lab)))
        # print *,'NLS ',Nls
        delta = 2. / Nls
        delz = (z2-z1)/Nls
        L = np.abs(delz)
        for n in range(1, Nls+1):
            d1 = -1. + (n-1) * delta
            d2 = -1. + n * delta
            if ((d2 < d1in) or (d1 > d2in)):
                continue
            d1 = np.max(np.array([d1, d1in]))
            d2 = np.min(np.array([d2, d2in]))
            za = z1 + (n-1) * delz
            zb = z1 + n * delz
            if (np.abs(z - 0.5*(za+zb)) < 0.5 * Lnear * L):  # Do integration
                qxqy = qxqy + \
                    bessells_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
            else:
                qxqy = qxqy + \
                    bessells_gauss_ho_qxqy_d1d2(
                        x, y, z1, z2, lab, order, d1, d2)
    return qxqy


@numba.njit(nogil=True)
def bessellsqxqyv(x, y, z1, z2, lab, order, R, nlab):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,R
    complex(kind=8), intent(in) :: z1,z2
    integer, intent(in) :: nlab
    real(kind=8) :: d1, d2
    complex(kind=8), dimension(nlab), intent(in) :: lab
    complex(kind=8), dimension(2*nlab*(order+1)) :: qxqy
    complex(kind=8), dimension(0:2*order+1) :: qxqylab
    integer :: n, nterms, nhalf
    """
    qxqy = np.zeros(2*nlab*(order+1), dtype=np.complex_)
    nterms = order+1
    nhalf = nlab*(order+1)
    d1, d2 = find_d1d2(z1, z2, np.complex(x, y), R*np.abs(lab[0]))
    for n in range(nlab):
        qxqylab = bessellsqxqy(x, y, z1, z2, lab[n], order, d1, d2)
        qxqy[(n)*nterms:(n+1)*nterms] = qxqylab[0:order+1]
        qxqy[(n)*nterms+nhalf:(n+1)*nterms +
             nhalf] = qxqylab[order+1:2*order+1+1]
    return qxqy


@numba.njit(nogil=True)
def bessellsqxqyv2(x, y, z1, z2, lab, order, R, nlab):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,R
    complex(kind=8), intent(in) :: z1,z2
    integer, intent(in) :: nlab
    real(kind=8) :: d1, d2
    complex(kind=8), dimension(nlab), intent(in) :: lab
    complex(kind=8), dimension(2*(order+1),nlab) :: qxqy
    complex(kind=8), dimension(0:2*order+1) :: qxqylab
    integer :: n, nterms, nhalf
    """
    qxqy = np.zeros((2*(order+1), nlab), dtype=np.complex_)
    nterms = order+1
    nhalf = nlab*(order+1)
    d1, d2 = find_d1d2(z1, z2, np.complex(x, y), R*np.abs(lab[0]))
    for n in range(nlab):
        qxqylab = bessellsqxqy(x, y, z1, z2, lab[n-1], order, d1, d2)
        qxqy[:nterms, n] = qxqylab[0:order+1]
        qxqy[nterms:2*nterms, n] = qxqylab[order+1:2*order+1+1]
    return qxqy


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
    if (L < Lnear*np.abs(lab)):  # No need to break integral up
        if (np.abs(z - 0.5*(z1+z2)) < 0.5 * Lnear * L):  # Do integration
            omega = bessells_int(x, y, z1, z2, lab)
        else:
            omega = bessells_gauss(x, y, z1, z2, lab)
    else:  # Break integral up in parts
        Nls = np.ceil(L / (Lnear*np.abs(lab)))
        delz = (z2-z1)/Nls
        L = np.abs(delz)
        for n in range(1, Nls+1):
            za = z1 + (n-1) * delz
            zb = z1 + n * delz
            if (np.abs(z - 0.5*(za+zb)) < 0.5 * Lnear * L):  # Do integration
                omega = omega + bessells_int(x, y, za, zb, lab)
            else:
                omega = omega + bessells_gauss(x, y, za, zb, lab)
    return omega


@numba.njit(nogil=True)
def bessellsuniv(x, y, z1, z2, lab, nlab):
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
    omega = np.zeros(nlab, dtype=np.complex_)
    for n in range(nlab):
        omega[n] = bessellsuni(x, y, z1, z2, lab[n])
    return omega


# %% Line Doublet Functions
@numba.njit(nogil=True)
def lapld_int_ho(x, y, z1, z2, order):
    """
    ! Near field only
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), dimension(0:order) :: omega, qm
    integer :: m, n
    real(kind=8) :: L
    complex(kind=8) :: z, zplus1, zmin1
    """

    omega = np.zeros(order+1, dtype=np.complex_)
    qm = np.zeros(order+1, dtype=np.complex_)

    L = np.abs(z2-z1)
    z = (2. * np.complex(x, y) - (z1+z2)) / (z2-z1)
    zplus1 = z + 1.
    zmin1 = z - 1.
    # Not sure if this gives correct answer at corner point (z also appears in qm); should really be caught in code that calls this function
    if np.abs(zplus1) < tiny:
        zplus1 = tiny
    if np.abs(zmin1) < tiny:
        zmin1 = tiny

    omega[0] = np.log(zmin1/zplus1)
    for n in range(1, order+1):
        omega[n] = z * omega[n-1]

    if order > 0:
        qm[1] = 2.
    for m in range(3, order+1, 2):
        qm[m] = qm[m-2] * z * z + 2. / m

    for m in range(2, order+1, 2):
        qm[m] = qm[m-1] * z

    omega = 1. / (np.complex(0., 2.) * np.pi) * (omega + qm)
    return omega


@numba.njit(nogil=True)
def lapld_int_ho_d1d2(x, y, z1, z2, order, d1, d2):
    """
    Near field only
    Returns integral from d1 to d2 along real axis while strength is still Delta^order from -1 to +1
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1,d2
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), dimension(0:order) :: omega, omegac
    integer :: n, m
    real(kind=8) :: xp, yp, dc, fac
    complex(kind=8) :: z1p,z2p,bigz1,bigz2
    """
    omega = np.zeros(order+1, dtype=np.complex_)

    bigz1 = np.complex(d1, 0.)
    bigz2 = np.complex(d2, 0.)
    z1p = 0.5 * (z2-z1) * bigz1 + 0.5 * (z1+z2)
    z2p = 0.5 * (z2-z1) * bigz2 + 0.5 * (z1+z2)
    omegac = lapld_int_ho(x, y, z1p, z2p, order)
    dc = (d1+d2) / (d2-d1)
    for n in range(order+1):
        for m in range(n+1):
            omega[n] = omega[n] + gam[n, m] * dc**(n-m) * omegac[m]
        omega[n] = (0.5*(d2-d1))**n * omega[n]

    return omega


@numba.njit(nogil=True)
def lapld_int_ho_wdis(x, y, z1, z2, order):
    """
    # Near field only
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), dimension(0:order) :: wdis
    complex(kind=8), dimension(0:10) :: qm  # Max order is 10
    integer :: m, n
    complex(kind=8) :: z, zplus1, zmin1, term1, term2, zterm
    """

    qm = np.zeros(11, dtype=np.complex_)
    wdis = np.zeros(order+1, dtype=np.complex_)

    z = (2. * np.complex(x, y) - (z1+z2)) / (z2-z1)
    zplus1 = z + 1.
    zmin1 = z - 1.
    # Not sure if this gives correct answer at corner point (z also appears in qm); should really be caught in code that calls this function
    if (np.abs(zplus1) < tiny):
        zplus1 = tiny
    if (np.abs(zmin1) < tiny):
        zmin1 = tiny

    qm[0:1] = 0.
    for m in range(2, order+1):
        qm[m] = 0.
        for n in range(1, m//2):
            qm[m] = qm[m] + (m-2*n+1) * z**(m-2*n) / (2*n-1)

    term1 = 1. / zmin1 - 1. / zplus1
    term2 = np.log(zmin1/zplus1)
    wdis[0] = term1
    zterm = np.complex(1., 0.)
    for m in range(1, order+1):
        wdis[m] = m * zterm * term2 + z * zterm * term1 + 2. * qm[m]
        zterm = zterm * z

    wdis = - wdis / (np.pi*np.complex(0., 1.)*(z2-z1))
    return wdis


@numba.njit(nogil=True)
def lapld_int_ho_wdis_d1d2(x, y, z1, z2, order, d1, d2):
    """
    # Near field only
    # Returns integral from d1 to d2 along real axis while strength is still Delta^order from -1 to +1
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1,d2
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), dimension(0:order) :: wdis, wdisc
    integer :: n, m
    real(kind=8) :: xp, yp, dc, fac
    complex(kind=8) :: z1p,z2p,bigz1,bigz2
    """

    wdis = np.zeros(order+1, dtype=np.complex_)

    bigz1 = np.complex(d1, 0.)
    bigz2 = np.complex(d2, 0.)
    z1p = 0.5 * (z2-z1) * bigz1 + 0.5 * (z1+z2)
    z2p = 0.5 * (z2-z1) * bigz2 + 0.5 * (z1+z2)
    wdisc = lapld_int_ho_wdis(x, y, z1p, z2p, order)
    dc = (d1+d2) / (d2-d1)
    wdis[0:order+1] = 0.
    for n in range(order+1):
        for m in range(n+1):
            wdis[n] = wdis[n] + gam[n, m] * dc**(n-m) * wdisc[m]
        wdis[n] = (0.5*(d2-d1))**n * wdis[n]
    return wdis


@numba.njit(nogil=True)
def besselld_int_ho(x, y, z1, z2, lab, order, d1, d2):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1,d2
    complex(kind=8), intent(in) :: z1,z2,lab
    complex(kind=8), dimension[0:order+1] :: omega
    real(kind=8) :: biglab, biga, L, ang, tol, bigy
    complex(kind=8) :: zeta, zetabar, log1, log2, term1, term2, d1minzeta, d2minzeta, bigz
    complex(kind=8) :: cm, biglabcomplex
    complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
    complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
    complex(kind=8), dimension[0:40+1] :: alpha, beta, alpha2
    complex(kind=8), dimension(0:50) :: alphanew, betanew, alphanew2 # Order fixed to 10

    integer :: m, n, p
    """

    zminzbar = np.zeros(21, dtype=np.complex_)
    omega = np.zeros(order+1, dtype=np.complex_)

    L = np.abs(z2-z1)
    bigz = (2. * np.complex(x, y) - (z1+z2)) / (z2-z1)
    bigy = bigz.imag
    biga = np.abs(lab)
    ang = np.arctan2(lab.imag, lab.real)
    biglab = 2. * biga / L
    biglabcomplex = 2.0 * lab / L

    tol = 1e-12

    exprange = np.exp(-np.complex(0, 2) * ang * nrange)
    anew = a1 * exprange
    bnew = (b1 - a1 * np.complex(0, 2) * ang) * exprange

    zeta = (2. * np.complex(x, y) - (z1+z2)) / (z2-z1) / biglab
    zetabar = np.conj(zeta)
    zminzbar[20] = 1.
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
    #d1minzeta = -1./biglab - zeta
    #d2minzeta = 1./biglab - zeta
    if (np.abs(d1minzeta) < tol):
        d1minzeta = d1minzeta + np.complex(tol, 0.)
    if (np.abs(d2minzeta) < tol):
        d2minzeta = d2minzeta + np.complex(tol, 0.)
    log1 = np.log(d1minzeta)
    log2 = np.log(d2minzeta)

    alphanew = np.zeros(51, dtype=np.complex_)
    alphanew2 = np.zeros(51, dtype=np.complex_)
    betanew = np.zeros(51, dtype=np.complex_)

    for p in range(order+1):
        alphanew[0:40+p+1] = 0.
        betanew[0:40+p+1] = 0.
        alphanew2[0:40+p+1] = 0.
        for m in range(p+1):
            cm = biglab**p * gam[p, m] * zeta**(p-m)
            alphanew[m:40+m+1] = alphanew[m:40+m+1] + cm * alpha[0:40+1]
            betanew[m:40+m+1] = betanew[m:40+m+1] + cm * beta[0:40+1]
            cm = biglab**p * gam[p, m] * zetabar**(p-m)
            alphanew2[m:40+m+1] = alphanew2[m:40+m+1] + cm * alpha2[0:40+1]

        omega[p] = 0.
        term1 = 1.
        term2 = 1.
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

    # omega = bigy * biglab / (2.*pi*biglabcomplex**2) * omega + real( lapld_int_ho_d1d2(x,y,z1,z2,order,d1,d2) )
    omega = bigy * biglab / (2.*np.pi*biglabcomplex**2) * \
        omega + lapld_int_ho_d1d2(x, y, z1, z2, order, d1, d2).real
    return omega


@numba.njit(nogil=True)
def besselld_gauss_ho(x, y, z1, z2, lab, order):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), intent(in) :: lab
    complex(kind=8), dimension(0:order) :: omega
    integer :: n, p
    real(kind=8) :: L, x0, r
    complex(kind=8) :: bigz, biglab
    complex(kind=8), dimension(8) :: k1overr
    """

    k1overr = np.zeros(8, dtype=np.complex_)
    omega = np.zeros(order+1, dtype=np.complex_)

    L = np.abs(z2-z1)
    biglab = 2. * lab / L
    bigz = (2. * np.complex(x, y) - (z1+z2)) / (z2-z1)
    for n in range(8):
        x0 = bigz.real - xg[n]
        r = np.sqrt(x0**2 + bigz.imag**2)
        k1overr[n] = besselk1(x0, bigz.imag, biglab) / r
    for p in range(order+1):
        omega[p] = np.complex(0., 0.)
        for n in range(8):
            omega[p] = omega[p] + wg[n] * xg[n]**p * k1overr[n]

        omega[p] = bigz.imag/(2.*np.pi*biglab) * omega[p]
    return omega


@numba.njit(nogil=True)
def besselld_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2):
    """
    # Returns integral from d1 to d2 along real axis while strength is still Delta^order from -1 to +1
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

    bigz1 = np.complex(d1, 0.)
    bigz2 = np.complex(d2, 0.)
    z1p = 0.5 * (z2-z1) * bigz1 + 0.5 * (z1+z2)
    z2p = 0.5 * (z2-z1) * bigz2 + 0.5 * (z1+z2)
    omegac = besselld_gauss_ho(x, y, z1p, z2p, lab, order)
    dc = (d1+d2) / (d2-d1)
    omega[0:order+1] = 0.
    for n in range(order+1):
        for m in range(n+1):
            omega[n] = omega[n] + gam[n, m] * dc**(n-m) * omegac[m]
        omega[n] = (0.5*(d2-d1))**n * omega[n]
    return omega


@numba.njit(nogil=True)
def besselld(x, y, z1, z2, lab, order, d1in, d2in):
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

    omega = np.zeros(order+1, dtype=np.complex_)

    Lnear = 3.
    z = np.complex(x, y)
    L = np.abs(z2-z1)
    if (L < Lnear*np.abs(lab)):  # No need to break integral up
        if (np.abs(z - 0.5*(z1+z2)) < 0.5 * Lnear * L):  # Do integration
            omega = besselld_int_ho(x, y, z1, z2, lab, order, d1in, d2in)
        else:
            omega = besselld_gauss_ho_d1d2(
                x, y, z1, z2, lab, order, d1in, d2in)
    else:  # Break integral up in parts
        Nls = np.ceil(L / (Lnear*np.abs(lab)))
        delta = 2. / Nls
        delz = (z2-z1)/Nls
        L = np.abs(delz)
        for n in range(1, Nls+1):
            d1 = -1. + (n-1) * delta
            d2 = -1. + n * delta
            if ((d2 < d1in) or (d1 > d2in)):
                continue
            d1 = max(d1, d1in)
            d2 = min(d2, d2in)
            za = z1 + (n-1) * delz
            zb = z1 + n * delz
            if (np.abs(z - 0.5*(za+zb)) < 0.5 * Lnear * L):  # Do integration
                omega = omega + \
                    besselld_int_ho(x, y, z1, z2, lab, order, d1, d2)
            else:
                omega = omega + \
                    besselld_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2)

    return omega


@numba.njit(nogil=True)
def besselldv(x, y, z1, z2, lab, order, R, nlab):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,R
    complex(kind=8), intent(in) :: z1,z2
    integer, intent(in) :: nlab
    real(kind=8) :: d1, d2
    complex(kind=8), dimension(nlab), intent(in) :: lab
    complex(kind=8), dimension(nlab*(order+1)) :: omega
    integer :: n, nterms
    """

    omega = np.zeros(nlab*(order+1), dtype=np.complex_)

    nterms = order+1
    # Check if endpoints need to be adjusted using the largest lambda (the first one)
    d1, d2 = find_d1d2(z1, z2, np.complex(x, y), R*np.abs(lab[0]))
    for n in range(nlab):
        omega[(n)*nterms:(n+1)*nterms] = besselld(x, y, z1, z2,
                                                  lab[n], order, d1, d2)
    return omega


@numba.njit(nogil=True)
def besselldv2(x, y, z1, z2, lab, order, R, nlab):
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

    omega = np.zeros((order+1, nlab), dtype=np.complex_)

    nterms = order+1
    # Check if endpoints need to be adjusted using the largest lambda (the first one)
    d1, d2 = find_d1d2(z1, z2, np.complex(x, y), R*np.abs(lab[0]))
    for n in range(nlab):
        omega[:nterms+1, n] = besselld(x, y, z1, z2, lab[n], order, d1, d2)

    return omega


# @numba.njit(nogil=True)
def besselldpart(x, y, z1, z2, lab, order, d1, d2):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1,d2
    complex(kind=8), intent(in) :: z1,z2,lab
    complex(kind=8), dimension(0:order) :: omega
    real(kind=8) :: biglab, biga, L, ang, tol, bigy
    complex(kind=8) :: zeta, zetabar, log1, log2, term1, term2, d1minzeta, d2minzeta, bigz
    complex(kind=8) :: cm, biglabcomplex
    complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
    complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
    complex(kind=8), dimension(0:40) :: alpha, beta, alpha2
    complex(kind=8), dimension(0:50) :: alphanew, betanew, alphanew2 ! Order fixed to 10
    integer :: m, n, p
    """
    zminzbar = np.zeros(21, dtype=np.complex_)

    L = np.abs(z2-z1)
    bigz = (2. * np.complex(x, y) - (z1+z2)) / (z2-z1)
    bigy = bigz.imag
    biga = np.abs(lab)
    ang = np.arctan2(lab.imag, lab.real)
    biglab = 2. * biga / L
    biglabcomplex = 2.0 * lab / L

    tol = 1e-12

    exprange = np.exp(-np.complex(0, 2) * ang * nrange)
    anew = a1 * exprange
    bnew = (b1 - a1 * np.complex(0, 2) * ang) * exprange

    zeta = (2. * np.complex(x, y) - (z1+z2)) / (z2-z1) / biglab
    zetabar = np.conj(zeta)
    zminzbar[-1] = 1.
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

    if (np.abs(d1minzeta) < tol):
        d1minzeta = d1minzeta + np.complex(tol, 0.)
    if (np.abs(d2minzeta) < tol):
        d2minzeta = d2minzeta + np.complex(tol, 0.)
    log1 = np.log(d1minzeta)
    log2 = np.log(d2minzeta)

    alphanew = np.zeros(51, dtype=np.complex_)
    alphanew2 = np.zeros(51, dtype=np.complex_)
    betanew = np.zeros(51, dtype=np.complex_)
    omega = np.zeros(order+1, dtype=np.complex_)

    for p in range(order+1):

        alphanew[0:40+p+1] = 0.
        betanew[0:40+p+1] = 0.
        alphanew2[0:40+p+1] = 0.
        for m in range(p+1):
            cm = biglab**p * gam[p, m] * zeta**(p-m)
            alphanew[m:40+m+1] = alphanew[m:40+m+1] + cm * alpha[0:40+1]
            betanew[m:40+m+1] = betanew[m:40+m+1] + cm * beta[0:40+1]
            cm = biglab**p * gam[p, m] * zetabar**(p-m)
            alphanew2[m:40+m+1] = alphanew2[m:40+m+1] + cm * alpha2[0:40+1]

        omega[p] = 0.
        term1 = 1.
        term2 = 1.
        for n in range(40+p+1):
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

    # + real( lapld_int_ho(x,y,z1,z2,order) )
    omega = biglab / (2.*np.pi*biglabcomplex**2) * omega
    # omega = real( lapld_int_ho(x,y,z1,z2,order) )

    return omega


# @numba.njit(nogil=True)
def besselld_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1,d2
    complex(kind=8), intent(in) :: z1,z2,lab
    complex(kind=8), dimension(0:2*order+1) :: qxqy
    complex(kind=8), dimension(0:order) :: rvz, rvzbar
    real(kind=8) :: biglab, biga, L, ang, angz, tol, bigy
    complex(kind=8) :: zeta, zetabar, log1, log2, term1, term2, d1minzeta, d2minzeta, bigz
    complex(kind=8) :: cm, biglabcomplex, azero
    complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
    complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
    complex(kind=8), dimension(0:40) :: alpha, beta, alpha2
    complex(kind=8), dimension(0:51) :: alphanew, betanew, alphanew2 ! Order fixed to 10
    complex(kind=8), dimension(0:order) :: omegalap, omegaom, wdis, qx, qy ! To store intermediate result
    complex(kind=8), dimension(0:order+1) :: omega ! To store intermediate result
    integer :: m, n, p
    """

    zminzbar = np.zeros(21, dtype=np.complex_)
    omega = np.zeros(order+2, dtype=np.complex_)
    qxqy = np.zeros(2*order+2, dtype=np.complex_)

    L = np.abs(z2-z1)
    bigz = (2 * np.complex(x, y) - (z1+z2)) / (z2-z1)
    bigy = bigz.imag
    biga = np.abs(lab)
    ang = np.arctan2(lab.imag, lab.real)
    angz = np.arctan2((z2-z1).imag, (z2-z1).real)
    biglab = 2 * biga / L
    biglabcomplex = 2.0 * lab / L

    tol = 1e-12

    exprange = np.exp(-np.complex(0, 2) * ang * nrange)
    anew = a1 * exprange
    bnew = (b1 - a1 * np.complex(0, 2) * ang) * exprange
    azero = anew[0]

    for n in range(20):
        bnew[n] = (n+1)*bnew[n+1] + anew[n+1]
        anew[n] = (n+1)*anew[n+1]

    anew[20] = 0  # This is a bit lazy
    bnew[20] = 0

    zeta = (2 * np.complex(x, y) - (z1+z2)) / (z2-z1) / biglab
    zetabar = np.conj(zeta)
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
    alpha2 = np.zeros(41, dtype=np.complex_)
    beta = np.zeros(41, dtype=np.complex_)
    alpha[0] = anew[0]
    beta[0] = bnew[0]
    alpha2[0] = anew[0]

    for n in range(1, 21):
        alpha[n:2*n+1] = alpha[n:2*n+1] + anew[n] * gamnew[n, 0:n+1]
        beta[n:2*n+1] = beta[n:2*n+1] + bnew[n] * gamnew[n, 0:n+1]
        alpha2[n:2*n+1] = alpha2[n:2*n+1] + anew[n] * gam2[n, 0:n+1]

    d1minzeta = d1/biglab - zeta
    d2minzeta = d2/biglab - zeta
    # d1minzeta = -1/biglab - zeta
    # d2minzeta = 1/biglab - zeta
    if (np.abs(d1minzeta) < tol):
        d1minzeta = d1minzeta + np.complex(tol, 0)
    if (np.abs(d2minzeta) < tol):
        d2minzeta = d2minzeta + np.complex(tol, 0)

    log1 = np.log(d1minzeta)
    log2 = np.log(d2minzeta)

    alphanew = np.zeros(52, dtype=np.complex_)
    alphanew2 = np.zeros(52, dtype=np.complex_)
    betanew = np.zeros(52, dtype=np.complex_)

    for p in range(order+2):

        alphanew[0:40+p+1] = 0
        betanew[0:40+p+1] = 0
        alphanew2[0:40+p+1] = 0
        for m in range(p+1):
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

    omegalap = lapld_int_ho_d1d2(
        x, y, z1, z2, order, d1, d2) / np.complex(0, 1)
    omegaom = besselldpart(x, y, z1, z2, lab, order, d1, d2)
    wdis = lapld_int_ho_wdis_d1d2(x, y, z1, z2, order, d1, d2)

    rvz = -biglab * bigy / (2*np.pi*biglabcomplex**2) * (omega[1:order+1+1]/biglab - zetabar * omega[0:order+1]) + \
        biglab * omegaom / np.complex(0, 2)
    rvzbar = -biglab * bigy / (2*np.pi*biglabcomplex**2) * (omega[1:order+1+1]/biglab - zeta * omega[0:order+1]) - \
        biglab * omegaom / np.complex(0, 2)
    # qxqy[0:order+1] = -2.0 / L * ( rvz + rvzbar ) / biglab  # As we need to take derivative w.r.t. z not zeta
    # qxqy[order+1:2*order+1+1] = -2.0 / L * np.complex(0,1) * (rvz-rvzbar) / biglab
    #
    # qxqy[0:order+1] = qxqy[0:order+1] - 2.0 / L / biglabcomplex**2 * azero * ( omegalap + np.conj(omegalap) )
    # qxqy[order+1:2*order+1+1] = qxqy[order+1:2*order+1+1] -  \
    #                          2.0 / L / biglabcomplex**2 * azero * np.complex(0,1) * (omegalap - np.conj(omegalap))
    #
    # qxqy[0:order+1] = qxqy[0:order+1] + real(wdis)
    # qxqy[order+1:2*order+1+1] = qxqy[order+1:2*order+1+1] - aimag(wdis)

    # As we need to take derivative w.r.t. z not zeta
    qx = -2.0 / L * (rvz + rvzbar) / biglab
    qy = -2.0 / L * np.complex(0, 1) * (rvz-rvzbar) / biglab

    qx = qx - 2.0 / L * bigy / biglabcomplex**2 * \
        azero * (omegalap + np.conj(omegalap))
    qy = qy - 2.0 / L * bigy / biglabcomplex**2 * azero * \
        np.complex(0, 1) * (omegalap - np.conj(omegalap))

    # qx = qx + real(wdis * (z2-z1) / L)
    # qy = qy - aimag(wdis * (z2-z1) / L)

    # print *,'angz ',angz
    # wdis already includes the correct rotation
    qxqy[0:order+1] = qx * np.cos(angz) - qy * np.sin(angz) + wdis.real
    qxqy[order+1:2*order+1+1] = qx * \
        np.sin(angz) + qy * np.cos(angz) - wdis.imag

    return qxqy


# @numba.njit(nogil=True)
def besselld_gauss_ho_qxqy(x, y, z1, z2, lab, order):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), intent(in) :: lab
    complex(kind=8), dimension(0:2*order+1) :: qxqy
    integer :: n, p
    real(kind=8) :: L, bigy, angz
    complex(kind=8) :: bigz, biglab
    real(kind=8), dimension(8) :: r, xmind
    complex(kind=8), dimension(8) :: k0,k1
    complex(kind=8), dimension(0:order) :: qx,qy
    """

    xmind = np.zeros(8, dtype=np.float_)
    r = np.zeros(8, dtype=np.float_)
    k0 = np.zeros(8, dtype=np.complex_)
    k1 = np.zeros(8, dtype=np.complex_)
    qxqy = np.zeros(2*order+2, dtype=np.complex_)

    L = np.abs(z2-z1)
    biglab = 2. * lab / L
    bigz = (2. * np.complex(x, y) - (z1+z2)) / (z2-z1)
    bigy = bigz.imag
    for n in range(8):
        xmind[n] = bigz.real - xg[n]
        r[n] = np.sqrt(xmind[n]**2 + bigz.imag**2)
        k0[n] = besselk0(xmind[n], bigz.imag, biglab)
        k1[n] = besselk1(xmind[n], bigz.imag, biglab)

    qx = np.zeros(order+1, dtype=np.complex_)
    qy = np.zeros(order+1, dtype=np.complex_)
    for p in range(order+1):
        for n in range(8):
            qx[p] = qx[p] + wg[n] * xg[n]**p * \
                (-bigy) * xmind[n] / r[n]**3 * \
                (r[n]*k0[n]/biglab + 2.*k1[n])
            qy[p] = qy[p] + wg[n] * xg[n]**p * \
                (k1[n]/r[n] - bigy**2 / r[n]**3 *
                 (r[n]*k0[n]/biglab + 2.*k1[n]))

    qx = -qx / (2*np.pi*biglab) * 2/L
    qy = -qy / (2*np.pi*biglab) * 2/L

    angz = np.arctan2((z2-z1).imag, (z2-z1).real)
    qxqy[0:order+1] = qx * np.cos(angz) - qy * np.sin(angz)
    qxqy[order+1:2*order+1+1] = qx * np.sin(angz) + qy * np.cos(angz)

    return qxqy


# @numba.njit(nogil=True)
def besselld_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2):
    """
    Returns integral from d1 to d2 along real axis while strength is still Delta^order from -1 to +1

    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1,d2
    complex(kind=8), intent(in) :: z1,z2,lab
    complex(kind=8), dimension(0:2*order+1) :: qxqy, qxqyc
    integer :: n, m
    real(kind=8) :: xp, yp, dc, fac
    complex(kind=8) :: z1p,z2p,bigz1,bigz2
    """

    qxqy = np.zeros(2*order+2, dtype=np.complex_)

    bigz1 = np.complex(d1, 0)
    bigz2 = np.complex(d2, 0)
    z1p = 0.5 * (z2-z1) * bigz1 + 0.5 * (z1+z2)
    z2p = 0.5 * (z2-z1) * bigz2 + 0.5 * (z1+z2)
    qxqyc = besselld_gauss_ho_qxqy(x, y, z1p, z2p, lab, order)
    dc = (d1+d2) / (d2-d1)
    for n in range(order+1):
        for m in range(n+1):
            qxqy[n] = qxqy[n] + gam[n, m] * dc**(n-m) * qxqyc[m]
            qxqy[n+order+1] = qxqy[n+order+1] + \
                gam[n, m] * dc**(n-m) * qxqyc[m+order+1]

        qxqy[n] = (0.5*(d2-d1))**n * qxqy[n]
        qxqy[n+order+1] = (0.5*(d2-d1))**n * qxqy[n+order+1]

    return qxqy


# @numba.njit(nogil=True)
def besselldqxqy(x, y, z1, z2, lab, order, d1in, d2in):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1in,d2in
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), intent(in) :: lab
    complex(kind=8), dimension(0:2*order+1) :: qxqy

    integer :: Nls, n
    real(kind=8) :: Lnear, L, d1, d2, delta
    complex(kind=8) :: z, delz, za, zb

    """
    Lnear = 3
    z = np.complex(x, y)
    qxqy = np.complex(0, 0)
    L = np.abs(z2-z1)

    # print *,'Lnear*np.abs(lab) ',Lnear*np.abs(lab)
    if (L < Lnear*np.abs(lab)):  # No need to break integral up
        if (np.abs(z - 0.5*(z1+z2)) < 0.5 * Lnear * L):  # Do integration
            qxqy = besselld_int_ho_qxqy(x, y, z1, z2, lab, order, d1in, d2in)
        else:
            qxqy = besselld_gauss_ho_qxqy_d1d2(
                x, y, z1, z2, lab, order, d1in, d2in)

    else:  # Break integral up in parts
        Nls = np.ceil(L / (Lnear*np.abs(lab)))
        # print *,'NLS ',Nls
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
                qxqy = qxqy + \
                    besselld_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
            else:
                qxqy = qxqy + \
                    besselld_gauss_ho_qxqy_d1d2(
                        x, y, z1, z2, lab, order, d1, d2)
    return qxqy


# @numba.njit(nogil=True)
def besselldqxqyv(x, y, z1, z2, lab, order, R, nlab):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,R
    complex(kind=8), intent(in) :: z1,z2
    integer, intent(in) :: nlab
    real(kind=8) :: d1, d2
    complex(kind=8), dimension(nlab), intent(in) :: lab
    complex(kind=8), dimension(2*nlab*(order+1)) :: qxqy
    complex(kind=8), dimension(0:2*order+1) :: qxqylab
    """

    qxqy = np.zeros(2*nlab*(order+1), dtype=np.complex_)

    nterms = order+1
    nhalf = nlab*(order+1)
    d1, d2 = find_d1d2(z1, z2, np.complex(x, y), R*np.abs(lab[0]))
    for n in range(nlab):
        qxqylab = besselldqxqy(x, y, z1, z2, lab[n], order, d1, d2)
        qxqy[n*nterms:(n+1)*nterms] = qxqylab[0:order+1]
        qxqy[n*nterms+nhalf:(n+1)*nterms +
             nhalf] = qxqylab[order+1:2*order+1+1]

    return qxqy


# @numba.njit(nogil=True)
def besselldqxqyv2(x, y, z1, z2, lab, order, R, nlab):
    """
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,R
    complex(kind=8), intent(in) :: z1,z2
    integer, intent(in) :: nlab
    real(kind=8) :: d1, d2
    complex(kind=8), dimension(nlab), intent(in) :: lab
    complex(kind=8), dimension(2*(order+1),nlab) :: qxqy
    complex(kind=8), dimension(0:2*order+1) :: qxqylab
    integer :: n, nterms, nhalf
    """
    qxqy = np.zeros((2*(order+1), nlab), dtype=np.complex_)

    nterms = order+1
    nhalf = nlab*(order+1)
    d1, d2 = find_d1d2(z1, z2, np.complex(x, y), R*np.abs(lab[0]))
    for n in range(nlab):
        qxqylab = besselldqxqy(x, y, z1, z2, lab[n], order, d1, d2)
        qxqy[:nterms, n] = qxqylab[0:order+1]
        qxqy[nterms:2*nterms, n] = qxqylab[order+1:2*order+1+1]
    return qxqy


@numba.njit(nogil=True)
def bessells_circcheck(x, y, z1in, z2in, lab):
    """
    implicit none
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1in,z2in
    complex(kind=8), intent(in) :: lab
    complex(kind=8) :: omega

    integer :: Npt, Nls, n
    real(kind=8) :: Lnear, Lzero, L, x1, y1, x2, y2
    complex(kind=8) :: z, z1, z2, delz, za, zb
    """
    Lnear = 3
    Lzero = 20
    z = np.complex(x, y)
    x1, y1, x2, y2, Npt = circle_line_intersection(
        z1in, z2in, z, Lzero*np.abs(lab))

    z1 = np.complex(x1, y1)
    z2 = np.complex(x2, y2)

    omega = np.complex(0, 0)

    if (Npt == 2):
        L = np.abs(z2-z1)
        if (L < Lnear*np.abs(lab)):  # No need to break integral up
            if (np.abs(z - 0.5*(z1+z2)) < 0.5 * Lnear * L):  # Do integration
                omega = bessells_int(x, y, z1, z2, lab)
            else:
                omega = bessells_gauss(x, y, z1, z2, lab)
        else:  # Break integral up in parts
            Nls = np.ceil(L / (Lnear*np.abs(lab)))
            delz = (z2-z1)/Nls
            L = np.abs(delz)
            for n in range(1, Nls+1):
                za = z1 + (n-1) * delz
                zb = z1 + n * delz
                if (np.abs(z - 0.5*(za+zb)) < 0.5 * Lnear * L):  # Do integration
                    omega = omega + bessells_int(x, y, za, zb, lab)
                else:
                    omega = omega + bessells_gauss(x, y, za, zb, lab)
    return omega


@numba.njit(nogil=True)
def is_too_far(z1, z2, zc, R):
    """
    Checks whether zc is more than R away from oval
    surrounding line element
    """
    
    Lover2 = np.abs(z2 - z1) / 2
    bigz = (2 * zc - (z1 + z2)) / (z2 - z1)
    Radj = R / Lover2
    
    rv = False
    if np.abs(bigz.imag) < Radj:
        if np.abs(bigz.real) < 1:
            rv = True
        elif np.abs(bigz.real) < 1 + Radj:
            if np.abs(zc - z1) < R:
                rv = True
            elif np.abs(zc - z2) < R:
                rv = True
    return rv

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
        d = np.sqrt(R**2 - bigz.imag**2)
        xa = bigz.real - d
        xb = bigz.real + d
        if ((xa < Lover2) and (xb > -Lover2)):
            N = 2
            if (xa < -Lover2):
                za = z1
            else:
                za = (xa * (z2-z1) / Lover2 + (z1+z2)) / 2.
            if (xb > Lover2):
                zb = z2
            else:
                zb = (xb * (z2-z1) / Lover2 + (z1+z2)) / 2.

    xouta = za.real
    youta = za.imag
    xoutb = zb.real
    youtb = zb.imag

    return xouta, youta, xoutb, youtb, N


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
    Lover2 = np.abs(z2-z1) / 2
    bigz = (2*zc - (z1+z2)) * np.abs(z2-z1) / (2*(z2-z1))

    if (np.abs(bigz.imag) < R):
        d = np.sqrt(R**2 - bigz.imag**2)
        xa = bigz.real - d
        xb = bigz.real + d
        if ((xa < Lover2) and (xb > -Lover2)):
            irv = 1
    return irv
