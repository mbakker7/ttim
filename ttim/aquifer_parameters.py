import numpy as np

def param_maq(kaq=[1],z=[1,0],c=[],Saq=[0.001],Sll=[0],topboundary='conf',phreatictop=False):
    # Computes the parameters for a TimModel from input for a maq model
    kaq = np.atleast_1d(kaq).astype('d')
    Naq = len(kaq)
    z = np.atleast_1d(z).astype('d')
    c = np.atleast_1d(c).astype('d')
    Saq = np.atleast_1d(Saq).astype('d')
    if len(Saq) == 1: Saq = Saq * np.ones(Naq)
    Sll = np.atleast_1d(Sll).astype('d')
    H = z[:-1] - z[1:]
    assert np.all(H >= 0), 'Error: Not all layers thicknesses are non-negative' + str(H) 
    if topboundary[:3] == 'con':
        assert len(z) == 2*Naq, 'Error: Length of z needs to be ' + str(2*Naq)
        assert len(c) == Naq-1, 'Error: Length of c needs to be ' + str(Naq-1)
        assert len(Saq) == Naq, 'Error: Length of Saq needs to be ' + str(Naq)
        if len(Sll) == 1: Sll = Sll * np.ones(Naq-1)
        assert len(Sll) == Naq-1, 'Error: Length of Sll needs to be ' + str(Naq-1)
        Haq = H[::2]
        #Saq = Saq * Haq
        #if phreatictop: Saq[0] = Saq[0] / H[0]
        Hll = H[1::2]
        #Sll = Sll * Hll
        c = np.hstack((1e100, c))  # changed (nan,c) to (1e100,c) as I get an error
        Sll = np.hstack((1e-30, Sll)) # Was: Sll = np.hstack((np.nan,Sll)), but that gives error when c approaches inf
        Hll = np.hstack((1e-30, Hll))
    else: # leaky layers on top
        assert len(z) == 2*Naq+1, 'Error: Length of z needs to be ' + str(2*Naq+1)
        assert len(c) == Naq, 'Error: Length of c needs to be ' + str(Naq)
        assert len(Saq) == Naq, 'Error: Length of Saq needs to be ' + str(Naq)
        if len(Sll) == 1: Sll = Sll * np.ones(Naq)
        assert len(Sll) == Naq, 'Error: Length of Sll needs to be ' + str(Naq)
        Haq = H[1::2]
        #Saq = Saq * Haq
        Hll = H[::2]
        #Sll = Sll * Hll
        #if phreatictop and (topboundary[:3]=='lea'): Sll[0] = Sll[0] / H[0]
    return kaq, Haq, Hll, c, Saq, Sll
    
def param_3d(kaq=[1], z=[1, 0], Saq=[0.001], kzoverkh=1, phreatictop=False, \
             topboundary='conf', topres=0, topthick=0, topSll=0):
    # Computes the parameters for a TimModel from input for a 3D model
    kaq = np.atleast_1d(kaq).astype('d')
    z = np.atleast_1d(z).astype('d')
    naq = len(z) - 1
    if len(kaq) == 1: kaq = kaq * np.ones(naq)
    Saq = np.atleast_1d(Saq).astype('d')
    if len(Saq) == 1: Saq = Saq * np.ones(naq)
    kzoverkh = np.atleast_1d(kzoverkh).astype('d')
    if len(kzoverkh) == 1: kzoverkh = kzoverkh * np.ones(naq)
    Haq = z[:-1] - z[1:]
    c = 0.5 * Haq[:-1] / (kzoverkh[:-1] * kaq[:-1]) + \
        0.5 * Haq[1:] /  (kzoverkh[1:] * kaq[1:])
    # Saq = Saq * H
    #if phreatictop:
    #    Saq[0] = Saq[0] / H[0]
    c = np.hstack((1e100, c))
    Hll = 1e-20 * np.ones(len(c))
    Sll = 1e-20 * np.ones(len(c))
    if (topboundary[:3] == 'sem') or (topboundary[:3] == 'lea'):
        c[0] = np.max([1e-20, topres])
        Hll[0] = np.max([1e-20, topthick])
        Sll[0] = np.max([1e-20, topSll])
    return kaq, Haq, Hll, c, Saq, Sll