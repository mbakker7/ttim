import numpy as np

def param_maq(kaq=[1], z=[1, 0], c=[], Saq=[0.001], Sll=[], 
              poraq=[0.3], porll=[], topboundary='conf', phreatictop=False):
    # Computes the parameters for a TimModel from input for a maq model
    kaq = np.atleast_1d(kaq).astype('d')
    naq = len(kaq)
    z = np.atleast_1d(z).astype('d')
    c = np.atleast_1d(c).astype('d')
    Saq = np.atleast_1d(Saq).astype('d')
    if len(Saq) == 1: 
        Saq = Saq * np.ones(naq)
    Sll = np.atleast_1d(Sll).astype('d')
    poraq = np.atleast_1d(poraq).astype('d')
    if len(poraq) == 1:
        poraq = poraq * np.ones(naq)
    porll = np.atleast_1d(porll).astype('d')
    H = z[:-1] - z[1:]
    assert np.all(H >= 0), 'Error: Not all layers thicknesses are' + \
                           ' non-negative' + str(H) 
    if topboundary[:3] == 'con':
        assert len(z) == 2 * naq, 'Error: Length of z needs to be ' + \
                                  str(2 * naq)
        assert len(c) == naq - 1, 'Error: Length of c needs to be ' + \
                                  str(naq - 1)
        assert len(Saq) == naq, 'Error: Length of Saq needs to be ' + \
                                str(naq)
        assert len(poraq) == naq, 'Error: Length of poraq needs to be ' + \
                                str(naq)
        assert len(Sll) == naq - 1, 'Error: Length of Sll needs to be ' + \
                                  str(naq - 1)
        assert len(porll) == naq - 1, 'Error: Length of porll needs to be ' + \
                                  str(naq - 1)
        Haq = H[::2]
        #Saq = Saq * Haq
        #if phreatictop: Saq[0] = Saq[0] / H[0]
        Hll = H[1::2]
        #Sll = Sll * Hll
        # changed (nan,c) to (1e100,c) as I get an error
        c = np.hstack((1e100, c))  
        # Was: Sll = np.hstack((np.nan,Sll)), but that gives error 
        # when c approaches inf
        Sll = np.hstack((1e-30, Sll)) 
        Hll = np.hstack((1e-30, Hll))
        porll = np.hstack((1e-30, porll))
        # layertype
        nlayers = len(z) - 1
        ltype = np.array(nlayers * ['a'])
        ltype[1::2] = 'l'
    else: # leaky layers on top
        assert len(z) == 2 * naq + 1, 'Error: Length of z needs to be ' + \
                                      str(2*naq+1)
        assert len(c) == naq, 'Error: Length of c needs to be ' + str(naq)
        assert len(Saq) == naq, 'Error: Length of Saq needs to be ' + str(naq)
        assert len(poraq) == naq, 'Error: Length of poraq needs to be ' + str(naq)
        if len(Sll) == 1: 
            Sll = Sll * np.ones(naq)
        assert len(Sll) == naq, 'Error: Length of Sll needs to be ' + str(naq)
        if len(porll) == 1: 
            porll = porll * np.ones(naq)
        assert len(porll) == naq, 'Error: Length of porll needs to be ' + \
                                   str(naq)
        Haq = H[1::2]
        #Saq = Saq * Haq
        Hll = H[::2]
        #Sll = Sll * Hll
        #if phreatictop and (topboundary[:3]=='lea'): Sll[0] = Sll[0] / H[0]
        # layertype
        nlayers = len(z) - 1
        ltype = np.array(nlayers * ['a'])
        ltype[0::2] = 'l'
    return kaq, Haq, Hll, c, Saq, Sll, poraq, porll, ltype
    
def param_3d(kaq=[1], z=[1, 0], Saq=[0.001], kzoverkh=1, poraq=0.3, 
             phreatictop=False, topboundary='conf', topres=0, topthick=0, 
             topSll=0, toppor=0.3):
    # Computes the parameters for a TimModel from input for a 3D model
    kaq = np.atleast_1d(kaq).astype('d')
    z = np.atleast_1d(z).astype('d')
    naq = len(z) - 1
    if len(kaq) == 1: 
        kaq = kaq * np.ones(naq)
    Saq = np.atleast_1d(Saq).astype('d')
    if len(Saq) == 1: 
        Saq = Saq * np.ones(naq)
    kzoverkh = np.atleast_1d(kzoverkh).astype('d')
    if len(kzoverkh) == 1: 
        kzoverkh = kzoverkh * np.ones(naq)
    poraq = np.atleast_1d(poraq).astype('d')
    if len(poraq) == 1:
        poraq = poraq * np.ones(naq)
    Haq = z[:-1] - z[1:]
    c = 0.5 * Haq[:-1] / (kzoverkh[:-1] * kaq[:-1]) + \
        0.5 * Haq[1:] /  (kzoverkh[1:] * kaq[1:])
    # Saq = Saq * H
    #if phreatictop:
    #    Saq[0] = Saq[0] / H[0]
    c = np.hstack((1e100, c))
    Hll = 1e-20 * np.ones(len(c))
    Sll = 1e-20 * np.ones(len(c))
    porll = np.zeros(len(c))
    nlayers = len(z) - 1
    ltype = np.array(nlayers * ['a'])
    if (topboundary[:3] == 'sem') or (topboundary[:3] == 'lea'):
        c[0] = np.max([1e-20, topres])
        Hll[0] = np.max([1e-20, topthick])
        Sll[0] = np.max([1e-20, topSll])
        porll[0] = toppor
        ltype = np.hstack(('l', ltype))
        z = np.hstack((z[0] + topthick, z))
    return kaq, Haq, Hll, c, Saq, Sll, poraq, porll, ltype, z