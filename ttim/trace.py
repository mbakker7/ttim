import numpy as np

def timtraceline(ml, xstart, ystart, zstart, tstart, delt,
                 nstep=100, silent=False, 
                 returnlayers=False, verbose=False):
    #verbose = True  # used for debugging
    # treating aquifer layers and leaky layers the same way
    terminate = False
    message = "no message"
    eps = 1e-10  # used to place point just above or below aquifer top or bottom
    aq = ml.aq.find_aquifer_data(xstart, ystart)
    if zstart > aq.z[0] or zstart < aq.z[-1]:
        terminate = True
        message = "starting z value not inside aquifer"
    ## fixed to 1 layer
    ## layer, ltype, modellayer = aq.findlayer(zstart)
    layer = 0
    ltype = 'a'
    modellayer = 0
    # slightly alter starting location not to get stuck in surpring points
    # starting at time 0
    xyzt = [np.array([xstart * (1 + eps), ystart * (1 + eps), zstart, tstart])]  
    layerlist = []  # to keep track of layers for plotting with colors
    for _ in range(nstep):
        if terminate:
            break
        x0, y0, z0, t0 = xyzt[-1]
        aq = ml.aq.find_aquifer_data(x0, y0)  # find new aquifer
        #layer, ltype, modellayer = aq.findlayer(z0)
        #layerlist.append(modellayer)
        v0 = ml.velo_one(x0, y0, z0, t0, aq, [layer, ltype])
        if verbose:
            print('xyz, layer', x0, y0, z0, layer)
            print('v0, layer, ltype', v0, layer, ltype)
        vx, vy, vz = v0
        x1 = x0 + delt * vx
        y1 = y0 + delt * vy
        z1 = z0 + delt * vz
        t1 = t0 + delt
        xyzt1 = np.array([x1, y1, z1, t1])
        # check if point needs to be changed
        correction = True
        ## not checked if correction is needed
#             for e in aq.elementlist:
#                 changed, terminate, xyztnew, changemessage = e.changetrace(
#                     xyzt[-1], xyzt1, aq, layer, ltype, modellayer, 
#                     direction, hstepmax)
#                 if changed or terminate:
#                     correction = False
#                     if changemessage:
#                         message = changemessage
#                     break
        if correction:  # correction step
            vnew = ml.velo_one(x1, y1, z1, t1, aq, [layer, ltype])
            v1 = 0.5 * (v0 + vnew)                            
            if verbose:
                print('xyz1, layer', x1, y1, z1, layer)
                print('correction vx, vy, vz', vx, vy, vz)
            vx, vy, vz = v1
            x1 = x0 + delt * vx
            y1 = y0 + delt * vy
            z1 = z0 + delt * vz
            t1 = t0 + delt
            # check again if point needs to be changed
                ## Again not checked
#                 for e in aq.elementlist:
#                     changed, terminate, xyztchanged, changemessage = \
#                         e.changetrace(xyzt[-1],  xyztnew[0], aq, layer, ltype,
#                                       modellayer, direction, hstepmax)
#                     if changed or terminate:
#                         xyztnew = xyztchanged
#                         if changemessage:
#                             message = changemessage
#                         break
        xyzt.append(np.array([x1, y1, z1, t1]))
    if not silent:
        print(message)
    result = {"trace": np.array(xyzt), "message": message, 
              "complete": terminate}
    return result







