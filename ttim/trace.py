import numpy as np

def timtracelines(ml, xstart, ystart, zstart, tstartend, tstartoffset, 
                  deltlist, nstepmax=100, hstepmax=10, silent=False, 
                  correctionstep=False):
    xyzt = np.array([[xstart, ystart, zstart, tstartend[0]]])
    if np.isscalar(deltlist):
        deltlist = len(tstartend) * [deltlist]
    for itrace in range(len(tstartend) - 1):
        x0, y0, z0, t0 = xyzt[-1]
        trace = timtraceline(ml, x0, y0, z0, t0 + tstartoffset, 
                             deltlist[itrace], tstartend[itrace + 1],
                             nstepmax=nstepmax, hstepmax=hstepmax, 
                             silent=silent, correctionstep=correctionstep)
        xyzt = np.vstack((xyzt, trace['trace']))
        if trace['message'] != 'reached maximum time tmax':
            break
    result = {"trace": np.array(xyzt), "message": trace['message'], 
              "complete": trace['complete']}
    return result                 
               

def timtraceline(ml, xstart, ystart, zstart, tstart, delt, tmax,
                 nstepmax=100, hstepmax=10, silent=False, 
                 correctionstep=False):
    # treating aquifer layers and leaky layers the same way
    direction = 1 # forward
    terminate = False
    message = "no message"
    eps = 1e-10  # used to place point just above or below aquifer top or bottom
    aq = ml.aq.find_aquifer_data(xstart, ystart)
    if zstart > aq.z[0] or zstart < aq.z[-1]:
        terminate = True
        message = "starting z value not inside aquifer"
    # slightly alter starting location not to get stuck in surpring points
    # starting at time 0
    xyzt = [np.array([xstart * (1 + eps), ystart * (1 + eps), zstart, tstart])]  
    layerlist = []  # to keep track of layers for plotting with colors
    for istep in range(nstepmax):
        if terminate:
            break
        do_correction = correctionstep # do correction step, unless do_correction changed to False
        x0, y0, z0, t0 = xyzt[-1]
        #aq = ml.aq.find_aquifer_data(x0, y0)  # find new aquifer
        layer, ltype, modellayer = aq.findlayer(z0)
        layerlist.append(modellayer)
        v0 = ml.velocomp(x0, y0, z0, t0, aq, [layer, ltype])
        vx, vy, vz = v0
        substep = 1 # take max 2 substeps
        
        for steps in range(2):
            if substep <= 2:
                            
                # check if max time reached
                if t0 + delt > tmax:
                    delt0 = tmax - t0
                else:
                    delt0 = delt

                # check if horizontal step larger than hstepmax
                hstep = np.sqrt(vx ** 2 + vy ** 2) * delt0
                if hstep > hstepmax:
                    delt0 = hstepmax / hstep * delt0

                # check if going to different layer
                z1 = z0 + delt0 * vz
                layer1, ltype1, modellayer1 = aq.findlayer(z1)
                if modellayer1 < modellayer: # step up to next layer
                    delt0 = (aq.z[modellayer] - z0) / (z1 - z0) * delt0
                    z1 = aq.z[modellayer] + eps
                    do_correction = False
                elif modellayer1 > modellayer: # step down to next layer
                    delt0 = (z0 - aq.z[modellayer + 1]) / (z0 - z1) * delt0
                    z1 = aq.z[modellayer1] - eps
                    do_correction = False

                # potential new location
                x1 = x0 + delt0 * vx
                y1 = y0 + delt0 * vy
                t1 = t0 + delt0
                xyzt1 = np.array([x1, y1, z1, t1])

                # check elements if point needs to be changed
                for e in ml.elementlist:
                    changed, terminate, xyztnew, changemessage = e.changetrace(
                        xyzt[-1], xyzt1, aq, layer, ltype, modellayer, 
                        direction, hstepmax)
                    if changed or terminate:
                        x1, y1, z1, t1 = xyztnew
                        do_correction = False
                        message = changemessage
                        break

                if t1 >= tmax:
                    terminate = True
                    message = 'reached maximum time tmax'
                if istep == nstepmax - 1:
                    terminate = True
                    message = 'reached maximum number of steps'
                if terminate: 
                    xyzt.append(np.array([x1, y1, z1, t1]))
                    break

                if substep == 1 and do_correction: # do correction step
                    vnew = ml.velocomp(x1, y1, z1, t1, aq, [layer, ltype])
                    v1 = 0.5 * (v0 + vnew)                            
                    vx, vy, vz = v1
                    substep = 2           
                else:
                    xyzt.append(np.array([x1, y1, z1, t1]))
                    break

    if not silent:
        print(message)
    result = {"trace": np.array(xyzt), "message": message, 
              "complete": terminate}
    return result

# test with tmult. didn't improve much
def timtraceline2(ml, xstart, ystart, zstart, tstart, tmax,
                  delt, deltmin, deltmax, tmult=1.1, 
                  nstepmax=100, hstepmax=10, silent=False, 
                  correct=False):
    # treating aquifer layers and leaky layers the same way
    direction = 1 # forward
    terminate = False
    message = "no message"
    eps = 1e-10  # used to place point just above or below aquifer top or bottom
    aq = ml.aq.find_aquifer_data(xstart, ystart)
    if zstart > aq.z[0] or zstart < aq.z[-1]:
        terminate = True
        message = "starting z value not inside aquifer"
    # slightly alter starting location not to get stuck in surpring points
    # starting at time 0
    xyzt = [np.array([xstart * (1 + eps), ystart * (1 + eps), zstart, tstart])]  
    layerlist = []  # to keep track of layers for plotting with colors
    speednew = 0.0
    for istep in range(nstepmax):
        if terminate:
            break
        do_correction = correct # do correction, unless do_correction changed to False
        x0, y0, z0, t0 = xyzt[-1]
        #aq = ml.aq.find_aquifer_data(x0, y0)  # find new aquifer
        layer, ltype, modellayer = aq.findlayer(z0)
        layerlist.append(modellayer)
        speedold = speednew
        v0 = ml.velocomp(x0, y0, z0, t0, aq, [layer, ltype])
        speednew = np.sqrt(np.sum(v0 ** 2))
        if speednew < speedold:
            delt *= 1.2
            if delt > deltmax:
                delt = deltmax
        else:
            delt /= 1.2
            if delt < deltmin:
                delt = deltmin
        print('delt:', delt)
        vx, vy, vz = v0
        substep = 1 # take max 2 substeps
        
        for steps in range(2):
            if substep <= 2:
                            
                # check if max time reached
                if t0 + delt > tmax:
                    delt0 = tmax - t0
                else:
                    delt0 = delt

                # check if horizontal step larger than hstepmax
                hstep = np.sqrt(vx ** 2 + vy ** 2) * delt0
                if hstep > hstepmax:
                    delt0 = hstepmax / hstep * delt0

                # check if going to different layer
                z1 = z0 + delt0 * vz
                layer1, ltype1, modellayer1 = aq.findlayer(z1)
                if modellayer1 < modellayer: # step up to next layer
                    delt0 = (aq.z[modellayer] - z0) / (z1 - z0) * delt0
                    z1 = aq.z[modellayer] + eps
                    do_correction = False
                elif modellayer1 > modellayer: # step down to next layer
                    delt0 = (z0 - aq.z[modellayer + 1]) / (z0 - z1) * delt0
                    z1 = aq.z[modellayer1] - eps
                    do_correction = False

                # potential new location
                x1 = x0 + delt0 * vx
                y1 = y0 + delt0 * vy
                t1 = t0 + delt0
                xyzt1 = np.array([x1, y1, z1, t1])

                # check elements if point needs to be changed
                for e in ml.elementlist:
                    changed, terminate, xyztnew, changemessage = e.changetrace(
                        xyzt[-1], xyzt1, aq, layer, ltype, modellayer, 
                        direction, hstepmax)
                    if changed or terminate:
                        x1, y1, z1, t1 = xyztnew
                        do_correction = False
                        message = changemessage
                        break

                if t1 >= tmax:
                    terminate = True
                    message = 'reached maximum time tmax'
                if istep == nstepmax - 1:
                    terminate = True
                    message = 'reached maximum number of steps'
                if terminate: 
                    xyzt.append(np.array([x1, y1, z1, t1]))
                    break

                if substep == 1 and do_correction: # do correction step
                    vnew = ml.velocomp(x1, y1, z1, t1, aq, [layer, ltype])
                    v1 = 0.5 * (v0 + vnew)                            
                    vx, vy, vz = v1
                    substep = 2           
                else:
                    xyzt.append(np.array([x1, y1, z1, t1]))
                    break

    if not silent:
        print(message)
    result = {"trace": np.array(xyzt), "message": message, 
              "complete": terminate}
    return result







