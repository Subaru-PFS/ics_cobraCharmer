import numpy as np
import sep
from ics.cobraCharmer import pfiDesign
import os
import sys

binSize = np.deg2rad(3.6)
regions = 112
phiThreshold = 0.6   # minimum phi angle to move in the beginning

def lazyIdentification(centers, spots, radii=None):
    n = len(centers)
    if radii is not None and len(radii) != n:
        raise RuntimeError("number of centers must match number of radii")
    ans = np.empty(n, dtype=int)
    for i in range(n):
        dist = np.absolute(spots - centers[i])
        j = np.argmin(dist)
        if radii is not None and np.absolute(centers[i] - spots[j]) > radii[i]:
            ans[i] = -1
        else:
            ans[i] = j
    return ans

def circle_fitting(p):
    x = np.real(p)
    y = np.imag(p)
    m = np.vstack([x, y, np.ones(len(p))]).T
    n = np.array(x*x + y*y)
    a, b, c = np.linalg.lstsq(m, n, rcond=None)[0]
    return a/2, b/2, np.sqrt(c+(a*a+b*b)/4)

def filtered_circle_fitting(fw, rv, threshold=1.0):
    data = np.array([], complex)
    for k in range(len(fw)):
        valid = np.where(np.abs(fw[k] - fw[k,-1]) > threshold)[0]
        if len(valid) > 0:
            last = valid[-1]
            data = np.append(data, fw[k,1:last+1])
    for k in range(len(rv)):
        valid = np.where(np.abs(rv[k] - rv[k,-1]) > threshold)[0]
        if len(valid) > 0:
            last = valid[-1]
            data = np.append(data, rv[k,1:last+1])
    if len(data) <= 3:
        return 0, 0, 0
    else:
        return circle_fitting(data)

def transform(origPoints, newPoints):
    """ return the tranformation parameters and a function that can convert origPoints to newPoints """
    origCenter = np.mean(origPoints)
    newCenter = np.mean(newPoints)
    origVectors = origPoints - origCenter
    newVectors = newPoints - newCenter
    scale = np.sum(np.abs(newVectors)) / np.sum(np.abs(origVectors))
    diffAngles = ((np.angle(newVectors) - np.angle(origVectors)) + np.pi) % (2*np.pi) - np.pi
    tilt = np.sum(diffAngles * np.abs(origVectors)) / np.sum(np.abs(origVectors))
    offset = -origCenter * scale * np.exp(tilt * (1j)) + newCenter
    def tr(x):
        return x * scale * np.exp(tilt * (1j)) + offset
    return offset, scale, tilt, tr

def diffAngle(angle1, angle2):
    return (angle1 - angle2 + np.pi) % (np.pi*2) - np.pi

def absDiffAngle(angle1, angle2):
    return np.abs(diffAngle(angle1, angle2))

def unwrappedAngle(angle, steps, fromAngle=np.nan, toAngle=np.nan, minSteps=50, delta=0.02, delta2=0.2):
    """ Adjust theta angles near 0 accounting for possible overshoots given the move """
    angle = angle % (np.pi*2)

    if np.isnan(fromAngle):
        # no clever way to unwrapped angle
        return angle

    if angle < np.pi*0.4:
        # check if the angle is over 2PI
        if absDiffAngle(angle, fromAngle) < delta and steps >= minSteps:
            # stuck at CW gard stop
            angle += np.pi*2
        elif angle < fromAngle - delta and steps >= minSteps:
            angle += np.pi*2
        elif angle < fromAngle - delta2 and steps > 0:
            angle += np.pi*2
        elif not np.isnan(toAngle) and toAngle >= np.pi*1.2:
            angle += np.pi*2
        elif np.isnan(toAngle) and fromAngle > np.pi*2.0:
            angle += np.pi*2
    elif angle > np.pi*1.8:
        # check if the angle is negative
        if absDiffAngle(angle, fromAngle) < delta and steps <= -minSteps:
            # stuck at CCW gard stop
            angle -= np.pi*2
        elif angle > fromAngle + delta and steps <= -minSteps:
            angle -= np.pi*2
        elif angle > fromAngle + delta2 and steps < 0:
            angle -= np.pi*2
        elif not np.isnan(toAngle) and toAngle < np.pi*0.8:
            angle -= np.pi*2
        elif np.isnan(toAngle) and fromAngle < 0:
            angle -= np.pi*2

    return angle

def thetaCenterAngles(thetaFW, thetaRV, threshold=1.0):
    """ calculate theta centers and angles """
    thetaAngFW = np.zeros(thetaFW.shape, dtype=float)
    thetaAngRV = np.zeros(thetaFW.shape, dtype=float)

    # measure centers
    x, y, r = filtered_circle_fitting(thetaFW, thetaRV)
#    data = np.concatenate((thetaFW.flatten(), thetaRV.flatten()))
#    x, y, r = circle_fitting(data)
    thetaCenter = x + y*(1j)
    thetaRadius = r

    # measure theta angles
    for n in range(thetaFW.shape[0]):
        thetaAngFW[n] = np.angle(thetaFW[n] - thetaCenter)
        thetaAngRV[n] = np.angle(thetaRV[n] - thetaCenter)
        home = thetaAngFW[n, 0]
        thetaAngFW[n] = (thetaAngFW[n] - home) % (np.pi*2)
        thetaAngRV[n] = (thetaAngRV[n] - home) % (np.pi*2)

        fwMid = np.argmin(abs(thetaAngFW[n] - np.pi))
        thetaAngFW[n, :fwMid][thetaAngFW[n, :fwMid] > 5.0] -= np.pi*2
        if fwMid < thetaAngFW.shape[1] - 1:
            thetaAngFW[n, fwMid:][thetaAngFW[n, fwMid:] < 1.0] += np.pi*2

        rvMid = np.argmin(abs(thetaAngRV[n] - np.pi))
        thetaAngRV[n, :rvMid][thetaAngRV[n, :rvMid] < 1.0] += np.pi*2
        if rvMid < thetaAngRV.shape[1] - 1:
            thetaAngRV[n, rvMid:][thetaAngRV[n, rvMid:] > 5.0] -= np.pi*2
#        if thetaAngRV[n, 0] < 1.0:
#            thetaAngRV[c, n] += np.pi*2

    # mark bad cobras by checking hard stops
    bad = np.any(thetaAngRV[:, 0] < np.pi*2)

    return thetaCenter, thetaRadius, thetaAngFW, thetaAngRV, bad

def phiCenterAngles(phiFW, phiRV):
    """ calculate phi centers and angles """
    phiAngFW = np.zeros(phiFW.shape, dtype=float)
    phiAngRV = np.zeros(phiFW.shape, dtype=float)

    # measure centers
    x, y, r = filtered_circle_fitting(phiFW, phiRV)
#    data = np.concatenate((phiFW.flatten(), phiRV.flatten()))
#    x, y, r = circle_fitting(data)
    phiCenter = x + y*(1j)
    phiRadius = r

    # measure phi angles
    phiAngFW = np.angle(phiFW - phiCenter)
    phiAngRV = np.angle(phiRV - phiCenter)
    home = np.copy(phiAngFW[:, 0, np.newaxis])
    phiAngFW = (phiAngFW - home + np.pi/2) % (np.pi*2) - np.pi/2
    phiAngRV = (phiAngRV - home + np.pi/2) % (np.pi*2) - np.pi/2

    # mark bad cobras by checking hard stops
    bad = np.any(phiAngRV[:, 0] < np.pi)

    return phiCenter, phiRadius, phiAngFW, phiAngRV, bad

def speed(angFW, angRV, steps, delta=0.1):
    # calculate average speed
    repeat = angFW.shape[0]
    iteration = angFW.shape[1] - 1

    fSteps = 0
    fAngle = 0
    rSteps = 0
    rAngle = 0
    for n in range(repeat):
        invalid = np.where(angFW[n] > angRV[n, 0] - delta)[0]
        last = iteration
        if len(invalid) > 0:
            last = invalid[0] - 2
            if last < 0:
                last = 0
        fSteps += last * steps
        fAngle += angFW[n, last]

        invalid = np.where(angRV[n] < delta)[0]
        last = iteration
        if len(invalid) > 0:
            last = invalid[0] - 2
            if last < 0:
                last = 0
        rSteps += last * steps
        rAngle += angRV[n, 0] - angRV[n, last]

    if fSteps > 0:
        speedFW = fAngle / fSteps
    else:
        speedFW = 0
    if rSteps > 0:
        speedRV = rAngle / rSteps
    else:
        speedRV = 0

    return speedFW, speedRV

def motorMaps(angFW, angRV, steps, delta=0.1):
    """ use Johannes weighting for motor maps, delta is the margin for detecting hard stops """
    mmFW = np.zeros(regions, dtype=float)
    mmRV = np.zeros(regions, dtype=float)
    bad = False
    repeat = angFW.shape[0]
    iteration = angFW.shape[1] - 1

    # calculate motor maps in Johannes way
    for b in range(regions):
        binMin = binSize * b
        binMax = binMin + binSize

        # forward motor maps
        fracSum = 0
        valueSum = 0
        for n in range(repeat):
            for k in range(iteration):
                if angFW[n, k+1] <= angFW[n, k] or angRV[n, 0] - angFW[n, k+1] < delta:
                    # hit hard stop or somethings went wrong, then skip it
                    continue
                if angFW[n, k] < binMax and angFW[n, k+1] > binMin:
                    moveSizeInBin = np.min([angFW[n, k+1], binMax]) - np.max([angFW[n, k], binMin])
                    entireMoveSize = angFW[n, k+1] - angFW[n, k]
                    fraction = moveSizeInBin * moveSizeInBin / entireMoveSize
                    fracSum += fraction
                    valueSum += fraction * entireMoveSize / steps
        if fracSum > 0:
            mmFW[b] = valueSum / fracSum
        else:
            mmFW[b] = 0

        # reverse motor maps
        fracSum = 0
        valueSum = 0
        for n in range(repeat):
            for k in range(iteration):
                if angRV[n, k+1] >= angRV[n, k] or angRV[n, k+1] < delta:
                    # hit hard stop or somethings went wrong, then skip it
                    continue
                if angRV[n, k] > binMin and angRV[n, k+1] < binMax:
                    moveSizeInBin = np.min([angRV[n, k], binMax]) - np.max([angRV[n, k+1], binMin])
                    entireMoveSize = angRV[n, k] - angRV[n, k+1]
                    fraction = moveSizeInBin * moveSizeInBin / entireMoveSize
                    fracSum += fraction
                    valueSum += fraction * entireMoveSize / steps
        if fracSum > 0:
            mmRV[b] = valueSum / fracSum
        else:
            mmRV[b] = 0

    # fill the zeros closed to hard stops
    nz = np.nonzero(mmFW)[0]
    if nz.size > 0:
        mmFW[:nz[0]] = mmFW[nz[0]]
        mmFW[nz[-1]+1:] = mmFW[nz[-1]]
    else:
        bad = True

    nz = np.nonzero(mmRV)[0]
    if nz.size > 0:
        mmRV[:nz[0]] = mmRV[nz[0]]
        mmRV[nz[-1]+1:] = mmRV[nz[-1]]
    else:
        bad = True

    return mmFW, mmRV, bad

def updateThetaMotorMaps(model, thetaMMFW, thetaMMRV, bad=None, slow=True):
    # update theta motor maps
    if bad is None:
        idx = np.arange(model.nCobras)
    else:
        idx = np.where(np.logical_not(bad))[0]

    if slow:
        mmFW = binSize / model.S1Pm
        mmRV = binSize / model.S1Nm
    else:
        mmFW = binSize / model.F1Pm
        mmRV = binSize / model.F1Nm

    mmFW[idx] = thetaMMFW[idx]
    mmRV[idx] = thetaMMRV[idx]

    model.updateMotorMaps(thtFwd=mmFW, thtRev=mmRV, useSlowMaps=slow)

def updatePhiMotorMaps(model, phiMMFW, phiMMRV, bad=None, slow=True):
    # update phi motor maps
    if bad is None:
        idx = np.arange(model.nCobras)
    else:
        idx = np.where(np.logical_not(bad))[0]

    if slow:
        mmFW = binSize / model.S2Pm
        mmRV = binSize / model.S2Nm
    else:
        mmFW = binSize / model.F2Pm
        mmRV = binSize / model.F2Nm

    mmFW[idx] = phiMMFW[idx]
    mmRV[idx] = phiMMRV[idx]

    model.updateMotorMaps(phiFwd=mmFW, phiRev=mmRV, useSlowMaps=slow)

def geometry(idx, thetaC, thetaR, thetaFW, thetaRV, phiC, phiR, phiFW, phiRV):
    """ calculate geometry from theta and phi motor maps process """
    nCobra = len(thetaC)

    # calculate arm legnths
    thetaL = np.absolute(phiC - thetaC)
    phiL = phiR

    # calculate phi hard stops
    phiCCW = np.full(nCobra, np.pi)
    phiCW = np.zeros(nCobra)
    y = idx

    s = np.angle(thetaC - phiC)[y]
    for n in range(phiFW.shape[1]):
        # CCW hard stops for phi arms
        t = (np.angle(phiFW[y, n, 0] - phiC[y]) - s + (np.pi/2)) % (np.pi*2) - (np.pi/2)
        p = np.where(t < phiCCW[y])[0]
        phiCCW[y[p]] = t[p]
        # CW hard stops for phi arms
        t = (np.angle(phiRV[y, n, 0] - phiC[y]) - s + (np.pi/2)) % (np.pi*2) - (np.pi/2)
        p = np.where(t > phiCW[y])[0]
        phiCW[y[p]] = t[p]

    # calculate theta hard stops
    thetaCCW = np.zeros(nCobra)
    thetaCW = np.zeros(nCobra)
    y = idx

    for n in range(thetaFW.shape[1]):
        # CCW hard stops for theta arms
        a = np.absolute(thetaFW[y, n, 0] - thetaC[y])
        s = np.arccos((thetaL[y]*thetaL[y] + a*a - phiL[y]*phiL[y]) / (2*a*thetaL[y]))
        t = (np.angle(thetaFW[y, n, 0] - thetaC[y]) + s) % (np.pi*2)
        if n == 0:
            thetaCCW[y] = t
        else:
            q = (t - thetaCCW[y] + np.pi) % (np.pi*2) - np.pi
            p = np.where(q < 0)[0]
            thetaCCW[y[p]] = t[p]

        # CW hard stops for theta arms
        a = np.absolute(thetaRV[y, n, 0] - thetaC[y])
        s = np.arccos((thetaL[y]*thetaL[y] + a*a - phiL[y]*phiL[y]) / (2*a*thetaL[y]))
        t = (np.angle(thetaRV[y, n, 0] - thetaC[y]) + s) % (np.pi*2)
        if n == 0:
            thetaCW[y] = t
        else:
            q = (t - thetaCW[y] + np.pi) % (np.pi*2) - np.pi
            p = np.where(q > 0)[0]
            thetaCW[y[p]] = t[p]

    return thetaL, phiL, thetaCCW, thetaCW, phiCCW, phiCW

def matchPositions(objects, guess, dist=None):
    """ Given a list of positions, return the closest ones to the guess.

    Args
    ----
    objects : `ndarray`, which includes an x and a y column.
       The _measured_ positions, from the camera.

    guess : `ndarray` of complex coordinates.
       Close to where we expect the spots to be.

    dist : `float`
       Maximum distance between detected objcect and guess, for matching.

    Returns
    -------
    pos : `ndarray` of complex coordinates
       The measured positions of the cobras.
       Note that (hmm), the cobra center is returned of there is not matching spot.

    indexMap : `ndarray` of ints
       Indices from our cobra array to the matching spots.
       -1 if there is no matching spot.
    """

    measPos = np.array(objects['x'] + objects['y']*(1j))
    target = lazyIdentification(guess, measPos, radii=dist)

    pos = np.zeros(len(guess), dtype=complex)
    for n, k in enumerate(target):
        if k >= 0:
            pos[n] = measPos[k]

    return pos, target

def mapDone(centers, points, limits, k,
             needAtEnd=3, closeEnough=np.deg2rad(1),
             limitTolerance=np.deg2rad(2)):
    """ Return a mask of the cobras which we deem at the axis limit.

    See thetaFWDone.
    """

    if centers is None or limits is None or k+1 < needAtEnd:
        return None, None

    lastAngles = np.angle(points[:,k-needAtEnd+2:k+2] - centers[:,None])
    atEnd = np.abs(lastAngles[:,-1] - limits) <= limitTolerance
    endDiff = np.abs(np.diff(lastAngles, axis=1))
    stable = np.all(endDiff <= closeEnough, axis=1)

    # Diagnostic: return the needAtEnd distances from the limit.
    anglesFromEnd = lastAngles - limits[:,None]

    return atEnd & stable, anglesFromEnd

def calculateScale(movedAngle, expectedAngle, scaleFactor=1):
    """ calculate scale from moved angle """

    ratio = expectedAngle / movedAngle
    scale = (ratio - 1) / scaleFactor + 1

    return scale

def calculateOntime(ontime, speedRatio, scaling, modelParameter, maxOntime):
    """ calculate on-time """

    if speedRatio <= 0:
        return ontime

    b0 = modelParameter
    b1 = ontime
    a0 = np.sqrt(b1*b1 + b0*b0) - b0
    a1 = a0*((speedRatio - 1) / scaling + 1) + b0

    r = min(np.sqrt(a1*a1 - b0*b0), maxOntime)
    if r > b1 * 1.5:
        r = b1 * 1.5
    return np.rint(r * 1000.0) / 1000.0

def smooth(x, window_len=11, window='hamming'):
    """ smooth the data """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w/w.sum(), s, mode='valid')

    return y[(window_len//2):-(window_len//2)]

def calMoveSegments(thetaMoved, phiMoved, thetaFrom, phiFrom,
                    mmTheta, mmPhi, maxSteps, nSegments, phiOffset=0,
                    trajectory=None, cId=None):
    """ calculate move segments """

    moved = 0
    n = 0
    if thetaMoved > 0:
        while moved < thetaMoved:
            idx = np.nanargmin(abs(mmTheta[0]['angle'] - thetaFrom - moved))
            moved += mmTheta[0,idx]['speed'] * maxSteps
            n += 1
    elif thetaMoved < 0:
         while moved > thetaMoved:
            idx = np.nanargmin(abs(mmTheta[1]['angle'] - thetaFrom - moved))
            moved += mmTheta[1,idx]['speed'] * maxSteps
            n += 1
    nSeg = n

    moved = 0
    n = 0
    if phiMoved > 0:
        while moved < phiMoved:
            idx = np.nanargmin(abs(mmPhi[0]['angle'] - phiFrom - moved))
            moved += mmPhi[0,idx]['speed'] * maxSteps
            n += 1
    elif phiMoved < 0:
         while moved > phiMoved:
            idx = np.nanargmin(abs(mmPhi[1]['angle'] - phiFrom - moved))
            moved += mmPhi[1,idx]['speed'] * maxSteps
            n += 1
    if nSeg < n:
        nSeg = n

    if nSeg < nSegments:
        nSeg = nSegments

    tSteps = np.zeros(nSeg, int)
    pSteps = np.zeros(nSeg, int)
    tOntimes = np.zeros(nSeg, float)
    pOntimes = np.zeros(nSeg, float)
    tMoves = np.zeros(nSeg, float)
    pMoves = np.zeros(nSeg, float)

    if thetaMoved > 0:
        # early scheme
        moved = 0
        n = 0
        while n < nSeg:
            idx = np.nanargmin(abs(mmTheta[0]['angle'] - thetaFrom - moved))
            speed = mmTheta[0,idx]['speed']
            tOntimes[n] = mmTheta[0,idx]['ontime']
            if speed * maxSteps < thetaMoved - moved:
                moved += speed * maxSteps
                tSteps[n] = maxSteps
                tMoves[n] = speed * maxSteps
                n += 1
            else:
                tSteps[n] = int((thetaMoved - moved) / speed)
                tMoves[n] = speed * tSteps[n]
                break

    elif thetaMoved < 0:
        # late scheme
        moved = thetaMoved
        n = nSeg - 1
        while n >= 0:
            idx = np.nanargmin(abs(mmTheta[1]['angle'] - thetaFrom - moved))
            speed = mmTheta[1,idx]['speed']
            tOntimes[n] = mmTheta[1,idx]['ontime']
            if speed * maxSteps > moved:
                moved -= speed * maxSteps
                tSteps[n] = -maxSteps
                tMoves[n] = speed * maxSteps
                n -= 1
            else:
                tSteps[n] = -int(moved / speed)
                tMoves[n] = -speed * tSteps[n]
                break

    if phiMoved > 0:
        # late scheme, but move away from center if too close
        left = np.min([phiThreshold - phiOffset - phiFrom, phiMoved])
        moved = 0
        n = 0
        while n < nSeg and moved < left:
            idx = np.nanargmin(abs(mmPhi[0]['angle'] - phiFrom - moved))
            speed = mmPhi[0,idx]['speed']
            pOntimes[n] = mmPhi[0,idx]['ontime']
            if speed * maxSteps < left - moved:
                moved += speed * maxSteps
                pSteps[n] = maxSteps
                pMoves[n] = speed * maxSteps
                n += 1
            else:
                pSteps[n] = int((left - moved) / speed)
                pMoves[n] = speed * pSteps[n]
                moved += pMoves[n]
                break

        moved = phiMoved - moved
        n = nSeg - 1
        while n >= 0:
            idx = np.nanargmin(abs(mmPhi[0]['angle'] - phiFrom - moved))
            speed = mmPhi[0,idx]['speed']
            pOntimes[n] = mmPhi[0,idx]['ontime']
            if speed * maxSteps < moved:
                moved -= speed * maxSteps
                pSteps[n] = maxSteps
                pMoves[n] = speed * maxSteps
                n -= 1
            else:
                pSteps[n] = int(moved / speed)
                pMoves[n] = speed * pSteps[n]
                break

    elif phiMoved < 0:
        # early scheme
        moved = 0
        n = 0
        while n < nSeg:
            idx = np.nanargmin(abs(mmPhi[1]['angle'] - phiFrom - moved))
            speed = mmPhi[1,idx]['speed']
            pOntimes[n] = mmPhi[1,idx]['ontime']
            if speed * maxSteps > phiMoved - moved:
                moved += speed * maxSteps
                pSteps[n] = -maxSteps
                pMoves[n] = speed * maxSteps
                n += 1
            else:
                pSteps[n] = -int((phiMoved - moved) / speed)
                pMoves[n] = -speed * pSteps[n]
                break

    if trajectory is not None:
        allTheta = interpAllConstantSpeedSteps(tSteps[:nSegments], tOntimes[:nSegments],
                                               tMoves[:nSegments], trajectory.getSpacing())
        allPhi = interpAllConstantSpeedSteps(pSteps[:nSegments], pOntimes[:nSegments],
                                             pMoves[:nSegments], trajectory.getSpacing())
        trajectory.addMotion(cId, allTheta, allPhi)

    return (tSteps[:nSegments], pSteps[:nSegments], tOntimes[:nSegments], pOntimes[:nSegments],
            np.sum(tMoves[:nSegments]), np.sum(pMoves[:nSegments]))

def calNSegments(angle, fromAngle, mm, maxSteps):
    moved = 0
    n = 0
    if angle > 0:
        while moved < angle:
            idx = np.nanargmin(abs(mm[0]['angle'] - fromAngle - moved))
            moved += mm[0,idx]['speed'] * maxSteps
            n += 1
    elif angle < 0:
         while moved > angle:
            idx = np.nanargmin(abs(mm[1]['angle'] - fromAngle - moved))
            moved += mm[1,idx]['speed'] * maxSteps
            n += 1

    return n

def calculateSteps(cId, maxSteps,
                   thetaAngle, phiAngle, fromTheta, fromPhi,
                   thetaFast, phiFast,
                   model, trajectory=None):
    # Get the integrated step maps for the theta angle
    if thetaAngle >= 0:
        if thetaFast:
            thetaModel = model.posThtSteps[cId]
        else:
            thetaModel = model.posThtSlowSteps[cId]
    else:
        if thetaFast:
            thetaModel = model.negThtSteps[cId]
        else:
            thetaModel = model.negThtSlowSteps[cId]

    # Get the integrated step maps for the phi angle
    if phiAngle >= 0:
        if phiFast:
            phiModel = model.posPhiSteps[cId]
        else:
            phiModel = model.posPhiSlowSteps[cId]
    else:
        if phiFast:
            phiModel = model.negPhiSteps[cId]
        else:
            phiModel = model.negPhiSlowSteps[cId]

    # Calculate the total number of motor steps for the theta movement
    stepsRange = np.interp([fromTheta, fromTheta + thetaAngle], model.thtOffsets[cId], thetaModel)
    if not np.all(np.isfinite(stepsRange)):
        raise ValueError(f"theta angle to step interpolation out of range: "
                         f"Cobra#{cId+1} {fromTheta}:{fromTheta + thetaAngle}")
    thetaSteps = np.rint(stepsRange[1] - stepsRange[0]).astype('i4')
    thetaFromSteps = stepsRange[0]

    # Calculate the total number of motor steps for the phi movement
    stepsRange = np.interp([fromPhi, fromPhi + phiAngle], model.phiOffsets[cId], phiModel)
    if not np.all(np.isfinite(stepsRange)):
        raise ValueError(f"phi angle to step interpolation out of range: "
                         f"Cobra#{cId+1} {fromPhi}:{fromPhi + phiAngle}")
    phiSteps = np.rint(stepsRange[1] - stepsRange[0]).astype('i4')
    phiFromSteps = stepsRange[0]

    # calculate phi motor steps away from center in the beginning
    safePhiSteps = 0
    if phiSteps > 0:
        safePhiAngle = phiThreshold - model.phiIn[cId] - np.pi
        if fromPhi < safePhiAngle:
            stepsRange = np.interp([fromPhi, safePhiAngle], model.phiOffsets[cId], phiModel)
            if not np.all(np.isfinite(stepsRange)):
                    raise ValueError(f"phi angle to step interpolation out of range: "
                                     f"Cobra#{cId+1} {fromPhi}:{safePhiAngle}")
            safePhiSteps = min(np.rint(stepsRange[1] - stepsRange[0]).astype('i4'), phiSteps, maxSteps)

    if abs(thetaSteps) > maxSteps or abs(phiSteps) > maxSteps:
        if abs(thetaSteps) >= abs(phiSteps):
            if phiSteps > 0 and thetaSteps > 0:
                phiSteps = max(phiSteps - thetaSteps + maxSteps, safePhiSteps)
                thetaSteps = maxSteps
            elif phiSteps > 0 and thetaSteps < 0:
                phiSteps = max(phiSteps + thetaSteps + maxSteps, safePhiSteps)
                thetaSteps = -maxSteps
            elif phiSteps < 0 and thetaSteps > 0:
                phiSteps = max(phiSteps, -maxSteps)
                thetaSteps = maxSteps
            elif phiSteps < 0 and thetaSteps < 0:
                phiSteps = max(phiSteps, -maxSteps)
                thetaSteps = -maxSteps
            elif thetaSteps > 0:
                thetaSteps = maxSteps
            else:
                thetaSteps = -maxSteps
        else:
            if phiSteps > 0 and thetaSteps > 0:
                thetaSteps = min(thetaSteps, maxSteps)
                phiSteps = maxSteps
            elif phiSteps > 0 and thetaSteps < 0:
                thetaSteps = min(thetaSteps + phiSteps - maxSteps, 0)
                phiSteps = maxSteps
            elif phiSteps < 0 and thetaSteps > 0:
                thetaSteps = min(thetaSteps, maxSteps)
                phiSteps = -maxSteps
            elif phiSteps < 0 and thetaSteps < 0:
                thetaSteps = min(thetaSteps - phiSteps - maxSteps, 0)
                phiSteps = -maxSteps
            elif phiSteps > 0:
                phiSteps = maxSteps
            else:
                phiSteps = -maxSteps

    if trajectory is not None:
        allTheta = interpStepsOnPath(thetaFromSteps, thetaSteps, thetaModel,
                                     model.thtOffsets[cId], trajectory.getSpacing())
        allPhi = interpStepsOnPath(phiFromSteps, phiSteps, phiModel,
                                   model.phiOffsets[cId], trajectory.getSpacing())
        trajectory.addMotion(cId, allTheta, allPhi)

    toTheta = np.interp(thetaFromSteps + thetaSteps, thetaModel, model.thtOffsets[cId])
    toPhi = np.interp(phiFromSteps + phiSteps, phiModel, model.phiOffsets[cId])
    return thetaSteps, phiSteps, toTheta - fromTheta, toPhi - fromPhi

def _spaceInterpolation(startSteps, moveSteps, spacing):
    return np.linspace(startSteps, startSteps + moveSteps, int(np.abs(moveSteps)//spacing))

def interpStepsOnPath(startSteps, moveSteps, stepMap, angleMap, spacing):
    """Return the angles taken along a motor's path, equally spaced in steps.

    Args
    ----
    startSteps : `int`
     the starting step position.
    moveSteps : `int`
     the number of steps to move (can be negative).
    stepMap : array
     the step array we interpolate into
    angleMap : array
     the angle array we want interpolations out of.
    spacing : `float`
     how tightly to space our interpolation

    Returns
    -------
    angles : array
     the
    """

    x = _spaceInterpolation(startSteps, moveSteps, spacing)
    return np.interp(x, stepMap, angleMap)

def interpAllConstantSpeedSteps(steps, ontimes, moves, spacing):
    raise NotImplementedError('have not done constant speed trajectories yet.')
