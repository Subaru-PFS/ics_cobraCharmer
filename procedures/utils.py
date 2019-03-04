from importlib import reload

import logging
import numpy as np

from ics.cobraCharmer import imageSet
from ics.cobraCharmer.camera import cameraFactory

reload(imageSet)

def lazyIdentification(centers, spots, radii=None):
    n = len(centers)
    if radii is not None:
        if np.isscalar(radii):
            radii = np.full(n, radii, dtype='f4')
        if len(radii) != n:
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

def getBrokenCobras(pfi, moduleName='unknown'):
    if moduleName == 'Spare1':
        broken = [1, 39, 43, 54]
    else:
        broken = []

    return broken

def targetThetasOut(pfi, cobras):
    """ Return targets moving theta arms pointing inward.  """

    # partition cobras into odd and even sets
    oddCobras = [c for c in cobras if (c.cobraNum%2 == 1)]
    evenCobras = [c for c in cobras if (c.cobraNum%2 == 0)]

    # Calculate up/down(outward) angles
    oddTheta = pfi.thetaToLocal(oddCobras, [np.deg2rad(270)]*len(oddCobras))
    # oddMoves[oddMoves>1.85*np.pi] = 0

    evenTheta = pfi.thetaToLocal(evenCobras, [np.deg2rad(90)]*len(evenCobras))
    # evenMoves[evenMoves>1.85*np.pi] = 0
    thetas = np.concatenate((evenTheta, oddTheta))
    phis = np.full(len(thetas), 0.0)

    outTargets = pfi.anglesToPositions(cobras, thetas, phis)

    return outTargets

def takePhiMap(pfi, imageDir,
               cobras,
               ontimes=None,
               setName=None,
               steps=50,
               phiRange=5000):

    dataset = imageSet.ImageSet(cameraFactory(), imageDir, setName=setName,
                                makeStack=True, saveSpots=True)

    # record the phi movements

    # forward phi motor maps
    n = 0
    dataset.expose(f'phiForward{n}Begin')
    for k in range(phiRange//steps):
        pfi.moveAllSteps(cobras, 0, steps)
        dataset.expose(f'phiForward{n}N{k}')

    # make sure it goes to the limit
    if ontimes is not None:
        pfi.calibModel.updateOntimes(*ontimes['fast'])
    pfi.moveAllSteps(cobras, 0, 5000)
    if ontimes is not None:
        pfi.calibModel.updateOntimes(*ontimes['normal'])
    dataset.expose(f'phiForward{n}End')

    # reverse phi motor maps
    dataset.expose(f'phiReverse{n}Begin')
    for k in range(phiRange//steps):
        pfi.moveAllSteps(cobras, 0, -steps)
        dataset.expose(f'phiReverse{n}N{k}')

    # At the end, make sure the cobra back to the hard stop
    if ontimes is not None:
        pfi.calibModel.updateOntimes(*ontimes['fast'])
    pfi.moveAllSteps(cobras, 0, -5000)
    if ontimes is not None:
        pfi.calibModel.updateOntimes(*ontimes['normal'])
    dataset.expose(f'phiReverse{n}End')

    dataset.saveStack(f'phiStack')

    return dataset

def measureSpots(centers, dataSet, positions, names=None, disp=None,
                 trackCenters=True, sigma=10.0, trackRadius=20.0):

    """ Measure a set of spots at known positions."""

    if any(positions != sorted(positions)):
        raise ValueError("positions must be sorted.")

    if names is None:
        names = dataSet.namelist.keys()
    if len(positions) != len(names):
        raise ValueError('number of positions (%d) must match the number of names (%d)' %
                         (len(positions), len(names)))

    nCobras = len(centers)
    res = np.zeros(len(positions),
                   dtype=dict(names=('name', 'pos', 'centers'),
                              formats=(np.object, np.float32, f'{nCobras}c8')))

    # We _start_ by matching the measured centers against the provided centers. If
    # the trackCenters flag is set, we then match to the previous position of the cobra.
    nearestCenters = centers
    radii = None
    for i in range(len(positions)):
        res[i]['name'] = names[i]
        res[i]['pos'] = positions[i]
        cs, _ = dataSet.spots(names[i], sigma=sigma, disp=disp)

        if len(cs) != len(centers):
            raise RuntimeError('in %s, number of spots (%d) != number of cobras (%d)' %
                               (names[i], len(cs), len(centers)))

        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = lazyIdentification(nearestCenters, spots, radii=radii)
        nomatch_w = (-1 == idx)
        if nomatch_w.sum() > 0:
            logging.warn(f'failed to match spots {np.where(nomatch_w)[0]} for {names[i]} at {positions[i]} steps')
        if trackCenters:
            nearestCenters = spots[idx]
            radii = trackRadius

        res[i]['centers'][:] = spots[idx]

    return res

def datasetNamesAndPositions(dataSet, stepSize):
    """ Given a dataSet, return the forward and reverse names and expected positions. """

    def name2steps(n):
        if n.find('Begin') >= 0:
            return 0.0
        sword = n[n.find('N')+1:]
        return stepSize*int(sword)

    allFiles = dataSet.namelist.keys()
    fwd = np.array([n for n in allFiles if 'Forward0N' in n or 'ForwardBegin' in n])
    fsteps = np.array([name2steps(n) for n in list(fwd)])
    f_w = np.argsort(fsteps)
    fsteps = fsteps[f_w]
    fwd = fwd[f_w]

    rev = np.array([n for n in allFiles if 'Reverse0N' in n or 'ReverseBegin' in n])
    rsteps = np.array([name2steps(n) for n in list(rev)])
    r_w = np.argsort(rsteps)
    rsteps = rsteps[r_w]
    rev = rev[r_w]

    return fwd, fsteps, rev, rsteps

def phiMeasure(centers, dataSet, stepSize):
    fnames, fsteps, rnames, rsteps = datasetNamesAndPositions(dataSet, stepSize)

    fwd = measureSpots(centers, dataSet, fsteps, names=fnames)
    rev = measureSpots(centers, dataSet, rsteps, names=rnames)

    return fwd['centers'].T, rev['centers'].T

def calcPhiGeometry(phiFW, phiRV, goodIdx=None):
    """ Calculate as much phi geometry as we can from arcs between stops.

    Args
    ----
    phiFW, phiRV : array
      As many spots on the arc as can be gathered. All assumed to be taken with
      theta at CCW home.

    Returns
    -------
    phiCenter : center of rotation
    phiAngFW : forward limit
    phiAngRV : reverse limit
    """

    cobCnt,posCnt = phiFW.shape
    if goodIdx is None:
        goodIdx = np.arange(cobCnt)

    phiCenter = np.zeros(cobCnt, dtype=complex)
    phiRadius = np.zeros(cobCnt, dtype=np.float32)
    phiAngFW = np.zeros((cobCnt, posCnt), dtype='f4')
    phiAngRV = np.zeros((cobCnt, posCnt), dtype='f4')

    # measure centers
    for c in goodIdx:
        data = np.concatenate((phiFW[c].flatten(), phiRV[c].flatten()))
        x, y, r = circle_fitting(data)
        phiCenter[c] = x + y*(1j)
        phiRadius[c] = r

    # measure phi angles
    for c in goodIdx:
        for k in range(posCnt):
            phiAngFW[c,k] = np.angle(phiFW[c,k] - phiCenter[c])
            phiAngRV[c,k] = np.angle(phiRV[c,k] - phiCenter[c])
        home = phiAngFW[c,0]
        phiAngFW[c] = (phiAngFW[c] - home + np.pi/2) % (np.pi*2) - np.pi/2
        phiAngRV[c] = (phiAngRV[c] - home + np.pi/2) % (np.pi*2) - np.pi/2

    return phiCenter, phiAngFW, phiAngRV, phiRadius

def calcPhiMotorMap(phiCenter, phiAngFW, phiAngRV, regions, steps, goodIdx=None):
    # calculate phi motor maps
    ncobras, cnt = phiAngFW.shape

    if goodIdx is None:
        goodIdx = np.arange(ncobras)

    # HACKS
    binSize = np.deg2rad(3.6)
    delta = np.deg2rad(10)

    phiMMFW = np.zeros((ncobras, regions), dtype=float)
    phiMMRV = np.zeros((ncobras, regions), dtype=float)

    for c in goodIdx:
        for b in range(regions):
            # forward motor maps
            binMin = binSize * b
            binMax = binMin + binSize
            fracSum = 0
            valueSum = 0
            for k in range(cnt-1):
                if phiAngFW[c,k] < binMax and phiAngFW[c,k+1] > binMin and phiAngFW[c,k+1] <= np.pi - delta:
                    moveSizeInBin = np.min([phiAngFW[c,k+1], binMax]) - np.max([phiAngFW[c,k], binMin])
                    entireMoveSize = phiAngFW[c,k+1] - phiAngFW[c,k]
                    fraction = moveSizeInBin * moveSizeInBin / entireMoveSize
                    fracSum += fraction
                    valueSum += fraction * entireMoveSize / steps
            if fracSum > 0:
                phiMMFW[c,b] = valueSum / fracSum
            else:
                phiMMFW[c,b] = phiMMFW[c,b-1]

            # reverse motor maps
            fracSum = 0
            valueSum = 0
            for k in range(cnt-1):
                if phiAngRV[c,k] > binMin and phiAngRV[c,k+1] < binMax and phiAngFW[c,k+1] >= delta:
                    moveSizeInBin = np.min([phiAngRV[c,k], binMax]) - np.max([phiAngRV[c,k+1], binMin])
                    entireMoveSize = phiAngRV[c,k] - phiAngRV[c,k+1]
                    fraction = moveSizeInBin * moveSizeInBin / entireMoveSize
                    fracSum += fraction
                    valueSum += fraction * entireMoveSize / steps
            if fracSum > 0:
                phiMMRV[c,b] = valueSum / fracSum
            else:
                phiMMRV[c,b] = phiMMFW[c,b-1]

    return phiMMFW, phiMMRV

def movePhiToSafeOut(pfi, goodCobras, output,
                     phiRange=5000, bootstrap=False):

    if bootstrap:
        # Be conservative until we tune ontimes: go to 50 not 60 degrees.
        # Also take images for posterity.

        targetAngle = 50.0
        dataset = imageSet.ImageSet(cameraFactory(), output.imageDir,
                                    setName='safeOut', makeStack=True,
                                    saveSpots=True)

        pfi.moveAllSteps(goodCobras, 0, -phiRange)
        dataset.expose(f'phiSafeBegin')

        phis = np.full(len(goodCobras), np.deg2rad(targetAngle))
        pfi.moveThetaPhi(goodCobras, phis*0, phis)
        dataset.expose(f'phiSafeEnd')
    else:
        targetAngle = 60.0
        pfi.moveAllSteps(goodCobras, 0, -phiRange)
        phis = np.full(len(goodCobras), np.deg2rad(targetAngle))
        pfi.moveThetaPhi(goodCobras, phis*0, phis)
