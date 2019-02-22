from importlib import reload

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

def takePhiMap(pfi, output,
               cobras,
               ontimes=None,
               setName=None,
               repeat=1,
               steps=50,
               phiRange=5000):

    dataset = imageSet.ImageSet(pfi, cameraFactory(), output, setName=setName,
                                makeStack=True, saveSpots=True)

    # record the phi movements
    for n in range(repeat):
        # forward phi motor maps
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
                 trackCenters=True, sigma=5.0):
    if names is None:
        names = dataSet.namelist.keys()
    if len(positions) != len(names):
        raise ValueError('number of positions (%d) must match the number of names (%d)' %
                         (len(positions), len(names)))

    nCobras = len(centers)
    res = np.zeros(len(positions),
                   dtype=dict(names=('name', 'pos', 'fiberIds', 'centers'),
                              formats=(np.object, np.float32, f'{nCobras}u2', f'{nCobras}c8')))

    # We _start_ by matching the measured centers against the provided centers. If
    # the trackCenters flag is set, we then match to the previous position of the cobra.
    nearestCenters = centers
    for i in range(len(positions)):
        res[i]['name'] = names[i]
        res[i]['pos'] = positions[i]
        cs, _ = dataSet.spots(names[i], sigma=sigma, disp=disp)

        if len(cs) != len(centers):
            raise RuntimeError('in %s, number of spots (%d) != number of cobras (%d)' %
                               (names[i], len(cs), len(centers)))

        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = lazyIdentification(nearestCenters, spots, radii=20.0)
        if trackCenters:
            nearestCenters = spots[idx]

        res[i]['centers'][:] = spots[idx]
        res[i]['fiberIds'][:] = idx

    return res

def phiMeasure(pfiModel, dataSet, phiRange, steps):
    """
    Given bootstrap phi data, pile up the measurements.

    This is bad: it should not know or care about how many samples are
    available. It should simply read them all.
    """

    phiFW = np.zeros((57, 1, phiRange//steps+2), dtype=complex)
    phiRV = np.zeros((57, 1, phiRange//steps+2), dtype=complex)
    centers = pfi.calibModel.centers

    # forward phi
    cnt = phiRange//steps
    for ds_i, dataSet in enumerate(dataSets):
        n = ds_i + 1

        cs, _ = dataSet.spots(f'phiForward0Begin')
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = lazyIdentification(centers, spots)
        phiFW[:,0,0] = spots[idx]
        for k in range(cnt):
            cs, _ = dataSet.spots(f'phiForward0N{k}')
            spots = np.array([c['x']+c['y']*(1j) for c in cs])
            idx = lazyIdentification(centers, spots)
            phiFW[:,0,k+1] = spots[idx]
        cs, _ = dataSet.spots(f'phiForward0End')
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = lazyIdentification(centers, spots)
        phiFW[:,0,k+2] = spots[idx]

    for ds_i, dataSet in enumerate(dataSets):
        n = ds_i + 1

        cs, _ = dataSet.spots(f'phiReverse0Begin')
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = lazyIdentification(centers, spots)
        phiRV[:,0,0] = spots[idx]
        for k in range(cnt):
            cs, _ = dataSet.spots(f'phiReverse0N{k}')
            spots = np.array([c['x']+c['y']*(1j) for c in cs])
            idx = lazyIdentification(centers, spots)
            phiRV[:,0,k+1] = spots[idx]
        cs, _  = dataSet.spots(f'phiReverse0End')
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = lazyIdentification(centers, spots)
        phiRV[:,0,k+2] = spots[idx]

    return phiFW, phiRV

def calcPhiGeometry(pfi, phiFW, phiRV, phiRange, steps, goodIdx=None):
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

    if goodIdx is None:
        goodIdx = np.arange(57)

    repeat = phiFW.shape[1]

    phiCenter = np.zeros(57, dtype=complex)
    phiAngFW = np.zeros((57, repeat, phiRange//steps+1), dtype=float)
    phiAngRV = np.zeros((57, repeat, phiRange//steps+1), dtype=float)

    # measure centers
    for c in goodIdx:
        data = np.concatenate((phiFW[c].flatten(), phiRV[c].flatten()))
        x, y, r = circle_fitting(data)
        phiCenter[c] = x + y*(1j)

    # measure phi angles
    cnt = phiRange//steps + 1
    for c in goodIdx:
        for n in range(repeat):
            for k in range(cnt):
                phiAngFW[c,n,k] = np.angle(phiFW[c,n,k] - phiCenter[c])
                phiAngRV[c,n,k] = np.angle(phiRV[c,n,k] - phiCenter[c])
            home = phiAngFW[c,n,0]
            phiAngFW[c,n] = (phiAngFW[c,n] - home + np.pi/2) % (np.pi*2) - np.pi/2
            phiAngRV[c,n] = (phiAngRV[c,n] - home + np.pi/2) % (np.pi*2) - np.pi/2

    return phiCenter, phiAngFW, phiAngRV

def calcPhiMotorMap(pfi, phiCenter, phiAngFW, phiAngRV, regions, steps, goodIdx=None):
    if goodIdx is None:
        goodIdx = np.arange(57)

    # calculate phi motor maps
    ncobras, repeat, cnt = phiAngFW.shape

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
            for n in range(repeat):
                for k in range(cnt-1):
                    if phiAngFW[c,n,k] < binMax and phiAngFW[c,n,k+1] > binMin and phiAngFW[c,n,k+1] <= np.pi - delta:
                        moveSizeInBin = np.min([phiAngFW[c,n,k+1], binMax]) - np.max([phiAngFW[c,n,k], binMin])
                        entireMoveSize = phiAngFW[c,n,k+1] - phiAngFW[c,n,k]
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
            for n in range(repeat):
                for k in range(cnt-1):
                    if phiAngRV[c,n,k] > binMin and phiAngRV[c,n,k+1] < binMax and phiAngFW[c,n,k+1] >= delta:
                        moveSizeInBin = np.min([phiAngRV[c,n,k], binMax]) - np.max([phiAngRV[c,n,k+1], binMin])
                        entireMoveSize = phiAngRV[c,n,k] - phiAngRV[c,n,k+1]
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
        # Also step out in parts and back and record images.

        targetAngle = 50.0
        dataset = imageSet.ImageSet(pfi, cameraFactory(), output,
                                    setName='safeOut', makeStack=True,
                                    saveSpots=True)

        pfi.moveAllSteps(goodCobras, 0, -phiRange)

        dataset.expose(f'phiSafeBegin')
        for s in range(1,3):
            ang = targetAngle//s
            phis = np.full(len(goodCobras), np.deg2rad(ang))
            pfi.moveThetaPhi(goodCobras, phis*0, phis)
            dataset.expose(f'phiSafe{ang}')

        pfi.moveAllSteps(goodCobras, 0, -phiRange)
        phis = np.full(len(goodCobras), np.deg2rad(targetAngle))
        pfi.moveThetaPhi(goodCobras, phis*0, phis)
        dataset.expose(f'phiSafeEnd')
    else:
        targetAngle = 60.0
        pfi.moveAllSteps(goodCobras, 0, -phiRange)
        phis = np.full(len(goodCobras), np.deg2rad(targetAngle))
        pfi.moveThetaPhi(goodCobras, phis*0, phis)
