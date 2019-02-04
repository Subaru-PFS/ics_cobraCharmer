import sys, os
from importlib import reload
import numpy as np
import time
import datetime
import logging
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import glob
from copy import deepcopy
from ics.cobraCharmer import pfi as pfiControl
from ics.cobraCharmer import imageSet
from ics.cobraCharmer.utils import fileManager
from ics.cobraCharmer.camera import cameraFactory

import utils

logger = logging.getLogger()

def getCobras(module, cobs):
    # cobs is 0-indexed list
    return pfiControl.PFI.allocateCobraList(zip(np.full(len(cobs), module), np.array(cobs) + 1))

# function to move cobras to target positions
def moveToXYfromHome(pfi, cobras, idx, targets, output, threshold=3.0, maxTries=10):
    pfi.moveXYfromHome(cobras, targets)

    dataSet = imageSet.ImageSet(pfi, cameraFactory(), output)
    ntries = 1
    while True:
        im, name = dataSet.expose(f'scratch_{ntries}')

        spots, _ = dataSet.spots(name)
        if len(spots) != len(cobras):
            logging.warn(f'wrong number of spots. expected={len(cobras)}, found={len(spots)}')

        match = utils.lazyIdentification(pfi.calibModel.centers[idx], spots['x'] + spots['y']*(1j))
        curPos = spots[match]['x'] + spots[match]['y']*(1j)
        print(curPos)

        # check position errors
        done = np.abs(curPos - targets) <= threshold
        if np.all(done):
            print('Convergence sequence done')
            break
        if ntries > maxTries:
            print(f'Reach max {maxTries} tries, gave up')
            break
        ntries += 1

        # move again
        pfi.moveXY(cobras, curPos, targets)

def getBrokenCobras(pfi, moduleName='unknown'):
    if moduleName == 'Spare1':
        broken = [1, 39, 43, 54]
    else:
        broken = []

    return broken

def targetThetasIn(pfi, cobras):
    """ Return targets moving theta arms inward.  """

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
               repeat=1,
               steps=50,
               phiRange=5000):

    dataset = imageSet.ImageSet(pfi, cameraFactory(), output, makeStack=True, saveSpots=True)

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

def phiMeasure(pfi, dataSets, phiRange, steps):
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
        idx = utils.lazyIdentification(centers, spots)
        phiFW[:,0,0] = spots[idx]
        for k in range(cnt):
            cs, _ = dataSet.spots(f'phiForward0N{k}')
            spots = np.array([c['x']+c['y']*(1j) for c in cs])
            idx = utils.lazyIdentification(centers, spots)
            phiFW[:,0,k+1] = spots[idx]
        cs, _ = dataSet.spots(f'phiForward0End')
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = utils.lazyIdentification(centers, spots)
        phiFW[:,0,k+2] = spots[idx]

    for ds_i, dataSet in enumerate(dataSets):
        n = ds_i + 1

        cs, _ = dataSet.spots(f'phiReverse0Begin')
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = utils.lazyIdentification(centers, spots)
        phiRV[:,0,0] = spots[idx]
        for k in range(cnt):
            cs, _ = dataSet.spots(f'phiReverse0N{k}')
            spots = np.array([c['x']+c['y']*(1j) for c in cs])
            idx = utils.lazyIdentification(centers, spots)
            phiRV[:,0,k+1] = spots[idx]
        cs, _  = dataSet.spots(f'phiReverse0End')
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = utils.lazyIdentification(centers, spots)
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
        x, y, r = utils.circle_fitting(data)
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

def savePhiGeometry(pfi, goodIdx, output, phiCircles, phiFW, phiRV):
    if goodIdx is None:
        goodIdx = len(phiCircles)

    # Calculate hard stops
    phiC = phiCircles[:,0] + phiCircles[:,1]*(1j)
    points = np.zeros((57, 4), dtype=complex)

    # phi hard stops
    phiCCW = (np.angle(points[:,0] - phiC) - np.angle(thetaC - phiC) + (np.pi/2)) % (2*np.pi) - (np.pi/2)
    phiCW = (np.angle(points[:,1] - phiC) - np.angle(thetaC - phiC) + (np.pi/2)) % (2*np.pi) - (np.pi/2)


def XXXXXstrangePhiGooFromPreciseNB():
    myIdx = goodIdx
    homes = pfi.calibModel.centers[myIdx]

    # theta CCW hard stops
    #thetaCCW = np.angle(phiC - thetaC) % (2*np.pi)
    a = np.absolute(points[:,2] - thetaC)
    b = np.absolute(thetaC - phiC)
    c = phiCircles[:,2]
    for k in range(57):
        if a[k]*b[k] != 0:
            thetaCCW[k] = (np.angle(points[k,2] - thetaC[k]) + np.arccos((a[k]*a[k] + b[k]*b[k] - c[k]*c[k]) / (2*a[k]*b[k]))) % (2*np.pi)
        else:
            thetaCCW[k] = 0
    # thetaCW hard stops
    thetaCW = (np.angle(points[:,3] - thetaC) - np.angle(points[:,2] - thetaC) + thetaCCW) % (2*np.pi)
    # save new configuration
    old = pfi.calibModel
    myConfig = deepcopy(old)

    # you can remove bad measurement here
    #idx = np.delete(goodIdx, np.argwhere(goodIdx==12))
    idx = goodIdx

    myConfig.centers[idx] = thetaC[idx]
    diff = np.absolute(thetaC - phiC)
    myConfig.L1[idx] = diff[idx]
    myConfig.L2[idx] = phiCircles[idx,2]

    myConfig.tht0[idx] = thetaCCW[idx]
    myConfig.tht1[idx] = thetaCW[idx]
    myConfig.phiIn[idx] = phiCCW[idx] - np.pi
    myConfig.phiOut[idx] = phiCW[idx] - np.pi

    old.updateGeometry(myConfig.centers, myConfig.L1, myConfig.L2)
    old.updateThetaHardStops(myConfig.tht0, myConfig.tht1)
    old.updatePhiHardStops(myConfig.phiIn + np.pi, myConfig.phiOut + np.pi)

    old.createCalibrationFile('../xml/precise5.xml')

def makeMotorMap(pfi, output, modules=None,
                 repeat=1,
                 steps=50,
                 phiRange=5000,
                 thetaRange=10000,
                 bootstrap=False,
                 reprocess=False,
                 updateModel=True):

    # We need to be more flexible....
    if modules is None:
        modules = [1]
    if len(modules) != 1 and modules[0] != 1:
        raise ValueError('sorry, lazy programmer cannot map several modules')

    module = modules[0]

    # Define the cobra range.
    allCobras = []
    for m in modules:
        allCobras.extend(pfiControl.PFI.allocateCobraModule(module))

    # CRAP: Move and flesh out. At least provide a commandline list.
    brokens = getBrokenCobras(pfi, module)

    # define the broken/good cobras
    visibles= [e for e in range(1,58) if e not in brokens]
    goodCobras = [c for c in allCobras if c.cobra not in brokens]
    badIdx = np.array(brokens) - 1
    goodIdx = np.array(visibles) - 1

    onTime = deepcopy([pfi.calibModel.motorOntimeFwd1,
                   pfi.calibModel.motorOntimeRev1,
                   pfi.calibModel.motorOntimeFwd2,
                   pfi.calibModel.motorOntimeRev2])

    fastOnTime = [np.full(57, 0.09)] * 4
    ontimes = dict(fast=fastOnTime, normal=onTime)

    # HACKS: use the same weighting as Johannes to calculate motor maps,
    #        plus some suspect definitions.
    binSize = np.deg2rad(3.6)
    regions = 112

    # Step 1: measure cobra centers at (0,0).
    # Step 2: take a phi motor map. (lets us get to 60 degress safely)
    # Step 3: move phi=60, take theta motor map.

    if reprocess:
        dataset = imageSet.ImageSet(pfi, camera=None, output=output)
        phiFW, phiRV = phiMeasure(pfi, [dataset], phiRange, steps=steps)
    else:
        pfi.reset()
        pfi.setFreq()

        if bootstrap:
            # Do __NOT__ home or move theta for bootstrapping.
            pfi.moveAllSteps(goodCobras, 0, -phiRange)
        else:
            # Home the good cobras
            pfi.moveAllSteps(goodCobras, -thetaRange, -phiRange)

            targets = targetThetasIn(pfi, goodCobras)
            moveToXYfromHome(pfi, goodCobras, goodIdx, targets, output)

        phiDataset = takePhiMap(pfi, output, goodCobras,
                                steps=steps, phiRange=phiRRange) # ontimes=ontimes
        phiFW, phiRV = phiMeasure(pfi, [phiDataset], phiRange=phiRange,, steps=steps)

    phiCenter, phiAngFW, phiAngRV = calcPhiGeometry(pfi, phiFW, phiRV, phiRange, steps, goodIdx=goodIdx)
    phiMMFW, phiMMRV = calcPhiMotorMap(pfi, phiCenter, phiAngFW, phiAngRV, regions, steps, goodIdx=None)

    if bootstrap:
        np.seterr(divide='raise')
        model = pfi.calibModel
        model.updateMotorMaps(phiFwd=phiMMFW, phiRev=phiMMRV)
        model.updateMotorMaps(phiFwd=phiMMFW, phiRev=phiMMRV, useSlowMaps=False)

        xmlPath = os.path.join(output.xmlDir, 'phiMM.xml')
        model.createCalibrationFile(xmlPath)

        pfi.loadModel(xmlPath)

        # Be conservative until we tune ontimes: go to 50 not 60 degrees.
        phis = np.full(len(goodCobras), np.deg2rad(50.0))
        pfi.moveThetaPhi(goodCobras, phis*0, phis)

        # No!! This can enable and execute (big!) theta moves!!!
        # moveToXYfromHome(pfi, goodCobras, goodIdx, targets, output)

    breakpoint()

    ## Everything below this breakpoint is unevaluated, but stripped for the stuff above.

    oldCenters = pfi.calibModel.centers[goodIdx]

    allMoves = targetThetasIn(pfi, module)
    allSteps, _ = pfi.calculateSteps(np.zeros(57), allMoves, np.zeros(57), np.zeros(57))

    # move visible positioners to outwards positions, phi arms are moved out for 60 degrees
    # (outTargets) otherwise we can't measure the theta angles
    thetas = np.empty(57, dtype=float)
    thetas[::2] = pfi.thetaToLocal(oddCobras, np.full(len(oddCobras), np.deg2rad(270)))
    thetas[1::2] = pfi.thetaToLocal(evenCobras, np.full(len(evenCobras), np.deg2rad(90)))
    phis = np.full(57, np.deg2rad(60.0))
    outTargets = pfi.anglesToPositions(allCobras, thetas, phis)

    # Home the good cobras
    pfi.moveAllSteps(goodCobras, -10000, -5000)
    pfi.moveAllSteps(goodCobras, -5000, -5000)

    # move to outTargets
    moveToXYfromHome(pfi, goodCobras, goodIdx, outTargets[goodIdx], dataPath)

    # move phi arms in
    pfi.moveAllSteps(goodCobras, 0, -5000)


    # move phi arms out for 60 degrees then home theta
    pfi.moveAllSteps(goodCobras, -10000, -5000)
    pfi.moveAllSteps(goodCobras, -5000, -5000)
    moveToXYfromHome(pfi, goodCobras, goodIdx, outTargets[goodIdx], dataPath)
    pfi.moveAllSteps(goodCobras, -10000, 0)
    pfi.moveAllSteps(goodCobras, -5000, 0)

    # record the theta movements
    for n in range(repeat):
        # forward theta motor maps
        expose(dataPath+f'/theta1Begin{n}_', dataPath+f'/theta2Begin{n}_')
        for k in range(thetaRange//steps):
            pfi.moveAllSteps(goodCobras, steps, 0)
            expose(dataPath+f'/theta1Forward{n}N{k}_', dataPath+f'/theta2Forward{n}N{k}_')

        # make sure it goes to the limit
        pfi.calibModel.updateOntimes(*fastOnTime)
        pfi.moveAllSteps(goodCobras, 10000, 0)
        pfi.calibModel.updateOntimes(*onTime)

        # reverse theta motor maps
        expose(dataPath+f'/theta1End{n}_', dataPath+f'/theta2End{n}_')
        for k in range(thetaRange//steps):
            pfi.moveAllSteps(goodCobras, -steps, 0)
            expose(dataPath+f'/theta1Reverse{n}N{k}_', dataPath+f'/theta2Reverse{n}N{k}_')

        # make sure it goes to the limit
        pfi.calibModel.updateOntimes(*fastOnTime)
        pfi.moveAllSteps(goodCobras, -10000, 0)
        pfi.calibModel.updateOntimes(*onTime)

    # variable declaration for position measurement
    thetaFW = np.zeros((57, repeat, thetaRange//steps+1), dtype=complex)
    thetaRV = np.zeros((57, repeat, thetaRange//steps+1), dtype=complex)

    # forward theta
    cnt = thetaRange//steps
    for n in range(repeat):
        data = fits.getdata(dataPath+f'/theta{nCam}Begin{n}_0001.fits')
        cs = sep.extract(data.astype(float), 50)
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = utils.lazyIdentification(centers, spots)
        thetaFW[myIdx,n,0] = spots[idx]
        stack_image = data
        for k in range(cnt):
            data = fits.getdata(dataPath+f'/theta{nCam}Forward{n}N{k}_0001.fits')
            cs = sep.extract(data.astype(float), 50)
            spots = np.array([c['x']+c['y']*(1j) for c in cs])
            idx = utils.lazyIdentification(centers, spots)
            thetaFW[myIdx,n,k+1] = spots[idx]
            stack_image = stack_image + data
        fits.writeto(prodctPath+f'/Cam{nCam}thetaForwardStack.fits',stack_image,overwrite=True)


    # reverse theta
    for n in range(repeat):
        data = fits.getdata(dataPath+f'/theta{nCam}End{n}_0001.fits')
        cs = sep.extract(data.astype(float), 50)
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = utils.lazyIdentification(centers, spots)
        thetaRV[myIdx,n,0] = spots[idx]
        stack_image = data
        for k in range(cnt):
            data = fits.getdata(dataPath+f'/theta{nCam}Reverse{n}N{k}_0001.fits')
            cs = sep.extract(data.astype(float), 50)
            spots = np.array([c['x']+c['y']*(1j) for c in cs])
            idx = utils.lazyIdentification(centers, spots)
            thetaRV[myIdx,n,k+1] = spots[idx]
            stack_image = stack_image + data
        fits.writeto(prodctPath+f'/Cam{nCam}thetaReverseStack.fits',stack_image,overwrite=True)



    # variable declaration for theta, phi angles
    thetaCenter = np.zeros(57, dtype=complex)
    thetaAngFW = np.zeros((57, repeat, thetaRange//steps+1), dtype=float)
    thetaAngRV = np.zeros((57, repeat, thetaRange//steps+1), dtype=float)

    # measure centers
    for c in goodIdx:
        data = np.concatenate((thetaFW[c].flatten(), thetaRV[c].flatten()))
        x, y, r = utils.circle_fitting(data)
        thetaCenter[c] = x + y*(1j)
        #x data = np.concatenate((phiFW[c].flatten(), phiRV[c].flatten()))
        #x x, y, r = utils.circle_fitting(data)
        #x phiCenter[c] = x + y*(1j)

    # measure theta angles
    cnt = thetaRange//steps
    for c in goodIdx:
        for n in range(repeat):
            for k in range(cnt+1):
                thetaAngFW[c,n,k] = np.angle(thetaFW[c,n,k] - thetaCenter[c])
                thetaAngRV[c,n,k] = np.angle(thetaRV[c,n,k] - thetaCenter[c])
            home = thetaAngFW[c,n,0]
            thetaAngFW[c,n] = (thetaAngFW[c,n] - home) % (np.pi*2)
            thetaAngRV[c,n] = (thetaAngRV[c,n] - home) % (np.pi*2)

    # fix over 2*pi angle issue
    for c in goodIdx:
        for n in range(repeat):
            for k in range(cnt):
                if thetaAngFW[c,n,k+1] < thetaAngFW[c,n,k]:
                    thetaAngFW[c,n,k+1] += np.pi*2
            for k in range(cnt):
                if thetaAngRV[c,n,k+1] > thetaAngRV[c,n,k]:
                    thetaAngRV[c,n,k] += np.pi*2
                else:
                    break
            for k in range(cnt):
                if thetaAngRV[c,n,k+1] > thetaAngRV[c,n,k]:
                    thetaAngRV[c,n,k+1] -= np.pi*2

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

    # use the same weighting as Johannes to calculate motor maps
    binSize = np.deg2rad(3.6)
    regions = 112

    thetaMMFW = np.zeros((57, regions), dtype=float)
    thetaMMRV = np.zeros((57, regions), dtype=float)
    phiMMFW = np.zeros((57, regions), dtype=float)
    phiMMRV = np.zeros((57, regions), dtype=float)

    delta = np.deg2rad(10)
    thetaHS = np.deg2rad(370)

    # calculate theta motor maps
    cnt = thetaRange//steps
    for c in goodIdx:
        for b in range(regions):
            # forward motor maps
            binMin = binSize * b
            binMax = binMin + binSize
            fracSum = 0
            valueSum = 0
            for n in range(repeat):
                for k in range(cnt):
                    if thetaAngFW[c,n,k] < binMax and thetaAngFW[c,n,k+1] > binMin and thetaAngFW[c,n,k+1] <= thetaHS:
                        moveSizeInBin = np.min([thetaAngFW[c,n,k+1], binMax]) - np.max([thetaAngFW[c,n,k], binMin])
                        entireMoveSize = thetaAngFW[c,n,k+1] - thetaAngFW[c,n,k]
                        fraction = moveSizeInBin * moveSizeInBin / entireMoveSize
                        fracSum += fraction
                        valueSum += fraction * entireMoveSize / steps
            if fracSum > 0:
                thetaMMFW[c,b] = valueSum / fracSum
            else:
                thetaMMFW[c,b] = thetaMMFW[c,b-1]

            # reverse motor maps
            fracSum = 0
            valueSum = 0
            for n in range(repeat):
                for k in range(cnt):
                    if thetaAngRV[c,n,k] > binMin and thetaAngRV[c,n,k+1] < binMax and thetaAngFW[c,n,k+1] >= delta:
                        moveSizeInBin = np.min([thetaAngRV[c,n,k], binMax]) - np.max([thetaAngRV[c,n,k+1], binMin])
                        entireMoveSize = thetaAngRV[c,n,k] - thetaAngRV[c,n,k+1]
                        fraction = moveSizeInBin * moveSizeInBin / entireMoveSize
                        fracSum += fraction
                        valueSum += fraction * entireMoveSize / steps
            if fracSum > 0:
                thetaMMRV[c,b] = valueSum / fracSum
            else:
                thetaMMRV[c,b] = thetaMMFW[c,b-1]

    # save new configuration for both slow nad fast motor maps
    old = pfi.calibModel

    sThetaFW = binSize / old.S1Pm
    sThetaRV = binSize / old.S1Nm
    fThetaFW = binSize / old.F1Pm
    fThetaRV = binSize / old.F1Nm
    sPhiFW = binSize / old.S2Pm
    sPhiRV = binSize / old.S2Nm
    fPhiFW = binSize / old.F2Pm
    fPhiRV = binSize / old.F2Nm

    # you can remove bad measurement here
    #idx = np.delete(goodIdx, np.argwhere(goodIdx==11))
    idx = goodIdx

    sThetaFW[idx] = thetaMMFW[idx]
    sThetaRV[idx] = thetaMMRV[idx]
    fThetaFW[idx] = thetaMMFW[idx]
    fThetaRV[idx] = thetaMMRV[idx]
    sPhiFW[idx] = phiMMFW[idx]
    sPhiRV[idx] = phiMMRV[idx]
    fPhiFW[idx] = phiMMFW[idx]
    fPhiRV[idx] = phiMMRV[idx]

    # update configuration
    old.updateMotorMaps(sThetaFW, sThetaRV, sPhiFW, sPhiRV, useSlowMaps=True)
    old.updateMotorMaps(fThetaFW, fThetaRV, fPhiFW, fPhiRV, useSlowMaps=False)

    # write to a new XML file
    #old.createCalibrationFile('../xml/motormaps.xml')
    old.createCalibrationFile(outputXML)

    print(f'{outputXML}  produced!')
    print("Process Finised")

def main(args=None):
    if isinstance(args, str):
        import shlex
        args = shlex.split(args)

    import argparse

    parser = argparse.ArgumentParser('makeMotorMap', add_help=True)
    parser.add_argument('moduleName', type=str,
                        help='the name of the module (e.g. "SC03", "Spare1", or "PFI")')

    parser.add_argument('--steps', type=int, default=50,
                        help='size of step to take for the motormaps')
    parser.add_argument('--phiRange', type=int, default=5000,
                        help='expected full range of phi motors')
    parser.add_argument('--thetaRange', type=int, default=10000,
                        help='expected full range of theta motors')
    parser.add_argument('--fpgaHost', type=str, default='localhost',
                        help='connect to the given FPGA host instead of the simulator.')
    parser.add_argument('--modelName', type=str, default=None,
                        help='load the given PFI model before calibrating.')
    parser.add_argument('--module', type=int, default=0,
                        help='calibrate the given module. Or all.')
    parser.add_argument('--saveModelFile', type=str, default='',
                        help='save the updated model in the given file.')
    parser.add_argument('--bootstrap', action='store_true',
                        help='Assume that the input XML file has unknown/bad geometry.')
    parser.add_argument('--reprocess', type=str, default=False,
                        help='do not acquire data, but re-process existing data in this directory.')

    opts = parser.parse_args(args)

    if opts.reprocess:
        cam = None
        output = fileManager.ProcedureDirectory.loadFromPath(opts.reprocess)
    else:
        cam = cameraFactory('cit')
        output = fileManager.ProcedureDirectory(opts.moduleName, experimentName='map')

    pfi = pfiControl.PFI(fpgaHost=opts.fpgaHost, logDir=output.logDir,
                         doLoadModel=False)
    pfi.loadModel(opts.modelName)

    pfi = makeMotorMap(pfi, output,
                       modules=[opts.module] if opts.module != 0 else None,
                       steps=opts.steps,
                       phiRange=opts.phiRange,
                       thetaRange=opts.thetaRange,
                       bootstrap=opts.bootstrap,
                       reprocess=opts.reprocess)

    if opts.saveModelFile:
        pfi.calibModel.createCalibrationFile(opts.saveModelFile)

if __name__ == "__main__":
    main()

def xxmain():
    datetoday=datetime.datetime.now().strftime("%Y%m%d")
    cobraCharmerPath='/home/pfs/mhs/devel/ics_cobraCharmer.cwen/'

    for steps in [50,100,200,400]:
        storagePath = '/data/pfs/'+datetoday+f'_step{steps}/'
        outputXML = cobraCharmerPath+'/xml/motormap_'+datetoday+f'_step{steps}.xml'
        runMotorMap(3, steps, storagePath, outputXML)


if __name__ == '__main__':
    main()
