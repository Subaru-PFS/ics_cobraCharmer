import sys, os
from importlib import reload
import numpy as np
import time
import datetime
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import glob
from copy import deepcopy
from ics.cobraCharmer import pfi as pfiControl

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

def getCobras(cobs):
    # cobs is 0-indexed list
    return pfiControl.PFI.allocateCobraList(zip(np.full(len(cobs), 1), np.array(cobs) + 1))


# function to move cobras to target positions
def moveToXYfromHome(pfi, idx, targets, dataPath, threshold=3.0, maxTries=8, cam_split=26):
    cobras = getCobras(idx)
    pfi.moveXYfromHome(cobras, targets)

    ntries = 1
    while True:
        # check current positions, first exposing
        p1 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "1", "-e", "18", "-f", dataPath+"/cam1_"], stdout=PIPE)
        p1.communicate()
        p2 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "2", "-e", "18", "-f", dataPath+"/cam2_"], stdout=PIPE)
        p2.communicate()

        # extract sources and fiber identification
        data1 = fits.getdata(dataPath+'/cam1_0001.fits').astype(float)
        ext1 = sep.extract(data1, 100)
        idx1 = lazyIdentification(pfi.calibModel.centers[idx[idx <= cam_split]], ext1['x'] + ext1['y']*(1j))
        data2 = fits.getdata(dataPath+'/cam2_0001.fits').astype(float)
        ext2 = sep.extract(data2, 100)
        idx2 = lazyIdentification(pfi.calibModel.centers[idx[idx > cam_split]], ext2['x'] + ext2['y']*(1j))
        curPos = np.concatenate((ext1[idx1]['x'] + ext1[idx1]['y']*(1j), ext2[idx2]['x'] + ext2[idx2]['y']*(1j)))
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

def setFiberUDPOS(XML, DataPath):
    # Define the cobra range.
    mod1Cobras = pfiControl.PFI.allocateCobraRange(range(1,2))
    allCobras = mod1Cobras
    oneCobra = pfiControl.PFI.allocateCobraList([(1,2)])
    twoCobras = pfiControl.PFI.allocateCobraList([(1,2), (1,5)])

    # partition module 1 cobras into non-interfering sets
    moduleCobras = {}
    for group in 1,2,3:
        cm = range(group,58,3)
        mod = [1]*len(cm)
        moduleCobras[group] = pfiControl.PFI.allocateCobraList(zip(mod,cm))
    group1Cobras = moduleCobras[1]
    group2Cobras = moduleCobras[2]
    group3Cobras = moduleCobras[3]

    # partition module 1 cobras into odd and even sets
    moduleCobras2 = {}
    for group in 1,2:
        cm = range(group,58,2)
        mod = [1]*len(cm)
        moduleCobras2[group] = pfiControl.PFI.allocateCobraList(zip(mod,cm))
    oddCobras = moduleCobras2[1]
    evenCobras = moduleCobras2[2]

    # Initializing COBRA module
    pfi = pfiControl.PFI(fpgaHost='128.149.77.24') #'fpga' for real device.
    #preciseXML=cobraCharmerPath+'/xml/updateOntime_'+datetoday+'.xml'

    if not os.path.exists(XML):
        print(f"Error: {XML} not presented!")
        sys.exit()
        
    pfi.loadModel(XML)
    pfi.setFreq(allCobras)

    # Calculate up/down(outward) angles
    oddMoves = pfi.thetaToLocal(oddCobras, [np.deg2rad(270)]*len(oddCobras))
    oddMoves[oddMoves>1.85*np.pi] = 0

    evenMoves = pfi.thetaToLocal(evenCobras, [np.deg2rad(90)]*len(evenCobras))
    evenMoves[evenMoves>1.85*np.pi] = 0

    allMoves = np.zeros(57)
    allMoves[::2] = oddMoves
    allMoves[1::2] = evenMoves

    allSteps, _ = pfi.calculateSteps(np.zeros(57), allMoves, np.zeros(57), np.zeros(57))

    # define the broken/good cobras
    brokens = [1, 39, 43, 54]
    visibles= [e for e in range(1,58) if e not in brokens]
    badIdx = np.array(brokens) - 1
    goodIdx = np.array(visibles) - 1

    # two groups for two cameras
    cam_split = 26
    group1 = goodIdx[goodIdx <= cam_split]
    group2 = goodIdx[goodIdx > cam_split]

    # three non-interfering groups for good cobras
    goodGroupIdx = {}
    for group in range(3):
        goodGroupIdx[group] = goodIdx[goodIdx%3==group]

    def getCobras(cobs):
        # cobs is 0-indexed list
        return pfiControl.PFI.allocateCobraList(zip(np.full(len(cobs), 1), np.array(cobs) + 1))

    # Home phi
    pfi.moveAllSteps(allCobras, 0, -5000)

    # Home theta
    pfi.moveAllSteps(allCobras, -10000, 0)

    # Move the bad cobras to up/down positions
    pfi.moveSteps(getCobras(badIdx), allSteps[badIdx], np.zeros(len(brokens)))

    # move visible positioners to outwards positions, phi arms are moved out for 60 degrees
    # (outTargets) otherwise we can't measure the theta angles
    thetas = np.empty(57, dtype=float)
    thetas[::2] = pfi.thetaToLocal(oddCobras, np.full(len(oddCobras), np.deg2rad(270)))
    thetas[1::2] = pfi.thetaToLocal(evenCobras, np.full(len(evenCobras), np.deg2rad(90)))
    phis = np.full(57, np.deg2rad(60.0))
    outTargets = pfi.anglesToPositions(allCobras, thetas, phis)

    # Home the good cobras
    pfi.moveAllSteps(getCobras(goodIdx), -10000, -5000)

    # move to outTargets
    moveToXYfromHome(pfi, goodIdx, outTargets[goodIdx], DataPath)

    # move phi arms in
    pfi.moveAllSteps(getCobras(goodIdx), 0, -5000)


def main():

    cobraCharmerPath='/home/pfs/mhs/devel/ics_cobraCharmer.cwen/'
    #xml=cobraCharmerPath+'/xml/motormaps_181205.xml'
    xml=cobraCharmerPath+'/xml/precise5.xml'

    datetoday=datetime.datetime.now().strftime("%Y%m%d")
    storagePath = '/data/pfs/'+datetoday
    dataPath = storagePath+'/image'
    prodctPath = storagePath+'/product'

    # Prepare the data path for the work
    if not (os.path.exists(storagePath)):
        os.makedirs(storagePath)
    if not (os.path.exists(dataPath)):
        os.makedirs(dataPath)
    if not (os.path.exists(prodctPath)):
        os.makedirs(prodctPath)


    setFiberUDPOS(xml, dataPath)


if __name__ == '__main__':
    main()