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
from copy import deepcopy

def expose(fn1, fn2):
    p1 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "1", "-e", "18", "-l", "3", "-f", fn1], stdout=PIPE)
    p1.communicate()
    p2 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "2", "-e", "18", "-l", "3", "-f", fn2], stdout=PIPE)
    p2.communicate()

def moveCobra(c, theta, phi):
    pfi.moveSteps([allCobras[c-1]], np.zeros(1)+theta, np.zeros(1)+phi)

def moveCobras(cs, theta, phi):
    cobs = []
    for c in cs:
        cobs.append(allCobras[c-1])
    pfi.moveSteps(cobs, np.array(theta), np.array(phi))

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

# function to move cobras to target positions
def moveToXYfromHome(idx, targets, dataPath, threshold=3.0, maxTries=12, cam_split=26):
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


datetoday=datetime.datetime.now().strftime("%Y%m%d")
#datetoday='20181219'
cobraCharmerPath='/home/pfs/mhs/devel/ics_cobraCharmer/'
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
preciseXML=cobraCharmerPath+'/xml/motormaps_181205.xml'
#preciseXML=cobraCharmerPath+'/xml/updateOntime_'+datetoday+'.xml'

if not os.path.exists(preciseXML):
    print(f"Error: {preciseXML} not presented!")
    sys.exit()
    
pfi.loadModel(preciseXML)
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
pfi.moveAllSteps(allCobras, -5000, 0)

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
pfi.moveAllSteps(getCobras(goodIdx), -5000, -5000)

# move to outTargets
moveToXYfromHome(goodIdx, outTargets[goodIdx], dataPath)

# move phi arms in
pfi.moveAllSteps(getCobras(goodIdx), 0, -5000)

# parameters declared here
repeat = 3
steps = 50
thetaSteps = 10000
phiSteps = 5000
myCobras = getCobras(goodIdx)


# Loading XML file once again.  Changing XML file name if you want to loading 
#  special setting.
ontimeXML=cobraCharmerPath+'/xml/updateOntime_20190124n.xml'
pfi.loadModel(ontimeXML)
pfi.setFreq(allCobras)

OnTime = deepcopy([pfi.calibModel.motorOntimeFwd1,
                   pfi.calibModel.motorOntimeRev1,
                   pfi.calibModel.motorOntimeFwd2,
                   pfi.calibModel.motorOntimeRev2])

fastOnTime = [np.full(57, 0.09)] * 4


#record the phi movements
for n in range(repeat):
    # forward phi motor maps
    expose(dataPath+f'/phi1Begin{n}_', dataPath+f'/phi2Begin{n}_')
    for k in range(phiSteps//steps):
        pfi.moveAllSteps(myCobras, 0, steps)
        expose(dataPath+f'/phi1Forward{n}N{k}_', dataPath+f'/phi2Forward{n}N{k}_')
    
    # make sure it goes to the limit
    pfi.calibModel.updateOntimes(*fastOnTime)
    pfi.moveAllSteps(myCobras, 0, 5000)
    pfi.calibModel.updateOntimes(*OnTime)

    # reverse phi motor maps
    expose(dataPath+f'/phi1End{n}_', dataPath+f'/phi2End{n}_')
    for k in range(phiSteps//steps):
        pfi.moveAllSteps(myCobras, 0, -steps)
        expose(dataPath+f'/phi1Reverse{n}N{k}_', dataPath+f'/phi2Reverse{n}N{k}_')

    # At the end, make sure the cobra back to the hard stop
    pfi.calibModel.updateOntimes(*fastOnTime)
    pfi.moveAllSteps(myCobras, 0, -5000)
    pfi.calibModel.updateOntimes(*OnTime)



# After the loop, set back to default XML for repositioning the fiber
pfi.loadModel(preciseXML)
pfi.setFreq(allCobras)


# move phi arms out for 60 degrees then home theta
pfi.moveAllSteps(myCobras, -10000, -5000)
pfi.moveAllSteps(myCobras, -5000, 0)
moveToXYfromHome(goodIdx, outTargets[goodIdx], dataPath)
pfi.moveAllSteps(myCobras, -10000, 0)
pfi.moveAllSteps(myCobras, -5000, 0)

# Loading XML file once again.  Changing XML file name if you want to loading 
#  special setting.
pfi.loadModel(ontimeXML)
pfi.setFreq(allCobras)


# record the theta movements
for n in range(repeat):
    # forward theta motor maps
    expose(dataPath+f'/theta1Begin{n}_', dataPath+f'/theta2Begin{n}_')
    for k in range(thetaSteps//steps):
        pfi.moveAllSteps(myCobras, steps, 0)
        expose(dataPath+f'/theta1Forward{n}N{k}_', dataPath+f'/theta2Forward{n}N{k}_')
    
    # make sure it goes to the limit
    pfi.calibModel.updateOntimes(*fastOnTime)
    pfi.moveAllSteps(myCobras, 10000, 0)
    pfi.calibModel.updateOntimes(*OnTime)
 
    # reverse theta motor maps
    expose(dataPath+f'/theta1End{n}_', dataPath+f'/theta2End{n}_')
    for k in range(thetaSteps//steps):
        pfi.moveAllSteps(myCobras, -steps, 0)
        expose(dataPath+f'/theta1Reverse{n}N{k}_', dataPath+f'/theta2Reverse{n}N{k}_')

    # make sure it goes to the limit
    pfi.calibModel.updateOntimes(*fastOnTime)
    pfi.moveAllSteps(myCobras, -10000, 0)
    pfi.calibModel.updateOntimes(*OnTime)

# variable declaration for position measurement
thetaFW = np.zeros((57, repeat, thetaSteps//steps+1), dtype=complex)
thetaRV = np.zeros((57, repeat, thetaSteps//steps+1), dtype=complex)
phiFW = np.zeros((57, repeat, phiSteps//steps+1), dtype=complex)
phiRV = np.zeros((57, repeat, phiSteps//steps+1), dtype=complex)

# first camera

# phi stages
for nCam in [1,2]:
    if (nCam == 1): myIdx = goodIdx[goodIdx <= cam_split]
    if (nCam == 2): myIdx = goodIdx[goodIdx > cam_split]
    centers = pfi.calibModel.centers[myIdx]

    # forward phi
    cnt = phiSteps//steps
    for n in range(repeat):
        data = fits.getdata(dataPath+f'/phi{nCam}Begin{n}_0001.fits')
        cs = sep.extract(data.astype(float), 50)
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = lazyIdentification(centers, spots)
        phiFW[myIdx,n,0] = spots[idx]
        stack_image = data   
        for k in range(cnt):
            data = fits.getdata(dataPath+f'/phi{nCam}Forward{n}N{k}_0001.fits')
            cs = sep.extract(data.astype(float), 50)
            spots = np.array([c['x']+c['y']*(1j) for c in cs])
            idx = lazyIdentification(centers, spots)
            phiFW[myIdx,n,k+1] = spots[idx]
            stack_image = stack_image + data
        fits.writeto(prodctPath+f'/Cam{nCam}phiForwardStack.fits',stack_image,overwrite=True)

    # reverse phi
    for n in range(repeat):
        data = fits.getdata(dataPath+f'/phi{nCam}End{n}_0001.fits')
        cs = sep.extract(data.astype(float), 50)
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = lazyIdentification(centers, spots)
        phiRV[myIdx,n,0] = spots[idx]
        stack_image = data   
        for k in range(cnt):
            data = fits.getdata(dataPath+f'/phi{nCam}Reverse{n}N{k}_0001.fits')
            cs = sep.extract(data.astype(float), 50)
            spots = np.array([c['x']+c['y']*(1j) for c in cs])
            idx = lazyIdentification(centers, spots)
            phiRV[myIdx,n,k+1] = spots[idx]
            stack_image = stack_image + data
        fits.writeto(prodctPath+f'/Cam{nCam}phiReverseStack.fits',stack_image,overwrite=True)

    # forward theta
    cnt = thetaSteps//steps
    for n in range(repeat):
        data = fits.getdata(dataPath+f'/theta{nCam}Begin{n}_0001.fits')
        cs = sep.extract(data.astype(float), 50)
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = lazyIdentification(centers, spots)
        thetaFW[myIdx,n,0] = spots[idx]
        stack_image = data   
        for k in range(cnt):
            data = fits.getdata(dataPath+f'/theta{nCam}Forward{n}N{k}_0001.fits')
            cs = sep.extract(data.astype(float), 50)
            spots = np.array([c['x']+c['y']*(1j) for c in cs])
            idx = lazyIdentification(centers, spots)
            thetaFW[myIdx,n,k+1] = spots[idx]
            stack_image = stack_image + data
        fits.writeto(prodctPath+f'/Cam{nCam}thetaForwardStack.fits',stack_image,overwrite=True)


    # reverse theta
    for n in range(repeat):
        data = fits.getdata(dataPath+f'/theta{nCam}End{n}_0001.fits')
        cs = sep.extract(data.astype(float), 50)
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = lazyIdentification(centers, spots)
        thetaRV[myIdx,n,0] = spots[idx]
        stack_image = data    
        for k in range(cnt):
            data = fits.getdata(dataPath+f'/theta{nCam}Reverse{n}N{k}_0001.fits')
            cs = sep.extract(data.astype(float), 50)
            spots = np.array([c['x']+c['y']*(1j) for c in cs])
            idx = lazyIdentification(centers, spots)
            thetaRV[myIdx,n,k+1] = spots[idx]
            stack_image = stack_image + data
        fits.writeto(prodctPath+f'/Cam{nCam}thetaReverseStack.fits',stack_image,overwrite=True)



# variable declaration for theta, phi angles
thetaCenter = np.zeros(57, dtype=complex)
phiCenter = np.zeros(57, dtype=complex)
thetaAngFW = np.zeros((57, repeat, thetaSteps//steps+1), dtype=float)
thetaAngRV = np.zeros((57, repeat, thetaSteps//steps+1), dtype=float)
phiAngFW = np.zeros((57, repeat, phiSteps//steps+1), dtype=float)
phiAngRV = np.zeros((57, repeat, phiSteps//steps+1), dtype=float)

# measure centers
for c in goodIdx:
    data = np.concatenate((thetaFW[c].flatten(), thetaRV[c].flatten()))
    x, y, r = circle_fitting(data)
    thetaCenter[c] = x + y*(1j)
    data = np.concatenate((phiFW[c].flatten(), phiRV[c].flatten()))
    x, y, r = circle_fitting(data)
    phiCenter[c] = x + y*(1j)

# measure theta angles
cnt = thetaSteps//steps
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
cnt = phiSteps//steps + 1
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
cnt = thetaSteps//steps
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

# calculate phi motor maps
cnt = phiSteps//steps
for c in goodIdx:
    for b in range(regions):
        # forward motor maps
        binMin = binSize * b
        binMax = binMin + binSize
        fracSum = 0
        valueSum = 0
        for n in range(repeat):
            for k in range(cnt):
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
            for k in range(cnt):
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
old.createCalibrationFile(cobraCharmerPath+'/xml/motormap_'+datetoday+'n.xml')


print(cobraCharmerPath+'/xml/motormap_'+datetoday+'n.xml  produced!')
print("Process Finised")