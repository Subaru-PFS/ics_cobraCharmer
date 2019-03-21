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
    x = p[:,0]
    y = p[:,1]
    m = np.vstack([x, y, np.ones(len(p))]).T
    n = np.array(x*x + y*y)
    a, b, c = np.linalg.lstsq(m, n, rcond=None)[0]
    return a/2, b/2, np.sqrt(c+(a*a+b*b)/4)

# function to move cobras to target positions
def moveToXYfromHome(idx, targets, dataPath, threshold=3.0, maxTries=8):
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


# Prepare the data path for the work
if not (os.path.exists(dataPath)):
    os.makedirs(dataPath)

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
coaseXML=cobraCharmerPath+'/xml/coarse_spare02_'+datetoday+'.xml'
if not os.path.exists(coaseXML):
    sys.exit()
    
pfi.loadModel(coaseXML)
pfi.setFreq(allCobras)


originalOnTime = deepcopy([pfi.calibModel.motorOntimeFwd1,
                   pfi.calibModel.motorOntimeRev1,
                   pfi.calibModel.motorOntimeFwd2,
                   pfi.calibModel.motorOntimeRev2])


# Using 1.5X on-time for geometry measurement
enhanceFactor = 2.0
fastOnTime = deepcopy([pfi.calibModel.motorOntimeFwd1*enhanceFactor,
                   pfi.calibModel.motorOntimeRev1*enhanceFactor,
                   pfi.calibModel.motorOntimeFwd2*enhanceFactor,
                   pfi.calibModel.motorOntimeRev2*enhanceFactor])

#pfi.calibModel.updateOntimes(*fastOnTime)
#pfi.setFreq(allCobras)


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
brokens = []
visibles= [e for e in range(1,58)]
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

def thetaFN(camId):
    return f'/theta{camId}_'

def phiFN(camId):
    return f'/phi{camId}_'

# Home phi
pfi.moveAllSteps(allCobras, 0, -5000)

# Home theta
pfi.moveAllSteps(allCobras, -10000, 0)

# move visible positioners to outwards positions, phi arms are moved out for 60 degrees
# (outTargets) so we can measure the positions
thetas = np.empty(57, dtype=float)
thetas[::2] = pfi.thetaToLocal(oddCobras, np.full(len(oddCobras), np.deg2rad(270)))
thetas[1::2] = pfi.thetaToLocal(evenCobras, np.full(len(evenCobras), np.deg2rad(90)))
phis = np.full(57, np.deg2rad(60.0))
outTargets = pfi.anglesToPositions(allCobras, thetas, phis)

# move to outTargets
moveToXYfromHome(goodIdx, outTargets[goodIdx], dataPath)

# move phi arms in
pfi.moveAllSteps(getCobras(goodIdx), 0, -5000)

# Using fast on-time before operation.
pfi.calibModel.updateOntimes(*fastOnTime)
pfi.setFreq(allCobras)

# record the theta and phi arm movements
myIdx = goodIdx
myCobras = getCobras(myIdx)

# take one image at limit
p1 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "1", "-e", "18", "-f", dataPath+"/cam1P1_"], stdout=PIPE)
p2 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "2", "-e", "18", "-f", dataPath+"/cam2P1_"], stdout=PIPE)
p1.communicate()
p2.communicate()
time.sleep(1.0)

# move phi out and capture the video
p1 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "1", "-e", "18", "-i", "100", "-l", "9999", "-f", dataPath+f"/{phiFN(1)}"], stdout=PIPE)
p2 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "2", "-e", "18", "-i", "100", "-l", "9999", "-f", dataPath+f"/{phiFN(2)}"], stdout=PIPE)
time.sleep(5.0)
pfi.moveAllSteps(myCobras, 0, 5000)
time.sleep(0.5)
p1.kill()
p2.kill()
p1.communicate()
p2.communicate()
pfi.moveAllSteps(myCobras, 0, 5000)


# take one image at limit
p1 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "1", "-e", "18", "-f", dataPath+f"/cam1P2_"], stdout=PIPE)
p2 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "2", "-e", "18", "-f", dataPath+f"/cam2P2_"], stdout=PIPE)
p1.communicate()
p2.communicate()
time.sleep(1.0)

pfi.calibModel.updateOntimes(*originalOnTime)
pfi.setFreq(allCobras)
# move phi arms to 60 degrees
pfi.moveAllSteps(myCobras, 0, -10000)
moveToXYfromHome(myIdx, outTargets[myIdx],dataPath)

pfi.calibModel.updateOntimes(*fastOnTime)
pfi.setFreq(allCobras)

# move theta arms to CCW hard stops and take one image
pfi.moveAllSteps(myCobras, -10000, 0)
p1 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "1", "-e", "18", "-f", dataPath+"/cam1P3_"], stdout=PIPE)
p2 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "2", "-e", "18", "-f", dataPath+"/cam2P3_"], stdout=PIPE)
p1.communicate()
p2.communicate()
time.sleep(1.0)

# move theta for a circle and capture the video
p1 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "1", "-e", "18", "-i", "100", "-l", "9999", "-f", dataPath+f"/{thetaFN(1)}"], stdout=PIPE)
p2 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "2", "-e", "18", "-i", "100", "-l", "9999", "-f", dataPath+f"/{thetaFN(2)}"], stdout=PIPE)
time.sleep(5.0)
pfi.moveAllSteps(myCobras, 10000, 0)
time.sleep(0.5)
p1.kill()
p2.kill()
p1.communicate()
p2.communicate()
pfi.moveAllSteps(myCobras, 10000, 0)

# take one image at limit
p1 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "1", "-e", "18", "-f", dataPath+f"/cam1P4_"], stdout=PIPE)
p1.communicate()
p2 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "2", "-e", "18", "-f", dataPath+f"/cam2P4_"], stdout=PIPE)
p2.communicate()

# move back to CCW hard stops
pfi.moveAllSteps(myCobras, -10000, -5000)

# variable declaration
phiCircles = np.zeros((57, 3), dtype=float)
thetaCircles = np.zeros((57, 3), dtype=float)
thetaCCW = np.zeros(57, dtype=float)
thetaCW = np.zeros(57, dtype=float)
phiCCW = np.zeros(57, dtype=float)
phiCW = np.zeros(57, dtype=float)


# first camera
myIdx = goodIdx[goodIdx <= cam_split]
homes = pfi.calibModel.centers[myIdx]

# phi stages
cnt = len(glob.glob(dataPath+f'/{phiFN(1)}*')) - 5
pos = np.zeros((len(myIdx), cnt, 2))

for i in range(cnt):
    data = fits.getdata(dataPath+f'/{phiFN(1)}{i+1:04d}.fits')
    cs = sep.extract(data.astype(float), 50)
    spots = np.array([(c['x'],c['y']) for c in cs])
    idx = lazyIdentification(homes, spots[:,0]+spots[:,1]*(1j))
    pos[:,i] = spots[idx]

# find centers
for i in range(len(myIdx)):
    x0, y0, r0 = circle_fitting(pos[i])
    phiCircles[myIdx[i]] = x0, y0, r0

# theta stages
cnt = len(glob.glob(dataPath+f'/{thetaFN(1)}*')) - 5
pos = np.zeros((len(myIdx), cnt, 2))
print(cnt)
for i in range(cnt):
    data = fits.getdata(dataPath+f'/{thetaFN(1)}{i+1:04d}.fits')
    cs = sep.extract(data.astype(float), 50)
    spots = np.array([(c['x'],c['y']) for c in cs])
    idx = lazyIdentification(homes, spots[:,0]+spots[:,1]*(1j))
    pos[:,i] = spots[idx]

# find centers
for i in range(len(myIdx)):
    x0, y0, r0 = circle_fitting(pos[i])
    thetaCircles[myIdx[i]] = x0, y0, r0

# second camera
myIdx = goodIdx[goodIdx > cam_split]
homes = pfi.calibModel.centers[myIdx]

# phi stages
cnt = len(glob.glob(dataPath+f'/{phiFN(2)}*')) - 5
pos = np.zeros((len(myIdx), cnt, 2))

for i in range(cnt):
    data = fits.getdata(dataPath+f'/{phiFN(2)}{i+1:04d}.fits')
    cs = sep.extract(data.astype(float), 50)
    spots = np.array([(c['x'],c['y']) for c in cs])
    idx = lazyIdentification(homes, spots[:,0]+spots[:,1]*(1j))
    pos[:,i] = spots[idx]

# find centers
for i in range(len(myIdx)):
    x0, y0, r0 = circle_fitting(pos[i])
    phiCircles[myIdx[i]] = x0, y0, r0

# theta stages
cnt = len(glob.glob(dataPath+f'/{thetaFN(2)}*')) - 5
pos = np.zeros((len(myIdx), cnt, 2))

for i in range(cnt):
    data = fits.getdata(dataPath+f'/{thetaFN(2)}{i+1:04d}.fits')
    cs = sep.extract(data.astype(float), 50)
    spots = np.array([(c['x'],c['y']) for c in cs])
    idx = lazyIdentification(homes, spots[:,0]+spots[:,1]*(1j))
    pos[:,i] = spots[idx]

# find centers
for i in range(len(myIdx)):
    x0, y0, r0 = circle_fitting(pos[i])
    thetaCircles[myIdx[i]] = x0, y0, r0

# second camera
myIdx = goodIdx[goodIdx > cam_split]
homes = pfi.calibModel.centers[myIdx]

# phi stages
cnt = len(glob.glob(dataPath+f'/{phiFN(2)}*')) - 5
pos = np.zeros((len(myIdx), cnt, 2))

for i in range(cnt):
    data = fits.getdata(dataPath+f'/{phiFN(2)}{i+1:04d}.fits')
    cs = sep.extract(data.astype(float), 50)
    spots = np.array([(c['x'],c['y']) for c in cs])
    idx = lazyIdentification(homes, spots[:,0]+spots[:,1]*(1j))
    pos[:,i] = spots[idx]

# find centers
for i in range(len(myIdx)):
    x0, y0, r0 = circle_fitting(pos[i])
    phiCircles[myIdx[i]] = x0, y0, r0

# theta stages
cnt = len(glob.glob(dataPath+f'/{thetaFN(2)}*')) - 5
pos = np.zeros((len(myIdx), cnt, 2))

for i in range(cnt):
    data = fits.getdata(dataPath+f'/{thetaFN(2)}{i+1:04d}.fits')
    cs = sep.extract(data.astype(float), 50)
    spots = np.array([(c['x'],c['y']) for c in cs])
    idx = lazyIdentification(homes, spots[:,0]+spots[:,1]*(1j))
    pos[:,i] = spots[idx]

# find centers
for i in range(len(myIdx)):
    x0, y0, r0 = circle_fitting(pos[i])
    thetaCircles[myIdx[i]] = x0, y0, r0

# Calculate hard stops
thetaC = thetaCircles[:,0] + thetaCircles[:,1]*(1j)
phiC = phiCircles[:,0] + phiCircles[:,1]*(1j)
points = np.zeros((57, 4), dtype=complex)

# process images
myIdx = goodIdx[goodIdx <= cam_split]
homes = pfi.calibModel.centers[myIdx]
for p in range(4):
    data = fits.getdata(dataPath+f'/cam1P{p+1}_0001.fits')
    cs = sep.extract(data.astype(float), 50)
    spots = np.array([(c['x'],c['y']) for c in cs])
    idx = lazyIdentification(homes, spots[:,0]+spots[:,1]*(1j))
    points[myIdx, p] = spots[idx,0] + spots[idx,1]*(1j)

myIdx = goodIdx[goodIdx > cam_split]
homes = pfi.calibModel.centers[myIdx]
for p in range(4):
    data = fits.getdata(dataPath+f'/cam2P{p+1}_0001.fits')
    cs = sep.extract(data.astype(float), 50)
    spots = np.array([(c['x'],c['y']) for c in cs])
    idx = lazyIdentification(homes, spots[:,0]+spots[:,1]*(1j))
    points[myIdx, p] = spots[idx,0] + spots[idx,1]*(1j)

# phi hard stops
phiCCW = (np.angle(points[:,0] - phiC) - np.angle(thetaC - phiC) + (np.pi/2)) % (2*np.pi) - (np.pi/2)
phiCW = (np.angle(points[:,1] - phiC) - np.angle(thetaC - phiC) + (np.pi/2)) % (2*np.pi) - (np.pi/2)

# theta CCW hard stops
thetaCCW = np.angle(phiC - thetaC) % (2*np.pi)
a = np.absolute(points[:,2] - thetaC)
b = np.absolute(thetaC - phiC)
c = phiCircles[:,2]

temp = a*b
inx = np.where(temp != 0)
ind = np.where(temp == 0)
print(a*b)
#if a*b != 0:
thetaCCW[inx] = (np.angle(points[:,2] - thetaC) + np.arccos((a*a + b*b - c*c) / (2*a*b))) % (2*np.pi)
#else:
thetaCCW[ind] = 0
# thetaCW hard stops
thetaCW = (np.angle(points[:,3] - thetaC) - np.angle(points[:,2] - thetaC) + thetaCCW) % (2*np.pi)    

# check if any thing is wrong here
print(phiCircles, thetaCircles)
print(phiCCW, phiCW, thetaCCW, (thetaCW-thetaCCW)%(2*np.pi))

#%matplotlib inline 
#%matplotlib qt
plt.figure(1)
plt.clf()

plt.subplot(211)
ax = plt.gca()
ax.plot(thetaCircles[group1,0], thetaCircles[group1,1], 'ro')
ax.plot(phiCircles[group1,0], phiCircles[group1,1], 'mo')
for idx in group1:
    c1 = plt.Circle((thetaCircles[idx,0], thetaCircles[idx,1]), thetaCircles[idx,2], color='g', fill=False)
    c2 = plt.Circle((phiCircles[idx,0], phiCircles[idx,1]), phiCircles[idx,2], color='b', fill=False)
    ax.add_artist(c1)
    ax.add_artist(c2)
ax.set_title(f'1st camera')

plt.subplot(212)
ax = plt.gca()
ax.plot(thetaCircles[group2,0], thetaCircles[group2,1], 'ro')
ax.plot(phiCircles[group2,0], phiCircles[group2,1], 'mo')
for idx in group2:
    c1 = plt.Circle((thetaCircles[idx,0], thetaCircles[idx,1]), thetaCircles[idx,2], color='g', fill=False)
    c2 = plt.Circle((phiCircles[idx,0], phiCircles[idx,1]), phiCircles[idx,2], color='b', fill=False)
    ax.add_artist(c1)
    ax.add_artist(c2)
ax.set_title(f'2nd camera')

plt.show()

# save new configuration
pfi.calibModel.updateOntimes(*originalOnTime)
old = pfi.calibModel
myConfig = deepcopy(old)

# you can remove bad measurement here
#idx = np.delete(goodIdx, np.argwhere(goodIdx==12))

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

old.createCalibrationFile(cobraCharmerPath+'/xml/precise_spare02_'+datetoday+'v1.xml')
print("Process Finised")

