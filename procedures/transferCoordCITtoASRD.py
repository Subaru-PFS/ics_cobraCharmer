from importlib import reload
import numpy as np
import os
import time
import datetime
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
from ics.cobraCharmer import pfi as pfiControl
from copy import deepcopy

#datetoday=datetime.datetime.now().strftime("%Y%m%d")
datetoday='20181219'
cobraCharmerPath='/home/pfs/mhs/devel/ics_cobraCharmer.cwen/'
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

# return the tranformation parameters and a function that can convert origPoints to newPoints
def makeTransformation(origPoints, newPoints):
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

pfi = pfiControl.PFI(fpgaHost='128.149.77.24') #'fpga' for real device.
#pfi = pfiControl.PFI(fpgaHost='localhost', doLoadModel=False)
pfi.loadModel(cobraCharmerPath+'xml/updatedLinksAndMaps.xml')
pfi.setFreq(allCobras)

# Home phi
pfi.homePhi(allCobras, nsteps=5000, dir='ccw')

# Home theta
pfi.homeTheta(allCobras, nsteps=10000, dir='ccw')

# define the broken fibers and two groups of cobras
mapping = np.array([e for e in range(1,58) if e not in {1, 39, 43, 54}]) - 1
n1 = 26
n2 = len(mapping) - n1
group1 = mapping[:n1]
group2 = mapping[n1:]

# take an image at home positions
p1 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "1", "-e", "18", "-f", dataPath+"/home1_"], stdout=PIPE)
p1.communicate()
p2 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "2", "-e", "18", "-f", dataPath+"/home2_"], stdout=PIPE)
p2.communicate()

# process the image from the 1st camera
data = fits.getdata(dataPath+'/home1_0001.fits').astype(float)
cs = sep.extract(data, 50)
cs_home = np.array(sorted([(c['x'], c['y']) for c in cs], key=lambda t: t[0], reverse=True))
homes = cs_home[:n1,0] + cs_home[:n1,1]*(1j)

old = pfi.calibModel.centers[group1]
offset1, scale1, tilt1, convert1 = makeTransformation(old, homes)
np.abs(homes - convert1(old))

# process the image from the 2nd camera
data = fits.getdata(dataPath+'/home2_0001.fits').astype(float)
cs = sep.extract(data, 50)
cs_home = np.array(sorted([(c['x'], c['y']) for c in cs], key=lambda t: t[0], reverse=True))
homes = cs_home[-n2:,0] + cs_home[-n2:,1]*(1j)

old = pfi.calibModel.centers[group2]
offset2, scale2, tilt2, convert2 = makeTransformation(old, homes)
np.abs(homes - convert2(old))

old = pfi.calibModel
n = mapping[n1]

myConfig = deepcopy(old)
myConfig.centers[:n] = convert1(old.centers[:n])
myConfig.tht0[:n] = (old.tht0[:n]+tilt1)%(2*np.pi)
myConfig.tht1[:n] = (old.tht1[:n]+tilt1)%(2*np.pi)
myConfig.L1[:n] = old.L1[:n]*scale1
myConfig.L2[:n] = old.L2[:n]*scale1
myConfig.centers[n:] = convert2(old.centers[n:])
myConfig.tht0[n:] = (old.tht0[n:]+tilt2)%(2*np.pi)
myConfig.tht1[n:] = (old.tht1[n:]+tilt2)%(2*np.pi)
myConfig.L1[n:] = old.L1[n:]*scale2
myConfig.L2[n:] = old.L2[n:]*scale2

old.updateGeometry(myConfig.centers, myConfig.L1, myConfig.L2)
old.updateThetaHardStops(myConfig.tht0, myConfig.tht1)
old.createCalibrationFile(cobraCharmerPath+'/xml/coarse'+datetoday+'.xml')

print(cobraCharmerPath+'/xml/coarse'+datetoday+'.xml  produced!')
print("Process Finised")