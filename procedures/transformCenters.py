from importlib import reload
import numpy as np
import os
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
from ics.cobraCharmer import pfi as pfiControl
from copy import deepcopy

from ics.cobraCharmer.utils import fileManager
from ics.cobraCharmer.utils import coordinates
from ics.cobraCharmer.camera import cameraFactory
from ics.cobraCharmer import imageSet

import utils

output = fileManager.ProcedureDirectory('SC03', experimentName='coords')
print("output to ", output.rootDir)

# Define the cobra range.
mod1Cobras = pfiControl.PFI.allocateCobraModule(1)
allCobras = mod1Cobras

pfi = pfiControl.PFI(fpgaHost='fpga', doLoadModel=False) # or 'localhost'
pfi.loadModel('SC03.xml')
pfi.setFreq()

cam = cameraFactory('cit')
dataSet = imageSet.ImageSet(pfi, cam, output)
im, name = dataSet.expose(name='preHome')

pfi.moveAllSteps(allCobras, -10000, -5000)

im, name = dataSet.expose(name='centers')
cs, im = dataSet.spots('centers')
imCenters = np.stack((cs['x'], cs['y']), 1)

print("nspots = %d" % (len(cs)))
oldCenters = pfi.calibModel.centers
modelCenters = np.stack((np.real(oldCenters), np.imag(oldCenters)), 1)

imIdx, _ = coordinates.laydown(imCenters)
modelIdx, _ = coordinates.laydown(modelCenters)

homes = imCenters[imIdx,0] + imCenters[imIdx,1]*(1j)

offset1, scale1, tilt1, convert1 = coordinates.makeTransform(oldCenters[modelIdx], homes)

old = pfi.calibModel

centers = convert1(old.centers)
tht0 = (old.tht0+tilt1)%(2*np.pi)
tht1 = (old.tht1+tilt1)%(2*np.pi)
L1 = old.L1*scale1
L2 = old.L2*scale1

old.updateGeometry(centers, L1, L2)
old.updateThetaHardStops(tht0, tht1)
old.createCalibrationFile(os.path.join(output.xmlDir, 'recentered.xml'))

print("Process Finised")
