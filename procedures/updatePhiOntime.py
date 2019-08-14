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
from astropy.table import Table
from scipy import stats
from copy import deepcopy


cobraCharmerPath='/Users/chyan/PythonCode/Instrument/ICS/ics_cobraCharmer'
#dataPath='/Users/chyan/Documents/workspace/ics_cobraCharmer/xml/'
for tms in range(20, 60, 10):
    xml=cobraCharmerPath+f'/xml/motormapPhiOntime_{tms}us_20190321.xml'
    
    # Initializing COBRA module
    pfi = pfiControl.PFI(fpgaHost='localhost', doConnect=False) 
    pfi.loadModel(xml)

    phiOnTime=np.full(57,(tms/1000.0))
    pfi.calibModel.updateOntimes(phiFwd=phiOnTime, phiRev=phiOnTime)

    pfi.calibModel.createCalibrationFile(cobraCharmerPath+f'/xml/motormapPhiOntime_{tms}us_20190321new.xml')

