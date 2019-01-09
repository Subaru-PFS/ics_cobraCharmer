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

class ontimeModel():

    def getTargetOnTime(self, target, modelSlope, onTime, angSpeed):
        #print(onTime)
        #print(angSpeed)
        
        size = len(onTime)
        newOntime_ms = np.zeros(size)

        onTime_ms = onTime*1000
        #if (target.any() > 0) :

        sumx = (target - angSpeed)/modelSlope    
        newOntime_ms= onTime_ms+sumx
        
    
        newOntime = newOntime_ms / 1000.0 

        return newOntime

    def __init__(self):
        self.j1fwd_slope = 0.0050
        self.j1rev_slope = -0.0046
        self.j2fwd_slope = 0.0090
        self.j2rev_slope = -0.0082

        self.j1fwd_itc = -0.12
        self.j1rev_itc = +0.10
        self.j2fwd_itc = -0.13
        self.j2rev_itc = +0.11
        
class adjustOnTime():

    def extractCalibModel(self, initXML):

        pfi = pfiControl.PFI(fpgaHost='localhost', doConnect=False) #'fpga' for real device.
        pfi.loadModel(initXML)
        
        return pfi.calibModel

    def updateOntimeWithDefaultSlope(self, originXML, newXML, thetaTable=False, phiTable=False):
        
        #datetoday=datetime.datetime.now().strftime("%Y%m%d")
        model = self.extractCalibModel(originXML)

        size = len(model.angularSteps)

        j1fwd_avg = np.zeros(size)
        j1rev_avg = np.zeros(size)
        j2fwd_avg = np.zeros(size)
        j2rev_avg = np.zeros(size)

        
        for i in range(size):
            
            # Calculate the limit index
            j1limit = 360/np.rad2deg(model.angularSteps[i])
            j2limit = 180/np.rad2deg(model.angularSteps[i])

            # The average should be in the range of 360 for theta, 180 for phi 
            j1fwd_avg[i] = np.mean(np.rad2deg(model.angularSteps[i]/model.S1Pm[i][:j1limit.astype(int)-1]))
            j1rev_avg[i] = -np.mean(np.rad2deg(model.angularSteps[i]/model.S1Nm[i][:j1limit.astype(int)-1]))
            j2fwd_avg[i] = np.mean(np.rad2deg(model.angularSteps[i]/model.S2Pm[i][:j2limit.astype(int)-1]))
            j2rev_avg[i] = -np.mean(np.rad2deg(model.angularSteps[i]/model.S2Nm[i][:j2limit.astype(int)-1]))

        otm = ontimeModel()
        newOntimeFwd1 = otm.getTargetOnTime(0.05,otm.j1fwd_slope, model.motorOntimeFwd1 ,j1fwd_avg)
        newOntimeFwd2 = otm.getTargetOnTime(0.07,otm.j2fwd_slope, model.motorOntimeFwd2 ,j2fwd_avg)

        newOntimeRev1 = otm.getTargetOnTime(-0.05,otm.j1rev_slope, model.motorOntimeRev1 ,j1rev_avg)
        newOntimeRev2 = otm.getTargetOnTime(-0.07,otm.j2rev_slope, model.motorOntimeRev2 ,j2rev_avg)


        if thetaTable is not False:
            t=Table([model.motorOntimeFwd1,j1fwd_avg,newOntimeFwd1,
                     model.motorOntimeRev1,j1rev_avg,newOntimeRev1],
                     names=('Ori Fwd OT', 'FWD sp', 'New Fwd OT','Ori Rev OT', 'REV sp', 'New Rev OT'),
                     dtype=('f4', 'f4', 'f4','f4', 'f4', 'f4'))
            t.write(thetaTable,format='ascii',overwrite=True)

        if phiTable is not False:
            t=Table([model.motorOntimeFwd2,j2fwd_avg,newOntimeFwd2,
                     model.motorOntimeRev2,j2rev_avg,newOntimeRev2],
                     names=('Ori Fwd OT', 'FWD sp', 'New Fwd OT','Ori Rev OT', 'REV sp', 'New Rev OT'),
                     dtype=('f4', 'f4', 'f4','f4', 'f4', 'f4'))
            t.write(phiTable,format='ascii',overwrite=True)

        model.updateOntimes(thtFwd=newOntimeFwd1, thtRev=newOntimeRev1, phiFwd=newOntimeFwd2, phiRev=newOntimeRev2)
        model.createCalibrationFile(newXML)

    def __init__(self):
        pass

def main():
    datetoday=datetime.datetime.now().strftime("%Y%m%d")    
    # cobraCharmerPath='/home/pfs/mhs/devel/ics_cobraCharmer.cwen/'
    cobraCharmerPath='/Users/chyan/Documents/workspace/ics_cobraCharmer/'
    adjot=adjustOnTime()
  
    initXML=cobraCharmerPath+'/xml/motormaps_181205.xml'
    newXML = cobraCharmerPath+'/xml/updateOntime_'+datetoday+'.xml'
    
    adjot.updateOntimeWithDefaultSlope(initXML, newXML, thetaTable='theta.tbl',phiTable='phi.tbl')


if __name__ == '__main__':
    main()