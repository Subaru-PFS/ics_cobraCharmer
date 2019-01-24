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

class ontimeModel():
    
    def getCalibModel(self, initXML):
    
        pfi = pfiControl.PFI(fpgaHost='localhost', doConnect=False) #'fpga' for real device.
        pfi.loadModel(initXML)
        
        return pfi.calibModel

    def getThetaFwdSlope(self, pid, modelArray):
        
        onTimeArray = []
        angSpdArray = []
        for m in modelArray:
            onTimeArray.append(m.motorOntimeFwd1[pid-1]*1000)
            angSpdArray.append(np.mean(np.rad2deg(m.angularSteps[pid-1]/m.S1Pm[pid-1])))
        
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(onTimeArray,angSpdArray)
        
        return slope
    
    def getThetaRevSlope(self, pid, modelArray):
        
        onTimeArray = []
        angSpdArray = []
        for m in modelArray:
            onTimeArray.append(m.motorOntimeRev1[pid-1]*1000)
            angSpdArray.append(-np.mean(np.rad2deg(m.angularSteps[pid-1]/m.S1Nm[pid-1])))

        slope, intercept, r_value, p_value, std_err = stats.linregress(onTimeArray,angSpdArray)
        
        return slope
  
    def getPhiFwdSlope(self, pid, modelArray):
        
        onTimeArray = []
        angSpdArray = []
        for m in modelArray:
            onTimeArray.append(m.motorOntimeFwd2[pid-1]*1000)
            angSpdArray.append(np.mean(np.rad2deg(m.angularSteps[pid-1]/m.S2Pm[pid-1])))
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(onTimeArray,angSpdArray)
        
        return slope
    
    def getPhiRevSlope(self, pid, modelArray):
        
        onTimeArray = []
        angSpdArray = []
        for m in modelArray:
            onTimeArray.append(m.motorOntimeRev2[pid-1]*1000)
            angSpdArray.append(-np.mean(np.rad2deg(m.angularSteps[pid-1]/m.S2Nm[pid-1])))
        slope, intercept, r_value, p_value, std_err = stats.linregress(onTimeArray,angSpdArray)
        
        return slope

    def buildModelfromXML(self, xmlArray, visibleFibers=False):
        
        # Reading all model and build a model list
        model = []
        for xml in xmlArray:
            model.append(self.getCalibModel(xml))

        j1fwd_slope = []
        j1rev_slope = []
        j2fwd_slope = []
        j2rev_slope = []

        for pid in range(1,58):
            j1fwd_slope.append(self.getThetaFwdSlope(pid,model))
            j1rev_slope.append(self.getThetaRevSlope(pid,model))
            j2fwd_slope.append(self.getPhiFwdSlope(pid,model))
            j2rev_slope.append(self.getPhiRevSlope(pid,model))
            

        self.j1fwd_slope = j1fwd_slope
        self.j1rev_slope = j1rev_slope
        self.j2fwd_slope = j2fwd_slope
        self.j2rev_slope = j2rev_slope

    
    def getTargetOnTime(self, target, modelSlope, onTime, angSpeed):
        #print(onTime)
        #print(angSpeed)
        
        size = len(onTime)
        newOntime_ms = np.zeros(size)
        sumx=np.zeros(size)

        onTime_ms = onTime*1000
        #if (target.any() > 0) :
        if isinstance(modelSlope, list) ==  True:
 
            for i in range(size):
                if modelSlope[i] == 0:
                    sumx[i]=0
                else:
                    sumx[i] = (target - angSpeed[i])/modelSlope[i]   

        else:
            sumx = (target - angSpeed)/modelSlope  

        newOntime_ms= onTime_ms+sumx

        newOntime = newOntime_ms / 1000.0 
        return newOntime

    def __init__(self):
        self.j1fwd_slope = 0.0050
        self.j1rev_slope = -0.0046
        self.j2fwd_slope = 0.0090
        self.j2rev_slope = -0.0082


        
class adjustOnTime():

    def extractCalibModel(self, initXML):

        pfi = pfiControl.PFI(fpgaHost='localhost', doConnect=False) #'fpga' for real device.
        pfi.loadModel(initXML)
        
        return pfi.calibModel
    

    def updateOntimeWithFiberSlope(self, originXML, newXML, xmlArray=False, thetaTable=False, phiTable=False):
        
        #datetoday=datetime.datetime.now().strftime("%Y%m%d")
        model = self.extractCalibModel(originXML)

        size = len(model.angularSteps)

        j1fwd_avg = np.zeros(size)
        j1rev_avg = np.zeros(size)
        j2fwd_avg = np.zeros(size)
        j2rev_avg = np.zeros(size)
        
        
        for i in range(size):
            
            # 
            j1_limit = (360/np.rad2deg(model.angularSteps[0])-1).astype(int)
            j2_limit = (180/np.rad2deg(model.angularSteps[0])-1).astype(int)
            
            j1fwd_avg[i] = np.mean(np.rad2deg(model.angularSteps[i]/model.S1Pm[i][:j1_limit]))
            j1rev_avg[i] = -np.mean(np.rad2deg(model.angularSteps[i]/model.S1Nm[i][:j1_limit]))
            j2fwd_avg[i] = np.mean(np.rad2deg(model.angularSteps[i]/model.S2Pm[i][:j2_limit]))
            j2rev_avg[i] = -np.mean(np.rad2deg(model.angularSteps[i]/model.S2Nm[i][:j2_limit]))

        # If xml files is given, use xml files to build the on-time model.
        if xmlArray is not False:
            otm = ontimeModel()
            otm.buildModelfromXML(xmlArray)

        newOntimeFwd1 = otm.getTargetOnTime(0.05,otm.j1fwd_slope, model.motorOntimeFwd1 ,j1fwd_avg)
        newOntimeFwd2 = otm.getTargetOnTime(0.07,otm.j2fwd_slope, model.motorOntimeFwd2 ,j2fwd_avg)

        newOntimeRev1 = otm.getTargetOnTime(-0.05,otm.j1rev_slope, model.motorOntimeRev1 ,j1rev_avg)
        newOntimeRev2 = otm.getTargetOnTime(-0.07,otm.j2rev_slope, model.motorOntimeRev2 ,j2rev_avg)
        pid = range(1,58)
        if thetaTable is not False:
            t=Table([pid, model.motorOntimeFwd1,j1fwd_avg, otm.j1fwd_slope, newOntimeFwd1, 
                     model.motorOntimeRev1,j1rev_avg, otm.j1rev_slope, newOntimeRev1],
                     names=('Fiber No','Ori Fwd OT', 'FWD sp', 'FWD slope', 'New Fwd OT',
                            'Ori Rev OT', 'REV sp', 'REV slope', 'New Rev OT'),
                     dtype=('i2','f4', 'f4', 'f4','f4', 'f4', 'f4', 'f4', 'f4'))
            t.write(thetaTable,format='ascii.ecsv',overwrite=True,
                    formats={'Fiber No':'%i','Ori Fwd OT': '%10.5f', 'FWD sp': '%10.5f', 'FWD slope': '%10.5f', 'New Fwd OT': '%10.5f',\
                             'Ori Rev OT': '%10.5f', 'REV sp': '%10.5f', 'REV slope': '%10.5f', 'New Rev OT': '%10.5f'})
  
        if phiTable is not False:
            t=Table([pid, model.motorOntimeFwd2,j2fwd_avg, otm.j2fwd_slope, newOntimeFwd2,
                     model.motorOntimeRev2,j2rev_avg, otm.j2rev_slope, newOntimeRev2],
                     names=('Fiber No','Ori Fwd OT', 'FWD sp', 'FWD slope', 'New Fwd OT',
                            'Ori Rev OT', 'REV sp', 'REV slope', 'New Rev OT'),
                     dtype=('i2','f4', 'f4', 'f4','f4', 'f4', 'f4', 'f4', 'f4'))
            t.write(phiTable,format='ascii.ecsv',overwrite=True, 
                    formats={'Fiber No':'%i','Ori Fwd OT': '%10.5f', 'FWD sp': '%10.5f', 'FWD slope': '%10.5f', 'New Fwd OT': '%10.5f',\
                             'Ori Rev OT': '%10.5f', 'REV sp': '%10.5f', 'REV slope': '%10.5f', 'New Rev OT': '%10.5f'})

        model.updateOntimes(thtFwd=newOntimeFwd1, thtRev=newOntimeRev1, phiFwd=newOntimeFwd2, phiRev=newOntimeRev2)
        model.createCalibrationFile(newXML)

    def __init__(self):
        pass

def main():
    xmlarray = []
    dataPath='/home/pfs/mhs/devel/ics_cobraCharmer/xml/'
    for tms in range(20, 60, 10):
        xml=dataPath+f'motormapOntime_{tms}us_20190123.xml'
        xmlarray.append(xml)
    
    datetoday=datetime.datetime.now().strftime("%Y%m%d")    
    # cobraCharmerPath='/home/pfs/mhs/devel/ics_cobraCharmer.cwen/'
    cobraCharmerPath='/home/pfs/mhs/devel/ics_cobraCharmer/'
    adjot=adjustOnTime()
  
    #initXML=cobraCharmerPath+'/xml/precise6.xml'
    initXML=cobraCharmerPath+'/xml/motormaps_181205.xml'
    newXML = cobraCharmerPath+'/xml/updateOntime_'+datetoday+'n.xml'
    
    adjot.updateOntimeWithFiberSlope(initXML, newXML, xmlArray=xmlarray, thetaTable='theta.tbl',phiTable='phi.tbl')

    m = adjot.extractCalibModel(newXML)
    OnTime = deepcopy([m.motorOntimeFwd1,
                   m.motorOntimeRev1,
                   m.motorOntimeFwd2,
                   m.motorOntimeRev2])
    
    # Taking care bad measurements
    OnTime[2][25]=0.035
    OnTime[3][25]=0.035

    OnTime[2][29]=0.022
    OnTime[3][29]=0.022

    OnTime[2][41]=0.0229
    OnTime[3][41]=0.0231
    
    OnTime[2][56]=0.020
    OnTime[3][56]=0.020
    m.updateOntimes(*OnTime)
    m.createCalibrationFile(newXML)

if __name__ == '__main__':
    main()