import logging
import pathlib
import sys, os
import numpy as np
import glob
import pandas as pd
import math
import pdb
from scipy import stats
import matplotlib.pyplot as plt
from astropy.io import fits

from bokeh.io import output_notebook, show, export_png,export_svgs, save
from bokeh.plotting import figure, show, output_file
import bokeh.palettes
from bokeh.layouts import column,gridplot
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper
from bokeh.models.glyphs import Text
from bokeh.palettes import Category20
from bokeh.palettes import YlGnBu8
from bokeh.transform import linear_cmap

import ontimeOptimize 
import visDianosticPlot
from moduleTest import ModuleTest
from ics.cobraCharmer import pfi as pfiControl
from ics.cobraCharmer import pfiDesign
from astropy.table import Table

class OntimeOptimize(object):

    def __init__(self, brokens=None, phiList=None, thetaList=None):
        self.brokens = []
        self.minSpeed = 0.001
        
        self.minOntime = 0.002

        if brokens is None:
            self.brokens = []
        else:
            self.brokens = brokens
     
        self.visibles= [e for e in range(1,58) if e not in brokens]
        self.badIdx = np.array(brokens) - 1
        self.goodIdx = np.array(self.visibles) - 1

        if thetaList is not None:
            self.datalist = thetaList
            self.loadFromThetaData(thetaList)
            self._buildModel()
        if phiList is not None:
            self.datalist = phiList
            self.loadFromPhiData(phiList)
            self._buildModel()
        if (phiList is None) and (thetaList is None):
            raise Exception
        
    #@classmethod
    def loadFromPhiData(self, phiList):
        #self = cls(runDirs)
        self.nPoints = len(phiList)
        self.axisNum = 2
        self.axisName = "Phi"
        self.maxOntime = 0.08
        self.slowTarget = 0.075
        self.fastTarget = 2.0*self.slowTarget
        self.minRange = 180
        self._buildDataFramefromData()

       # return self

    #@classmethod
    def loadFromThetaData(self, thetaList):
        #self = cls(thetaList)
        self.nPoints = len(thetaList)
        self.axisNum = 1
        self.axisName = "Theta"
        self.maxOntime = 0.08
        self.slowTarget = 0.075
        self.fastTarget = 2.0*self.slowTarget
        self.minRange = 360
        self._buildDataFramefromData()

        #return self
    
    def getSlope(self, pid, direction):
        dataframe = self.dataframe
        loc1 = dataframe.loc[dataframe['fiberNo'] == pid].loc
        loc2 = loc1[dataframe[direction].abs() > self.minSpeed].loc
        ndf = loc2[dataframe[f'range{direction}'].abs() > self.minRange]
        
        # When nPoint=1, using simple ratio to determine next point.  Otherwise, fitting data 
        #   with full ROM only
        
        if self.nPoints == 1:
            if (len(ndf[f'onTime{direction}'].values) == 0):
                slope = 0
                intercept = 0
            else:
            
                slope = ndf[f'{direction}'].values/ndf[f'onTime{direction}'].values
                intercept = 0
        else:
       
            if len(ndf[f'onTime{direction}'].values) > 1:

                onTimeArray = ndf[f'onTime{direction}'].values
                angSpdArray = ndf[f'{direction}'].values

                slope, intercept, r_value, p_value, std_err = stats.linregress(onTimeArray,angSpdArray)
            
            else:
                slope = 0.0
                intercept = 0.0
        
        if np.isnan(slope):
            slope = 0.0
            intercept = 0.0

        if direction == 'Fwd' and slope <= 0:
            slope = 0.0
            intercept = 0
        
        if direction == 'Rev' and slope >= 0:
            slope = 0.0
            intercept = 0

        return slope,intercept
    
    def getFwdSlope(self, pid):
        return self.getSlope(pid, 'Fwd')

    def getRevSlope(self, pid):
        return self.getSlope(pid, 'Rev')

    def fwdOntimeSlowModel(self, model):
        if self.axisNum == 1:
            return model.motorOntimeSlowFwd1[self.goodIdx]
        else:
            return model.motorOntimeSlowFwd2[self.goodIdx]

    def revOntimeSlowModel(self, model):
        if self.axisNum == 1:
            return model.motorOntimeSlowRev1[self.goodIdx]
        else:
            return model.motorOntimeSlowRev2[self.goodIdx]
    
    def fwdOntimeModel(self, model):
        if self.axisNum == 1:
            return model.motorOntimeFwd1[self.goodIdx]
        else:
            return model.motorOntimeFwd2[self.goodIdx]

    def revOntimeModel(self, model):
        if self.axisNum == 1:
            return model.motorOntimeRev1[self.goodIdx]
        else:
            return model.motorOntimeRev2[self.goodIdx]
   
    def _buildDataFramefromData(self):
        pidarray=[]
        otfarray=[]
        otrarray=[]
        fwarray=[]
        rvarray=[]
        dafarray=[]
        dararray=[]

        fwdminarray=[]
        fwdmaxarray=[]
        revminarray=[]
        revmaxarray=[]
        #imageSet = self.datalist

        for i in self.datalist:
            fw_file=f'{i}/data/'+f'{self.axisName.lower()}SpeedFW.npy'
            rv_file=f'{i}/data/'+f'{self.axisName.lower()}SpeedRV.npy'
            af = np.load(f'{i}/data/' + f'{self.axisName.lower()}AngFW.npy')
            ar = np.load(f'{i}/data/' + f'{self.axisName.lower()}AngRV.npy')
            fwdmm = np.rad2deg(np.load(f'{i}/data/' + f'{self.axisName.lower()}MMFW.npy'))
            revmm = np.rad2deg(np.load(f'{i}/data/' + f'{self.axisName.lower()}MMRV.npy'))
            fwd=np.load(fw_file)*180.0/math.pi 
            rev=-np.load(rv_file)*180.0/math.pi
            
            daf=[]
            dar=[]

            fwdmin = []
            fwdmax = []
            revmin = []
            revmax = []
            for p in self.goodIdx:
                daf.append((af[p,0,-1]-af[p,0,0])*180/3.14159)
                dar.append((ar[p,0,-1]-ar[p,0,0])*180/3.14159)
                fwdmin.append(np.min(fwdmm[p]))
                fwdmax.append(np.max(fwdmm[p]))
                revmin.append(-np.min(revmm[p]))
                revmax.append(-np.max(revmm[p]))

            xml = glob.glob(f'{i}/output/'+'*.xml')[0]
            model = pfiDesign.PFIDesign(xml)
            
            ontimeFwd = self.fwdOntimeSlowModel(model)
            ontimeRev = self.revOntimeSlowModel(model)

            pid=np.array(self.visibles)
            pidarray.append(pid)
            otfarray.append(ontimeFwd)
            otrarray.append(ontimeRev)
            dafarray.append(daf)
            dararray.append(dar)
            fwdminarray.append(fwdmin)
            fwdmaxarray.append(fwdmax)
            revminarray.append(revmin)
            revmaxarray.append(revmax)
            fwarray.append(fwd[self.goodIdx])
            rvarray.append(rev[self.goodIdx])
            
        d={'fiberNo': np.array(pidarray).flatten(),
           'onTimeFwd': np.array(otfarray).flatten(),
           'onTimeRev': np.array(otrarray).flatten(),
           'minFwd':np.array(fwdminarray).flatten(),
           'maxFwd':np.array(fwdmaxarray).flatten(),
           'minRev':np.array(revminarray).flatten(),
           'maxRev':np.array(revmaxarray).flatten(),
           'rangeFwd':np.array(dafarray).flatten(),
           'rangeRev':np.array(dararray).flatten(),           
           'Fwd': np.array(fwarray).flatten(),
           'Rev': np.array(rvarray).flatten()}
        
        self.dataframe = pd.DataFrame(d)

        self.groupMotors('Fwd')
        self.groupMotors('Rev')

    # This function separate motors into two groups, motors with small and large 
    #  speed variation.   
    def groupMotors(self,direction):
        try:
            self.dataframe
        except AttributeError:
            raise Exception('Dataframe is not existing')

        self.dataframe[f'dev{direction}'] = (self.dataframe[f'max{direction}']-
            self.dataframe[f'min{direction}'])/np.abs(self.dataframe[f'{direction}'])   
        
        self.dataframe[f'group{direction}'] = np.zeros(len(self.dataframe['fiberNo'])).astype(int)
        self.dataframe.loc[self.dataframe[f'dev{direction}'].abs() > 1.0, f'group{direction}'] = 1

    def _buildModel(self):

        self.fwd_slope = fwd_slope = np.zeros(57)
        self.fwd_int = fwd_int = np.zeros(57)

        self.rev_slope = rev_slope = np.zeros(57)
        self.rev_int = rev_int = np.zeros(57)

        for pid in self.visibles:
            #if 0 in self.dataframe.loc[self.dataframe['fiberNo'] == pid]['groupFwd'].tolist():
            # Apply fitting only when data are all good.
            datapoints = len(self.dataframe.loc[self.dataframe['fiberNo'] == pid]['groupFwd'].values)
            if np.sum(self.dataframe.loc[self.dataframe['fiberNo'] == pid]['groupFwd'].values) < 0.5*datapoints:   
                fw_s, fw_i = self.getFwdSlope(pid)
                fwd_slope[pid-1] = fw_s
                fwd_int[pid-1] = fw_i
            if np.sum(self.dataframe.loc[self.dataframe['fiberNo'] == pid]['groupRev'].values) < 0.5*datapoints:
                rv_s, rv_i = self.getRevSlope(pid)
                rev_slope[pid-1] = rv_s
                rev_int[pid-1] = rv_i

    def _pickCobraBestSpeed(self, fiberInx, direction, targetSpeed):
        dataframe = self.dataframe
        loc1 = dataframe.loc[dataframe['fiberNo'] == fiberInx+1].loc
        loc2 = loc1[dataframe[direction].abs() > self.minSpeed].loc
        ndf = loc2[dataframe[f'range{direction}'].abs() > self.minRange]
        # First, make sure there is good speed data.
        ind=np.argmin(np.abs((np.abs(ndf[f'{direction}']) - targetSpeed).values))
        newOntime = ndf[f'onTime{direction}'].values[ind]

        return newOntime

    def _pickupForSpeed(self, targetSpeed=None):
        newOntimeFwd = np.full(len(self.fwd_int), 0.08)
        newOntimeRev = np.full(len(self.fwd_int), 0.08)

        for i in self.goodIdx:
            
            newOntimeFwd[i]=self._pickCobraBestSpeed(i,'Fwd',targetSpeed)    
            newOntimeRev[i]=self._pickCobraBestSpeed(i,'Rev',targetSpeed)
        return newOntimeFwd,newOntimeRev

    def _searchNextGoodSpeed(self, fiberInx, direction, targetSpeed):
        dataframe = self.dataframe
        if self.nPoints == 1:
            ndf = dataframe.loc[dataframe['fiberNo'] == fiberInx+1]
            if np.abs(ndf[f'{direction}'].values[0]) > np.abs(targetSpeed):
                newOntime = ndf[f'onTime{direction}'].values[0] - 0.005
            else:
                newOntime = ndf[f'onTime{direction}'].values[0] + 0.005

        elif self.nPoints == 2:        
            loc1 = dataframe.loc[dataframe['fiberNo'] == fiberInx+1].loc
            ndf = loc1[dataframe[direction].abs() > self.minSpeed]
            if np.sum(ndf[f'group{direction}']) > 0:
                if np.abs(ndf[f'{direction}'].values[-1]) <  np.abs(targetSpeed):
                    newOntime = np.mean(ndf[f'onTime{direction}'].values)
                else:
                    newOntime = ndf[f'onTime{direction}'].values[-1] - 0.005
            else:
            
                if np.abs(ndf[f'{direction}'].values[-1]) <  np.abs(targetSpeed):
                    newOntime = ndf[f'onTime{direction}'].values[-1] + 0.01
                else:
                    newOntime = ndf[f'onTime{direction}'].values[-1] - 0.01
        else:
            loc1 = dataframe.loc[dataframe['fiberNo'] == fiberInx+1].loc
            loc2 = loc1[dataframe[direction].abs() > self.minSpeed].loc
            ndf = loc2[dataframe[f'range{direction}'].abs() > self.minRange]
            
            if (len(ndf[f'onTime{direction}'].values) == 0):
                loc1 = dataframe.loc[dataframe['fiberNo'] == fiberInx+1].loc
                ndf = loc1[dataframe[direction].abs() > self.minSpeed]
            
            # First, make sure there is good speed data.
            ind=np.argmin(np.abs((ndf[f'{direction}'] -  targetSpeed).values))
            if np.abs(ndf[f'{direction}'].values[ind]-targetSpeed) < 0.02:
                newOntime = ndf[f'onTime{direction}'].values[ind]
            else:
                ind = np.argmin(ndf[f'onTime{direction}'].values)
                if np.abs(ndf[f'{direction}'].values[ind]) > np.abs(targetSpeed):
                    newOntime = ndf[f'onTime{direction}'].values[ind] - 0.010
                else:
                    newOntime = ndf[f'onTime{direction}'].values[ind] + 0.010 

        return newOntime
    
    def checkOnTimeROM(self,fiberInx, Ontime, direction):
        dataframe = self.dataframe
        ndf = dataframe.loc[dataframe['fiberNo'] == fiberInx+1]
        ndd = ndf.loc[ndf[f'onTime{direction}'] == Ontime]
        if len(ndd[f'range{direction}']) != 0:
            if np.abs(ndd[f'range{direction}'].values[0]) < self.minRange:
                newOntime = 0.01 + Ontime
                return newOntime
            else:
                return Ontime    
        else:
            return Ontime
    def _solveForSpeed(self, targetSpeed=None):
        fwd_target = np.full(len(self.fwd_int), targetSpeed)
        rev_target = np.full(len(self.fwd_int), -targetSpeed)
        
        newOntimeFwd = (fwd_target - self.fwd_int)/self.fwd_slope
        newOntimeRev = (rev_target - self.rev_int)/self.rev_slope

        # If there is only one dataset, add 10us to first guess for 
        #  better change to complete full ROM
        if self.nPoints == 1:
            newOntimeFwd[np.where(np.isfinite(newOntimeFwd))[0]] += 0.010
            newOntimeRev[np.where(np.isfinite(newOntimeRev))[0]] += 0.010

        # Now, dealing with inf values.  Those values are due to bad fitting results for the
        #  following cases.
        #   1. No data in good motor groups
        #   2. Negative slope in Fwd or pasitive slope in Rev
        #   3. 
        inx = np.where(np.isinf(newOntimeFwd))[0].tolist()
        for i in inx:
            if i not in self.badIdx:
                newOntimeFwd[i] = self._searchNextGoodSpeed(i, 'Fwd', targetSpeed)
            else:
                newOntimeFwd[i] = self.maxOntime

        inx = np.where(np.isinf(newOntimeRev))[0].tolist()
        for i in inx:
            if i not in self.badIdx:
                newOntimeRev[i] = self._searchNextGoodSpeed(i, 'Rev', -targetSpeed)
            else:
                newOntimeRev[i] = self.maxOntime


        fastOnes = np.where((newOntimeFwd > self.maxOntime) | (newOntimeRev > self.maxOntime))
        if len(fastOnes[0]) > 0:
            logging.warn(f'some motors too fast: {fastOnes[0]}: '
                         f'fwd:{newOntimeFwd[fastOnes]} rev:{newOntimeRev[fastOnes]}')

        slowOnes = np.where((newOntimeFwd < self.minOntime) | (newOntimeRev < self.minOntime))
        if len(slowOnes[0]) > 0:
            logging.warn(f'some motors too slow {slowOnes[0]}: '
                         f'fwd:{newOntimeFwd[slowOnes]} rev:{newOntimeRev[slowOnes]}')

        slowOnes = np.where((newOntimeFwd < 0))
        if len(slowOnes[0]) > 0: 
            for i in slowOnes[0]:
                newOntimeFwd[i] = self._searchNextGoodSpeed(i, 'Fwd', targetSpeed)
        
        slowOnes = np.where((newOntimeRev < 0))
        if len(slowOnes[0]) > 0: 
            for i in slowOnes[0]:
                newOntimeRev[slowOnes] = self._searchNextGoodSpeed(i, 'Rev', -targetSpeed)

        for i in self.goodIdx:
            newOntimeFwd[i] = self.checkOnTimeROM(i, newOntimeFwd[i], 'Fwd') 
            newOntimeRev[i] = self.checkOnTimeROM(i, newOntimeRev[i], 'Rev') 

        return newOntimeFwd, newOntimeRev

    def pickForSlowSpeed(self, targetSpeed=None):
        if targetSpeed is None:
            targetSpeed = self.slowTarget
        newMaps = self._pickupForSpeed(targetSpeed=targetSpeed)
        self.newOntimeSlowFwd, self.newOntimeSlowRev = newMaps

        return newMaps

    def pickForFastSpeed(self, targetSpeed=None):
        if targetSpeed is None:
            targetSpeed = self.fastTarget
        
        fastOntimeFwd = np.zeros(57)
        fastOntimeRev = np.zeros(57)
        _,_=self.pickForSlowSpeed()
        pickFastOntimeFwd, pickFastOntimeRev = self._pickupForSpeed(targetSpeed=targetSpeed)
        sloveFastOntimeFwd, solveFastOntimeRev = self._solveForSpeed(targetSpeed=targetSpeed)
        
        for i in range(57):
            # Including result from solving fast speed
            if sloveFastOntimeFwd[i] > 0.08:
                fastOntimeFwd[i]=pickFastOntimeFwd[i]
            else:
                fastOntimeFwd[i]=np.max([pickFastOntimeFwd[i],sloveFastOntimeFwd[i]])
                if np.abs(fastOntimeFwd[i]-self.newOntimeSlowFwd[i])< 0.005:
                    fastOntimeFwd[i]=0.08
            
            if solveFastOntimeRev[i] > 0.08:
                fastOntimeRev[i]=pickFastOntimeRev[i]
            else:    
                fastOntimeRev[i]=np.max([pickFastOntimeRev[i],solveFastOntimeRev[i]])
                if np.abs(fastOntimeRev[i]-self.newOntimeSlowRev[i])< 0.005:
                    fastOntimeRev[i]=0.08

        self.newOntimeFwd, self.newOntimeRev = (fastOntimeFwd,fastOntimeRev)

        return fastOntimeFwd,fastOntimeRev
    
    def solveForSlowSpeed(self, targetSpeed=None):
        if targetSpeed is None:
            targetSpeed = self.slowTarget
        newMaps = self._solveForSpeed(targetSpeed=targetSpeed)
        self.newOntimeSlowFwd, self.newOntimeSlowRev = newMaps

        return newMaps

    def solveForFastSpeed(self, targetSpeed=None):
        if targetSpeed is None:
            targetSpeed = self.fastTarget
        newMaps = self._solveForSpeed(targetSpeed=targetSpeed)
        self.newOntimeFwd, self.newOntimeRev = newMaps

        return newMaps

    def visBestOntime(self, fiberInx, direction, predict=True):
        dd=self.dataframe.loc[self.dataframe['fiberNo'] == fiberInx+1]

        
        TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']
        title_string=f"Fiber {fiberInx+1} {self.axisName} {direction}"
        
        xmin = 0
        xmax = 1.5*np.max(dd[f'onTime{direction}'])
        if direction is 'Fwd': 
            ymin =0
            ymax = 1.5*np.max(dd[f'{direction}']) 
        else: 
            ymax = 0
            ymin = 1.2*np.min(dd[f'{direction}'])
        
        
        
        p = figure(tools=TOOLS, x_range=[xmin,xmax], y_range=[ymin,ymax],
                plot_height=400, plot_width=500,title=title_string)
        p.xaxis.axis_label = 'On Time'
        p.yaxis.axis_label = 'Speed'

        gooddata = dd.loc[dd[f'{direction}'].abs() > 
             self.minSpeed].loc[dd[f'range{direction}'].abs() > self.minRange]
        
        #color_array = YlGnBu8[:len(dd[direction].values)]
        c_array = ['blue', 'saddlebrown', 'darkviolet', 
                        'seagreen', 'yellow','tan']
        color_array = c_array[:len(dd[direction].values)]
        if direction is 'Fwd':
            xrange=np.arange(self.newOntimeSlowFwd[fiberInx]*0.8,1.2*np.max(dd[f'onTime{direction}']),0.001)
            newOntime = self.newOntimeSlowFwd[fiberInx] 
            y_predicted = [self.fwd_slope[fiberInx]*i + self.fwd_int[fiberInx]  for i in xrange]
            #if predict == True:
            p.circle(x=[newOntime],y=[self.slowTarget],color='red', size=15)
        
    
        if direction is 'Rev':
            xrange=np.arange(self.newOntimeSlowRev[fiberInx]*0.8,1.2*np.max(dd[f'onTime{direction}']),0.001)

            newOntime = self.newOntimeSlowRev[fiberInx]
            y_predicted = [self.rev_slope[fiberInx]*i + self.rev_int[fiberInx]  for i in xrange]
            #if predict == True:
            p.circle(x=[newOntime],y=[-self.slowTarget],color='red', size=15)
        
        if predict == True:
            p.line(xrange,y_predicted,color='red')
        
        p.circle(x=dd[f'onTime{direction}'],y=dd[f'{direction}'], size=10, color=color_array)
        p.circle(x=gooddata[f'onTime{direction}'],y=gooddata[direction],color='red',fill_color=None, size= 10)
        p.segment(dd[f'onTime{direction}'], dd[f'max{direction}'], dd[f'onTime{direction}'], dd[f'min{direction}'], color="black")
        return p


    def updateXML(self, initXML, newXML, solve=True, table=None):
        
        model = pfiDesign.PFIDesign(initXML)

        if solve is True:
            SlowOntimeFwd, SlowOntimeRev = self.solveForSlowSpeed()
            OntimeFwd, OntimeRev = self.solveForFastSpeed()
        else:
            SlowOntimeFwd, SlowOntimeRev = self.pickForSlowSpeed()
            OntimeFwd, OntimeRev = self.pickForFastSpeed()


        # If there is a broken fiber, set on-time to original value
        if len(self.badIdx) != 0:
            if self.axisNum == 1: 
                SlowOntimeFwd[self.badIdx] =  model.motorOntimeSlowFwd1[self.badIdx]
                SlowOntimeRev[self.badIdx] =  model.motorOntimeSlowRev1[self.badIdx]

                OntimeFwd[self.badIdx] = model.motorOntimeFwd1[self.badIdx]
                OntimeRev[self.badIdx] = model.motorOntimeRev1[self.badIdx]
            else:
                SlowOntimeFwd[self.badIdx] =  model.motorOntimeSlowFwd2[self.badIdx]
                SlowOntimeRev[self.badIdx] =  model.motorOntimeSlowRev2[self.badIdx]

                OntimeFwd[self.badIdx] = model.motorOntimeFwd2[self.badIdx]
                OntimeRev[self.badIdx] = model.motorOntimeRev2[self.badIdx]

        if table is not None:
            pid=range(len(OntimeFwd)) 
            t=Table([pid, self.fwdOntimeSlowModel(model), self.fwd_int, self.fwd_slope, SlowOntimeFwd,
                self.revOntimeSlowModel(model),self.rev_int, self.rev_slope, SlowOntimeRev],
                names=('Fiber No','Ori Fwd OT', 'FWD int', 'FWD slope', 'New Fwd OT',
                'Ori Rev OT', 'REV int', 'REV slope', 'New Rev OT'),
                dtype=('i2','f4', 'f4', 'f4','f4', 'f4', 'f4', 'f4', 'f4'))
            t.write(table,format='ascii.ecsv',overwrite=True, 
                formats={'Fiber No':'%i','Ori Fwd OT': '%10.5f', 'FWD int': '%10.5f', 'FWD slope': '%10.5f', 'New Fwd OT': '%10.5f',\
                'Ori Rev OT': '%10.5f', 'REV int': '%10.5f', 'REV slope': '%10.5f', 'New Rev OT': '%10.5f'})

        if self.axisNum == 1:
            model.updateOntimes(thtFwd=SlowOntimeFwd, thtRev=SlowOntimeRev, fast=False)
            model.updateOntimes(thtFwd=OntimeFwd, thtRev=OntimeRev, fast=True)
        else:
            model.updateOntimes(phiFwd=SlowOntimeFwd, phiRev=SlowOntimeRev, fast=False)
            model.updateOntimes(phiFwd=OntimeFwd, phiRev=OntimeRev, fast=True)

        model.createCalibrationFile(newXML)

        pass

    def visMaps(self, direction, filename=None, pngfile= None, predict = True):
        if filename is not None:
            output_file(filename)

        parray=[]
        for f in self.goodIdx:
           
            p=self.visBestOntime(f,direction, predict=predict)
            parray.append(p)

        grid = gridplot(parray, ncols=3, plot_width=400, plot_height=400)

        
        if pngfile is not None:
            export_png(grid, filename=pngfile)
        
        show(grid)
        

def exploreModuleOntime(arm=None, 
    brokens=None, iteration=4, XML=None, stepsize=250, repeat=3):
    
    logger = logging.getLogger('ontimeOptimize')
    logger.info(f'Starting to run on-time optimization')

    dataPath = '/data/MCS/'
    
    if brokens is None:
        brokens = []

    # Define the beginning point for on-time searching
    thetaOnTimeMax = 0.065    
    phiOnTimeMax = 0.05

    datalist =[]

    for itr in range(iteration):
        #currentpath = dataPath+f'run{itr}/'
        #if not (os.path.exists(currentpath)):
        #    os.mkdir(currentpath)
        outXML = f'temp.xml'
        curXML = f'{arm}_run{itr}.xml'

        logger.info(f'Output XML = {dataPath}{outXML}')

        #fwdhtml = currentpath+f'{arm}_fwd{itr}.html'
        #revhtml = currentpath+f'{arm}_rev{itr}.html'
        
        #fwdpng = currentpath+f'{arm}_fwd{itr}.png'
        #revpng = currentpath+f'{arm}_rev{itr}.png'
        
        #datalist.append(currentpath)
        
        if itr == 0:
            
            thetaOntime = [np.full(57, thetaOnTimeMax)] * 2 
            phiOntime =  [np.full(57, phiOnTimeMax)] * 2 
            xml = XML
            
            
            mt = ModuleTest('fpga', xml, brokens=brokens)
            mt._connect()
            if arm is 'phi':
                #mt.pfi.moveAllSteps(mt.allCobras, 0, -5000)
                #mt.pfi.moveAllSteps(mt.allCobras, 0, -1000)
                logger.info(f'Running {itr+1} iteration')
                currentpath = mt.makePhiMotorMap(f'{curXML}',
                        phiOnTime=phiOntime, repeat = repeat, fast=False, 
                        steps = stepsize, totalSteps = 6000)

            else:
                mt.pfi.moveAllSteps(mt.allCobras, -10000, 0)
                mt.pfi.moveAllSteps(mt.allCobras, -2000, 0)

                currentpath = mt.makeThetaMotorMap(f'{curXML}',
                        thetaOnTime=thetaOntime, repeat = repeat, fast=False, 
                        steps = stepsize, totalSteps = 12000)
                
            curXML = XML
        else:
            
            xml = pathlib.Path(f'{dataPath}{outXML}')
            mt = ModuleTest('fpga', xml, brokens=brokens)
            mt._connect()

            if arm is 'phi':
                #mt.pfi.moveAllSteps(mt.allCobras, 0,-5000)
                #nt.pfi.moveAllSteps(mt.allCobras, 0, -1000)
                currentpath = mt.makePhiMotorMap(f'{curXML}',
                        repeat = repeat, fast=False, 
                        steps = stepsize, totalSteps = 6000)

            else:
                pfi.moveAllSteps(mt.allCobras, -10000, 0)
                pfi.moveAllSteps(mt.allCobras, -2000, 0)

                currentpath = mt.makeThetaMotorMap(f'{curXML}', 
                    repeat = repeat, fast=False,totalSteps = 12000, 
                    limitOnTime = 0.08, steps = stepsize)

            curXML = f'{currentpath}/output/{curXML}'
            logger.info(f'Current XML = {curXML}')
            
            vis = visDianosticPlot.VisDianosticPlot(f'{currentpath}/data/', brokens=brokens, camSplit=28)
            vis.visAngleMovement(figPath=f'{currentpath}/output/',
                arm=arm,pdffile=f'{currentpath}/output/AngleMove.pdf')
            
            del(vis)

        datalist.append(f'{currentpath}/')
        logger.info(f'{datalist}')
        print(datalist)
       
        if arm is 'phi':
            otm = ontimeOptimize.OntimeOptimize(brokens=brokens, phiList = datalist)
        else:
            otm = ontimeOptimize.OntimeOptimize(brokens=brokens, thetaList = datalist)
        otm.solveForSlowSpeed()
        
        otm.updateXML(curXML,f'{dataPath}{outXML}')
        logger.info(f'Updating XML = {dataPath}{outXML}')

        fwdhtml = f'{currentpath}/output/{arm}_fwd{itr}.html'
        revhtml = f'{currentpath}/output/{arm}_rev{itr}.html'
        
        otm.visMaps('Fwd',filename=f'{fwdhtml}',)
        otm.visMaps('Rev',filename=f'{revhtml}')

        
        del(mt)
    
    logger.info(f'Starting to work on optimization based on all data')
    # Make the plot at the very end.
    if arm is 'phi':
        otm = ontimeOptimize.OntimeOptimize(brokens=brokens, phiList = datalist)
    else:
        otm = ontimeOptimize.OntimeOptimize(brokens=brokens, thetaList = datalist)
    
    otm.pickForSlowSpeed()

    fwdhtml = f'{currentpath}/output/{arm}_fwd{itr}.html'
    revhtml = f'{currentpath}/output/{arm}rev{itr}.html'
    fwdpng = f'{currentpath}/output/{arm}_fwd{itr}.png'
    revpng = f'{currentpath}/output/{arm}_fwd{itr}.png'  
        #fwdpng = currentpath+f'{arm}_fwd{itr}.png'
        #revpng = currentpath+f'{arm}_rev{itr}.png'
    
    
    # Update XML with best on-time
    logger.info(f'Producing final XML = {currentpath}/output/{arm}_final.xml')

    otm.updateXML(curXML,f'{currentpath}/output/{arm}_final.xml', solve=False)
    otm.visMaps('Fwd',filename=f'{fwdhtml}',pngfile=f'{fwdpng}',predict=False)
    otm.visMaps('Rev',filename=f'{revhtml}',pngfile=f'{revpng}',predict=False)

    # Using the last XML to run motor map
    xml = pathlib.Path(f'{currentpath}/output/{arm}_final.xml')
    mt = ModuleTest('fpga', xml, brokens=brokens)
    mt._connect()
    if arm is 'phi':
        path = mt.makePhiMotorMap(f'{arm}_250step.xml', 
            repeat = 3, fast=False,totalSteps = 6000, limitOnTime = 0.08, steps = 250)
    else:
        path = mt.makeThetaMotorMap(f'{arm}_250step.xml', 
            repeat = 3, fast=False,totalSteps = 12000, limitOnTime = 0.08, steps = 250)    
    
    vis = visDianosticPlot.VisDianosticPlot(f'{path}/data/', brokens=brokens, camSplit=28)
    vis.visAngleMovement(figPath=f'{path}/output/',
                 arm=arm,pdffile=f'{path}/output/AngleMove.pdf')
            
    del(vis)

    return datalist
    # currentpath = dataPath+'finalMM/'
    # if not (os.path.exists(currentpath)):
    #     os.mkdir(currentpath)
    # curXML='final.xml'
    # mt = ModuleTest(f'{fpgaHost}', 
    #             f'{dataPath}{outXML}', brokens=brokens,camSplit=28)

    # if arm is 'phi':
    #     pfi.moveAllSteps(mt.allCobras, 0,-5000)
    #     pfi.moveAllSteps(mt.allCobras, 0, -1000)
    #     mt.makePhiMotorMap(f'{curXML}',f'{currentpath}', 
    #         repeat = repeat, fast=False,totalSteps = 6000, limitOnTime = 0.08, steps = stepsize)

    # else:
    #     pfi.moveAllSteps(mt.allCobras, -10000, 0)
    #     pfi.moveAllSteps(mt.allCobras, -2000, 0)

    #     mt.makeThetaMotorMap(f'{curXML}',f'{currentpath}', 
    #         repeat = repeat, fast=False,totalSteps = 12000, limitOnTime = 0.08, steps = stepsize)

    # vis = visDianosticPlot.VisDianosticPlot(currentpath, brokens=brokens, camSplit=28)
    # vis.visAngleMovement(figPath=f'{currentpath}',
    #             arm=arm,pdffile=f'{currentpath}AngleMove.pdf')
            
    # del(vis)
