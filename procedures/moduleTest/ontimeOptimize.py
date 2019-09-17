import logging

import sys, os
import numpy as np
import glob
import pandas as pd
import math

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
from bokeh.transform import linear_cmap

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
        self.minRange = 130
        self._buildDataFramefromData()

       # return self

    #@classmethod
    def loadFromThetaData(self, thetaList):
        #self = cls(thetaList)
        self.nPoints = len(thetaList)
        self.axisNum = 1
        self.axisName = "Theta"
        self.maxOntime = 0.08
        self.slowTarget = 0.05
        self.fastTarget = 2.0*self.slowTarget
        self.minRange = 360
        self._buildDataFramefromData()

        #return self
    
    def getSlope(self, pid, direction):
        dataframe = self.dataframe
        loc1 = dataframe.loc[dataframe['fiberNo'] == pid].loc
        loc2 = loc1[dataframe[direction].abs() > self.minSpeed].loc
        ndf = loc2[dataframe[f'range{direction}'].abs() > self.minRange]
        
        # When nPoint=1, using simple ratio to determine next point.
        if self.nPoints == 1:
            if (len(ndf[f'onTime{direction}'].values) == 0):
                loc1 = dataframe.loc[dataframe['fiberNo'] == pid].loc
                ndf = loc1[dataframe[direction].abs() > self.minSpeed]
            
     
            slope = ndf[f'{direction}'].values/ndf[f'onTime{direction}'].values
            intercept = 0
        else:

            if (len(ndf[f'onTime{direction}'].values) <= 1):
                loc1 = dataframe.loc[dataframe['fiberNo'] == pid].loc
                ndf = loc1[dataframe[direction].abs() > self.minSpeed]
        
            if len(ndf[f'onTime{direction}'].values) > 1:

                onTimeArray = ndf[f'onTime{direction}'].values
                angSpdArray = ndf[(f'{direction}')].values

                slope, intercept, r_value, p_value, std_err = stats.linregress(onTimeArray,angSpdArray)
            
            else:
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
            fw_file=f'{i}'+f'{self.axisName}SpeedFW.npy'
            rv_file=f'{i}'+f'{self.axisName}SpeedRV.npy'
            af = np.load(f'{i}' + f'{self.axisName}AngFW.npy')
            ar = np.load(f'{i}' + f'{self.axisName}AngRV.npy')
            fwdmm = np.rad2deg(np.load(f'{i}' + f'{self.axisName}MMFW.npy'))
            revmm = np.rad2deg(np.load(f'{i}' + f'{self.axisName}MMRV.npy'))
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

            xml = glob.glob(f'{i}'+'*.xml')[0]
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
            if 0 in self.dataframe.loc[self.dataframe['fiberNo'] == pid]['groupFwd'].tolist():
                
                fw_s, fw_i = self.getFwdSlope(pid)
                fwd_slope[pid-1] = fw_s
                fwd_int[pid-1] = fw_i
            if 0 in self.dataframe.loc[self.dataframe['fiberNo'] == pid]['groupRev'].tolist():
                rv_s, rv_i = self.getRevSlope(pid)
                rev_slope[pid-1] = rv_s
                rev_int[pid-1] = rv_i
            

    def _solveForSpeed(self, targetSpeed=None):
        fwd_target = np.full(len(self.fwd_int), targetSpeed)
        rev_target = np.full(len(self.fwd_int), -targetSpeed)
        
        newOntimeFwd = (fwd_target - self.fwd_int)/self.fwd_slope
        newOntimeRev = (rev_target - self.rev_int)/self.rev_slope

        # New, dealing with inf values.
        inx = np.where(np.isinf(newOntimeFwd))[0].tolist()
        for i in inx:
            if i not in self.badIdx:
                # Make sure if there is any good on-time value in this collection
                ndf = self.dataframe.loc[self.dataframe.fiberNo == i+1].loc[self.dataframe['Fwd'].abs() > self.minSpeed]
                ind=np.argmin(np.abs((ndf.Fwd -  targetSpeed).values))
                if np.abs(ndf['Fwd'].values[ind]-targetSpeed) < 0.01:
                    newOntimeFwd[i] = ndf['onTimeFwd'].values[ind]
                else:
                    if ndf['Fwd'].values[ind] > targetSpeed:
                        newOntimeFwd[i] = ndf.onTimeFwd.values[ind] - 0.005
                    else:
                        newOntimeFwd[i] = ndf.onTimeFwd.values[ind] + 0.005
            else:
                newOntimeFwd[i] = self.maxOntime

        inx = np.where(np.isinf(newOntimeRev))[0].tolist()
        for i in inx:
            if i not in self.badIdx:
                ndf = self.dataframe.loc[self.dataframe.fiberNo == i+1].loc[self.dataframe['Rev'].abs() > self.minSpeed]
                ind=np.argmin(np.abs((ndf.Rev +  targetSpeed).values))
                if (np.abs(ndf['Rev'].values[ind]+targetSpeed)) < 0.01:
                    newOntimeRev[i] = ndf['onTimeRev'].values[ind]
                else:
                    if ndf['Rev'].values[ind] > -targetSpeed:
                        newOntimeRev[i] = ndf.onTimeRev.values[ind] - 0.005
                    else:
                        newOntimeRev[i] = ndf.onTimeRev.values[ind] + 0.005 
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
        if len(slowOnes[0]) > 0: newOntimeFwd[slowOnes] = 0.02
        
        slowOnes = np.where((newOntimeRev < 0))
        if len(slowOnes[0]) > 0: newOntimeRev[slowOnes] = 0.02

        return newOntimeFwd, newOntimeRev

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

    def visBestOntime(self, fiberInx, direction):
        dd=self.dataframe.loc[self.dataframe['fiberNo'] == fiberInx+1]

        
        TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']
        title_string=f"Fiber {fiberInx+1} Theta {direction}"
        
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
        
       

        if direction is 'Fwd':
            xrange=np.arange(self.newOntimeSlowFwd[fiberInx]*0.8,1.2*np.max(dd[f'onTime{direction}']),0.001)
            newOntime = self.newOntimeSlowFwd[fiberInx] 
            y_predicted = [self.fwd_slope[fiberInx]*i + self.fwd_int[fiberInx]  for i in xrange]
            p.circle(x=[newOntime],y=[self.slowTarget],color='red', size=15)
        
    
        if direction is 'Rev':
            xrange=np.arange(self.newOntimeSlowRev[fiberInx]*0.8,1.2*np.max(dd[f'onTime{direction}']),0.001)

            newOntime = self.newOntimeSlowRev[fiberInx]
            y_predicted = [self.rev_slope[fiberInx]*i + self.rev_int[fiberInx]  for i in xrange]
            p.circle(x=[newOntime],y=[-self.slowTarget],color='red', size=15)
        
        p.line(xrange,y_predicted,color='red')
        p.circle(x=gooddata[f'onTime{direction}'],y=gooddata[direction])
        p.circle(x=dd[f'onTime{direction}'],y=dd[f'{direction}'], size=7, color='red',fill_color=None)
        p.segment(dd[f'onTime{direction}'], dd[f'max{direction}'], dd[f'onTime{direction}'], dd[f'min{direction}'], color="black")
        return p

    def updateXML(self, initXML, newXML, table=None):
        
        model = pfiDesign.PFIDesign(initXML)

        SlowOntimeFwd, SlowOntimeRev = self.solveForSlowSpeed()
        OntimeFwd, OntimeRev = self.solveForFastSpeed()
        
        # If there is a broken fiber, set on-time to original value 
        SlowOntimeFwd[self.badIdx] = self.fwdOntimeSlowModel(model)[self.badIdx]
        SlowOntimeRev[self.badIdx] = self.revOntimeSlowModel(model)[self.badIdx]

        OntimeFwd[self.badIdx] = self.fwdOntimeModel(model)[self.badIdx]
        OntimeRev[self.badIdx] = self.revOntimeModel(model)[self.badIdx]

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

    def visMaps(self, direction, filename=None, pngfile= None):
        if filename is not None:
            output_file(filename)

        parray=[]
        for f in self.goodIdx:
           
            p=self.visBestOntime(f,direction)
            parray.append(p)

        grid = gridplot(parray, ncols=3, plot_width=400, plot_height=400)

        
        if pngfile is not None:
            export_png(grid, filename=pngfile)
        
        show(grid)
        