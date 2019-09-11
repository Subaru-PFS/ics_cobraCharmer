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
        self.axisNum = 1
        self.axisName = "Theta"
        self.maxOntime = 0.08
        self.slowTarget = 0.05
        self.fastTarget = 2.0*self.slowTarget
        self.minRange = 270
        self._buildDataFramefromData()

        #return self
    
    def getSlope(self, pid, direction):
        dataframe = self.dataframe
        loc1 = dataframe.loc[dataframe['fiberNo'] == pid].loc
        loc2 = loc1[dataframe[direction].abs() > self.minSpeed].loc
        ndf = loc2[dataframe[f'range{direction}'].abs() > self.minRange]
        
        # If there is any exception, assuming the slope to be 1

        if (len(ndf[f'onTime{direction}'].values) >= 2):

            onTimeArray = ndf[f'onTime{direction}'].values
            angSpdArray = ndf[(f'{direction}')].values

            slope, intercept, r_value, p_value, std_err = stats.linregress(onTimeArray,angSpdArray)

        else:
            slope = 1.0
            intercept = 0

        if direction == 'Fwd' and slope <= 0:
            slope = 1.0
            intercept = np.nan
        
        if direction == 'Rev' and slope >= 0:
            slope = -1.0
            intercept = 0

        return slope,intercept
    
    def getFwdSlope(self, pid):
        return self.getSlope(pid, 'Fwd')

    def getRevSlope(self, pid):
        return self.getSlope(pid, 'Rev')

    def fwdOntimeModel(self, model):
        if self.axisNum == 1:
            return model.motorOntimeSlowFwd1[self.goodIdx]
        else:
            return model.motorOntimeSlowFwd2[self.goodIdx]

    def revOntimeModel(self, model):
        if self.axisNum == 1:
            return model.motorOntimeSlowRev1[self.goodIdx]
        else:
            return model.motorOntimeSlowRev2[self.goodIdx]
   
    def _buildDataFramefromData(self):
        pidarray=[]
        otfarray=[]
        otrarray=[]
        fwarray=[]
        rvarray=[]
        dafarray=[]
        dararray=[]
        #imageSet = self.datalist

        for i in self.datalist:
            fw_file=f'{i}'+f'/{self.axisName}SpeedFW.npy'
            rv_file=f'{i}'+f'/{self.axisName}SpeedRV.npy'
            af = np.load(f'{i}' + f'/{self.axisName}AngFW.npy')
            ar = np.load(f'{i}' + f'/{self.axisName}AngRV.npy')
            fwd=np.load(fw_file)*180.0/math.pi 
            rev=-np.load(rv_file)*180.0/math.pi
            
            daf=[]
            dar=[]
            for p in self.goodIdx:
                daf.append((af[p,0,-1]-af[p,0,0])*180/3.14159)
                dar.append((ar[p,0,-1]-ar[p,0,0])*180/3.14159)


            xml = glob.glob(f'{i}'+'/*.xml')[0]
            model = pfiDesign.PFIDesign(xml)
            
            ontimeFwd = self.fwdOntimeModel(model)
            ontimeRev = self.revOntimeModel(model)

            pid=np.array(self.visibles)
            pidarray.append(pid)
            otfarray.append(ontimeFwd)
            otrarray.append(ontimeRev)
            dafarray.append(daf)
            dararray.append(dar)
            fwarray.append(fwd[self.goodIdx])
            rvarray.append(rev[self.goodIdx])
            
        d={'fiberNo': np.array(pidarray).flatten(),
           'onTimeFwd': np.array(otfarray).flatten(),
           'onTimeRev': np.array(otrarray).flatten(),
           'rangeFwd':np.array(dafarray).flatten(),
           'rangeRev':np.array(dararray).flatten(),           
           'Fwd': np.array(fwarray).flatten(),
           'Rev': np.array(rvarray).flatten()}
        
        self.dataframe = pd.DataFrame(d)

    def _buildModel(self):

        self.fwd_slope = fwd_slope = np.zeros(57)
        self.fwd_int = fwd_int = np.zeros(57)

        self.rev_slope = rev_slope = np.zeros(57)
        self.rev_int = rev_int = np.zeros(57)

        for pid in self.visibles:
            fw_s, fw_i = self.getFwdSlope(pid)
            
            rv_s, rv_i = self.getRevSlope(pid)
            #if fw_s <= 0 or rv_s >= 0:
            #    breakpoint()

            fwd_slope[pid-1] = fw_s
            fwd_int[pid-1] = fw_i

            rev_slope[pid-1] = rv_s
            rev_int[pid-1] = rv_i
        

    def _solveForSpeed(self, targetSpeed=None):
        fwd_target = np.full(len(self.fwd_int), targetSpeed)
        rev_target = np.full(len(self.fwd_int), -targetSpeed)
        
        newOntimeFwd = (fwd_target - self.fwd_int)/self.fwd_slope
        newOntimeRev = (rev_target - self.rev_int)/self.rev_slope

        fastOnes = np.where((newOntimeFwd > self.maxOntime) | (newOntimeRev > self.maxOntime))
        if len(fastOnes[0]) > 0:
            logging.warn(f'some motors too fast: {fastOnes[0]}: '
                         f'fwd:{newOntimeFwd[fastOnes]} rev:{newOntimeRev[fastOnes]}')

            ind = np.where(newOntimeFwd > self.maxOntime)
            newOntimeFwd[ind] = self.maxOntime
            ind = np.where(newOntimeRev > self.maxOntime)
            newOntimeRev[ind] = self.maxOntime

        slowOnes = np.where((newOntimeFwd < self.minOntime) | (newOntimeRev < self.minOntime))
        if len(slowOnes[0]) > 0:
            logging.warn(f'some motors too slow {slowOnes[0]}: '
                         f'fwd:{newOntimeFwd[slowOnes]} rev:{newOntimeRev[slowOnes]}')

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

    def visBestOntime(self, fiberNo, direction):
        dd=self.dataframe.loc[self.dataframe['fiberNo'] == fiberNo+1]

        


        TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']
        title_string=f"Fiber {fiberNo+1} Theta {direction}"
        
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
        
        xrange = (np.arange(100)/2000) + np.min(gooddata[f'onTime{direction}'])
        #xrange = range(np.min(gooddata[f'onTime{direction}']).astype(int),
        #            np.max(gooddata[f'onTime{direction}']).astype(int))

        if direction is 'Fwd':
            newOntime = (self.slowTarget - self.fwd_int[fiberNo])/self.fwd_slope[fiberNo]
            y_predicted = [self.fwd_slope[fiberNo]*i + self.fwd_int[fiberNo]  for i in xrange]
            p.circle(x=[newOntime],y=[self.slowTarget],color='red', size=15)

        if direction is 'Rev':
            newOntime = (-self.slowTarget - self.rev_int[fiberNo])/self.rev_slope[fiberNo]
            y_predicted = [self.rev_slope[fiberNo]*i + self.rev_int[fiberNo]  for i in xrange]
            p.circle(x=[newOntime],y=[-self.slowTarget],color='red', size=15)
        #breakpoint()
        #p.circle(x=[newOntime],y=[self.slowTarget],color='red', size=15)
        p.line(xrange,y_predicted,color='red')
        p.circle(x=gooddata[f'onTime{direction}'],y=gooddata[direction])
        p.circle(x=dd[f'onTime{direction}'],y=dd[f'{direction}'], size=7, color='red',fill_color=None)
        
        return p

    def updateXML(self, initXML, newXML, table=None):
        
        model = pfiDesign.PFIDesign(initXML)

        SlowOntimeFwd, SlowOntimeRev = self.solveForSlowSpeed()
        OntimeFwd, OntimeRev = self.solveForFastSpeed()

        if table is not None:
            pid=range(len(OntimeFwd))
            t=Table([pid, self.fwdOntimeModel(model), self.fwd_int, self.fwd_slope, SlowOntimeFwd,
                self.revOntimeModel(model),self.rev_int, self.rev_slope, SlowOntimeRev],
                names=('Fiber No','Ori Fwd OT', 'FWD int', 'FWD slope', 'New Fwd OT',
                'Ori Rev OT', 'REV int', 'REV slope', 'New Rev OT'),
                dtype=('i2','f4', 'f4', 'f4','f4', 'f4', 'f4', 'f4', 'f4'))
            t.write(table,format='ascii.ecsv',overwrite=True, 
                formats={'Fiber No':'%i','Ori Fwd OT': '%10.5f', 'FWD int': '%10.5f', 'FWD slope': '%10.5f', 'New Fwd OT': '%10.5f',\
                'Ori Rev OT': '%10.5f', 'REV int': '%10.5f', 'REV slope': '%10.5f', 'New Rev OT': '%10.5f'})


        model.updateOntimes(thtFwd=SlowOntimeFwd, thtRev=SlowOntimeRev, fast=False)
        model.updateOntimes(thtFwd=OntimeFwd, thtRev=OntimeRev, fast=True)

        model.createCalibrationFile(newXML)

        pass

    def visMaps(self, direction, filename=None):
        if filename is not None:
            output_file(filename)

        parray=[]
        for f in self.goodIdx:
            p=self.visBestOntime(f,direction)
            parray.append(p)

        grid = gridplot(parray, ncols=3, plot_width=400, plot_height=400)
        show(grid)
        