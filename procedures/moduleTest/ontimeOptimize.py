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
        self.minRange = 270
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
            pass
        if (phiList is None) and (thetaList is None):
            raise Exception
        
    #@classmethod
    def loadFromPhiData(self, runDirs):
        #self = cls(runDirs)
        self.axisNum = 2
        self.axisName = "Phi"
        self.maxOntime = 0.08
        self.slowTarget = 0.075
        self.fastTarget = 2.0*self.slowTarget
        self._buildDataFramefromData()

       # return self

    #@classmethod
    def loadFromThetaData(self, thetaList):
        #self = cls(thetaList)
        self.axisNum = 1
        self.axisName = "Theta"
        self.maxOntime = 0.09
        self.slowTarget = 0.05
        self.fastTarget = 2.0*self.slowTarget
        self._buildDataFramefromData()

        #return self
    
    def getSlope(self, pid, direction):
        dataframe = self.dataframe
        loc1 = dataframe.loc[dataframe['fiberNo'] == pid].loc
        loc2 = loc1[dataframe[direction].abs() > self.minSpeed].loc
        ndf = loc2[dataframe[f'range{direction}'].abs() > self.minRange]
        
        if (len(ndf[f'onTime{direction}'].values) >= 2):

            onTimeArray = ndf[f'onTime{direction}'].values
            angSpdArray = ndf[(f'{direction}')].values

            slope, intercept, r_value, p_value, std_err = stats.linregress(onTimeArray,angSpdArray)

        else:
            slope = np.nan
            intercept = np.nan

        if direction == 'Fwd' and slope <= 0:
            slope = np.nan
            intercept = np.nan
        
        if direction == 'Rev' and slope >= 0:
            slope = np.nan
            intercept = np.nan

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
        

    def solveForSpeed(self, targetSpeed=None):
        fwd_target = np.full(len(self.fwd_int), targetSpeed)
        rev_target = np.full(len(self.fwd_int), -targetSpeed)
        
        newOntimeFwd = []
        for idx,item in enumerate(fwd_target): 
            if np.isnan(self.fwd_int[idx]) and np.isnan(self.fwd_slope[idx]):
                

        newOntimeFwd = (fwd_target - self.fwd_int)/self.fwd_slope
        newOntimeRev = (rev_target - self.rev_int)/self.rev_slope

        fastOnes = np.where((newOntimeFwd < self.minOntime) | (newOntimeRev < self.minOntime))
        if len(fastOnes[0]) > 0:
            logging.warn(f'some motors too fast: {fastOnes[0]}: '
                         f'fwd:{newOntimeFwd[fastOnes]} rev:{newOntimeRev[fastOnes]}')
        
        
        slowOnes = np.where((newOntimeFwd > self.maxOntime) | (newOntimeRev > self.maxOntime))
        if len(slowOnes[0]) > 0:
            logging.warn(f'some motors too slow {slowOnes[0]}: '
                         f'fwd:{newOntimeFwd[slowOnes]} rev:{newOntimeRev[slowOnes]}')

        return newOntimeFwd, newOntimeRev


    def plotBestThetaOnTimeFwd(self, DataFrame, OntimeModel, fiberNo):
        TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']

        title_string=f"Fiber {fiberNo+1} Theta Fwd"
        p = figure(tools=TOOLS, x_range=[0,60], y_range=[0,0.4],
                plot_height=400, plot_width=500,title=title_string)
        p.xaxis.axis_label = 'On Time'
        p.yaxis.axis_label = 'Speed'
        dd=df.loc[df['fiberNo'] == fiberNo+1]

        newOntimeFwd1 = (0.05-otm.j1fwd_int[fiberNo])/otm.j1fwd_slope[fiberNo]

        #print(newOntimeFwd1)
        x = range(5,70)
        y_predicted = [otm.j1fwd_slope[fiberNo]*i + otm.j1fwd_int[fiberNo]  for i in x]
        p.circle(x=newOntimeFwd1,y=[0.05],color='red', fill_color=None, radius=1.0)
        #p.circle(x=calibModel.motorOntimeSlowFwd2[fiberNo-1]*1000.0,y=speed_fwd[fiberNo-1]*180/math.pi,
        #color='blue', fill_color=None, radius=1.0)

        p.circle(x=dd['J1onTimeFwd'],y=dd['J1_fwd'])
        p.line(x,y_predicted,color='red')
        
        return p

    def plotBestThetaOnTimeRev(self, DataFrame, OntimeModel, fiberNo):
        TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']

        title_string=f"Fiber {fiberNo} Theta Rev"
        p = figure(tools=TOOLS, x_range=[0,60], y_range=[-0.4,0],
                plot_height=400, plot_width=500,title=title_string)
        p.xaxis.axis_label = 'On Time'
        p.yaxis.axis_label = 'Speed'
        dd=df.loc[df['fiberNo'] == fiberNo+1]

        newOntimeRev1 = (-0.05-otm.j1rev_int[fiberNo])/otm.j1rev_slope[fiberNo]


        x = range(5,70)
        y_predicted = [otm.j1rev_slope[fiberNo]*i + otm.j1rev_int[fiberNo]  for i in x]
        p.circle(x=newOntimeRev1,y=[-0.05],color='red', fill_color=None, radius=1.0)
        #p.circle(x=calibModel.motorOntimeSlowRev1[fiberNo-1]*1000.0,y=-speed_rev[fiberNo-1]*180/math.pi,
        #         color='blue', fill_color=None, radius=1.0)

        p.circle(x=dd['J1onTimeRev'],y=dd['J1_rev'])
        p.line(x,y_predicted,color='red')
        
        return p


