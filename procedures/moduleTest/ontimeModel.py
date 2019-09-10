import logging

import numpy as np
import pandas as pd
from scipy import stats

from ics.cobraCharmer import pfiDesign
from ics.cobraCharmer import cobraState

class ontimeModel():

    def __init__(self, runDirs):
        self.runDirs = runDirs
        self.model = None

        self.minSpeed = 0.001
        self.minOntime = 0.002

        self.newOntimeSlowFwd = self.newOntimeSlowRev = None
        self.newOntimeFastFwd = self.newOntimeFastRev = None

    @property
    def nCobras(self):
        return len(self.dataframe)//len(self.runDirs)

    @classmethod
    def loadFromPhiData(cls, runDirs):
        self = cls(runDirs)
        self.axisNum = 2
        self.axisName = "Phi"
        self.maxOntime = 0.08
        self.slowTarget = 0.09
        self.fastTarget = 2.0*self.slowTarget
        self._buildModel()

        return self

    @classmethod
    def loadFromThetaData(cls, runDirs):
        self = cls(runDirs)
        self.axisNum = 1
        self.axisName = "Theta"
        self.maxOntime = 0.09
        self.slowTarget = 0.07
        self.fastTarget = 2.0*self.slowTarget
        self._buildModel()

        return self

    def getSlope(self, pid, direction):
        dataframe = self.dataframe
        loc = dataframe.loc[dataframe['fiberNo'] == pid].loc
        ndf = loc[dataframe[direction].abs() > self.minSpeed]

        onTimeArray = ndf[f'onTime{direction}'].values
        angSpdArray = ndf[(f'{direction}')].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(onTimeArray,angSpdArray)

        # If the slope is nan, that means the linear regression failed.  Return zero instead of nan.
        # Really? -- CPL
        if np.isnan(slope):
            slope = 0

        if direction == 'Fwd' and slope <= 0:
            breakpoint()

        return slope,intercept

    def solveForSpeed(self, targetSpeed=None):
        fwd_target = np.full(len(self.fwd_int), targetSpeed)
        rev_target = np.full(len(self.fwd_int), -targetSpeed)

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

    def solveForSlowSpeed(self, targetSpeed=None):
        if targetSpeed is None:
            targetSpeed = self.slowTarget
        newMaps = self.solveForSpeed(targetSpeed=targetSpeed)
        self.newOntimeSlowFwd, self.newOntimeSlowRev = newMaps

        return newMaps

    def solveForFastSpeed(self, targetSpeed=None):
        if targetSpeed is None:
            targetSpeed = self.fastTarget
        newMaps = self.solveForSpeed(targetSpeed=targetSpeed)
        self.newOntimeFwd, self.newOntimeRev = newMaps

        return newMaps

    def saveOptMap(self, mapPath):
        if (self.newOntimeSlowFwd is None or self.newOntimeSlowRev is None or
            self.newOntimeFwd is None or self.newOntimeRev is None):

            raise RuntimeError('for now, will only write all models with all four new maps')

        motor = self.axisName.lower()
        updateArgs = dict(fast=True)
        updateArgs[f'{motor}Fwd'] = self.newOntimeFwd
        updateArgs[f'{motor}Rev'] = self.newOntimeRev
        self.model.updateOntimes(**updateArgs)

        updateArgs = dict(fast=False)
        updateArgs[f'{motor}Fwd'] = self.newOntimeSlowFwd
        updateArgs[f'{motor}Rev'] = self.newOntimeSlowRev
        self.model.updateOntimes(**updateArgs)

        self.model.createCalibrationFile(mapPath)

    def getFwdSlope(self, pid):
        return self.getSlope(pid, 'Fwd')

    def getRevSlope(self, pid):
        return self.getSlope(pid, 'Rev')

    def fwdSpeedFile(self, runDir):
        return runDir / 'data' / f'{self.axisName}SpeedFW.npy'

    def revSpeedFile(self, runDir):
        return runDir / 'data' / f'{self.axisName}SpeedRV.npy'

    def fwdOntimeModel(self, model):
        if self.axisNum == 1:
            return model.motorOntimeSlowFwd1
        else:
            return model.motorOntimeSlowFwd2

    def revOntimeModel(self, model):
        if self.axisNum == 1:
            return model.motorOntimeSlowRev1
        else:
            return model.motorOntimeSlowRev2

    def loadData(self):
        pidarray=[]
        otfarray=[]
        otrarray=[]
        fwdarray=[]
        revarray=[]

        for runDir in self.runDirs:
            fwd = np.rad2deg(np.load(self.fwdSpeedFile(runDir)))
            rev = -np.rad2deg(np.load(self.revSpeedFile(runDir)))

            xml = list((runDir / 'output').glob('*.xml'))[0]
            model = pfiDesign.PFIDesign(xml)
            if self.model is None:
                self.model = model

            # Get from model
            try:
                pid = self.visibleFibers
            except AttributeError:
                pid = self.visibleFibers = np.arange(model.nCobras)

            ontimeFwd = self.fwdOntimeModel(model)
            ontimeRev = self.revOntimeModel(model)

            pidarray.append(pid)
            otfarray.append(ontimeFwd)
            otrarray.append(ontimeRev)

            fwdarray.append(fwd)
            revarray.append(rev)

        d={'fiberNo': np.array(pidarray).flatten(),
           'onTimeFwd': np.array(otfarray).flatten(),
           'onTimeRev': np.array(otrarray).flatten(),
           'Fwd': np.array(fwdarray).flatten(),
           'Rev': np.array(revarray).flatten()}
        self.dataframe = pd.DataFrame(d)

    def _buildModel(self):
        self.loadData()

        self.fwd_slope = fwd_slope = np.zeros(self.nCobras)
        self.fwd_int = fwd_int = np.zeros(self.nCobras)

        self.rev_slope = rev_slope = np.zeros(self.nCobras)
        self.rev_int = rev_int = np.zeros(self.nCobras)

        for pid in self.visibleFibers:
            fw_s, fw_i = self.getFwdSlope(pid)
            rv_s, rv_i = self.getRevSlope(pid)
            if fw_s <= 0 or rv_s >= 0:
                breakpoint()

            fwd_slope[pid-1] = fw_s
            fwd_int[pid-1] = fw_i

            rev_slope[pid-1] = rv_s
            rev_int[pid-1] = rv_i
