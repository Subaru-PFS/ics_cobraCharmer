import logging

import numpy as np
import pandas as pd
from scipy import stats

from ics.cobraCharmer import pfiDesign
from ics.cobraCharmer import cobraState
from ics.cobraCharmer.utils import butler

class ontimeModel():

    def __init__(self, modelPath, runDirs, logLevel=logging.INFO):
        self.runDirs = runDirs
        self.model = pfiDesign.PFIDesign(butler.mapForRun(modelPath))
        self.logger = logging.getLogger('ontimeModel')
        self.logger.setLevel(logLevel)

        self.minSpeed = 0.001
        self.minOntime = 0.002

        self.newOntimeSlowFwd = self.newOntimeSlowRev = None
        self.newOntimeFastFwd = self.newOntimeFastRev = None

    @property
    def nCobras(self):
        return len(self.dataframe)//len(self.runDirs)

    @classmethod
    def loadFromPhiData(cls, modelPath, runDirs):
        self = cls(modelPath, runDirs)
        self.axisNum = 2
        self.axisName = "Phi"
        self.maxOntime = 0.08
        self.slowTarget = 0.09
        self.minRange = np.deg2rad(182.0)
        self.fastTarget = 2.0*self.slowTarget
        self._buildModel()

        return self

    @classmethod
    def loadFromThetaData(cls, modelPath, runDirs):
        self = cls(modelPath, runDirs)
        self.axisNum = 1
        self.axisName = "Theta"
        self.maxOntime = 0.08
        self.slowTarget = 0.07
        self.fastTarget = 2.0*self.slowTarget
        self.minRange = np.deg2rad(375.0)
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

        return slope,intercept

    def solveForSpeed(self, targetSpeed=None):
        fwd_target = np.full(len(self.fwd_int), targetSpeed)
        rev_target = np.full(len(self.fwd_int), -targetSpeed)

        newOntimeFwd = (fwd_target - self.fwd_int)/self.fwd_slope
        newOntimeRev = (rev_target - self.rev_int)/self.rev_slope

        fastOnes = np.where((newOntimeFwd < self.minOntime) | (newOntimeRev < self.minOntime))
        if len(fastOnes[0]) > 0:
            self.logger.warn(f'some motors too fast: {fastOnes[0]}: '
                             f'fwd:{newOntimeFwd[fastOnes]} rev:{newOntimeRev[fastOnes]}')
        slowOnes = np.where((newOntimeFwd > self.maxOntime) | (newOntimeRev > self.maxOntime))
        if len(slowOnes[0]) > 0:
            self.logger.warn(f'some motors too slow {slowOnes[0]}: '
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

    def saveOptMap(self, mapPath, scaleBy=1.1):
        if (self.newOntimeSlowFwd is None or self.newOntimeSlowRev is None or
            self.newOntimeFwd is None or self.newOntimeRev is None):

            raise RuntimeError('for now, will only write all models with all four new maps')

        motor = self.axisName.lower()
        updateArgs = dict(fast=True)
        updateArgs[f'{motor}Fwd'] = np.clip(self.newOntimeFwd*scaleBy, self.maxOntime*0.75, self.maxOntime)
        updateArgs[f'{motor}Rev'] = np.clip(self.newOntimeRev*scaleBy, self.maxOntime*0.75, self.maxOntime)
        self.model.updateOntimes(**updateArgs)

        updateArgs = dict(fast=False)
        updateArgs[f'{motor}Fwd'] = np.clip(self.newOntimeSlowFwd*scaleBy, None, self.maxOntime)
        updateArgs[f'{motor}Rev'] = np.clip(self.newOntimeSlowRev*scaleBy, None, self.maxOntime)
        self.model.updateOntimes(**updateArgs)

        self.model.createCalibrationFile(mapPath)

    def getFwdSlope(self, pid):
        return self.getSlope(pid, 'Fwd')

    def getRevSlope(self, pid):
        return self.getSlope(pid, 'Rev')

    def fwdAngleFile(self, runDir):
        return runDir / 'data' / f'{self.axisName.lower()}AngFW.npy'

    def revAngleFile(self, runDir):
        return runDir / 'data' / f'{self.axisName.lower()}AngRV.npy'

    def fwdSpeedFile(self, runDir):
        return runDir / 'data' / f'{self.axisName.lower()}SpeedFW.npy'

    def revSpeedFile(self, runDir):
        return runDir / 'data' / f'{self.axisName.lower()}SpeedRV.npy'

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

    def getSlowestGoodOntimes(self, closeEnough=np.deg2rad(2)):
        """ Return the slowest ontimes for which each cobra completes its range.

        For each cobra, scan the ontime maps looking for the slowest one where the limits are reached.
        """

        nCobras = 57
        self.fwOntimes = np.zeros(nCobras, dtype='i4')
        self.rvOntimes = np.zeros(nCobras, dtype='i4')

        ontimes = sorted(self.fwAngles.keys())
        for c_i in range(len(self.fwAngles[ontimes[0]])):
            for ot in ontimes[::-1]:
                lastFwd = self.fwAngles[ot][c_i, 0, -2]
                firstRev = self.rvAngles[ot][c_i, 0, 0]
                lastRev = self.rvAngles[ot][c_i, 0, -2]
                firstFwd = self.fwAngles[ot][c_i, 0, 0]

                # Basically, we trust the limits as found at the start of the two sweeps.
                # So we check those against the angles at the _end_ of the other sweeps.
                fwdOK = np.abs(firstRev - lastFwd) < closeEnough
                if fwdOK:
                    self.fwOntimes[c_i] = ot

                revOK = np.abs(firstFwd - lastRev) < closeEnough
                if revOK:
                    self.rvOntimes[c_i] = ot

                self.logger.debug(f'{c_i:02d} {ot:0.2f}: '
                                  f'{np.rad2deg(lastFwd):0.2f} {np.rad2deg(firstRev):0.2f} {fwdOK}    '
                                  f'{np.rad2deg(lastRev):0.2f} {np.rad2deg(firstFwd):0.2f} {revOK}')

        slowFw_w = np.where(self.fwOntimes == 0)[0]
        if len(slowFw_w) > 0:
            self.logger.error(f'no valid forward speed for cobra ids {[i+1 for i in slowFw_w]}. '
                              f'Setting to {max(ontimes)}')
            self.fwOntimes[slowFw_w] = max(ontimes)
        slowRv_w = np.where(self.rvOntimes == 0)[0]
        if len(slowRv_w) > 0:
            self.logger.error(f'no valid reverse speed for cobra ids {[i+1 for i in slowRv_w]}. '
                              f'Setting to {max(ontimes)}')
            self.rvOntimes[slowRv_w] = max(ontimes)

        self.newOntimeSlowFwd = self.fwOntimes / 1000
        self.newOntimeSlowRev = self.rvOntimes / 1000
        self.newOntimeFwd = np.clip(self.newOntimeSlowFwd*2, None, self.maxOntime)
        self.newOntimeRev = np.clip(self.newOntimeSlowRev*2, None, self.maxOntime)

        return self.fwOntimes, self.rvOntimes

    def saveNewMap(self, mapPath, scaleBy=1.1):
        if (self.newOntimeSlowFwd is None or self.newOntimeSlowRev is None or
            self.newOntimeFwd is None or self.newOntimeRev is None):

            raise RuntimeError('for now, will only write all models with all four new maps')

        # Build a new model by selecting the appropriate motor map for each motor.
        #
        for i in range(57):
            fwOntime = self.fwOntimes[i]
            rvOntime = self.rvOntimes[i]
            self.model.copyMotorMap(self.models[fwOntime], i,
                                    doThetaFwd=(self.axisName == "Theta"),
                                    doPhiFwd=(self.axisName == "Phi"), doFast=False)
            self.model.copyMotorMap(self.models[rvOntime], i,
                                    doThetaRev=(self.axisName == "Theta"),
                                    doPhiRev=(self.axisName == "Phi"), doFast=False)
            self.model.copyMotorMap(self.models[fwOntime], i,
                                    doThetaFwd=(self.axisName == "Theta"),
                                    doPhiFwd=(self.axisName == "Phi"), doFast=True)
            self.model.copyMotorMap(self.models[rvOntime], i,
                                    doThetaRev=(self.axisName == "Theta"),
                                    doPhiRev=(self.axisName == "Phi"), doFast=True)

        motor = self.axisName.lower()
        updateArgs = dict(fast=True)
        updateArgs[f'{motor}Fwd'] = np.clip(self.newOntimeFwd*scaleBy, self.maxOntime*0.75, self.maxOntime)
        updateArgs[f'{motor}Rev'] = np.clip(self.newOntimeRev*scaleBy, self.maxOntime*0.75, self.maxOntime)
        self.model.updateOntimes(**updateArgs)

        updateArgs = dict(fast=False)
        updateArgs[f'{motor}Fwd'] = np.clip(self.newOntimeSlowFwd*scaleBy, None, self.maxOntime)
        updateArgs[f'{motor}Rev'] = np.clip(self.newOntimeSlowRev*scaleBy, None, self.maxOntime)
        self.model.updateOntimes(**updateArgs)

        self.model.createCalibrationFile(mapPath)

    def loadData(self):
        pidarray=[]
        otfarray=[]
        otrarray=[]
        fwdarray=[]
        revarray=[]

        self.fwAngles = dict()
        self.rvAngles = dict()
        self.models = dict()

        for ontime in sorted(self.runDirs.keys())[::-1]:
            runDir = self.runDirs[ontime]
            self.fwAngles[ontime] = np.load(self.fwdAngleFile(runDir))
            self.rvAngles[ontime] = np.load(self.revAngleFile(runDir))
            fwd = np.rad2deg(np.load(self.fwdSpeedFile(runDir)))
            rev = -np.rad2deg(np.load(self.revSpeedFile(runDir)))

            xml = butler.mapForRun(runDir)
            model = pfiDesign.PFIDesign(xml)
            self.models[ontime] = model

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
                self.logger.warn(f'pid {pid}: fw={fw_s},{fw_i} rv={rv_s},{rv_i}')

            fwd_slope[pid-1] = fw_s
            fwd_int[pid-1] = fw_i

            rev_slope[pid-1] = rv_s
            rev_int[pid-1] = rv_i
