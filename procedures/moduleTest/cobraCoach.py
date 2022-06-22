from importlib import reload
import logging
from bokeh.transform import factor_cmap
import numpy as np
from astropy.io import fits
import pathlib
import time

from procedures.moduleTest import calculus
from procedures.moduleTest.speedModel import SpeedModel

from procedures.moduleTest.mcs import camera
from ics.cobraCharmer import pfi as pfiControl
import ics.cobraCharmer.pfiDesign as pfiDesign
from ics.cobraCharmer import func
from ics.cobraCharmer.utils import butler as cbutler
from pfs.utils import butler 
import pfs.utils.coordinates.transform as transformUtils

from opdb import opdb

class CobraCoach():
    nCobrasPerModule = 57
    nModules = 42
    thetaHomeSteps = 10000
    phiHomeSteps = 5000

    constantSpeedMode = False
    mmTheta = None
    mmPhi = None
    mmThetaSlow = None
    mmPhiSlow = None
    maxSegments = 10
    maxStepsPerSeg = 100
    maxTotalSteps = 2000

    """
    There are three modes for cobra operation:
    normalMode 
        The default mode, cobras can move both theta and phi arms.
    thetaMode:
        In this mode, only theta arms can be moved. Use this mode to generate theta motormaps
        and 1-Dim theta convergence test.
    phiMode:
        In this mode, only phi arms can be moved. Use this mode to generate phi motormaps
        and do 1-Dim phi convergence test.
    """
    normalMode = 0
    thetaMode = 1
    phiMode = 2

    thetaModelParameter = 0.088
    phiModelParameter = 0.072

    thetaDtype = np.dtype(dict(names=['center', 'ccwHome', 'cwHome', 'angle'],
                               formats=['c8', 'f4', 'f4', 'f4']))
    phiDtype = np.dtype(dict(names=['center', 'ccwHome', 'cwHome', 'angle'],
                             formats=['c8', 'f4', 'f4', 'f4']))
    cobraDtype = np.dtype(dict(names=['position', 'thetaAngle', 'phiAngle'],
                               formats=['c8', 'f4', 'f4']))
    moveDtype = np.dtype(dict(names=['thetaSteps', 'movedTheta', 'expectedTheta', 'thetaOntime', 'thetaFast',
                                     'phiSteps', 'movedPhi', 'expectedPhi', 'phiOntime', 'phiFast'],
                              formats=['i4', 'f4', 'f4', 'f4', '?', 'i4', 'f4', 'f4', 'f4', '?']))

    def __init__(self, fpgaHost='localhost', loadModel=True, logLevel=logging.INFO,
                 trajectoryMode=False, rootDir=None, actor=None, cmd=None,
                 simDataPath=None):
        self.logger = logging.getLogger('cobraCoach')
        self.logger.setLevel(logLevel)

        self.runManager = cbutler.RunTree(doCreate=False, rootDir=rootDir)
        self.pfi = None
        self.cam = None
        self.fpgaHost = fpgaHost

        if loadModel:
            self.loadModel()

        # scaling model
        self.useScaling = True
        self.thetaScaleFactor = 4.0
        self.phiScaleFactor = 4.0
        self.minThetaStepsForScaling = 10
        self.minPhiStepsForScaling = 10
        self.constantThetaSpeed = np.deg2rad(0.06)
        self.constantPhiSpeed = np.deg2rad(0.06)
        self.thetaModel = SpeedModel(p1=self.thetaModelParameter)
        self.phiModel = SpeedModel(p1=self.phiModelParameter)

        # create cobra trajectory
        self.trajectoryMode = trajectoryMode
        self.trajectory = None
        
        self.simMode=False
        if simDataPath is not None:
            self.simMode = True
            self.simDataPath = simDataPath
        
        self.actor = actor
        self.cmd = cmd

        self.frameNum = None
        self.expTime = None

    def loadModel(self, file=None, version=None, moduleVersion=None, camSplit=28):
        if file is not None:
            self.calibModel = pfiDesign.PFIDesign(file)
        else:
            self.calibModel = pfiDesign.PFIDesign.loadPfi(version, moduleVersion)
        
        self.calibModel.fixModuleIds()

        self.camSplit = camSplit

        # define the cobras and index
        self._setupCobras()

        # cobra position and movement status
        self.mode = self.normalMode
        self.thetaInfo = np.zeros(self.nCobras, dtype=self.thetaDtype)
        self.phiInfo = np.zeros(self.nCobras, dtype=self.phiDtype)
        self.cobraInfo = np.zeros(self.nCobras, dtype=self.cobraDtype)
        self.moveInfo = np.zeros(self.nCobras, dtype=self.moveDtype)
        self.thetaInfoIsValid = False
        self.phiInfoIsValid = False

        self.connect()

    def setScaling(self, enabled=True, thetaScaleFactor=None, phiScaleFactor=None,
                   minThetaSteps=None, minPhiSteps=None, thetaScaling=None, phiScaling=None):
        """ enable/disable motor scaling """
        self.useScaling = enabled
        if thetaScaleFactor is not None:
            self.thetaScaleFactor = thetaScaleFactor
        if phiScaleFactor is not None:
            self.phiScaleFactor = phiScaleFactor
        if minThetaSteps is not None:
            self.minThetaStepsForScaling = minThetaSteps
        if minPhiSteps is not None:
            self.minPhiStepsForScaling = minPhiSteps
        if thetaScaling is not None:
            if len(thetaScaling) != self.nCobras:
                ValueError(f'len(thetaScaling) should equal to {self.nCobras}')
            self.thetaScaling = thetaScaling
        if phiScaling is not None:
            if len(phiScaling) != self.nCobras:
                ValueError(f'len(phiScaling) should equal to {self.nCobras}')
            self.phiScaling = phiScaling

    def _setupCobras(self):
        """ define the broken/good cobras """
        cobras = []
        for i in self.calibModel.findAllCobras():
            c = func.Cobra(self.calibModel.moduleIds[i],
                           self.calibModel.positionerIds[i])
            cobras.append(c)
        self.allCobras = np.array(cobras)
        self.nCobras = len(self.allCobras)

        brokens = [i+1 for i,c in enumerate(self.allCobras) if
                   self.calibModel.fiberIsBroken(c.cobraNum, c.module)]
        visibles = [e for e in range(1, self.nCobras+1) if e not in brokens]
        if len(brokens) > 0:
            self.logger.warn("setting invisible cobras: %s", brokens)

        self.invisibleIdx = np.array(brokens, dtype='i4') - 1
        self.visibleIdx = np.array(visibles, dtype='i4') - 1
        self.invisibleCobras = self.allCobras[self.invisibleIdx]
        self.visibleCobras = self.allCobras[self.visibleIdx]

        goodNums = [i+1 for i,c in enumerate(self.allCobras) if
                   self.calibModel.cobraIsGood(c.cobraNum, c.module)]
        badNums = [e for e in range(1, self.nCobras+1) if e not in goodNums]
        if len(badNums) > 0:
            self.logger.warn("setting bad cobras: %s", badNums)

        self.goodIdx = np.array(goodNums, dtype='i4') - 1
        self.badIdx = np.array(badNums, dtype='i4') - 1
        self.goodCobras = self.allCobras[self.goodIdx]
        self.badCobras = self.allCobras[self.badIdx]

        # partition cobras into odd and even sets
        self.oddCobras = []
        self.evenCobras = []
        for c in self.allCobras:
            if c.cobraNum % 2 == 0:
                self.evenCobras.append(c)
            else:
                self.oddCobras.append(c)
        self.oddCobras = np.array(self.oddCobras)
        self.evenCobras = np.array(self.evenCobras)

        # default is all enabled for scaling
        self.thetaScaling = np.full(self.nCobras, True)
        self.phiScaling = np.full(self.nCobras, True)



    def showStatus(self):
        """ show current cobra status """
        if self.mode == self.thetaMode:
            self.logger.info('Operate in THETA mode')

            if self.thetaInfoIsValid:
                localAngles = np.rad2deg(self.thetaInfo['angle'])
                self.logger.info(f'angles from CCW hard stops: {np.round(localAngles,1)}')
                globalAngles = np.rad2deg((self.thetaInfo['angle'] + self.calibModel.tht0) % (np.pi*2))
                self.logger.info(f'angles in global coordinate: {np.round(globalAngles,1)}')
            else:
                self.logger.info('Theta geometry is not set')

        elif self.mode == self.phiMode:
            self.logger.info('Operate in PHI mode')

            if self.phiInfoIsValid:
                localAngles = np.rad2deg(self.phiInfo['angle'])
                self.logger.info(f'angles from CCW hard stops: {np.round(localAngles,1)}')
                globalAngles = np.rad2deg(self.phiInfo['angle'] + self.calibModel.phiIn + np.pi)
                self.logger.info(f'angles from theta arms: {np.round(globalAngles,1)}')
            else:
                self.logger.info('Phi geometry is not set')

        else:
            self.logger.info('Operate in NORMAL mode')

            localAngles = np.rad2deg(self.cobraInfo['thetaAngle'])
            self.logger.info(f'Theta angles from CCW hard stops: {np.round(localAngles,1)}')
            globalAngles = np.rad2deg((self.cobraInfo['thetaAngle'] + self.calibModel.tht0) % (np.pi*2))
            self.logger.info(f'Theta angles in global coordinate: {np.round(globalAngles,1)}')

            localAngles = np.rad2deg(self.cobraInfo['phiAngle'])
            self.logger.info(f'Phi angles from CCW hard stops: {np.round(localAngles,1)}')
            globalAngles = np.rad2deg(self.cobraInfo['phiAngle'] + self.calibModel.phiIn + np.pi)
            self.logger.info(f'Phi angles in global coordinate: {np.round(globalAngles,1)}')

        self.logger.info(f'Current positions: {np.round(self.cobraInfo["position"],1)}')
        if self.useScaling:
            self.logger.info(f'Scaling enabled, Scaling Factor: {self.thetaScaleFactor}/{self.phiScaleFactor}, '
                             f'Min Steps: {self.minThetaStepsForScaling}/{self.minPhiStepsForScaling}')
        else:
            self.logger.info(f'Scaling disabled')

    def setMode(self, mode):
        if mode == 'normal':
            if self.mode != self.normalMode:
                self.mode = self.normalMode
            else:
                self.logger.info('Already in normal mode')
        elif mode == 'theta':
            if self.mode != self.thetaMode:
                self.mode = self.thetaMode
                self.thetaInfoIsValid = False
            else:
                self.logger.info('Already in theta mode')
        elif mode == 'phi':
            if self.mode != self.phiMode:
                self.mode = self.phiMode
                self.phiInfoIsValid = False
            else:
                self.logger.info('Already in phi mode')
        else:
            raise ValueError(f'parameter should be [normal, theta, phi]: {mode}')

    def getMode(self):
        if self.mode == self.phiMode:
            return 'phi'
        elif self.mode == self.thetaMode:
            return 'theta'
        else:
            return 'normal'

    def connect(self, setFreq=True):
        self.logger.info(f'cc connecting to {self.fpgaHost}....')
        self.runManager.newRun()

        # Initializing COBRA module
        reload(pfiControl)
        if self.trajectoryMode:
            self.pfi = pfiControl.PFI(doLoadModel=False,
                                      doConnect=False,
                                      logDir=self.runManager.logDir)
        else:
            self.pfi = pfiControl.PFI(fpgaHost=self.fpgaHost,
                                      doLoadModel=False,
                                      logDir=self.runManager.logDir)
        self.pfi.calibModel = self.calibModel

        if self.trajectoryMode:
            return

        if setFreq:
            if True:
                self.pfi.diag()
                self.pfi.power(0)
                time.sleep(1)
                self.pfi.reset()
                time.sleep(1)
                self.pfi.diag()
            else:
                self.pfi.power(0x3f)
                time.sleep(1)
                self.pfi.power(0x3f)
                time.sleep(1)
                self.pfi.reset()
                time.sleep(1)
                self.pfi.diag()
                time.sleep(1)
                self.pfi.power(0)
                time.sleep(1)
                self.pfi.power(0)
                time.sleep(1)
                self.pfi.reset()
                time.sleep(1)
                self.pfi.diag()
            
            self.pfi.calibModel = self.calibModel
            self.pfi.setFreq()        # initialize cameras
        try:
            #if self.actor is not None:
            #    self.cam = camera.cameraFactory(name='mcs',doClear=True, 
            #        runManager=self.runManager, actor=self.actor)
            #else:    
            if self.actor is not None:
                if self.simMode is False:
                    self.logger.info(f'MCS actor is given, try using mcsActor camera. simMode = {self.simMode}')
                    self.cam = camera.cameraFactory(name='mcsActor',doClear=True, runManager=self.runManager,
                                                    actor=self.actor, cmd=self.cmd)
                else:
                    self.logger.info(f'MCS actor is given, try using mcsActor camera in simulation mode')
                    self.cam = camera.cameraFactory(name='mcsActor',doClear=True,
                                                    runManager=self.runManager, actor=self.actor,
                                                    cmd=self.cmd, simulationPath=self.simDataPath)
            else:
                self.logger.info('MCS actor is not present, using RMOD camera')    
                self.cam = camera.cameraFactory(name='rmod',doClear=True, runManager=self.runManager)
        except Exception as e:
            self.logger.warn(f'Problem when connecting to camera: {e}')
            #self.cam = None

    def _getIndex(self, cobras):
        cIds = np.zeros(len(cobras), 'int')
        for c_i, c in enumerate(cobras):
            cIds[c_i] = self.calibModel.findCobraByModuleAndPositioner(c.module, c.cobraNum)

        return cIds

    def exposeAndExtractPositions(self, name=None, guess=None, tolerance=None, 
                                  exptime=None, dbMatch = True, writeData = None):
        """ Take an exposure, measure centroids, match to cobras, save info.

        Args
        ----
        name : `str`
           Additional name for saved image file. File _always_ gets PFS-compliant name.
        guess : `ndarray` of complex coordinates
           Where to center searches. By default uses the cobra center.
        tolerance : `float`
           Additional factor to scale search region by. 1 = cobra radius (phi+theta)
        exptime : `float`
           What exposure time to use.

        Returns
        -------
        positions : `ndarray` of complex
           The measured positions of our cobras.
           If no matching spot found, return the cobra center.

        Note
        ----
        Not at all convinced that we should return anything if no matching spot found.

        """

        cmd = self.actor.visitor.cmd
        frameNum = self.actor.visitor.getNextFrameNum()
        centroids, filename, bkgd = self.cam.expose(name,
                                                    exptime=self.expTime,
                                                    frameNum=frameNum,
                                                    cmd=cmd)

        self.frameNum = frameNum
        self.logger.info(f'Exposure done and filename = {self.frameNum}')


        if dbMatch is True:
            '''
                We extract matchin table from databse instead of calling matching sequence.
            '''
            db=opdb.OpDB(hostname='db-ics', port=5432,
                   dbname='opdb',
                   username='pfs')
            match = db.bulkSelect('cobra_match','select * from cobra_match where '
                      f'mcs_frame_id = {frameNum}').sort_values(by=['cobra_id']).reset_index()
            

            positions = match['pfi_center_x_mm'].values+match['pfi_center_y_mm'].values*(1j)
            
            self.logger.info(f'Returing result from matching table ncen={len(positions)} ')
            
        else:
            self.logger.info(f'dbMatching is turned off! THIS HAS TO BE MOTOR MAP RUN!')

            self.cameraName = 'canon'
            self.logger.info(f'Current camera name is {self.cameraName}')
            
            #idx = self.visibleIdx
            # Here we start to process the centroid.  Converting them from pixel to mm
            fids = butler.Butler().get('fiducials')
            db=opdb.OpDB(hostname='db-ics', port=5432,
                   dbname='opdb',
                   username='pfs')
            mcsData = db.bulkSelect('mcs_data','select * from mcs_data where '
                        f'mcs_frame_id = {frameNum} and spot_id > 0').sort_values(by=['spot_id'])
            teleInfo = db.bulkSelect('mcs_exposure','select altitude, insrot from mcs_exposure where '
                        f'mcs_frame_id = {frameNum}')


            if 'rmod' in self.cameraName.lower():
                altitude = 90.0
                insrot = 0
            else: 
                altitude = teleInfo['altitude'].values[0]
                insrot = teleInfo['insrot'].values[0]

            pfiTransform = transformUtils.fromCameraName(self.cameraName, 
            altitude=altitude, insrot=insrot,nsigma=0, alphaRot=1)
        
            outerRingIds = [29, 30, 31, 61, 62, 64, 93, 94, 95, 96]
            fidsOuterRing = fids[fids.fiducialId.isin(outerRingIds)]

            pfiTransform.updateTransform(mcsData, fidsOuterRing, matchRadius=8.0, nMatchMin=0.1)

            nsigma = 0
            pfiTransform.nsigma = nsigma
            pfiTransform.alphaRot = 0

            for i in range(2):
                ffid, dist = pfiTransform.updateTransform(mcsData, fids, matchRadius=4.2,nMatchMin=0.1)

            xx , yy = pfiTransform.mcsToPfi(mcsData['mcs_center_x_pix'],mcsData['mcs_center_y_pix'])
            centroids=np.rec.array([xx,yy],formats='float,float',names='x,y')
            
            if tolerance is not None:
                radii = (self.calibModel.L1 + self.calibModel.L2) * (1 + tolerance)
            else:
                radii = None

            if guess is None:
                centers = self.calibModel.centers
            elif len(guess) == self.nCobras:
                centers = guess
            elif len(guess) != len(idx):
                raise RuntimeError('len(guess) should be equal to the visible cobras or total cobras')
            else:
                centers = guess

        
            self.logger.info('Running object matching!')
            positions, indexMap = calculus.matchPositions(centroids, guess=centers, dist=radii)
            self.logger.info(f'Total detected spots={len(centroids)} Npos={len(positions)} nguess={len(centers)}')
        return positions

    def getCurrentPositionsUNUSED(self):
        """
        expose and get current cobra positions
        """
        if self.trajectoryMode:
            raise RuntimeError('roundTrip command not available in trajectoryMode mode!')

        pos = np.zeros(self.nCobras, 'complex')
        pos[self.visibleIdx] = self.exposeAndExtractPositions()
        thetas, phis, flags = self.pfi.positionsToAngles(self.allCobras, pos)

        self.cobraInfo['position'] = pos
        self.cobraInfo['thetaAngle'] = thetas[:, 0]
        self.cobraInfo['phiAngle'] = phis[:, 0]

    def moveSteps(self, cobras, thetaSteps, phiSteps, thetaFast=False, phiFast=False,
                  expectedThetas=None, expectedPhis=None, force=False,
                  thetaOntimes=None, phiOntimes=None, nSegments=0,
                  thetaSpeeds=None, phiSpeeds=None):
        """
        move cobras by the given steps and update current positions

        parameters:
        expectedThetas, expectedPhis: expected theta, phi angles to move
        force: can move both theta/phi arms even in theta/phi mode
        nSegments: number of moves
        """
        if self.mode == self.thetaMode and np.any(phiSteps != 0) and not force:
            raise RuntimeError('Move phi arms in theta mode!')
        elif self.mode == self.phiMode and np.any(thetaSteps != 0) and not force:
            raise RuntimeError('Move theta arms in phi mode!')
        if np.all(thetaSteps == 0) and np.all(phiSteps == 0):
            self.logger.info('all theta and phi steps are 0, not moving!')
            return

        if nSegments == 0:
            if np.isscalar(thetaSteps):
                thetaSteps = np.full(len(cobras), thetaSteps)
            elif len(thetaSteps) != len(cobras):
                raise RuntimeError("number of theta steps must match number of cobras")
            if np.isscalar(phiSteps):
                phiSteps = np.full(len(cobras), phiSteps)
            elif len(phiSteps) != len(cobras):
                raise RuntimeError("number of phi steps must match number of cobras")
            if thetaOntimes is not None:
                if np.isscalar(thetaOntimes):
                    thetaOntimes = np.full(len(cobras), thetaOntimes)
                elif len(thetaOntimes) != len(cobras):
                    raise RuntimeError("number of thetaOntimes must match number of cobras")
            if phiOntimes is not None:
                if np.isscalar(phiOntimes):
                    phiOntimes = np.full(len(cobras), phiOntimes)
                elif len(phiOntimes) != len(cobras):
                    raise RuntimeError("number of phiOntimes must match number of cobras")
        elif nSegments > 0:
            if np.isscalar(thetaSteps):
                thetaSteps = np.full((nSegments, len(cobras)), thetaSteps)
            elif thetaSteps.shape != (nSegments, len(cobras)):
                raise RuntimeError("number of theta steps must match (nSegments, nCobras)")
            if np.isscalar(phiSteps):
                phiSteps = np.full(len(cobras), phiSteps)
            elif phiSteps.shape != (nSegments, len(cobras)):
                raise RuntimeError("number of phi steps must match (nSegments, nCobras)")
            if thetaOntimes is None:
                raise RuntimeError("thetaOntimes is required for multiple segment moves")
            elif np.isscalar(thetaOntimes):
                thetaOntimes = np.full((nSegments, len(cobras)), thetaOntimes)
            elif thetaOntimes.shape != (nSegments, len(cobras)):
                raise RuntimeError("shape of thetaOntimes must match (nSegments, nCobras)")
            if phiOntimes is None:
                raise RuntimeError("phiOntimes is required for multiple segment moves")
            elif np.isscalar(phiOntimes):
                phiOntimes = np.full((nSegments, len(cobras)), phiOntimes)
            elif phiOntimes.shape != (nSegments, len(cobras)):
                raise RuntimeError("shape of phiOntimes must match (nSegments, nCobras)")
        else:
            raise RuntimeError("nSegments should be positive integer")

        if expectedThetas is None:
            expectedThetas = np.full(len(cobras), np.nan)
        elif len(expectedThetas) != len(cobras):
            raise RuntimeError("number of expected theta angles must match number of cobras")
        if expectedPhis is None:
            expectedPhis = np.full(len(cobras), np.nan)
        elif len(expectedPhis) != len(cobras):
            raise RuntimeError("number of expected phi angles must match number of cobras")

        if isinstance(thetaFast, bool):
            thetaFast = np.full(len(cobras), thetaFast)
        elif len(cobras) != len(thetaFast):
            raise RuntimeError("number of thetaFast must match number of cobras")
        if isinstance(phiFast, bool):
            phiFast = np.full(len(cobras), phiFast)
        elif len(cobras) != len(phiFast):
            raise RuntimeError("number of phiFast must match number of cobras")
        cIds = self._getIndex(cobras)

        # move and expose
        thetaSteps = np.clip(thetaSteps, -10000, 10000)
        phiSteps = np.clip(phiSteps, -6000, 6000)

        if self.trajectoryMode:
            timeStep = self.trajectory.getTimeStep()
            if nSegments == 0:
                thetaT, phiT = calculus.interpolateMoves(cIds, timeStep,
                                                         self.cobraInfo['thetaAngle'][cIds],
                                                         self.cobraInfo['phiAngle'][cIds],
                                                         thetaSteps, phiSteps, thetaFast, phiFast,
                                                         self.calibModel)
            else:
                thetaT, phiT = calculus.interpolateSegments(timeStep,
                                                            self.cobraInfo['thetaAngle'][cIds],
                                                            self.cobraInfo['phiAngle'][cIds],
                                                            thetaSteps, phiSteps,
                                                            thetaSpeeds, phiSpeeds)
            self.trajectory.addMovement(cIds, thetaT, phiT)

        if not self.trajectoryMode and self.cam.filePrefix == 'PFAC':
            self.cam.startRecord()
        if nSegments == 0:
            self.pfi.moveSteps(cobras, thetaSteps, phiSteps,
                               thetaFast=thetaFast, phiFast=phiFast,
                               thetaOntimes=thetaOntimes, phiOntimes=phiOntimes,
                               trajectoryMode=self.trajectoryMode)
        else:
            for n in range(nSegments):
                self.pfi.moveSteps(cobras, thetaSteps[n], phiSteps[n],
                                   thetaOntimes=thetaOntimes[n], phiOntimes=phiOntimes[n],
                                   trajectoryMode=self.trajectoryMode)
        if not self.trajectoryMode and self.cam.filePrefix == 'PFAC':
            self.cam.stopRecord()

        if self.mode == self.thetaMode:
            thetas = self.thetaInfo['angle'] + self.thetaInfo['ccwHome']
            thetas[cIds] += expectedThetas
            L1 = np.abs(self.cobraInfo['position'] - self.thetaInfo['center'])
            targets = self.thetaInfo['center'] + L1 * np.exp(thetas * 1j)
        elif self.mode == self.phiMode:
            phis = self.phiInfo['angle'] + self.phiInfo['ccwHome']
            phis[cIds] += expectedPhis
            L2 = np.abs(self.cobraInfo['position'] - self.phiInfo['center'])
            targets = self.phiInfo['center'] + L2 * np.exp(phis * 1j)
        else:
            thetas = np.copy(self.cobraInfo['thetaAngle'])
            phis = np.copy(self.cobraInfo['phiAngle'])
            thetas[cIds] += expectedThetas
            phis[cIds] += expectedPhis
            targets = self.pfi.anglesToPositions(self.allCobras, thetas, phis)

        invalid = np.isnan(targets)
        if self.mode == self.thetaMode and self.thetaInfoIsValid:
            targets[invalid] = self.thetaInfo['center'][invalid]
        elif self.mode == self.phiMode and self.phiInfoIsValid:
            targets[invalid] = self.phiInfo['center'][invalid]

        invalid = np.isnan(targets)
        targets[invalid] = self.calibModel.centers[invalid]

        pos = np.zeros(self.nCobras, 'complex')
        if self.trajectoryMode:
            pos[self.visibleIdx] = targets[self.visibleIdx]
        else:
            #pos[self.visibleIdx] = self.exposeAndExtractPositions(guess=targets[self.visibleIdx])
            pos[self.visibleIdx] = self.exposeAndExtractPositions(dbMatch=True)[self.visibleIdx]

            #np.save('/tmp/cobraCharmer_pos',pos)
        # update status
        thetas, phis, flags = self.pfi.positionsToAngles(self.allCobras, pos)
        for c_i in range(len(cobras)):
            cId = cIds[c_i]
            if nSegments == 0:
                tSteps = thetaSteps[c_i]
                pSteps = phiSteps[c_i]
            else:
                tSteps = np.sum(thetaSteps[:,c_i])
                pSteps = np.sum(phiSteps[:,c_i])

            self.cobraInfo['position'][cId] = pos[cId]
            self.moveInfo['thetaSteps'][cId] = tSteps
            self.moveInfo['phiSteps'][cId] = pSteps
            self.moveInfo['expectedTheta'][cId] = expectedThetas[c_i]
            self.moveInfo['expectedPhi'][cId] = expectedPhis[c_i]
            self.moveInfo['thetaFast'][cId] = thetaFast[c_i]
            self.moveInfo['phiFast'][cId] = phiFast[c_i]
            if tSteps == 0:
                self.moveInfo['thetaOntime'][cId] = 0
            elif nSegments == 0:
                self.moveInfo['thetaOntime'][cId] = cobras[c_i].p.pulses[0] / 1000
            else:
                tOn = thetaOntimes[:,c_i]
                self.moveInfo['thetaOntime'][cId] = np.average(tOn[np.nonzero(tOn)])
            if pSteps == 0:
                self.moveInfo['phiOntime'][cId] = 0
            elif nSegments == 0:
                self.moveInfo['phiOntime'][cId] = cobras[c_i].p.pulses[1] / 1000
            else:
                pOn = phiOntimes[:,c_i]
                self.moveInfo['phiOntime'][cId] = np.average(pOn[np.nonzero(pOn)])

            # check theta/phi solution
            if flags[cId, 0] & self.pfi.SOLUTION_OK != 0:
                theta = thetas[cId, 0]
            elif flags[cId, 0] & self.pfi.TOO_FAR_FROM_CENTER != 0:
                self.logger.warn(f'Cobra#{cId+1} is too far from center')
                theta = thetas[cId, 0]
            #elif flags[cId, 0] & self.pfi.TOO_CLOSE_TO_CENTER != 0:
            else:
                self.logger.warn(f'Cobra#{cId+1} is too close to center')
                theta = self.cobraInfo['thetaAngle'][cId] + expectedThetas[c_i]
            phi = phis[cId, 0]

            #self.logger.info(f'print {theta}')
            if not np.isnan(theta):
                angle = calculus.unwrappedAngle(theta, tSteps, self.cobraInfo['thetaAngle'][cId],
                                                self.cobraInfo['thetaAngle'][cId] + expectedThetas[c_i])
                self.moveInfo['movedTheta'][cId] = angle - self.cobraInfo['thetaAngle'][cId]
                self.moveInfo['movedPhi'][cId] = phi - self.cobraInfo['phiAngle'][cId]
                self.cobraInfo['thetaAngle'][cId] = angle
                self.cobraInfo['phiAngle'][cId] = phi

                if self.useScaling and self.mode == self.normalMode:
                    if self.thetaScaling[cId] and not np.isnan(expectedThetas[c_i]) and not np.isnan(self.moveInfo['movedTheta'][cId]) and abs(tSteps) > self.minThetaStepsForScaling:
                        direction = 'cw' if expectedThetas[c_i] > 0 else 'ccw'
                        scale = expectedThetas[c_i] / self.moveInfo['movedTheta'][cId]
                        if scale < 0:
                            self.logger.warn(f'Theta scale negative: Cobra#{cId+1}, steps:{tSteps}, angle:{angle}')
                        else:
                            scale = (scale - 1) / self.thetaScaleFactor + 1
                            self.pfi.scaleMotorOntime(cobras[c_i], 'theta', direction, scale)
#                            self.pfi.scaleMotorOntimeBySpeed(cobras[c_i], 'theta', direction, thetaFast[c_i], scale, self.moveInfo['thetaOntime'][cId])
                    if self.phiScaling[cId] and not np.isnan(expectedPhis[c_i]) and not np.isnan(self.moveInfo['movedPhi'][cId]) and abs(pSteps) > self.minPhiStepsForScaling:
                        direction = 'cw' if expectedPhis[c_i] > 0 else 'ccw'
                        scale = expectedPhis[c_i] / self.moveInfo['movedPhi'][cId]
                        if scale < 0:
                            self.logger.warn(f'Phi scale negative: Cobra#{cId+1}, steps:{pSteps}, angle:{phis[cId, 0]}')
                        else:
                            scale = (scale - 1) / self.phiScaleFactor + 1
                            self.pfi.scaleMotorOntime(cobras[c_i], 'phi', direction, scale)
#                            self.pfi.scaleMotorOntimeBySpeed(cobras[c_i], 'phi', direction, phiFast[c_i], scale, self.moveInfo['phiOntime'][cId])
            else:
                self.cobraInfo['thetaAngle'][cId] = np.nan
                self.cobraInfo['phiAngle'][cId] = np.nan
                self.moveInfo['movedTheta'][cId] = np.nan
                self.moveInfo['movedPhi'][cId] = np.nan

            if self.mode == self.thetaMode and self.thetaInfoIsValid:
                angle = np.angle(pos[cId] - self.thetaInfo['center'][cId]) - self.thetaInfo['ccwHome'][cId]
                angle = calculus.unwrappedAngle(angle, tSteps, self.thetaInfo['angle'][cId],
                                                self.thetaInfo['angle'][cId] + expectedThetas[c_i])
                self.moveInfo['movedTheta'][cId] = angle - self.thetaInfo['angle'][cId]
                self.moveInfo['movedPhi'][cId] = 0
                self.thetaInfo['angle'][cId] = angle

                if self.useScaling and self.thetaScaling[cId] and not np.isnan(expectedThetas[c_i]) and abs(tSteps) > self.minThetaStepsForScaling:
                    direction = 'cw' if expectedThetas[c_i] > 0 else 'ccw'
                    scale = expectedThetas[c_i] / self.moveInfo['movedTheta'][cId]
                    if scale < 0:
                        self.logger.warn(f'scale is negative: Cobra#{cId+1}, steps:{tSteps}, angle:{angle}')
                    else:
                        scale = (scale - 1) / self.thetaScaleFactor + 1
                        self.pfi.scaleMotorOntime(cobras[c_i], 'theta', direction, scale)
#                        self.pfi.scaleMotorOntimeBySpeed(cobras[c_i], 'theta', direction, thetaFast[c_i], scale, self.moveInfo['thetaOntime'][cId])

            elif self.mode == self.phiMode and self.phiInfoIsValid:
                angle = np.angle(pos[cId] - self.phiInfo['center'][cId]) - self.phiInfo['ccwHome'][cId]
                angle = (angle + np.pi/2) % (np.pi*2) - np.pi/2
                self.moveInfo['movedTheta'][cId] = 0
                self.moveInfo['movedPhi'][cId] = angle - self.phiInfo['angle'][cId]
                self.phiInfo['angle'][cId] = angle

                if self.useScaling and self.phiScaling[cId] and not np.isnan(expectedPhis[c_i]) and abs(pSteps) > self.minPhiStepsForScaling:
                    direction = 'cw' if expectedPhis[c_i] > 0 else 'ccw'
                    scale = expectedPhis[c_i] / self.moveInfo['movedPhi'][cId]
                    if scale < 0:
                        self.logger.warn(f'scale is negative: Cobra#{cId+1}, steps:{pSteps}, angle:{angle}')
                    else:
                        scale = (scale - 1) / self.phiScaleFactor + 1
                        self.pfi.scaleMotorOntime(cobras[c_i], 'phi', direction, scale)
#                        self.pfi.scaleMotorOntimeBySpeed(cobras[c_i], 'phi', direction, phiFast[c_i], scale, self.moveInfo['phiOntime'][cId])

    def moveDeltaAngles(self, cobras, thetaAngles=None, phiAngles=None, thetaFast=False, phiFast=False):
        """ move cobras by the given amount of theta and phi angles """
        if self.constantSpeedMode:
            if self.mmThetaSlow is None or self.mmPhiSlow is None:
                raise RuntimeError('Please slow set on-time maps first!')
            if self.mmTheta is None or self.mmPhi is None:
                raise RuntimeError('Please set on-time maps first!')
        if np.all(self.cobraInfo['position'] == 0.0):
            raise RuntimeError('Last position is unkown! Run moveToHome or setCurrentAngles')

        if self.mode == self.thetaMode:
            if phiAngles is not None:
                raise RuntimeError('Move phi arms in theta mode!')
            if not self.thetaInfoIsValid:
                raise RuntimeError('Please set theta geometry first!')
        elif self.mode == self.phiMode:
            if thetaAngles is not None:
                raise RuntimeError('Move theta arms in phi mode!')
            if not self.phiInfoIsValid:
                raise RuntimeError('Please set phi geometry first!')
        if thetaAngles is None and phiAngles is None:
            self.logger.info('both theta and phi angles are None, not moving!')
            return

        if thetaAngles is not None:
            if np.isscalar(thetaAngles):
                thetaAngles = np.full(len(cobras), thetaAngles)
            elif len(thetaAngles) != len(cobras):
                raise RuntimeError("number of theta angles must match number of cobras")
        else:
            thetaAngles = np.zeros(len(cobras))
        if phiAngles is not None:
            if np.isscalar(phiAngles):
                phiAngles = np.full(len(cobras), phiAngles)
            elif len(phiAngles) != len(cobras):
                raise RuntimeError("number of phi angles must match number of cobras")
        else:
            phiAngles = np.zeros(len(cobras))

        if isinstance(thetaFast, bool):
            thetaFast = np.full(len(cobras), thetaFast)
        elif len(cobras) != len(thetaFast):
            raise RuntimeError("number of thetaFast must match number of cobras")
        if isinstance(phiFast, bool):
            phiFast = np.full(len(cobras), phiFast)
        elif len(cobras) != len(phiFast):
            raise RuntimeError("number of phiFast must match number of cobras")

        # calculate steps
        fromTheta = np.zeros(len(cobras), 'float')
        fromPhi = np.zeros(len(cobras), 'float')
        cIds = self._getIndex(cobras)

        # get last theta/phi angles
        if self.mode == self.thetaMode:
            badFromThetaIdx = cIds[np.isnan(self.thetaInfo['angle'][cIds])]
            if len(badFromThetaIdx) > 0:
                self.logger.warn(f'Last theta angle is unknown: {badFromThetaIdx}')

            fromPhi[:] = 0
            for c_i in range(len(cobras)):
                cId = cIds[c_i]
                fromTheta[c_i] = self.thetaInfo['angle'][cId]
                if np.isnan(fromTheta[c_i]):
                    # well, assume in the CCW or CW hard stop to calculate steps
                    if thetaAngles[c_i] >= 0:
                        fromTheta[c_i] = 0
                    else:
                        fromTheta[c_i] = (self.calibModel.tht1[cId] - self.calibModel.tht0[cId] + np.pi) % (np.pi*2) + np.pi

        elif self.mode == self.phiMode:
            badFromPhiIdx = cIds[np.isnan(self.phiInfo['angle'][cIds])]
            if len(badFromPhiIdx) > 0:
                self.logger.warn(f'Last phi angle is unknown: {badFromPhiIdx}')

            fromTheta[:] = 0
            fromPhi[:] = self.phiInfo['angle'][cIds]
            for c_i in range(len(cobras)):
                cId = cIds[c_i]
                fromPhi[c_i] = self.phiInfo['angle'][cId]
                if np.isnan(fromPhi[c_i]):
                    # well, assume in the CCW or CW hard stop to calculate steps
                    if phiAngles[c_i] >= 0:
                        fromPhi[c_i] = 0
                    else:
                        fromPhi[c_i] = (self.calibModel.phiOut[cId] - self.calibModel.phiIn[cId]) % (np.pi*2)

        else:
            # normal mode
            badFromThetaIdx = cIds[np.isnan(self.cobraInfo['thetaAngle'][cIds])]
            if len(badFromThetaIdx) > 0:
                self.logger.warn(f'Last theta angle is unknown: {badFromThetaIdx}')
            badFromPhiIdx = cIds[np.isnan(self.cobraInfo['phiAngle'][cIds])]
            if len(badFromPhiIdx) > 0:
                self.logger.warn(f'Last phi angle is unknown: {badFromPhiIdx}')

            for c_i in range(len(cobras)):
                cId = cIds[c_i]

                fromTheta[c_i] = self.cobraInfo['thetaAngle'][cId]
                if np.isnan(fromTheta[c_i]):
                    # well, assume in the CCW or CW hard stop to calculate steps
                    if thetaAngles[c_i] >= 0:
                        fromTheta[c_i] = 0
                    else:
                        fromTheta[c_i] = (self.calibModel.tht1[cId] - self.calibModel.tht0[cId] + np.pi) % (np.pi*2) + np.pi

                fromPhi[c_i] = self.cobraInfo['phiAngle'][cId]
                if np.isnan(fromPhi[c_i]):
                    # well, assume in the CCW or CW hard stop to calculate steps
                    if phiAngles[c_i] >= 0:
                        fromPhi[c_i] = 0
                    else:
                        fromPhi[c_i] = (self.calibModel.phiOut[cId] - self.calibModel.phiIn[cId]) % (np.pi*2)

        if not self.constantSpeedMode:
            # calculate steps
            thetaSteps = np.zeros(len(cobras))
            phiSteps = np.zeros(len(cobras))
            for c_i in range(len(cobras)):
                thetaSteps[c_i], phiSteps[c_i], thetaAngles[c_i], phiAngles[c_i] = \
                    calculus.calculateSteps(cIds[c_i], self.maxTotalSteps, thetaAngles[c_i], phiAngles[c_i],
                                            fromTheta[c_i], fromPhi[c_i], thetaFast[c_i], phiFast[c_i], self.calibModel)

            # send move command
            #self.logger.info(f'thetaSteps = {thetaSteps}')
            self.moveSteps(cobras, thetaSteps, phiSteps, thetaFast, phiFast, thetaAngles, phiAngles)

        else:
            # determine the maximum number of required segments
            nSegments = 0
            for c_i in range(len(cobras)):
                cId = cIds[c_i]
                _mmTheta = self.mmTheta[cId] if thetaFast[c_i] else self.mmThetaSlow[cId]
                _mmPhi = self.mmPhi[cId] if phiFast[c_i] else self.mmPhiSlow[cId]
                
                self.logger.warn(f'''Calulate theta arm {c_i}/{len(cobras)}, {thetaAngles[c_i]}, 
                    {fromTheta[c_i]} {self.maxStepsPerSeg}''')
                
                n = calculus.calNSegments(thetaAngles[c_i], fromTheta[c_i], _mmTheta, self.maxStepsPerSeg)
                if nSegments < n:
                    nSegments = n

                self.logger.warn(f'''Calulate phi arm {c_i}/{len(cobras)}, {phiAngles[c_i]}, 
                    {fromPhi[c_i]} {self.maxStepsPerSeg}''')
                
                n = calculus.calNSegments(phiAngles[c_i], fromPhi[c_i], _mmPhi, self.maxStepsPerSeg)
                if nSegments < n:
                    nSegments = n

            if nSegments > self.maxSegments:
                nSegments = self.maxSegments
            thetaOntimes = np.zeros((nSegments, len(cIds)), float)
            phiOntimes = np.zeros((nSegments, len(cIds)), float)
            thetaSteps = np.zeros((nSegments, len(cIds)), int)
            phiSteps = np.zeros((nSegments, len(cIds)), int)
            thetaSpeeds = np.zeros((nSegments, len(cIds)), float)
            phiSpeeds = np.zeros((nSegments, len(cIds)), float)

            for c_i in range(len(cobras)):
                cId = cIds[c_i]
                phiOffset = self.calibModel.phiIn[cId] + np.pi
                _mmTheta = self.mmTheta[cId] if thetaFast[c_i] else self.mmThetaSlow[cId]
                _mmPhi = self.mmPhi[cId] if phiFast[c_i] else self.mmPhiSlow[cId]

                thetaSteps[:,c_i], phiSteps[:,c_i], thetaOntimes[:,c_i], phiOntimes[:,c_i], \
                    thetaAngles[c_i], phiAngles[c_i], thetaSpeeds[:,c_i], phiSpeeds[:,c_i] = \
                    calculus.calMoveSegments(thetaAngles[c_i], phiAngles[c_i], fromTheta[c_i], fromPhi[c_i],
                                             _mmTheta, _mmPhi, self.maxStepsPerSeg, nSegments, phiOffset)

            dataPath = self.runManager.dataDir
            nowSecond = int(time.time())
            segmentsDtype = np.dtype([('thetaSteps', 'i2'), ('thetaOntimes', 'f4'), ('thetaSpeeds', 'f4'), (
                                       'phiSteps', 'i2'), ('phiOntimes', 'f4'), ('phiSpeeds', 'f4')])
            segments = np.empty((nSegments, len(cIds)), dtype=segmentsDtype)
            segments['thetaSteps'] = thetaSteps
            segments['thetaOntimes'] = thetaOntimes
            segments['thetaSpeeds'] = thetaSpeeds
            segments['phiSteps'] = phiSteps
            segments['phiOntimes'] = phiOntimes
            segments['phiSpeeds'] = phiSpeeds
            np.savez(dataPath / f'segments_{nowSecond}', idx=cIds, segs=segments)

            self.moveSteps(cobras, thetaSteps, phiSteps, thetaFast, phiFast, thetaAngles, phiAngles, False, thetaOntimes, phiOntimes, nSegments, thetaSpeeds, phiSpeeds)

        return self.moveInfo['movedTheta'][cIds], self.moveInfo['movedPhi'][cIds]

    def moveToAngles(self, cobras, thetaAngles=None, phiAngles=None, thetaFast=False, phiFast=False, local=True):
        """ move cobras to the given theta and phi angles
            If local is True, both theta and phi angles are measured from CCW hard stops,
            otherwise theta angles are in global coordinate and phi angles are
            measured from the phi arms.
        """
        if self.mode == self.thetaMode and phiAngles is not None:
            raise RuntimeError('Move phi arms in theta mode!')
        elif self.mode == self.phiMode and thetaAngles is not None:
            raise RuntimeError('Move theta arms in phi mode!')
        if thetaAngles is None and phiAngles is None:
            self.logger.info('both theta and phi angles are None, not moving!')
            return
        if not local and self.mode != self.normalMode:
            raise RuntimeError('In theta/phi mode (local) must be True!')

        if thetaAngles is not None:
            if np.isscalar(thetaAngles):
                thetaAngles = np.full(len(cobras), thetaAngles)
            elif len(thetaAngles) != len(cobras):
                raise RuntimeError("number of theta angles must match number of cobras")
        if phiAngles is not None:
            if np.isscalar(phiAngles):
                phiAngles = np.full(len(cobras), phiAngles)
            elif len(phiAngles) != len(cobras):
                raise RuntimeError("number of phi angles must match number of cobras")

        # calculate theta and phi moving amount
        cIds = self._getIndex(cobras)

        if thetaAngles is not None:
            if not local:
                toTheta = (thetaAngles - self.calibModel.tht0[cIds]) % (np.pi*2)
            else:
                toTheta = thetaAngles

            if self.mode == self.thetaMode:
                fromTheta = self.thetaInfo['angle'][cIds]
            else:
                fromTheta = self.cobraInfo['thetaAngle'][cIds]

            deltaTheta = toTheta - fromTheta
            badThetaIdx = np.where(np.isnan(deltaTheta))[0]
            if len(badThetaIdx) > 0:
                self.logger.warn(f'Last theta angle is unknown, not moving: {cIds[badThetaIdx]}')
                deltaTheta[badThetaIdx] = 0
        else:
            deltaTheta = None

        if phiAngles is not None:
            if not local:
                toPhi = phiAngles - self.calibModel.phiIn[cIds] - np.pi
            else:
                toPhi = phiAngles

            if self.mode == self.phiMode:
                fromPhi = self.phiInfo['angle'][cIds]
            else:
                fromPhi = self.cobraInfo['phiAngle'][cIds]

            deltaPhi = toPhi - fromPhi
            badPhiIdx = np.where(np.isnan(deltaPhi))[0]
            if len(badPhiIdx) > 0:
                self.logger.warn(f'Last phi angle is unknown, not moving: {cIds[badPhiIdx]}')
                deltaPhi[badPhiIdx] = 0
        else:
            deltaPhi = None

        # send the command
        #self.logger.debug(f'Moving deltaTheta = {deltaTheta}')
        #self.logger.debug(f'Moving deltaPhi = {deltaPhi}')
        self.moveDeltaAngles(cobras, deltaTheta, deltaPhi, thetaFast, phiFast)
        if local:
            if self.mode == self.thetaMode:
                return self.thetaInfo['angle'][cIds], np.zeros(len(cobras))
            elif self.mode == self.phiMode:
                return np.zeros(len(cobras)), self.phiInfo['angle'][cIds]
            else:
                return self.cobraInfo['thetaAngle'][cIds], self.cobraInfo['phiAngle'][cIds]
        else:
            return ((self.cobraInfo['thetaAngle'][cIds] + self.calibModel.tht0[cIds]) % (np.pi*2),
                    self.cobraInfo['phiAngle'][cIds] + self.calibModel.phiIn[cIds] + np.pi)

    def moveToPosition(self, cobras, positions, thetaFast=False, phiFast=False):
        """ move cobras to the given positions """
        if self.mode != self.normalMode:
            raise RuntimeError('Please switch to normal mode first!')
        if len(positions) != len(cobras):
            raise RuntimeError("number of positions must match number of cobras")

        thetas, phis, flags = self.pfi.positionsToAngles(cobras, positions)
        valid = (flags[:,0] & self.pfi.SOLUTION_OK) != 0
        if not np.all(valid):
            raise RuntimeError(f"Given positions are invalid: {np.where(valid)[0]}")
        self.moveToAngles(cobras, thetas[:,0], phis[:,0], thetaFast, phiFast)

        cIds = self._getIndex(cobras)
        return self.moveInfo['position'][cIds]

    def moveToHome(self, cobras, thetaEnable=False, phiEnable=False, thetaCCW=True):
        """ move arms to hard stop positions """
        if not thetaEnable and not phiEnable:
            self.logger.info('Both arms are disabled, ignore the command')
            return

        if thetaEnable:
            if self.mode == self.phiMode:
                raise RuntimeError('Home theta arms in phi mode!')
            if thetaCCW:
                thetaSteps = -self.thetaHomeSteps
            else:
                thetaSteps = self.thetaHomeSteps
        else:
            thetaSteps = 0

        if phiEnable:
            if self.mode == self.thetaMode:
                raise RuntimeError('Home phi arms in theta mode!')
            phiSteps = -self.phiHomeSteps
        else:
            phiSteps = 0

        cIds = self._getIndex(cobras)

        # move cobras and update information
        self.logger.info(f'home cobras: theta={thetaSteps}, phi={phiSteps}')
        if self.trajectoryMode:
            if thetaEnable:
                if thetaCCW:
                    thetas = np.zeros(len(cobras))
                else:
                    thetas = ((self.calibModel.tht1[cIds] - self.calibModel.tht0[cIds] + np.pi)
                              % (np.pi*2) + np.pi)
            else:
                thetas = self.cobraInfo['thetaAngle'][cIds]
            if phiEnable:
                phis = np.zeros(len(cobras))
            else:
                phis = self.cobraInfo['phiAngle'][cIds]
            self.cobraInfo['position'][cIds] = self.pfi.anglesToPositions(cobras, thetas, phis)

        else:
            self.pfi.moveAllSteps(cobras, thetaSteps, phiSteps, thetaFast=True, phiFast=True)
            # update current positions
            # Here we need to think about how to deal with it. 
            # There are dots, so we may wanto to skip this
            self.cobraInfo['position'][self.visibleIdx] = self.exposeAndExtractPositions(dbMatch = False)[self.visibleIdx]

        diff = None

        if thetaEnable:
            if thetaCCW:
                thetas = np.zeros(len(cobras))
            else:
                thetas = (self.calibModel.tht1[cIds] - self.calibModel.tht0[cIds] + np.pi) % (np.pi*2) + np.pi
            self.cobraInfo['thetaAngle'][cIds] = thetas
            if self.mode == self.thetaMode and self.thetaInfoIsValid:
                thetas2 = (np.angle(self.cobraInfo['position'] - self.thetaInfo['center'])
                           - self.thetaInfo['ccwHome'])[cIds]
                if thetaCCW:
                    self.thetaInfo['angle'][cIds] = (thetas2 + np.pi) % (np.pi*2) - np.pi
                    diff = self.thetaInfo['angle'][cIds]
                else:
                    self.thetaInfo['angle'][cIds] = (thetas2 + np.pi) % (np.pi*2) + np.pi
                    diff = calculus.diffAngle(self.thetaInfo['angle'],
                                              self.thetaInfo['cwHome'] - self.thetaInfo['ccwHome'])[cIds]
        else:
            thetas = self.cobraInfo['thetaAngle'][cIds]

        if phiEnable:
            phis = np.zeros(len(cobras))
            self.cobraInfo['phiAngle'][cIds] = phis
            if self.mode == self.phiMode and self.phiInfoIsValid:
                diff = calculus.diffAngle(np.angle(self.cobraInfo['position'] - self.phiInfo['center']),
                                          self.phiInfo['ccwHome'])[cIds]
                self.phiInfo['angle'][cIds] = diff
        else:
            phis = self.cobraInfo['phiAngle'][cIds]

        if self.mode == self.normalMode and thetaEnable and phiEnable:
            diff = np.abs(self.cobraInfo['position'][cIds] - self.pfi.anglesToPositions(cobras, thetas,
                                                                                        phis))

        return diff

    def setCurrentAngles(self, cobras, thetaAngles=None, phiAngles=None):
        """ set current theta and phi angles """
        if thetaAngles is not None:
            if np.isscalar(thetaAngles):
                thetaAngles = np.full(len(cobras), thetaAngles)
            elif len(thetaAngles) != len(cobras):
                raise RuntimeError("number of theta angles must match number of cobras")
        elif self.mode != self.phiMode:
            raise RuntimeError('Missing theta angles parameter!')

        if phiAngles is not None:
            if np.isscalar(phiAngles):
                phiAngles = np.full(len(cobras), phiAngles)
            elif len(phiAngles) != len(cobras):
                raise RuntimeError("number of phi angles must match number of cobras")
        elif self.mode != self.thetaMode:
            raise RuntimeError('Missing phi angles parameter!')

        cIds = self._getIndex(cobras)
        if self.mode == self.thetaMode:
            self.thetaInfo['angle'][cIds] = thetaAngles
        elif self.mode == self.phiMode:
            self.phiInfo['angle'][cIds] = phiAngles
        else:
            self.cobraInfo['thetaAngle'][cIds] = thetaAngles
            self.cobraInfo['phiAngle'][cIds] = phiAngles
        if thetaAngles is not None and phiAngles is not None:
            self.cobraInfo['position'][cIds] = self.pfi.anglesToPositions(cobras, thetaAngles, phiAngles)

    def setPhiGeometryFromRun(self, geometryRun, angle=0, onlyIfClear=True):
        """ set current phi center and hard stops """
        if onlyIfClear and self.phiInfoIsValid:
            return
        if isinstance(geometryRun, str):
            geometryRun = pathlib.Path(geometryRun)

        center = np.load(geometryRun / 'data' / 'phiCenter.npy')
        phiFW = np.load(geometryRun / 'data' / 'phiFW.npy')
        phiRV = np.load(geometryRun / 'data' / 'phiRV.npy')
        angRV = np.load(geometryRun / 'data' / 'phiAngRV.npy')

        cwHome = np.zeros(self.nCobras)
        ccwHome = np.zeros(self.nCobras)
        for m in range(self.nCobras):
            maxR = 0
            maxIdx = 0
            for n in range(phiFW.shape[1]):
                ccwH = np.angle(phiFW[m,n,0] - center[m])
                cwH = np.angle(phiRV[m,n,0] - center[m])
                curR = (cwH - ccwH) % (np.pi*2)
                if curR > maxR:
                    maxR = curR
                    maxIdx = n
            if self.phiInfoIsValid:
                lastR = (self.phiInfo['cwHome'][m] - self.phiInfo['ccwHome'][m]) % (np.pi*2)
            else:
                lastR = 0
            if curR >= lastR:
                ccwHome[m] = np.angle(phiFW[m,maxIdx,0] - center[m]) % (np.pi*2)
                cwHome[m] = np.angle(phiRV[m,maxIdx,0] - center[m]) % (np.pi*2)
            else:
                ccwHome[m] = self.phiInfo['ccwHome'][m]
                cwHome[m] = self.phiInfo['cwHome'][m]
                center[m] = self.phiInfo['center'][m]

        self.phiInfo['center'] = center
        self.phiInfo['ccwHome'] = ccwHome
        self.phiInfo['cwHome'] = cwHome
        self.phiInfo['angle'] = angle

        dAng = (self.phiInfo['cwHome'] - self.phiInfo['ccwHome']) % (np.pi*2)
        stopped = np.where(dAng < np.deg2rad(180.0))[0]
        if len(stopped) > 0:
            self.logger.error(f"phi ranges for cobras {stopped+1} are too small: "
                              f"CW={np.rad2deg(self.phiInfo['cwHome'][stopped])} "
                              f"CCW={np.rad2deg(self.phiInfo['ccwHome'][stopped])}")
            self.logger.error(f"     ranges={np.round(np.rad2deg(dAng[stopped]), 2)}")

        if not self.phiInfoIsValid:
            self.phiInfoIsValid = True

    def setPhiGeometry(self, center, ccwHome, cwHome, angle=0, onlyIfClear=True):
        if onlyIfClear and self.phiInfoIsValid:
            return

        self.phiInfo['center'] = center
        self.phiInfo['ccwHome'] = ccwHome
        self.phiInfo['cwHome'] = cwHome
        self.phiInfo['angle'] = angle

        dAng = (cwHome - ccwHome) % (np.pi*2)
        stopped = np.where(dAng < np.deg2rad(180.0))[0]
        if len(stopped) > 0:
            self.logger.error(f"phi ranges for cobras {stopped+1} are too small: "
                              f"CW={np.rad2deg(cwHome[stopped])} "
                              f"CCW={np.rad2deg(ccwHome[stopped])}")
            self.logger.error(f"ranges={np.round(np.rad2deg(dAng[stopped]), 2)}")

        if not self.phiInfoIsValid:
            self.phiInfoIsValid = True

    def setThetaGeometryFromRun(self, geometryRun, angle=0, onlyIfClear=True):
        if onlyIfClear and self.thetaInfoIsValid:
            return
        if isinstance(geometryRun, str):
            geometryRun = pathlib.Path(geometryRun)

        center = np.load(geometryRun / 'data' / 'thetaCenter.npy')
        thetaFW = np.load(geometryRun / 'data' / 'thetaFW.npy')
        thetaRV = np.load(geometryRun / 'data' / 'thetaRV.npy')
        angRV = np.load(geometryRun / 'data' / 'thetaAngRV.npy')

        cwHome = np.zeros(self.nCobras)
        ccwHome = np.zeros(self.nCobras)
        for m in range(self.nCobras):
            maxR = 0
            maxIdx = 0
            for n in range(thetaFW.shape[1]):
                ccwH = np.angle(thetaFW[m,n,0] - center[m])
                cwH = np.angle(thetaRV[m,n,0] - center[m])
                curR = (cwH - ccwH + np.pi) % (np.pi*2) + np.pi
                if curR > maxR:
                    maxR = curR
                    maxIdx = n
            if self.thetaInfoIsValid:
                lastR = (self.thetaInfo['cwHome'][m] - self.thetaInfo['ccwHome'][m] + np.pi) % (np.pi*2) + np.pi
            else:
                lastR = 0
            if curR >= lastR:
                ccwHome[m] = np.angle(thetaFW[m,maxIdx,0] - center[m]) % (np.pi*2)
                cwHome[m] = np.angle(thetaRV[m,maxIdx,0] - center[m]) % (np.pi*2)
            else:
                ccwHome[m] = self.thetaInfo['ccwHome'][m]
                cwHome[m] = self.thetaInfo['cwHome'][m]
                center[m] = self.thetaInfo['center'][m]

        self.thetaInfo['center'] = center
        self.thetaInfo['ccwHome'] = ccwHome
        self.thetaInfo['cwHome'] = cwHome
        self.thetaInfo['angle'] = angle

        dAng = (self.thetaInfo['cwHome'] - self.thetaInfo['ccwHome'] + np.pi) % (np.pi*2) + np.pi
        stopped = np.where(dAng < np.deg2rad(370.0))[0]
        if len(stopped) > 0:
            self.logger.error(f"theta ranges for cobras {stopped+1} are too small: "
                              f"CW={np.rad2deg(self.thetaInfo['cwHome'][stopped])} "
                              f"CCW={np.rad2deg(self.thetaInfo['ccwHome'][stopped])}")
            self.logger.error(f"     {np.round(np.rad2deg(dAng[stopped]), 2)}")

        if not self.thetaInfoIsValid:
            self.thetaInfoIsValid = True

    def setThetaGeometry(self, center, ccwHome, cwHome, angle=0, onlyIfClear=True):
        if onlyIfClear and self.thetaInfoIsValid:
            return

        self.thetaInfo['center'] = center
        self.thetaInfo['ccwHome'] = ccwHome
        self.thetaInfo['cwHome'] = cwHome
        self.thetaInfo['angle'] = angle

        dAng = (cwHome - ccwHome + np.pi) % (np.pi*2) + np.pi
        stopped = np.where(dAng < np.deg2rad(370.0))[0]
        if len(stopped) > 0:
            self.logger.error(f"theta ranges for cobras {stopped+1} are too small: "
                              f"CW={np.rad2deg(cwHome[stopped])} "
                              f"CCW={np.rad2deg(ccwHome[stopped])}")
            self.logger.error(f"range={np.round(np.rad2deg(dAng[stopped]), 2)}")

        if not self.thetaInfoIsValid:
            self.thetaInfoIsValid = True

    def thetaFWDone(self, thetas, k, needAtEnd=3,
                    closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(5)):
        """ Return a mask of the cobras which we deem at the FW theta limit.

        Args
        ----
        thetas : `np.array` of `complex`
          2 or 3d array of measured positions.
          0th axis is cobra, last axis is iteration
        k : integer
          the iteration we just made.
        needAtEnd : integer
          how many iterations we require to be at the same position
        closeEnough : radians
          how close the last needAtEnd point must be to each other.
        limitTolerance : radians
          how close to the known FW limit the last (kth) point must be.

        Returns
        -------
        doneMask : array of `bool`
          True for the cobras which are at the FW limit.
        endDiffs : array of radians
          The last `needAtEnd` angles to the limit
        """

        if self.thetaInfoIsValid:
            return calculus.mapDone(self.thetaInfo['center'], thetas, self.thetaInfo['cwHome'], k,
                                 needAtEnd=needAtEnd, closeEnough=closeEnough,
                                 limitTolerance=limitTolerance)
        else:
            return None, None

    def thetaRVDone(self, thetas, k, needAtEnd=3,
                    closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(5)):
        """ Return a mask of the cobras which we deem at the RV theta limit.

        See `thetaFWDone`
        """
        if self.thetaInfoIsValid:
            return calculus.mapDone(self.thetaInfo['center'], thetas, self.thetaInfo['ccwHome'], k,
                                 needAtEnd=needAtEnd, closeEnough=closeEnough,
                                 limitTolerance=limitTolerance)
        else:
            return None, None

    def phiFWDone(self, phis, k, needAtEnd=3, closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(5)):
        """ Return a mask of the cobras which we deem at the FW phi limit.

        See `thetaFWDone`
        """
        if self.phiInfoIsValid:
            return calculus.mapDone(self.phiInfo['center'], phis, self.phiInfo['cwHome'], k,
                                 needAtEnd=needAtEnd, closeEnough=closeEnough,
                                 limitTolerance=limitTolerance)
        else:
            return None, None

    def phiRVDone(self, phis, k, needAtEnd=3, closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(5)):
        """ Return a mask of the cobras which we deem at the RV phi limit.

        See `thetaFWDone`
        """
        if self.phiInfoIsValid:
            return calculus.mapDone(self.phiInfo['center'], phis, self.phiInfo['ccwHome'], k,
                                 needAtEnd=needAtEnd, closeEnough=closeEnough,
                                 limitTolerance=limitTolerance)
        else:
            return None, None

    def roundTripForPhi(self,
                        steps=250,
                        totalSteps=5000,
                        repeat=1,
                        fast=False,
                        phiOnTime=None,
                        limitOnTime=0.08,
                        limitSteps=5000):
        """ move all phi arms from CCW to CW hard stops and then back, in steps and return the positions """
        if self.trajectoryMode:
            raise RuntimeError('roundTrip command not available in trajectoryMode mode!')
        if self.mode != self.phiMode:
            raise RuntimeError('Switch to phi mode for this operation!')
        self.connect(False)

        # backup current on-times
        defaultOnTimeFast = np.copy([self.calibModel.motorOntimeFwd2,
                                      self.calibModel.motorOntimeRev2])
        defaultOnTimeSlow = np.copy([self.calibModel.motorOntimeSlowFwd2,
                                      self.calibModel.motorOntimeSlowRev2])

        # set fast on-time to a large value so it can cover the whole range, set slow on-time to the desired value.
        fastOnTime = [np.full(self.nCobras, limitOnTime)] * 2
        if phiOnTime is not None:
            if np.isscalar(phiOnTime):
                slowOnTime = [np.full(self.nCobras, phiOnTime)] * 2
            else:
                slowOnTime = phiOnTime
        elif fast:
            slowOnTime = defaultOnTimeFast
        else:
            slowOnTime = defaultOnTimeSlow

        # store steps and on-times
        dataPath = self.runManager.dataDir
        np.save(dataPath / 'steps', steps)
        np.save(dataPath / 'ontime', slowOnTime)

        # update ontimes for test
        self.calibModel.updateOntimes(phiFwd=fastOnTime[0], phiRev=fastOnTime[1], fast=True)
        self.calibModel.updateOntimes(phiFwd=slowOnTime[0], phiRev=slowOnTime[1], fast=False)

        iteration = totalSteps // steps
        phiFW = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)
        phiRV = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)

        self.pfi.resetMotorScaling(self.goodCobras, 'phi')
        self.logger.info(f'phi home {-limitSteps} steps')
        self.pfi.moveAllSteps(self.goodCobras, 0, -limitSteps)  # default is fast
        for n in range(repeat):
            self.cam.resetStack(f'phiForwardStack{n}.fits')

            # forward phi motor maps
            phiFW[self.visibleIdx, n, 0] = self.exposeAndExtractPositions(f'phiBegin{n}.fits', 
                dbMatch=True)[self.visibleIdx]

            self.cobraInfo['position'][self.visibleIdx] = phiFW[self.visibleIdx, n, 0]

            notdoneMask = np.zeros(len(phiFW), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} phi forward to {(k+1)*steps}')
                self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, steps, phiFast=False)
                # Here we turn-off dbMatch because we need good matching 
                phiFW[self.visibleIdx, n, k+1] = self.exposeAndExtractPositions(f'phiForward{n}N{k}.fits',
                                                    guess=phiFW[:, n, k], 
                                                    tolerance=1.0, dbMatch = True)[self.visibleIdx]

                self.cobraInfo['position'][self.visibleIdx] = phiFW[self.visibleIdx, n, k+1]
                doneMask, lastAngles = self.phiFWDone(phiFW[:,n,:], k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    phiFW[self.visibleIdx, n, k+2:] = phiFW[self.visibleIdx, n, k+1][:,None]
                    break
            if doneMask is not None and np.any(notdoneMask) and self.phiInfoIsValid:
                self.logger.warn(f'{(notdoneMask == True).sum()} cobras did not reach phi CW limit:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    with np.printoptions(precision=2, suppress=True):
                        self.logger.warn(f'  {str(c)}: {d}')

            # make sure it goes to the limit
            self.logger.info(f'{n+1}/{repeat} phi forward {limitSteps} to limit')
            self.pfi.moveAllSteps(self.goodCobras, 0, limitSteps)  # fast to limit

            # calculate phi centers
            centers = np.zeros(self.nCobras, 'complex')
            isValid = np.zeros(self.nCobras, 'bool')
            for c_i in self.goodIdx:
                x, y, r = calculus.circle_fitting(phiFW[c_i,n])
                ratio = r / self.calibModel.L2[c_i]
                if ratio < 1.25 and ratio > 0.8:
                    centers[c_i] = x + y * (1j)
                    isValid[c_i] = True
            for c_i in self.visibleIdx:
                if not isValid[c_i]:
                    modId = c_i // self.nCobrasPerModule
                    brdId = (c_i - modId*self.nCobrasPerModule) % 2
                    neighbors = np.arange(modId*self.nCobrasPerModule+brdId, (modId+1)*self.nCobrasPerModule, 2)
                    validNeighbors = neighbors[isValid[neighbors]]
                    if len(validNeighbors) > 0:
                        offset = np.average(centers[validNeighbors] - self.calibModel.centers[validNeighbors])
                        centers[c_i] = self.calibModel.centers[c_i] + offset
                    else:
                        centers[c_i] = np.average(phiFW[c_i, n])

            # reverse phi motor maps
            self.cam.resetStack(f'phiReverseStack{n}.fits')
            phiRV[self.visibleIdx, n, 0] = self.exposeAndExtractPositions(f'phiEnd{n}.fits', tolerance=1.0,
                                                        guess=centers, dbMatch=True)[self.visibleIdx]
            self.cobraInfo['position'][self.visibleIdx] = phiRV[self.visibleIdx, n, 0]

            notdoneMask = np.zeros(len(phiRV), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} phi backward to {(k+1)*steps}')
                self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, -steps, phiFast=False)
                # Use the last position for guess.
                phiRV[self.visibleIdx, n, k+1] = self.exposeAndExtractPositions(f'phiReverse{n}N{k}.fits',
                                                            guess=phiRV[:, n, k],
                                                            tolerance=1.0, dbMatch=True)[self.visibleIdx]
                self.cobraInfo['position'][self.visibleIdx] = phiRV[self.visibleIdx, n, k+1]
                doneMask, lastAngles = self.phiRVDone(phiRV[:,n,:], k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    phiRV[self.visibleIdx, n, k+2:] = phiRV[self.visibleIdx, n, k+1][:,None]
                    break

            if doneMask is not None and np.any(notdoneMask) and self.phiInfoIsValid:
                self.logger.warn(f'{(notdoneMask == True).sum()} did not reach phi CCW limit:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    with np.printoptions(precision=2, suppress=True):
                        self.logger.warn(f'  {str(c)}: {d}')

            # At the end, make sure the cobra back to the hard stop
            self.logger.info(f'{n+1}/{repeat} phi reverse {-limitSteps} steps to limit')
            self.pfi.moveAllSteps(self.goodCobras, 0, -limitSteps)  # fast to limit

        # restore ontimes after test
        self.calibModel.updateOntimes(phiFwd=defaultOnTimeFast[0], phiRev=defaultOnTimeFast[1], fast=True)
        self.calibModel.updateOntimes(phiFwd=defaultOnTimeSlow[0], phiRev=defaultOnTimeSlow[1], fast=False)

        # save calculation result
        np.save(dataPath / 'phiFW', phiFW)
        np.save(dataPath / 'phiRV', phiRV)

        self.setCurrentAngles(self.goodCobras, phiAngles=0)
        return dataPath, phiFW, phiRV

    def roundTripForTheta(self,
            steps=500,
            totalSteps=10000,
            repeat=1,
            fast=False,
            thetaOnTime=None,
            limitOnTime=0.08,
            limitSteps=10000,
            force=False
        ):
        """ move all theta arms from CCW to CW hard stops and then back, in steps and return the positions """
        if self.trajectoryMode:
            raise RuntimeError('roundTrip command not available in trajectoryMode mode!')
        if self.mode != self.thetaMode:
            raise RuntimeError('Switch to theta mode for this operation!')
        self.connect(False)

        # backup current on-times
        defaultOnTimeFast = np.copy([self.calibModel.motorOntimeFwd1,
                                      self.calibModel.motorOntimeRev1])
        defaultOnTimeSlow = np.copy([self.calibModel.motorOntimeSlowFwd1,
                                      self.calibModel.motorOntimeSlowRev1])

        # set fast on-time to a large value so it can cover the whole range, set slow on-time to the desired value.
        fastOnTime = [np.full(self.nCobras, limitOnTime)] * 2
        if thetaOnTime is not None:
            if np.isscalar(thetaOnTime):
                slowOnTime = [np.full(self.nCobras, thetaOnTime)] * 2
            else:
                slowOnTime = thetaOnTime
        elif fast:
            slowOnTime = defaultOnTimeFast
        else:
            slowOnTime = defaultOnTimeSlow

        # store steps and on-times
        dataPath = self.runManager.dataDir
        np.save(dataPath / 'steps', steps)
        np.save(dataPath / 'ontime', slowOnTime)

        # update ontimes for test
        self.calibModel.updateOntimes(thetaFwd=fastOnTime[0], thetaRev=fastOnTime[1], fast=True)
        self.calibModel.updateOntimes(thetaFwd=slowOnTime[0], thetaRev=slowOnTime[1], fast=False)

        iteration = totalSteps // steps
        thetaFW = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)
        thetaRV = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)

        self.pfi.resetMotorScaling(self.goodCobras, 'theta')
        self.logger.info(f'theta home {-limitSteps} steps')
        self.pfi.moveAllSteps(self.goodCobras, -limitSteps, 0)  # default is fast
        for n in range(repeat):
            self.cam.resetStack(f'thetaForwardStack{n}.fits')

            # forward theta motor maps
            thetaFW[self.visibleIdx, n, 0] = self.exposeAndExtractPositions(f'thetaBegin{n}.fits',dbMatch=False)[self.visibleIdx]

            self.cobraInfo['position'][self.visibleIdx] = thetaFW[self.visibleIdx, n, 0]

            notdoneMask = np.zeros(self.nCobras, 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} theta forward to {(k+1)*steps}')
                self.pfi.moveAllSteps(self.allCobras[notdoneMask], steps, 0, thetaFast=False)
                #thetaFW[self.visibleIdx, n, k+1] = self.exposeAndExtractPositions(f'thetaForward{n}N{k}.fits',
                #                                        arm='theta',guess=thetaFW[self.visibleIdx, n, k],tolerance=0.02)
                thetaFW[self.visibleIdx, n, k+1] = self.exposeAndExtractPositions(f'thetaForward{n}N{k}.fits',dbMatch=False)[self.visibleIdx]


                self.cobraInfo['position'][self.visibleIdx] = thetaFW[self.visibleIdx, n, k+1]

                doneMask, lastAngles = self.thetaFWDone(thetaFW[:,n,:], k)
                if doneMask is not None and not force:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    thetaFW[self.visibleIdx, n, k+2:] = thetaFW[self.visibleIdx, n, k+1][:,None]
                    break
            if doneMask is not None and np.any(notdoneMask) and self.thetaInfoIsValid and not force:
                self.logger.warn(f'{(notdoneMask == True).sum()} cobras did not reach theta CW limit:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    with np.printoptions(precision=2, suppress=True):
                        self.logger.warn(f'  {str(c)}: {d}')

            # make sure it goes to the limit
            self.logger.info(f'{n+1}/{repeat} theta forward {limitSteps} to limit')
            self.pfi.moveAllSteps(self.goodCobras, limitSteps, 0)  # fast to limit

            # reverse theta motor maps
            self.cam.resetStack(f'thetaReverseStack{n}.fits')
            #thetaRV[self.visibleIdx, n, 0] = self.exposeAndExtractPositions(f'thetaEnd{n}.fits',arm='theta',
            #                                        guess=thetaFW[self.visibleIdx, n, -1],tolerance=0.02)
            thetaRV[self.visibleIdx, n, 0] = self.exposeAndExtractPositions(f'thetaEnd{n}.fits',dbMatch=False)[self.visibleIdx]


            self.cobraInfo['position'][self.visibleIdx] = thetaRV[self.visibleIdx, n, 0]

            notdoneMask = np.zeros(self.nCobras, 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} theta backward to {(k+1)*steps}')
                self.pfi.moveAllSteps(self.allCobras[notdoneMask], -steps, 0, thetaFast=False)
                #thetaRV[self.visibleIdx, n, k+1] = self.exposeAndExtractPositions(f'thetaReverse{n}N{k}.fits',
                #                                    arm='theta',guess=thetaRV[self.visibleIdx, n, k],tolerance=0.02)
                thetaRV[self.visibleIdx, n, k+1] = self.exposeAndExtractPositions(f'thetaReverse{n}N{k}.fits',dbMatch=False)[self.visibleIdx]


                self.cobraInfo['position'][self.visibleIdx] = thetaRV[self.visibleIdx, n, k+1]
                doneMask, lastAngles = self.thetaRVDone(thetaRV[:,n,:], k)
                if doneMask is not None and not force:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    thetaRV[self.visibleIdx, n, k+2:] = thetaRV[self.visibleIdx, n, k+1][:,None]
                    break

            if doneMask is not None and np.any(notdoneMask) and self.thetaInfoIsValid and not force:
                self.logger.warn(f'{(notdoneMask == True).sum()} did not reach theta CCW limit:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    with np.printoptions(precision=2, suppress=True):
                        self.logger.warn(f'  {str(c)}: {d}')

            # At the end, make sure the cobra back to the hard stop
            self.logger.info(f'{n+1}/{repeat} theta reverse {-limitSteps} steps to limit')
            self.pfi.moveAllSteps(self.goodCobras, -limitSteps, 0)  # fast to limit

        # restore ontimes after test
        self.calibModel.updateOntimes(thetaFwd=defaultOnTimeFast[0], thetaRev=defaultOnTimeFast[1], fast=True)
        self.calibModel.updateOntimes(thetaFwd=defaultOnTimeSlow[0], thetaRev=defaultOnTimeSlow[1], fast=False)

        # save calculation result
        np.save(dataPath / 'thetaFW', thetaFW)
        np.save(dataPath / 'thetaRV', thetaRV)

        self.setCurrentAngles(self.goodCobras, thetaAngles=0)
        return dataPath, thetaFW, thetaRV

    def camResetStack(self, doStack=False):
        if not self.trajectoryMode:
            self.cam.resetStack(doStack)
