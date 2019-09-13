from importlib import reload
import logging
import numpy as np
from astropy.io import fits
import sep
from copy import deepcopy
import calculation
reload(calculation)

from mcs import camera
reload(camera)
from ics.cobraCharmer import pfi as pfiControl
from ics.cobraCharmer.utils import butler
from ics.cobraCharmer.fpgaState import fpgaState
from ics.cobraCharmer import cobraState

class Camera():
    def __init__(self, devId):
        from idsCamera import idsCamera
        self.devId = devId
        self.camera = idsCamera(devId)
        self.camera.setExpoureTime(20)
        self.data = None

    def expose(self, fn=None):
        self.data = self.camera.getCurrentFrame()
        if fn is not None:
            fits.writeto(fn, self.data, overwrite=True)
        return self.data

    def reload(self):
        del self.camera
        self.camera = idsCamera(self.devId)
        self.camera.setExpoureTime(20)
        self.data = None

class ModuleTest():
    def __init__(self, fpgaHost, xml, brokens=None, cam1Id=1, cam2Id=2, camSplit=26, logLevel=logging.INFO):

        self.logger = logging.getLogger('moduleTest')
        self.logger.setLevel(logLevel)

        self.runManager = butler.RunTree(doCreate=False)

        """ Init module 1 cobras """

        # NO, not 1!! Pass in moduleName, etc. -- CPL
        reload(pfiControl)
        self.allCobras = pfiControl.PFI.allocateCobraModule(1)
        self.fpgaHost = fpgaHost
        self.xml = xml

        # partition module 1 cobras into odd and even sets
        moduleCobras = {}
        for group in 1, 2:
            cm = range(group, 58, 2)
            mod = [1]*len(cm)
            moduleCobras[group] = pfiControl.PFI.allocateCobraList(zip(mod, cm))
        self.oddCobras = moduleCobras[1]
        self.evenCobras = moduleCobras[2]

        self.pfi = None
        self.cam = None

        self.thetaCenter = None
        self.thetaCCWHome = None
        self.thetaCWHome = None
        self.phiCenter = None
        self.phiCCWHome = None
        self.phiCWHome = None

    def _connect(self):
        self.runManager.newRun()
        # Initializing COBRA module
        self.pfi = pfiControl.PFI(fpgaHost=self.fpgaHost,
                                  doLoadModel=False,
                                  logDir=self.runManager.logDir)
        self.pfi.loadModel(self.xml)
        self.pfi.setFreq()

        # define the broken/good cobras
        self.setBrokenCobras()

        # initialize cameras
        self.cam = camera.cameraFactory(doClear=True, runManager=self.runManager)

        # init calculation library
        self.cal = calculation.Calculation(self.xml, None, None)

    def setBrokenCobras(self, brokens=None):
        """ define the broken/good cobras """
        if brokens is None:
            brokens = []
        visibles = [e for e in range(1, 58) if e not in brokens]
        self.badIdx = np.array(brokens) - 1
        self.goodIdx = np.array(visibles) - 1
        self.badCobras = np.array(getCobras(self.badIdx))
        self.goodCobras = np.array(getCobras(self.goodIdx))

        if hasattr(self, 'cal'):
            self.cal.setBrokenCobras(brokens)

    movesDtype = np.dtype(dict(names=['expId', 'spotId',
                                      'module', 'cobra',
                                      'phiSteps', 'phiOntime', 'phiOntimeScale',
                                      'thetaSteps','thetaOntime', 'thetaOntimeScale'],
                               formats=['U12', 'i4',
                                        'i2', 'i2',
                                        'f4', 'f4', 'f4',
                                        'f4', 'f4', 'f4']))

    def _saveMoveTable(self, expId, positions, indexMap):
        """ Save cobra move and spot information to a file.

        Args
        ----
        expId : `str`
          An exposure identifier. We want "PFxxNNNNNNNN".
        positions : `ndarray` of complex coordinates.
          What the matcher thinks is the cobra position.
        indexMap : `ndarray` of `int`
          For each of our cobras, the index of measured spot

        """
        moveTable = np.zeros(len(positions), dtype=self.movesDtype)
        moveTable['expId'][:] = expId
        if len(positions) != len(self.allCobras):
            raise RuntimeError("Craig is confused about cobra lists")

        for pos_i, pos in enumerate(positions):
            cobraInfo = self.allCobras[pos_i]
            cobraNum = self.pfi.calibModel.findCobraByModuleAndPositioner(cobraInfo.module,
                                                                          cobraInfo.cobraNum)
            moveInfo = fpgaState.cobraLastMove(cobraInfo)

            phiMotorId = cobraState.mapId(cobraNum, 'phi', 'ccw' if moveInfo['phiSteps'] < 0 else 'cw')
            thetaMotorId = cobraState.mapId(cobraNum, 'theta', 'ccw' if moveInfo['thetaSteps'] < 0 else 'cw')
            phiScale = cobraState.motorScales.get(phiMotorId, 1.0)
            thetaScale = cobraState.motorScales.get(thetaMotorId, 1.0)
            moveTable['spotId'][pos_i] = indexMap[pos_i]
            moveTable['module'][pos_i] = cobraInfo.module
            moveTable['cobra'][pos_i] = cobraInfo.cobraNum
            for field in ('phiSteps', 'phiOntime',
                          'thetaSteps', 'thetaOntime'):
                moveTable[field][pos_i] = moveInfo[field]
            moveTable['thetaOntimeScale'] = thetaScale
            moveTable['phiOntimeScale'] = phiScale

        movesPath = self.runManager.outputDir / "moves.npz"
        self.logger.debug(f'saving {len(moveTable)} moves to {movesPath}')
        if movesPath.exists():
            with open(movesPath, 'rb') as f:
                oldMoves = np.load(f)['moves']
            allMoves = np.concatenate([oldMoves, moveTable])
        else:
            allMoves = moveTable

        with open(movesPath, 'wb') as f:
            np.savez_compressed(f, moves=allMoves)

    def extractPositions(self, data1, data2=None, guess=None, tolerance=None):
        return self.cal.extractPositions(data1, data2, guess, tolerance)

    def exposeAndExtractPositions(self, name=None, guess=None, tolerance=None):
        """ Take an exposure, measure centroids, match to cobras, save info.

        Args
        ----
        name : `str`
           Additional name for saved image file. File _always_ gets PFS-compliant name.
        guess : `ndarray` of complex coordinates
           Where to center searches. By default uses the cobra center.
        tolerance : `float`
           Additional factor to scale search region by. 1 = cobra radius (phi+theta)

        Returns
        -------
        positions : `ndarray` of complex
           The measured positions of our cobras.
           If no matching spot found, return the cobra center.

        Note
        ----
        Not at all convinced that we should return anything if no matching spot found.

        """
        centroids, filename, bkgd = self.cam.expose(name)
        positions, indexMap = self.cal.matchPositions(centroids, guess=guess, tolerance=tolerance)
        self._saveMoveTable(filename.stem, positions, indexMap)

        return positions

    @staticmethod
    def dPhiAngle(target, source, doWrap=False, doAbs=False):
        d = np.atleast_1d(target - source)

        if doAbs:
            d[d<0] += 2*np.pi
            d[d>=2*np.pi] -= 2*np.pi

            return d

        if doWrap:
            lim = np.pi
        else:
            lim = 2*np.pi

        # d[d > lim] -= 2*np.pi
        d[d < -lim] += 2*np.pi

        return d

    @staticmethod
    def dThetaAngle(target, source, doWrap=False, doAbs=False):
        d = np.atleast_1d(target - source)

        if doAbs:
            d[d<0] += 2*np.pi
            d[d>=2*np.pi] -= 2*np.pi

            return d

        return d

    @staticmethod
    def _fullAngle(toPos, fromPos=None):
        """ Return ang of vector, 0..2pi """
        if fromPos is None:
            fromPos = 0+0j
        a = np.angle(toPos - fromPos)
        if np.isscalar(a):
            if a < 0:
                a += 2*np.pi
            if a >= 2*np.pi:
                a -= 2*np.pi
        else:
            a[a<0] += 2*np.pi
            a[a>=2*np.pi] -= 2*np.pi

        return a

    def getPhiCenters(self, fixOntime=None):
        outputDir = self.makePhiMotorMap('quickPhiScan.xml',
                                         repeat=1, phiOnTime=fixOntime, steps=200,
                                         totalSteps=4000, fast=False)
        dataDir = outputDir / 'data'

        phiCenters = np.load(dataDir / 'phiCenter.npy')
        return phiCenters

    def setPhiCentersFromRun(self, geometryRun):
        self.phiCenters = np.load(geometryRun / 'data' / 'phiCenter.npy')

    def setThetaCentersFromRun(self, geometryRun):
        self.thetaCenter = np.load(geometryRun / 'data' / 'thetaCenter.npy')

    def setThetaGeometryFromRun(self, geometryRun):
        self.setThetaCentersFromRun(geometryRun)

        FW = np.load(geometryRun / 'data' / 'thetaFW.npy')
        RV = np.load(geometryRun / 'data' / 'thetaRV.npy')
        self.thetaCCWHome = np.angle(FW[:,0,0] - self.thetaCenter)
        self.thetaCWHome = np.angle(RV[:,0,0] - self.thetaCenter)

    def moveToPhiAngle(self, idx=None, angle=60.0,
                       keepExistingPosition=False,
                       tolerance=1.0, maxTries=7, scaleFactor=8,
                       doFast=False):
        """
        Robustly move to a given phi angle.

        This uses only the angle between the phi center and the
        measured spot to determine where the phi motor is, and only
        the phi motor is moved. The significant drawback is that it
        requires the location of the phi center, which is not always
        known. But for the initial, post-phiMap move, we do.

        EXPECTS TO BE AT PHI HOME.

        Args
        ----
        idx : index or index array
          Which cobras to limit the move to.
        angle : `float`
          Degrees we want to move to from the CCW limit.
        keepExistingPosition : bool
          Do not reset the phi home position to where we are.
        tolerance : `float`
          How close we want to get, in degrees.
        maxTries: `int`
          How many moves to attempt.
        doFast : bool
          For the first move, use the fast map?
        """

        dtype = np.dtype(dict(names=['iteration', 'cobra', 'target', 'position', 'left', 'done'],
                              formats=['i2', 'i2', 'f4', 'f4', 'f4', 'i1']))

        # We do want a new stack of these images.
        self._connect()
        self.cam.resetStack(doStack=True)

        cobras = self.goodCobras
        cobras = np.array(cobras)
        if idx is not None:
            cobras = cobras[idx]
        moveList = []
        moves0 = np.zeros(len(cobras), dtype=dtype)

        try:
            phiCenters = self.phiCenters
        except AttributeError:
            raise RuntimeError("moduleTest needs to have been to told the phi Centers")

        tolerance = np.deg2rad(tolerance)

        # extract sources and fiber identification
        curPos = self.exposeAndExtractPositions(tolerance=0.2)
        if idx is not None:
            curPos = curPos[idx]
            phiCenters = phiCenters[idx]
        if keepExistingPosition:
            homeAngles = self.phiHomes
            curAngles = self._fullAngle(curPos, phiCenters)
            lastAngles = self.dPhiAngle(curAngles, homeAngles, doAbs=True)
        else:
            homeAngles = self._fullAngle(curPos, phiCenters)
            curAngles = homeAngles
            lastAngles = np.zeros(len(homeAngles))
            self.phiHomes = homeAngles

        targetAngles = np.full(len(homeAngles), np.deg2rad(angle))
        thetaAngles = targetAngles*0
        ntries = 1
        notDone = targetAngles != 0
        left = self.dPhiAngle(targetAngles,lastAngles, doWrap=True)

        moves = moves0.copy()
        moveList.append(moves)
        for i in range(len(cobras)):
            moveIdx = i
            cobraNum = cobras[i].cobraNum
            moves['iteration'][moveIdx] = 0
            moves['cobra'][moveIdx] = cobraNum
            moves['target'][moveIdx] = angle
            moves['position'][moveIdx] = curAngles[i]
            moves['left'][moveIdx] = left[i]
            moves['done'][moveIdx] = not notDone[i]
        while True:
            with np.printoptions(precision=2, suppress=True):
                self.logger.debug("to: %s", np.rad2deg(targetAngles)[notDone])
                self.logger.debug("at: %s", np.rad2deg(lastAngles)[notDone])
                self.logger.info("left try %d/%d, %d/%d: %s",
                                 ntries, maxTries,
                                 notDone.sum(), len(cobras),
                                 np.rad2deg(left)[notDone])
            self.pfi.moveThetaPhi(cobras[notDone],
                                  thetaAngles[notDone],
                                  left[notDone],
                                  phiFroms=lastAngles[notDone],
                                  phiFast=(doFast and ntries==1))

            # extract sources and fiber identification
            curPos = self.exposeAndExtractPositions(tolerance=0.2)
            if idx is not None:
                curPos = curPos[idx]
            a1 = self._fullAngle(curPos, phiCenters)
            atAngles = self.dPhiAngle(a1, homeAngles, doAbs=True)
            left = self.dPhiAngle(targetAngles,atAngles, doWrap=True)

            # check position errors
            notDone = np.abs(left) > tolerance

            moves = moves0.copy()
            moveList.append(moves)
            for i in range(len(cobras)):
                moveIdx = i
                cobraNum = cobras[i].cobraNum
                moves['iteration'][moveIdx] = ntries
                moves['cobra'][moveIdx] = cobraNum
                moves['target'][moveIdx] = angle
                moves['position'][moveIdx] = atAngles[i]
                moves['left'][moveIdx] = left[i]
                moves['done'][moveIdx] = not notDone[i]

            if not np.any(notDone):
                self.logger.info(f'Convergence sequence done after {ntries} iterations')
                break

            for c_i in np.where(notDone)[0]:

                tryDist = self.dPhiAngle(targetAngles[c_i], lastAngles[c_i], doWrap=True)[0]
                gotDist = self.dPhiAngle(atAngles[c_i], lastAngles[c_i], doWrap=True)[0]
                rawScale = abs(tryDist/gotDist)
                if abs(tryDist) > np.deg2rad(2) and (rawScale < 0.9 or rawScale > 1.1):
                    direction = 'ccw' if tryDist < 0 else 'cw'

                    if rawScale > 1:
                        scale = 1 + (rawScale - 1)/scaleFactor
                    else:
                        scale = 1/(1 + (1/rawScale - 1)/scaleFactor)

                    self.logger.info(f'{c_i+1} try={np.rad2deg(tryDist):0.3f} '
                                     f'at={np.rad2deg(atAngles[c_i]):0.2f} '
                                     f'got={np.rad2deg(gotDist):0.3f} '
                                     f'rawScale={rawScale:0.2f} scale={scale:0.2f}')
                    self.pfi.scaleMotorOntime(cobras[c_i], 'phi', direction, scale)

            lastAngles = atAngles
            if ntries >= maxTries:
                self.logger.warn(f'Reached max {maxTries} tries, '
                                 f'left: {[str(c) for c in cobras[np.where(notDone)]]}: '
                                 f'{np.rad2deg(left)[notDone]}')
                break
            ntries += 1

        moves = np.concatenate(moveList)
        movesPath = self.runManager.outputDir / "phiConvergence.npy"
        np.save(movesPath, moves)

        return self.runManager.runDir

    def moveToThetaAngle(self, idx=None, angle=60.0,
                         keepExistingPosition=False,
                         tolerance=1.0, maxTries=7, scaleFactor=10,
                         doFast=False):
        """
        Robustly move to a given theta angle.

        This uses only the angle between the theta center and the
        measured spot to determine where the theta motor is, and only
        the theta motor is moved.

        Args
        ----
        idx : index or index array
          Which cobras to limit the move to.
        angle : `float`
          Degrees we want to move to from the CCW limit.
        tolerance : `float`
          How close we want to get, in degrees.
        maxTries: `int`
          How many moves to attempt.
        doFast : bool
          For the first move, use the fast map?
        """

        dtype = np.dtype(dict(names=['iteration', 'cobra', 'target', 'position', 'left', 'done'],
                              formats=['i2', 'i2', 'f4', 'f4', 'f4', 'i1']))

        # We do want a new stack of these images.
        self._connect()
        self.cam.resetStack(doStack=True)

        cobras = self.goodCobras
        cobras = np.array(cobras)
        if idx is not None:
            cobras = cobras[idx]
        moveList = []
        moves0 = np.zeros(len(cobras), dtype=dtype)

        try:
            thetaCenters = self.thetaCenters
        except AttributeError:
            raise RuntimeError("moduleTest needs to have been told the thetaCenters")

        tolerance = np.deg2rad(tolerance)

        # extract sources and fiber identification
        curPos = self.exposeAndExtractPositions(tolerance=0.2)
        if idx is not None:
            curPos = curPos[idx]
            thetaCenters = thetaCenters[idx]
        if keepExistingPosition:
            homeAngles = self.thetaHomes
            curAngles = self._fullAngle(curPos, thetaCenters)
            lastAngles = self.dThetaAngle(curAngles, homeAngles, doAbs=True)
        else:
            homeAngles = self._fullAngle(curPos, thetaCenters)
            curAngles = homeAngles
            lastAngles = np.zeros(len(homeAngles))
            self.thetaHomes = homeAngles

        targetAngles = np.full(len(homeAngles), np.deg2rad(angle))
        phiAngles = targetAngles*0
        ntries = 1
        notDone = targetAngles != 0
        left = self.dThetaAngle(targetAngles,lastAngles, doWrap=True)

        moves = moves0.copy()
        moveList.append(moves)
        for i in range(len(cobras)):
            moveIdx = i
            cobraNum = cobras[i].cobraNum
            moves['iteration'][moveIdx] = 0
            moves['cobra'][moveIdx] = cobraNum
            moves['target'][moveIdx] = angle
            moves['position'][moveIdx] = curAngles[i]
            moves['left'][moveIdx] = left[i]
            moves['done'][moveIdx] = not notDone[i]
        while True:
            with np.printoptions(precision=2, suppress=True):
                self.logger.debug("to: %s", np.rad2deg(targetAngles)[notDone])
                self.logger.debug("at: %s", np.rad2deg(lastAngles)[notDone])
                self.logger.info("left try %d/%d, %d/%d: %s",
                                 ntries, maxTries,
                                 notDone.sum(), len(cobras),
                                 np.rad2deg(left)[notDone])
            self.pfi.moveThetaPhi(cobras[notDone],
                                  left[notDone],
                                  phiAngles[notDone],
                                  thetaFroms=lastAngles[notDone],
                                  thetaFast=(doFast and ntries==1))

            # extract sources and fiber identification
            curPos = self.exposeAndExtractPositions(tolerance=0.2)
            if idx is not None:
                curPos = curPos[idx]
            a1 = self._fullAngle(curPos, thetaCenters)
            atAngles = self.dThetaAngle(a1, homeAngles, doAbs=True)
            left = self.dThetaAngle(targetAngles,atAngles, doWrap=True)

            # check position errors
            notDone = np.abs(left) > tolerance

            moves = moves0.copy()
            moveList.append(moves)
            for i in range(len(cobras)):
                moveIdx = i
                cobraNum = cobras[i].cobraNum
                moves['iteration'][moveIdx] = ntries
                moves['cobra'][moveIdx] = cobraNum
                moves['target'][moveIdx] = angle
                moves['position'][moveIdx] = atAngles[i]
                moves['left'][moveIdx] = left[i]
                moves['done'][moveIdx] = not notDone[i]

            if not np.any(notDone):
                self.logger.info(f'Convergence sequence done after {ntries} iterations')
                break

            for c_i in np.where(notDone)[0]:

                tryDist = self.dThetaAngle(targetAngles[c_i], lastAngles[c_i], doWrap=True)[0]
                gotDist = self.dThetaAngle(atAngles[c_i], lastAngles[c_i], doWrap=True)[0]
                rawScale = abs(tryDist/gotDist)
                if abs(tryDist) > np.deg2rad(2) and (rawScale < 0.9 or rawScale > 1.1):
                    direction = 'ccw' if tryDist < 0 else 'cw'

                    if rawScale > 1:
                        scale = 1 + (rawScale - 1)/scaleFactor
                    else:
                        scale = 1/(1 + (1/rawScale - 1)/scaleFactor)

                    self.logger.info(f'{c_i+1} try={np.rad2deg(tryDist):0.3f} '
                                     f'at={np.rad2deg(atAngles[c_i]):0.2f} '
                                     f'got={np.rad2deg(gotDist):0.3f} '
                                     f'rawScale={rawScale:0.2f} scale={scale:0.2f}')
                    self.pfi.scaleMotorOntime(cobras[c_i], 'theta', direction, scale)

            lastAngles = atAngles
            if ntries >= maxTries:
                self.logger.warn(f'Reached max {maxTries} tries, '
                                 f'left: {[str(c) for c in cobras[np.where(notDone)]]}: '
                                 f'{np.rad2deg(left)[notDone]}')
                break
            ntries += 1

        moves = np.concatenate(moveList)
        movesPath = self.runManager.outputDir / "thetaConvergence.npy"
        np.save(movesPath, moves)

        return self.runManager.runDir

    def moveToXYfromHome(self, idx, targets, threshold=3.0, maxTries=8):
        """ function to move cobras to target positions """

        cobras = getCobras(idx)
        if idx is not None:
            targets = targets[idx]

        self.pfi.moveXYfromHome(cobras, targets, thetaThreshold=threshold, phiThreshold=threshold)

        ntries = 1
        keepMoving = targets != 0
        while True:
            # extract sources and fiber identification
            curPos = self.exposeAndExtractPositions(tolerance=0.2)
            if idx is not None:
                curPos = curPos[idx]
            # check position errors
            self.logger.info("to: %s", targets[keepMoving])
            self.logger.info("at: %s", curPos[keepMoving])
            self.logger.info("left (%d/%d): %s", keepMoving.sum(), len(targets),
                             targets[keepMoving] - curPos[keepMoving])

            notDone = np.abs(curPos - targets) > threshold
            if not np.any(notDone):
                print('Convergence sequence done')
                break
            if ntries > maxTries:
                print(f'Reach max {maxTries} tries, gave up, gave up on {np.where(notDone)}')
                break
            self.logger.info("left (%d/%d)", keepMoving.sum(), len(targets))

            ntries += 1

            # move again, skip bad center measurement
            # Yikes. No! Was wondering where replacement in calculations.py got used. -- CPL
            good = (curPos != self.pfi.calibModel.centers[idx])

            keepMoving = good & notDone
            self.pfi.moveXY(cobras[keepMoving], curPos[keepMoving], targets[keepMoving],
                            thetaThreshold=threshold, phiThreshold=threshold)

    def moveToXY(self, idx, theta, phi, threshold=3.0, maxTries=8):
        """ move positioners to given theta, phi angles.
        """

        cobras = getCobras(idx)
        if np.isscalar(theta):
            thetaAngles = np.full(len(cobras), theta, dtype='f4')
        elif idx is not None:
            thetaAngles = theta[idx]
        else:
            thetaAngles = theta

        if np.isscalar(phi):
            phiAngles = np.full(len(cobras), phi, dtype='f4')
        elif idx is not None:
            phiAngles = phi[idx]
        else:
            phiAngles = phi

        thetaAngles = self.pfi.thetaToLocal(cobras, thetaAngles)
        outTargets = self.pfi.anglesToPositions(self.allCobras, thetaAngles, phiAngles)

        # move to outTargets
        self.moveToXYfromHome(self.goodIdx, outTargets, threshold=threshold, maxTries=maxTries)

    def moveToThetaPhi(self, idx, theta, phi, threshold=3.0, maxTries=8):
        """ move positioners to given theta, phi angles.
        """

        cobras = getCobras(None)
        if np.isscalar(theta):
            thetaAngles = np.full(len(cobras), theta, dtype='f4')
        else:
            thetaAngles = theta
        if len(thetaAngles) != len(cobras):
            raise RuntimeError('number of thetas must match _total_ number of cobras')

        if np.isscalar(phi):
            phiAngles = np.full(len(cobras), phi, dtype='f4')
        else:
            phiAngles = phi
        if len(phiAngles) != len(cobras):
            raise RuntimeError('number of phis must match _total_ number of cobras')

        thetaAngles = self.pfi.thetaToLocal(cobras, thetaAngles)
        outTargets = self.pfi.anglesToPositions(self.allCobras,
                                                np.deg2rad(thetaAngles),
                                                np.deg2rad(phiAngles))

        # move to outTargets
        self.moveToXYfromHome(idx, outTargets, threshold=threshold, maxTries=maxTries)

    def moveBadCobrasOut(self):
        """ move bad cobras to point outwards """
        if len(self.badIdx) <= 0:
            return

        # Calculate up/down(outward) angles
        oddMoves = self.pfi.thetaToLocal(self.oddCobras, [np.deg2rad(270)]*len(self.oddCobras))
        #oddMoves[oddMoves>1.9*np.pi] = 0

        evenMoves = self.pfi.thetaToLocal(self.evenCobras, [np.deg2rad(90)]*len(self.evenCobras))
        #evenMoves[evenMoves>1.9*np.pi] = 0

        allMoves = np.zeros(57)
        allMoves[::2] = oddMoves
        allMoves[1::2] = evenMoves

        allSteps, _ = self.pfi.calculateSteps(np.zeros(57), allMoves, np.zeros(57), np.zeros(57))

        # Home
        self.pfi.moveAllSteps(self.badCobras, -10000, -5000)
        self.pfi.moveAllSteps(self.badCobras, -10000, -5000)

        # Move the bad cobras to up/down positions
        self.pfi.moveSteps(self.badCobras, allSteps[self.badIdx], np.zeros(len(self.badIdx)))

    def moveGoodCobrasOut(self, threshold=3.0, maxTries=8):
        """ move visible positioners to outwards positions, phi arms are moved out for 60 degrees
            (outTargets) so we can measure the arm angles
        """
        thetas = np.empty(57, dtype=float)
        thetas[::2] = self.pfi.thetaToLocal(self.oddCobras, np.full(len(self.oddCobras), np.deg2rad(270)))
        thetas[1::2] = self.pfi.thetaToLocal(self.evenCobras, np.full(len(self.evenCobras), np.deg2rad(90)))
        phis = np.full(57, np.deg2rad(60.0))
        outTargets = self.pfi.anglesToPositions(self.allCobras, thetas, phis)

        # Home the good cobras
        self.pfi.moveAllSteps(self.goodCobras, -10000, -5000)
        self.pfi.moveAllSteps(self.goodCobras, -10000, -5000)

        # move to outTargets
        self.moveToXYfromHome(self.goodIdx, outTargets[self.goodIdx], threshold=threshold, maxTries=maxTries)

        # move phi arms in
        self.pfi.moveAllSteps(self.goodCobras, 0, -5000)

    def makePhiMotorMap(
            self,
            newXml,
            repeat=3,
            steps=100,
            totalSteps=5000,
            fast=True,
            phiOnTime=None,
            updateGeometry=False,
            limitOnTime=0.06,
            resetScaling=True,
            delta=0.1):
        """ generate phi motor maps, it accepts custom phiOnTIme parameter.
            it assumes that theta arms have been move to up/down positions to avoid collision
            if phiOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime

            Example:
                makePhiMotorMap(xml, path, fast=True)             // update fast motor maps
                makePhiMotorMap(xml, path, fast=False)            // update slow motor maps
                makePhiMotorMap(xml, path, phiOnTime=0.06)        // motor maps for on-time=0.06
        """
        self._connect()
        defaultOnTime = deepcopy([self.pfi.calibModel.motorOntimeFwd1,
                                  self.pfi.calibModel.motorOntimeRev1,
                                  self.pfi.calibModel.motorOntimeFwd2,
                                  self.pfi.calibModel.motorOntimeRev2])
        defaultOnTimeSlow = deepcopy([self.pfi.calibModel.motorOntimeSlowFwd1,
                                      self.pfi.calibModel.motorOntimeSlowRev1,
                                      self.pfi.calibModel.motorOntimeSlowFwd2,
                                      self.pfi.calibModel.motorOntimeSlowRev2])

        # set fast on-time to a large value so it can move over whole range, set slow on-time to the test value.
        fastOnTime = [np.full(57, limitOnTime)] * 4
        if phiOnTime is not None:
            slowOnTime = defaultOnTimeSlow[:2] + [np.full(57, phiOnTime)] * 2
        elif fast:
            slowOnTime = defaultOnTimeSlow[:2] + defaultOnTime[2:]
        else:
            slowOnTime = defaultOnTimeSlow

        # update one-time for test
        self.pfi.calibModel.updateOntimes(*fastOnTime, fast=True)
        self.pfi.calibModel.updateOntimes(*slowOnTime, fast=False)

        # variable declaration for position measurement
        iteration = totalSteps // steps
        phiFW = np.zeros((57, repeat, iteration+1), dtype=complex)
        phiRV = np.zeros((57, repeat, iteration+1), dtype=complex)

        if resetScaling:
            self.pfi.resetMotorScaling(cobras=None, motor='phi')

        # record the phi movements
        dataPath = self.runManager.dataDir
        self.logger.info(f'phi home {-totalSteps} steps')
        self.pfi.moveAllSteps(self.goodCobras, 0, -totalSteps)
        for n in range(repeat):
            self.cam.resetStack(f'phiForwardStack{n}.fits')

            # forward phi motor maps
            phiFW[self.goodIdx, n, 0] = self.exposeAndExtractPositions(f'phiBegin{n}.fits')

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, 0, steps, phiFast=False)
                phiFW[self.goodIdx, n, k+1] = self.exposeAndExtractPositions(f'ph1Forward{n}N{k}.fits',
                                                                             guess=phiFW[self.goodIdx, n, k])

            # make sure it goes to the limit
            self.pfi.moveAllSteps(self.goodCobras, 0, 5000)

            # reverse phi motor maps
            self.cam.resetStack(f'phiReverseStack{n}.fits')
            phiRV[self.goodIdx, n, 0] = self.exposeAndExtractPositions(f'phiEnd{n}.fits',
                                                                       guess=phiFW[self.goodIdx, n, iteration])

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, 0, -steps, phiFast=False)
                phiRV[self.goodIdx, n, k+1] = self.exposeAndExtractPositions(f'phiReverse{n}N{k}.fits',
                                                                             guess=phiRV[self.goodIdx, n, k])

            # At the end, make sure the cobra back to the hard stop
            self.pfi.moveAllSteps(self.goodCobras, 0, -5000)

        # save calculation result
        np.save(dataPath / 'phiFW', phiFW)
        np.save(dataPath / 'phiRV', phiRV)

        # calculate centers and phi angles
        phiCenter, phiRadius, phiAngFW, phiAngRV, badRange = self.cal.phiCenterAngles(phiFW, phiRV)
        np.save(dataPath / 'phiCenter', phiCenter)
        np.save(dataPath / 'phiRadius', phiRadius)
        np.save(dataPath / 'phiAngFW', phiAngFW)
        np.save(dataPath / 'phiAngRV', phiAngRV)
        np.save(dataPath / 'badRange', badRange)

        # calculate average speeds
        phiSpeedFW, phiSpeedRV = self.cal.speed(phiAngFW, phiAngRV, steps, delta)
        np.save(dataPath / 'phiSpeedFW', phiSpeedFW)
        np.save(dataPath / 'phiSpeedRV', phiSpeedRV)

        # calculate motor maps by Johannes weighting
        phiMMFW, phiMMRV, bad = self.cal.motorMaps(phiAngFW, phiAngRV, steps, delta)
        bad[badRange] = True
        np.save(dataPath / 'phiMMFW', phiMMFW)
        np.save(dataPath / 'phiMMRV', phiMMRV)
        np.save(dataPath / 'bad', np.where(bad)[0])

        # calculate motor maps by average speeds
        phiMMFW2, phiMMRV2, bad2 = self.cal.motorMaps2(phiAngFW, phiAngRV, steps, delta)
        bad2[badRange] = True
        np.save(dataPath / 'phiMMFW2', phiMMFW2)
        np.save(dataPath / 'phiMMRV2', phiMMRV2)
        np.save(dataPath / 'bad2', np.where(bad2)[0])

        # update XML file, using Johannes weighting
        slow = not fast
        self.cal.updatePhiMotorMaps(phiMMFW, phiMMRV, bad, slow)
        if phiOnTime is not None:
            onTime = np.full(57, phiOnTime)
            self.cal.calibModel.updateOntimes(phiFwd=onTime, phiRev=onTime, fast=fast)
        if updateGeometry:
            self.cal.calibModel.updateGeometry(centers=phiCenter, phiArms=phiRadius)
        self.cal.calibModel.createCalibrationFile(self.runManager.outputDir / newXml, name='phiModel')
        self.cal.restoreConfig()

        # restore default setting
        self.pfi.loadModel(self.xml)

        return self.runManager.outputDir

    def thetaFWDone(self, thetas, k, needAtEnd=4, closeEnough=0.01, limitTolerance=np.deg2rad(2)):
        if self.thetaCenter is None or self.thetaCWHome is None or k+1 < needAtEnd:
            return np.zeros(len(thetas), bool)

        lastAngles = np.angle(thetas[:,0,k-needAtEnd+1:k+1] - self.thetaCenter[:,None])
        atEnd = np.abs(lastAngles[:,-1] - self.thetaCWHome) <= limitTolerance
        endDiff = np.abs(np.diff(lastAngles, axis=1))
        stable = np.all(endDiff <= closeEnough, axis=1)

        return atEnd & stable

    def thetaRVDone(self, thetas, k, needAtEnd=4, closeEnough=0.01, limitTolerance=np.deg2rad(2)):
        if self.thetaCenter is None or self.thetaCCWHome is None or k+1 < needAtEnd:
            return np.zeros(len(thetas), bool)

        lastAngles = np.angle(thetas[:,0,k-needAtEnd+1:k+1] - self.thetaCenter[:,None])
        atEnd = np.abs(lastAngles[:,-1] - self.thetaCCWHome) <= limitTolerance
        endDiff = np.abs(np.diff(lastAngles, axis=1))
        stable = np.all(endDiff <= closeEnough, axis=1)

        return atEnd & stable

    def makeThetaMotorMap(
            self,
            newXml,
            repeat=3,
            steps=100,
            totalSteps=10000,
            fast=True,
            thetaOnTime=None,
            updateGeometry=False,
            phiRunDir=None,
            limitOnTime=0.08,
            resetScaling=True,
            delta=np.deg2rad(5.0)
        ):
        # generate theta motor maps, it accepts custom thetaOnTIme parameter.
        # it assumes that phi arms have been move to ~60 degrees out to avoid collision
        # if thetaOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime
        # Example:
        #     makethetaMotorMap(xml, path, fast=True)               // update fast motor maps
        #     makethetaMotorMap(xml, path, fast=False)              // update slow motor maps
        #     makethetaMotorMap(xml, path, thetaOnTime=0.06)        // motor maps for on-time=0.06
        self._connect()
        defaultOnTime = deepcopy([self.pfi.calibModel.motorOntimeFwd1,
                                  self.pfi.calibModel.motorOntimeRev1,
                                  self.pfi.calibModel.motorOntimeFwd2,
                                  self.pfi.calibModel.motorOntimeRev2])
        defaultOnTimeSlow = deepcopy([self.pfi.calibModel.motorOntimeSlowFwd1,
                                      self.pfi.calibModel.motorOntimeSlowRev1,
                                      self.pfi.calibModel.motorOntimeSlowFwd2,
                                      self.pfi.calibModel.motorOntimeSlowRev2])

        if updateGeometry and phiRunDir is None:
            raise RuntimeError('To write geometry, need to be told the phiRunDir')

        # set fast on-time to a large value so it can move over whole range, set slow on-time to the test value.
        fastOnTime = [np.full(57, limitOnTime)] * 4
        if thetaOnTime is not None:
            slowOnTime = [np.full(57, thetaOnTime)] * 2 + defaultOnTimeSlow[2:]
        elif fast:
            slowOnTime = defaultOnTime[:2] + defaultOnTimeSlow[2:]
        else:
            slowOnTime = defaultOnTimeSlow

        # update one-time for test
        self.pfi.calibModel.updateOntimes(*fastOnTime, fast=True)
        self.pfi.calibModel.updateOntimes(*slowOnTime, fast=False)

        # variable declaration for position measurement
        iteration = totalSteps // steps
        thetaFW = np.zeros((57, repeat, iteration+1), dtype=complex)
        thetaRV = np.zeros((57, repeat, iteration+1), dtype=complex)

        if resetScaling:
            self.pfi.resetMotorScaling(cobras=None, motor='theta')

        #record the theta movements
        dataPath = self.runManager.dataDir
        self.logger.info(f'theta home {-totalSteps} steps')
        self.pfi.moveAllSteps(self.goodCobras, -totalSteps, 0)
        for n in range(repeat):
            self.cam.resetStack(f'thetaForwardStack{n}.fits')

            # forward theta motor maps
            thetaFW[self.goodIdx, n, 0] = self.exposeAndExtractPositions(f'thetaBegin{n}.fits')

            notdoneMask = np.zeros(len(thetaFW), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} theta forward to {(k+1)*steps}')
                self.pfi.moveAllSteps(self.goodCobras[np.where(notdoneMask)], steps, 0, thetaFast=False)
                thetaFW[self.goodIdx, n, k+1] = self.exposeAndExtractPositions(f'thetaForward{n}N{k}.fits',
                                                                               guess=thetaFW[self.goodIdx, n, k])
                doneMask = self.thetaFWDone(thetaFW, k)
                newlyDone = doneMask & notdoneMask
                if np.any(newlyDone):
                    notdoneMask &= ~doneMask
                    self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')

                # Should, if geometry is available,
            # make sure it goes to the limit
            self.logger.info(f'{n+1}/{repeat} theta forward {totalSteps} to limit')
            self.pfi.moveAllSteps(self.goodCobras, totalSteps, 0)

            # reverse theta motor maps
            self.cam.resetStack(f'thetaReverseStack{n}.fits')
            thetaRV[self.goodIdx, n, 0] = self.exposeAndExtractPositions(f'thetaEnd{n}.fits',
                                                                         guess=thetaFW[self.goodIdx, n, iteration])

            notdoneMask = np.zeros(len(thetaFW), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} theta backward to {(k+1)*steps}')
                self.pfi.moveAllSteps(self.goodCobras[np.where(notdoneMask)], -steps, 0, thetaFast=False)
                thetaRV[self.goodIdx, n, k+1] = self.exposeAndExtractPositions(f'thetaReverse{n}N{k}.fits',
                                                                               guess=thetaRV[self.goodIdx, n, k])

                doneMask = self.thetaRVDone(thetaRV, k)
                newlyDone = doneMask & notdoneMask
                if np.any(newlyDone):
                    notdoneMask &= ~doneMask
                    self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')

            # At the end, make sure the cobra back to the hard stop
            self.logger.info(f'{n+1}/{repeat} theta reverse {-totalSteps} steps to limit')
            self.pfi.moveAllSteps(self.goodCobras, -totalSteps, 0)

        # save calculation result
        np.save(dataPath / 'thetaFW', thetaFW)
        np.save(dataPath / 'thetaRV', thetaRV)

        # calculate centers and theta angles
        thetaCenter, thetaRadius, thetaAngFW, thetaAngRV, badRange = self.cal.thetaCenterAngles(thetaFW, thetaRV)
        np.save(dataPath / 'thetaCenter', thetaCenter)
        np.save(dataPath / 'thetaRadius', thetaRadius)
        np.save(dataPath / 'thetaAngFW', thetaAngFW)
        np.save(dataPath / 'thetaAngRV', thetaAngRV)
        np.save(dataPath / 'badRange', badRange)

        self.thetaCenters = thetaCenter
        self.thetaCCWHome = np.angle(thetaFW[:,0,0] - thetaCenter)
        self.thetaCCWHome = np.angle(thetaRV[:,0,0] - thetaCenter)

        # calculate average speeds
        thetaSpeedFW, thetaSpeedRV = self.cal.speed(thetaAngFW, thetaAngRV, steps, delta)
        np.save(dataPath / 'thetaSpeedFW', thetaSpeedFW)
        np.save(dataPath / 'thetaSpeedRV', thetaSpeedRV)

        # calculate motor maps in Johannes weighting
        thetaMMFW, thetaMMRV, bad = self.cal.motorMaps(thetaAngFW, thetaAngRV, steps, delta)
        bad[badRange] = True
        np.save(dataPath / 'thetaMMFW', thetaMMFW)
        np.save(dataPath / 'thetaMMRV', thetaMMRV)
        np.save(dataPath / 'bad', np.where(bad)[0])

        # calculate motor maps by average speeds
        thetaMMFW2, thetaMMRV2, bad2 = self.cal.motorMaps2(thetaAngFW, thetaAngRV, steps, delta)
        bad2[badRange] = True
        np.save(dataPath / 'thetaMMFW2', thetaMMFW2)
        np.save(dataPath / 'thetaMMRV2', thetaMMRV2)
        np.save(dataPath / 'bad2', np.where(bad2)[0])

        # update XML file, using Johannes weighting
        slow = not fast
        self.cal.updateThetaMotorMaps(thetaMMFW, thetaMMRV, bad, slow)
        if thetaOnTime is not None:
            onTime = np.full(57, thetaOnTime)
            self.cal.calibModel.updateOntimes(thetaFwd=onTime, thetaRev=onTime, fast=fast)
        if updateGeometry:
            phiCenter = np.load(phiRunDir / 'data' / 'phiCenter.npy')
            phiRadius = np.load(phiRunDir / 'data' / 'phiRadius.npy')
            phiFW = np.load(phiRunDir / 'data' / 'phiFW.npy')
            phiRV = np.load(phiRunDir / 'data' / 'phiRV.npy')

            thetaL, phiL, thetaCCW, thetaCW, phiCCW, phiCW = self.cal.geometry(thetaCenter, thetaRadius,
                                                                               thetaFW, thetaRV,
                                                                               phiCenter, phiRadius,
                                                                               phiFW, phiRV)
            self.cal.calibModel.updateGeometry(thetaCenter, thetaL, phiL)
            self.cal.calibModel.updateThetaHardStops(thetaCCW, thetaCW)
            self.cal.calibModel.updatePhiHardStops(phiCCW, phiCW)

            self.cal.calibModel.createCalibrationFile(self.runManager.outputDir / newXml)
        self.cal.restoreConfig()

        # restore default setting
        self.pfi.loadModel(self.xml)

        return self.runManager.outputDir

    def phiConvergenceTest(self, dataPath, margin=15.0, runs=50, tries=8, fast=True, finalAngle=None):
        # variable declaration for center measurement
        steps = 200
        iteration = 4000 // steps
        phiFW = np.zeros((57, iteration+1), dtype=complex)
        phiRV = np.zeros((57, iteration+1), dtype=complex)

        #record the phi movements
        self.pfi.moveAllSteps(self.goodCobras, 0, -5000, phiFast=True)
        data1 = self.cam1.expose()
        data2 = self.cam2.expose()
        phiFW[self.goodIdx, 0] = self.extractPositions(data1, data2)
        stack_image1 = data1
        stack_image2 = data2

        for k in range(iteration):
            self.pfi.moveAllSteps(self.goodCobras, 0, steps, phiFast=False)
            data1 = self.cam1.expose()
            data2 = self.cam2.expose()
            phiFW[self.goodIdx, k+1] = self.extractPositions(data1, data2, guess=phiFW[self.goodIdx, k])
            stack_image1 += data1
            stack_image2 += data2
        fits.writeto(dataPath + f'/phi1ForwardStack.fits.gz', stack_image1, overwrite=True)
        fits.writeto(dataPath + f'/phi2ForwardStack.fits.gz', stack_image2, overwrite=True)

        # make sure it goes to the limit
        self.pfi.moveAllSteps(self.goodCobras, 0, 5000, phiFast=True)

        # reverse phi motors
        data1 = self.cam1.expose()
        data2 = self.cam2.expose()
        phiRV[self.goodIdx, 0] = self.extractPositions(data1, data2, guess=phiFW[self.goodIdx, iteration])
        stack_image1 = data1
        stack_image2 = data2

        for k in range(iteration):
            self.pfi.moveAllSteps(self.goodCobras, 0, -steps, phiFast=False)
            data1 = self.cam1.expose()
            data2 = self.cam2.expose()
            phiRV[self.goodIdx, k+1] = self.extractPositions(data1, data2, guess=phiRV[self.goodIdx, k])
            stack_image1 += data1
            stack_image2 += data2
        fits.writeto(dataPath + f'/phi1ReverseStack.fits.gz', stack_image1, overwrite=True)
        fits.writeto(dataPath + f'/phi2ReverseStack.fits.gz', stack_image2, overwrite=True)

        # At the end, make sure the cobra back to the hard stop
        self.pfi.moveAllSteps(self.goodCobras, 0, -5000, phiFast=True)

        # save calculation result
        np.save(dataPath / 'phiFW', phiFW)
        np.save(dataPath / 'phiRV', phiRV)

        # variable declaration
        phiCenter = np.zeros(57, dtype=complex)
        phiRadius = np.zeros(57, dtype=float)
        phiHS = np.zeros(57, dtype=float)

        # measure centers
        for c in self.goodIdx:
            data = np.concatenate((phiFW[c].flatten(), phiRV[c].flatten()))
            x, y, r = calculation.circle_fitting(data)
            phiCenter[c] = x + y*(1j)
            phiRadius[c] = r

        # measure phi hard stops
        for c in self.goodIdx:
            phiHS[c] = np.angle(phiFW[c, 0] - phiCenter[c])

        # save calculation result
        np.save(dataPath / 'phiCenter', phiCenter)
        np.save(dataPath / 'phiRadius', phiRadius)
        np.save(dataPath / 'phiHS', phiHS)

        # convergence test
        phiData = np.zeros((57, runs, tries, 3))
        goodIdx = self.goodIdx
        zeros = np.zeros(len(goodIdx))
        centers = phiCenter[goodIdx]
        homes = phiHS[goodIdx]

        for i in range(runs):
            if runs > 1:
                angle = np.deg2rad(margin + (180 - 2 * margin) * i / (runs - 1))
            else:
                angle = np.deg2rad(90)
            self.pfi.moveThetaPhi(self.goodCobras, zeros, zeros + angle, phiFast=fast)
            cAngles, cPositions = self.measureAngles(centers, homes)
            phiData[goodIdx, i, 0, 0] = cAngles
            phiData[goodIdx, i, 0, 1] = np.real(cPositions)
            phiData[goodIdx, i, 0, 2] = np.imag(cPositions)

            for j in range(tries - 1):
                self.pfi.moveThetaPhi(self.goodCobras, zeros, angle - cAngles, phiFroms=cAngles, phiFast=fast)
                cAngles, cPositions = self.measureAngles(centers, homes)
                cAngles[cAngles>np.pi*(3/2)] -= np.pi*2
                phiData[goodIdx, i, j+1, 0] = cAngles
                phiData[goodIdx, i, j+1, 1] = np.real(cPositions)
                phiData[goodIdx, i, j+1, 2] = np.imag(cPositions)

            # home phi
            self.pfi.moveAllSteps(self.goodCobras, 0, -5000, phiFast=True)

        # save calculation result
        np.save(dataPath / 'phiData', phiData)

        if finalAngle is not None:
            angle = np.deg2rad(finalAngle)
            self.pfi.moveThetaPhi(self.goodCobras, zeros, zeros + angle, phiFast=fast)
            cAngles, cPositions = self.measureAngles(centers, homes)

            for j in range(tries - 1):
                self.pfi.moveThetaPhi(self.goodCobras, zeros, angle - cAngles, phiFroms=cAngles, phiFast=fast)
                cAngles, cPositions = self.measureAngles(centers, homes)
                cAngles[cAngles>np.pi*(3/2)] -= np.pi*2

    def thetaConvergenceTest(self, dataPath, margin=15.0, runs=50, tries=8, fast=True):
        # variable declaration for center measurement
        steps = 300
        iteration = 6000 // steps
        thetaFW = np.zeros((57, iteration+1), dtype=complex)
        thetaRV = np.zeros((57, iteration+1), dtype=complex)

        #record the theta movements
        self.pfi.moveAllSteps(self.goodCobras, -10000, 0, thetaFast=True)
        data1 = self.cam1.expose()
        data2 = self.cam2.expose()
        thetaFW[self.goodIdx, 0] = self.extractPositions(data1, data2)
        stack_image1 = data1
        stack_image2 = data2

        for k in range(iteration):
            self.pfi.moveAllSteps(self.goodCobras, steps, 0, thetaFast=False)
            data1 = self.cam1.expose()
            data2 = self.cam2.expose()
            thetaFW[self.goodIdx, k+1] = self.extractPositions(data1, data2)
            stack_image1 += data1
            stack_image2 += data2
        fits.writeto(dataPath + f'/theta1ForwardStack.fits.gz', stack_image1, overwrite=True)
        fits.writeto(dataPath + f'/theta2ForwardStack.fits.gz', stack_image2, overwrite=True)

        # make sure it goes to the limit
        self.pfi.moveAllSteps(self.goodCobras, 10000, 0, thetaFast=True)

        # reverse theta motors
        data1 = self.cam1.expose()
        data2 = self.cam2.expose()
        thetaRV[self.goodIdx, 0] = self.extractPositions(data1, data2)
        stack_image1 = data1
        stack_image2 = data2

        for k in range(iteration):
            self.pfi.moveAllSteps(self.goodCobras, -steps, 0, thetaFast=False)
            data1 = self.cam1.expose()
            data2 = self.cam2.expose()
            thetaRV[self.goodIdx, k+1] = self.extractPositions(data1, data2)
            stack_image1 += data1
            stack_image2 += data2
        fits.writeto(dataPath + f'/theta1ReverseStack.fits.gz', stack_image1, overwrite=True)
        fits.writeto(dataPath + f'/theta2ReverseStack.fits.gz', stack_image2, overwrite=True)

        # At the end, make sure the cobra back to the hard stop
        self.pfi.moveAllSteps(self.goodCobras, -10000, 0, thetaFast=True)

        # save calculation result
        np.save(dataPath / 'thetaFW', thetaFW)
        np.save(dataPath / 'thetaRV', thetaRV)

        # variable declaration
        thetaCenter = np.zeros(57, dtype=complex)
        thetaRadius = np.zeros(57, dtype=float)
        thetaHS = np.zeros(57, dtype=float)

        # measure centers
        for c in self.goodIdx:
            data = np.concatenate((thetaFW[c].flatten(), thetaRV[c].flatten()))
            x, y, r = calculation.circle_fitting(data)
            thetaCenter[c] = x + y*(1j)
            thetaRadius[c] = r

        # measure theta hard stops
        for c in self.goodIdx:
            thetaHS[c] = np.angle(thetaFW[c, 0] - thetaCenter[c])

        # save calculation result
        np.save(dataPath / 'thetaCenter', thetaCenter)
        np.save(dataPath / 'thetaRadius', thetaRadius)
        np.save(dataPath / 'thetaHS', thetaHS)

        # convergence test
        thetaData = np.zeros((57, runs, tries, 3))
        goodIdx = self.goodIdx
        zeros = np.zeros(len(goodIdx))
        centers = thetaCenter[goodIdx]
        homes = thetaHS[goodIdx]
        tGaps = ((self.pfi.calibModel.tht1 - self.pfi.calibModel.tht0) % (np.pi*2))[goodIdx]

        for i in range(runs):
            if runs > 1:
                angle = np.deg2rad(margin + (360 - 2 * margin) * i / (runs - 1))
            else:
                angle = np.deg2rad(180)
            self.pfi.moveThetaPhi(self.goodCobras, zeros + angle, zeros, thetaFast=fast)
            cAngles, cPositions = self.measureAngles(centers, homes)
            for k in range(len(goodIdx)):
                if angle > np.pi + tGaps[k] and cAngles[k] < tGaps[k] + 0.1:
                    cAngles[k] += np.pi*2
            thetaData[goodIdx, i, 0, 0] = cAngles
            thetaData[goodIdx, i, 0, 1] = np.real(cPositions)
            thetaData[goodIdx, i, 0, 2] = np.imag(cPositions)

            for j in range(tries - 1):
                dirs = angle > cAngles
                self.pfi.moveThetaPhi(self.goodCobras, angle - cAngles, zeros, thetaFroms=cAngles, thetaFast=fast)
                cAngles, cPositions = self.measureAngles(centers, homes)
                for k in range(len(goodIdx)):
                    lastAngle = thetaData[goodIdx[k], i, j, 0]
                    if dirs[k] and cAngles[k] < lastAngle - 0.01 and cAngles[k] < tGaps[k] + 0.1:
                        cAngles[k] += np.pi*2
                    elif not dirs[k] and cAngles[k] > lastAngle + 0.01 and cAngles[k] > np.pi*2 - 0.1:
                        cAngles[k] -= np.pi*2
                thetaData[goodIdx, i, j+1, 0] = cAngles
                thetaData[goodIdx, i, j+1, 1] = np.real(cPositions)
                thetaData[goodIdx, i, j+1, 2] = np.imag(cPositions)

            # home theta
            self.pfi.moveAllSteps(self.goodCobras, -10000, 0, thetaFast=True)

        # save calculation result
        np.save(dataPath / 'thetaData', thetaData)

    def measureAngles(self, centers, homes):
        """ measure positions and angles for good cobras """

        data1 = self.cam1.expose()
        data2 = self.cam2.expose()
        curPos = self.extractPositions(data1, data2)
        angles = (np.angle(curPos - centers) - homes) % (np.pi*2)
        return angles, curPos

    def convertXML(self, newXml):
        """ convert old XML to a new coordinate by taking 'phi homed' images
            assuming the cobra module is in horizontal setup
        """
        idx = self.goodIdx
        idx1 = idx[idx <= self.camSplit]
        idx2 = idx[idx > self.camSplit]
        oldPos = self.cal.calibModel.centers
        newPos = np.zeros(57, dtype=complex)

        # go home and measure new positions
        self.pfi.moveAllSteps(self.goodCobras, 0, -5000)
        data1 = sep.extract(self.cam1.expose().astype(float), 200)
        data2 = sep.extract(self.cam2.expose().astype(float), 200)
        home1 = np.array(sorted([(c['x'], c['y']) for c in data1], key=lambda t: t[0], reverse=True))
        home2 = np.array(sorted([(c['x'], c['y']) for c in data2], key=lambda t: t[0], reverse=True))
        newPos[idx1] = home1[:len(idx1), 0] + home1[:len(idx1), 1] * (1j)
        newPos[idx2] = home2[-len(idx2):, 0] + home2[-len(idx2):, 1] * (1j)

        # calculation tranformation
        offset1, scale1, tilt1, convert1 = calculation.transform(oldPos[idx1], newPos[idx1])
        offset2, scale2, tilt2, convert2 = calculation.transform(oldPos[idx2], newPos[idx2])

        split = self.camSplit + 1
        old = self.cal.calibModel
        new = deepcopy(self.cal.calibModel)
        new.centers[:split] = convert1(old.centers[:split])
        new.tht0[:split] = (old.tht0[:split]+tilt1) % (2*np.pi)
        new.tht1[:split] = (old.tht1[:split]+tilt1) % (2*np.pi)
        new.L1[:split] = old.L1[:split] * scale1
        new.L2[:split] = old.L2[:split] * scale1
        new.centers[split:] = convert2(old.centers[split:])
        new.tht0[split:] = (old.tht0[split:]+tilt2) % (2*np.pi)
        new.tht1[split:] = (old.tht1[split:]+tilt2) % (2*np.pi)
        new.L1[split:] = old.L1[split:] * scale2
        new.L2[split:] = old.L2[split:] * scale2

        # create a new XML file
        old.updateGeometry(new.centers, new.L1, new.L2)
        old.updateThetaHardStops(new.tht0, new.tht1)
        old.createCalibrationFile(newXml)
        self.cal.restoreConfig()

    def convertXML1(self, newXml):
        """ convert old XML to a new coordinate by taking 'phi homed' images
            assuming the cobra module is in horizontal setup
        """
        idx = self.goodIdx
        oldPos = self.cal.calibModel.centers
        newPos = np.zeros(57, dtype=complex)

        # go home and measure new positions
        self.pfi.moveAllSteps(self.goodCobras, 0, -5000)
        data = sep.extract(self.cam.expose().astype(float), 200)
        home = np.array(sorted([(c['x'], c['y']) for c in data1], key=lambda t: t[0], reverse=True))
        newPos[idx] = home[:len(idx), 0] + home[:len(idx), 1] * (1j)

        # calculation tranformation
        offset, scale, tilt, convert = calculation.transform(oldPos[idx], newPos[idx])

        old = self.cal.calibModel
        new = deepcopy(self.cal.calibModel)
        new.centers[:] = convert(old.centers)
        new.tht0[:] = (old.tht0 + tilt) % (2*np.pi)
        new.tht1[:] = (old.tht1 + tilt) % (2*np.pi)
        new.L1[:] = old.L1*scale
        new.L2[:] = old.L2*scale

        # create a new XML file
        old.updateGeometry(new.centers, new.L1, new.L2)
        old.updateThetaHardStops(new.tht0, new.tht1)
        old.createCalibrationFile(newXml)
        self.cal.restoreConfig()


def getCobras(cobs):
    # cobs is 0-indexed list
    return pfiControl.PFI.allocateCobraList(zip(np.full(len(cobs), 1), np.array(cobs) + 1))

