from importlib import reload
import logging
import numpy as np
from astropy.io import fits
import sep
from copy import deepcopy

from procedures.moduleTest import calculation
reload(calculation)
from procedures.moduleTest.speedModel import SpeedModel

from procedures.moduleTest.mcs import camera
reload(camera)
from ics.cobraCharmer import pfi as pfiControl
from ics.cobraCharmer.utils import butler
from ics.cobraCharmer.fpgaState import fpgaState
from ics.cobraCharmer import cobraState

def unwrappedAngle(angle, fromAngle, toAngle,
                   tripAngle=np.pi, allowAngle=np.pi/6):
    """ Adjust angles near 0 accounting for possible overshoots given the move.

    Args
    ----
    angle : radians
       The angle from CCW limit which we want to adjust.
    fromAngle : radians
       The start of the last move
    toAngle : radians
       The destination of the last move
    tripAngle : radians
       If the destination is within `tripAngle` of 0 or 2pi,
       possibly adjust `angle` if we believe 0 was crossed.
    allowAngle : radians
       If `tripAngle` was crossed and `angle` is within `allowAngle`
       of 0 on the other side, convert `angle` to be continuous with motion from
       `fromAngle` to `toAngle`.

    Returns
    -------
    angle : radians
       The full angle, which can be both slightly negative and slightly above 2pi.

    Given tripAngle is 90 and allowAngle is 30:

    If moving DOWN past 90 or UP from < 0 and target < 90
      turn 360..330 to 0..-30

    If moving UP past 270 or DOWN from > 360 and target > 270
      turn 0..45 to 360..405
    """

    angle = np.atleast_1d(angle)

    motion = toAngle - fromAngle
    motion = np.atleast_1d(motion)

    AND = np.logical_and

    # Allow motion down past 0: turn angle negative
    down_w = AND(motion < 0, toAngle < tripAngle)
    down_w |= AND(motion > 0, AND(fromAngle < 0, toAngle < tripAngle))
    down_w &= angle > (2*np.pi - allowAngle)
    angle[down_w] -= 2*np.pi

    # Allow motion up past 0: let angle go past 360
    up_w = AND(motion > 0, toAngle > 2*np.pi - tripAngle)
    up_w |= AND(motion < 0, AND(fromAngle > 2*np.pi, toAngle > 2*np.pi - tripAngle))
    up_w &= (angle < allowAngle)
    angle[up_w] += 2*np.pi

    return angle

def unwrappedPosition(pos, center, homeAngle, fromAngle, toAngle,
                      tripAngle=np.pi, allowAngle=np.pi/4):
    """ Return the angle for a given position accounting for overshoots across 0.

    See unwrappedAngle.
    """
    # Get the pos angle from the center, normalized to 0..2pi
    rawAngle = np.angle(pos - center)
    rawAngle = np.atleast_1d(rawAngle)
    rawAngle[rawAngle<0] += 2*np.pi

    # Get the angle w.r.t. home, normalized to 0..2pi
    diffAngle = rawAngle - homeAngle
    diffAngle[diffAngle<0] += 2*np.pi
    diffAngle[diffAngle>=2*np.pi] -= 2*np.pi

    return unwrappedAngle(diffAngle, fromAngle, toAngle,
                          tripAngle=tripAngle, allowAngle=allowAngle)

class ModuleTest():
    nCobrasPerModule = 57
    nModules = 42

    def __init__(self, fpgaHost, xml=None, version=None, moduleVersion=None, brokens=None, cam1Id=1,
                 cam2Id=2, camSplit=28, logLevel=logging.INFO):

        self.logger = logging.getLogger('moduleTest')
        self.logger.setLevel(logLevel)

        self.runManager = butler.RunTree(doCreate=False)

        """ Init module """
        reload(pfiControl)
        self.fpgaHost = fpgaHost
        self.xml = xml
        self.version = version
        self.moduleVersion = moduleVersion
        self.brokens = brokens
        self.camSplit = camSplit

        self.pfi = None
        self.cam = None

        self.thetaCenter = None
        self.thetaCCWHome = None
        self.thetaCWHome = None
        self.phiCenter = None
        self.phiCCWHome = None
        self.phiCWHome = None

        self.minScalingAngle = np.deg2rad(2.0)
        self.thetaModel = SpeedModel(p1=0.09)
        self.phiModel = SpeedModel(p1=0.07)

    def _connect(self):
        self.runManager.newRun()
        # Initializing COBRA module
        self.pfi = pfiControl.PFI(fpgaHost=self.fpgaHost,
                                  doLoadModel=False,
                                  logDir=self.runManager.logDir)
        if self.xml is not None:
            self.pfi.loadModel(self.xml)
        else:
            self.pfi.loadModel(None, self.version, self.moduleVersion)
            self.pfi.calibModel.fixModuleIds()
        self.pfi.setFreq()
        self.allCobras = np.array(self.pfi.getAllDefinedCobras())
        self.nCobras = len(self.allCobras)

        # define the broken/good cobras
        self.setBrokenCobras(self.brokens)

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

        # initialize cameras
        try:
            self.cam = camera.cameraFactory(doClear=True, runManager=self.runManager)
        except:
            self.cam = None

        # init calculation library
        self.cal = calculation.Calculation(self.pfi.calibModel, self.invisibleIdx+1, self.camSplit, self.badIdx+1)

    def setBrokenCobras(self, brokens=None):
        """ define the broken/good cobras """
        if brokens is None:
            brokens = [i+1 for i,c in enumerate(self.allCobras) if
                       self.pfi.calibModel.fiberIsBroken(c.cobraNum, c.module)]
        else:
            # update cobra status
            for c_i in brokens:
                moduleId = (c_i - 1) // self.nCobrasPerModule + 1
                cobraId = (c_i - 1) % self.nCobrasPerModule + 1
                self.pfi.calibModel.setCobraStatus(cobraId, moduleId, invisible=True)
        if len(brokens) > 0:
            self.logger.warn("setting invisible cobras: %s", brokens)

        visibles = [e for e in range(1, self.nCobras+1) if e not in brokens]
        self.invisibleIdx = np.array(brokens, dtype='i4') - 1
        self.visibleIdx = np.array(visibles, dtype='i4') - 1
        self.invisibleCobras = self.allCobras[self.invisibleIdx]
        self.visibleCobras = self.allCobras[self.visibleIdx]

        goodNums = [i+1 for i,c in enumerate(self.allCobras) if
                   self.pfi.calibModel.cobraIsGood(c.cobraNum, c.module)]
        badNums = [e for e in range(1, self.nCobras+1) if e not in goodNums]
        self.goodIdx = np.array(goodNums, dtype='i4') - 1
        self.badIdx = np.array(badNums, dtype='i4') - 1
        self.goodCobras = self.allCobras[self.goodIdx]
        self.badCobras = self.allCobras[self.badIdx]
        if len(badNums) > 0:
            self.logger.warn("setting bad cobras: %s", badNums)


    movesDtype = np.dtype(dict(names=['expId', 'spotId',
                                      'module', 'cobra',
                                      'thetaSteps','thetaOntime', 'thetaOntimeScale',
                                      'phiSteps', 'phiOntime', 'phiOntimeScale'],
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
        if len(positions) != len(self.visibleCobras):
            raise RuntimeError("Craig is confused about cobra lists")

        for pos_i, pos in enumerate(positions):
            cobraInfo = self.visibleCobras[pos_i]
            cobraNum = self.pfi.calibModel.findCobraByModuleAndPositioner(cobraInfo.module,
                                                                          cobraInfo.cobraNum)
            moveInfo = fpgaState.cobraLastMove(cobraInfo)

            phiMotorId = cobraState.mapId(cobraNum, 'phi', 'ccw' if moveInfo['phiSteps'] < 0 else 'cw')
            thetaMotorId = cobraState.mapId(cobraNum, 'theta', 'ccw' if moveInfo['thetaSteps'] < 0 else 'cw')
            phiScale = self.pfi.ontimeScales.get(phiMotorId, 1.0)
            thetaScale = self.pfi.ontimeScales.get(thetaMotorId, 1.0)
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

    def exposeAndExtractPositions(self, name=None, guess=None, tolerance=None, doFibreID=True):
        """ Take an exposure, measure centroids, match to cobras, save info.

        Args
        ----
        name : `str`
           Additional name for saved image file. File _always_ gets PFS-compliant name.
        guess : `ndarray` of complex coordinates
           Where to center searches. By default uses the cobra center.
        tolerance : `float`
           Additional factor to scale search region by. 1 = cobra radius (phi+theta)
        doFibreID : `bool`
          If set, do spot matching. Currently requires fiducial fibers.

        Returns
        -------
        positions : `ndarray` of complex
           The measured positions of our cobras.
           If no matching spot found, return the cobra center.

        Note
        ----
        Not at all convinced that we should return anything if no matching spot found.

        """
        centroids, filename, bkgd = self.cam.expose(name, doFibreID=doFibreID)
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

    @staticmethod
    def dAngle(angle):
        """ return angle between Pi and -Pi """
        d = (angle + np.pi) % (np.pi*2) - np.pi

        return d

    def getPhiCenters(self, fixOntime=None):
        outputDir, bad = self.makePhiMotorMap('quickPhiScan.xml',
                                         repeat=1, phiOnTime=fixOntime, steps=200,
                                         totalSteps=4000, fast=False)
        dataDir = outputDir / 'data'

        phiCenters = np.load(dataDir / 'phiCenter.npy')
        return phiCenters

    def setPhiCentersFromRun(self, geometryRun):
        self.phiCenter = np.load(geometryRun / 'data' / 'phiCenter.npy')

    def setPhiGeometryFromRun(self, geometryRun, onlyIfClear=True):
        if (onlyIfClear and (self.phiCenter is not None
                             and self.phiCWHome is not None
                             and self.phiCCWHome is not None)):
            return
        self.setPhiCentersFromRun(geometryRun)

        phiFW = np.load(geometryRun / 'data' / 'phiFW.npy')
        phiRV = np.load(geometryRun / 'data' / 'phiRV.npy')
        self.phiCCWHome = np.angle(phiFW[:,0,0] - self.phiCenter[:])
        self.phiCWHome = np.angle(phiRV[:,0,0] - self.phiCenter[:])
        dAng = self.phiCWHome - self.phiCCWHome
        dAng[dAng < 0] += 2*np.pi
        stopped = np.where(dAng < np.deg2rad(180.0))[0]
        if len(stopped) > 0:
            self.logger.error(f"phi ranges for cobras {stopped+1} are too small: "
                              f"CW={np.rad2deg(self.phiCWHome[stopped])} "
                              f"CCW={np.rad2deg(self.phiCCWHome[stopped])}")
            self.logger.error(f"     ranges={np.round(np.rad2deg(dAng[stopped]), 2)}")

    def setThetaCentersFromRun(self, geometryRun):
        self.thetaCenter = np.load(geometryRun / 'data' / 'thetaCenter.npy')

    def setThetaGeometryFromRun(self, geometryRun, onlyIfClear=True):
        if (onlyIfClear and (self.thetaCenter is not None
                             and self.thetaCWHome is not None
                             and self.thetaCCWHome is not None)):
            return

        self.setThetaCentersFromRun(geometryRun)

        thetaFW = np.load(geometryRun / 'data' / 'thetaFW.npy')
        thetaRV = np.load(geometryRun / 'data' / 'thetaRV.npy')
        self.thetaCCWHome = np.angle(thetaFW[:,0,0] - self.thetaCenter[:])
        self.thetaCWHome = np.angle(thetaRV[:,0,0] - self.thetaCenter[:])

        dAng = (self.thetaCWHome - self.thetaCCWHome + np.pi) % (np.pi*2) + np.pi
        stopped = np.where(dAng < np.deg2rad(370.0))[0]
        if len(stopped) > 0:
            self.logger.error(f"theta ranges for cobras {stopped+1} are too small: "
                              f"CW={np.rad2deg(self.thetaCWHome[stopped])} "
                              f"CCW={np.rad2deg(self.thetaCCWHome[stopped])}")
            self.logger.error(f"     {np.round(np.rad2deg(dAng[stopped]), 2)}")

    def moveToPhiAngle(self, idx=None, angle=60.0,
                       keepExistingPosition=False,
                       tolerance=np.rad2deg(0.005), maxTries=8,
                       scaleFactor=5,
                       doFast=False,
                       fromPhiCCW=True):
        """
        Robustly move to a given phi angle.

        This uses only the angle between the phi center and the
        measured spot to determine where the phi motor is, and only
        the phi motor is moved. The significant drawback is that it
        requires the location of the phi center, which is not always
        known. But for the initial, post-phiMap move, we do.

        EXPECTS TO BE AT PHI HOME if keepExistingPosition is False.

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
        scaleFactor: `float`
          What fraction of the motion error to apply to the motor scale. 1/scalefactor
        doFast : bool
          For the first move, use the fast map?
        fromPhiCCW : bool
          True if angle is measured from Phi CCW hard stop, otherwise from the theta arm
        """

        dtype = np.dtype(dict(names=['iteration', 'cobra', 'target', 'position', 'left', 'steps', 'done'],
                              formats=['i2', 'i2', 'f4', 'f4', 'f4', 'i4', 'i1']))

        # We do want a new stack of these images.
        self._connect()
        self.cam.resetStack(doStack=True)

        if idx is None:
            idx = self.goodIdx
        _idx = self.getIndexInGoodCobras(idx)
        cobras = np.array(self.allCobras[idx])
        moveList = []
        moves0 = np.zeros(len(cobras), dtype=dtype)

        if np.isscalar(angle):
            angle = np.full(len(cobras), angle)
        elif len(angle) == self.nCobras:
            angle = angle[idx]

        angle = np.deg2rad(angle)
        if not fromPhiCCW:
            angle -= np.pi + self.pfi.calibModel.phiIn[idx]

        if self.phiCenter is not None:
            phiCenters = self.phiCenter
        else:
            raise RuntimeError("moduleTest needs to have been to told the phi Centers")
        phiCenters = phiCenters[idx]

        tolerance = np.deg2rad(tolerance)

        # extract sources and fiber identification
        curPos = self.exposeAndExtractPositions(tolerance=0.2)
        curPos = curPos[_idx]
        if keepExistingPosition and hasattr(self, 'phiHomes'):
            homeAngles = self.phiHomes[idx]
            curAngles = self._fullAngle(curPos, phiCenters)
            lastAngles = self.dPhiAngle(curAngles, homeAngles, doAbs=True)
        else:
            homeAngles = self._fullAngle(curPos, phiCenters)
            curAngles = homeAngles
            lastAngles = np.zeros(len(homeAngles))
            if not hasattr(self, 'phiHomes'):
                self.phiHomes = np.zeros(self.nCobras)
            self.phiHomes[idx] = homeAngles

        targetAngles = np.full(len(homeAngles), angle)
        thetaAngles = targetAngles * 0
        ntries = 1
        notDone = targetAngles != 0
        left = self.dPhiAngle(targetAngles, lastAngles, doWrap=True)

        moves = moves0.copy()
        moveList.append(moves)
        for i in range(len(cobras)):
            cobraNum = cobras[i].cobraNum
            moves['iteration'][i] = 0
            moves['cobra'][i] = cobraNum
            moves['target'][i] = targetAngles[i]
            moves['position'][i] = lastAngles[i]
            moves['left'][i] = left[i]
            moves['done'][i] = not notDone[i]

        with np.printoptions(precision=2, suppress=True):
            self.logger.info("to: %s", np.rad2deg(targetAngles)[notDone])
            self.logger.info("at: %s", np.rad2deg(lastAngles)[notDone])
        while True:
            with np.printoptions(precision=2, suppress=True):
                self.logger.debug("to: %s", np.rad2deg(targetAngles)[notDone])
                self.logger.debug("at: %s", np.rad2deg(lastAngles)[notDone])
                self.logger.debug("try %d/%d, %d/%d cobras left: %s",
                                  ntries, maxTries,
                                  notDone.sum(), len(cobras),
                                  np.rad2deg(left)[notDone])
                self.logger.info("try %d/%d, %d/%d cobras left",
                                 ntries, maxTries,
                                 notDone.sum(), len(cobras))
            _, phiSteps = self.pfi.moveThetaPhi(cobras[notDone],
                                                thetaAngles[notDone],
                                                left[notDone],
                                                phiFroms=lastAngles[notDone],
                                                phiFast=(doFast and ntries==1))
            allPhiSteps = np.zeros(len(cobras), dtype='i4')
            allPhiSteps[notDone] = phiSteps

            # extract sources and fiber identification
            curPos = self.exposeAndExtractPositions(tolerance=0.2)
            curPos = curPos[_idx]
            a1 = self._fullAngle(curPos, phiCenters)
            atAngles = self.dPhiAngle(a1, homeAngles, doAbs=True)
            left = self.dPhiAngle(targetAngles, atAngles, doWrap=True)

            # Any cobras which were 0 steps away on the last move are done.
            lastNotDone = notDone.copy()
            tooCloseToMove = (allPhiSteps == 0)
            notDone[tooCloseToMove] = False

            # check position errors
            closeEnough = np.abs(left) <= tolerance
            notDone[closeEnough] = False

            moves = moves0.copy()
            for i in range(len(cobras)):
                cobraNum = cobras[i].cobraNum
                moves['iteration'][i] = ntries
                moves['cobra'][i] = cobraNum
                moves['target'][i] = targetAngles[i]
                moves['position'][i] = atAngles[i]
                moves['left'][i] = left[i]
                moves['steps'][i] = allPhiSteps[i]
                moves['done'][i] = not notDone[i]
            moveList[-1]['steps'][lastNotDone] = phiSteps
            moveList.append(moves)

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

                    if scale <= 0.75 or scale >= 1.25:
                        logCall = self.logger.info
                    else:
                        logCall = self.logger.debug

                    logCall(f'{c_i+1} at={np.rad2deg(atAngles[c_i]):0.2f} '
                            f'try={np.rad2deg(tryDist):0.2f} '
                            f'got={np.rad2deg(gotDist):0.2f} '
                            f'rawScale={rawScale:0.2f} scale={scale:0.2f}')
                    self.pfi.scaleMotorOntime(cobras[c_i], 'phi', direction, scale)

            lastAngles = atAngles
            if ntries >= maxTries:
                self.logger.warn(f'Reached max {maxTries} tries, {notDone.sum()} cobras left')
                self.logger.warn(f'   cobras: {[c.cobraNum for c in cobras[np.where(notDone)]]}')
                self.logger.warn(f'   left: {np.round(np.rad2deg(left)[notDone], 2)}')

                _, phiSteps = self.pfi.moveThetaPhi(cobras[notDone],
                                                    thetaAngles[notDone],
                                                    left[notDone],
                                                    phiFroms=lastAngles[notDone],
                                                    phiFast=(doFast and ntries==1),
                                                    doRun=False)
                self.logger.warn(f'   steps: {phiSteps}')

                break
            ntries += 1

        moves = np.concatenate(moveList)
        movesPath = self.runManager.outputDir / "phiConvergence.npy"
        np.save(movesPath, moves)

        return self.runManager.runDir

    def gotoSafeFromPhi60(self, phiAngle=60.0, tolerance=np.rad2deg(0.05)):
        """ Move cobras to nominal safe position: thetas OUT, phis in.
        Assumes phi is at 60deg and that we know thetaPositions.

        """

        angle = (180.0 - phiAngle) / 2.0
        thetaAngles = np.full(self.nCobras, -angle, dtype='f4')
        thetaAngles[np.arange(0,self.nCobras,2)] += 270
        thetaAngles[np.arange(1,self.nCobras,2)] += 90

        if not hasattr(self, 'thetaHomes'):
            keepExisting = False
        else:
            keepExisting = True

        run = self.moveToThetaAngle(None, angle=thetaAngles, tolerance=tolerance,
                                    keepExistingPosition=keepExisting, globalAngles=True)
        return run

    def gotoShippingFromPhi60(self, phiAngle=60.0, tolerance=np.rad2deg(0.05)):
        """ Move cobras to nominal safe shipping position: thetas IN, phis in.
        Assumes phi is at 60deg and that we know thetaPositions.

        """

        angle = (180.0 - phiAngle) / 2.0
        thetaAngles = np.full(len(self.allCobras), -angle, dtype='f4')
        thetaAngles[np.arange(0,self.nCobras,2)] += 90
        thetaAngles[np.arange(1,self.nCobras,2)] += 270

        if not hasattr(self, 'thetaHomes'):
            keepExisting = False
        else:
            keepExisting = True

        run = self.moveToThetaAngle(None, angle=thetaAngles, tolerance=tolerance,
                                    keepExistingPosition=keepExisting, globalAngles=True)
        return run

    def moveToThetaAngle(self, idx=None, angle=60.0,
                         keepExistingPosition=False,
                         tolerance=1.0, maxTries=7, scaleFactor=5,
                         globalAngles=False,
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
        globalAngle : `bool`
          Whether to use limit-based or module-based angles.
        tolerance : `float`
          How close we want to get, in degrees.
        maxTries: `int`
          How many moves to attempt.
        doFast : bool
          For the first move, use the fast map?
        """

        dtype = np.dtype(dict(names=['iteration', 'cobra', 'target', 'position', 'left', 'steps', 'done'],
                              formats=['i2', 'i2', 'f4', 'f4', 'f4', 'i4', 'i1']))

        # We do want a new stack of these images.
        self._connect()
        self.cam.resetStack(doStack=True)

        if idx is None:
            idx = self.goodIdx
        _idx = self.getIndexInGoodCobras(idx)
        cobras = np.array(self.allCobras[idx])
        moveList = []
        moves0 = np.zeros(len(cobras), dtype=dtype)

        if np.isscalar(angle):
            angle = np.full(len(cobras), angle)
        elif len(angle) == self.nCobras:
            angle = angle[idx]

        if self.thetaCenter is not None:
            thetaCenters = self.thetaCenter
        else:
            thetaCenters = self.pfi.calibModel.centers
        thetaCenters =  thetaCenters[idx]

        tolerance = np.deg2rad(tolerance)

        if not keepExistingPosition or not hasattr(self, 'thetaHomes'):
            # extract sources and fiber identification
            self.logger.info(f'theta backward -10000 steps to limit')
            self.pfi.moveAllSteps(cobras, -10000, 0)
            allCurPos = np.zeros(self.nCobras, dtype='complex')
            allCurPos[self.visibleIdx] = self.exposeAndExtractPositions(tolerance=0.2)
            homeAngles = self._fullAngle(allCurPos[idx], thetaCenters)
            if not hasattr(self, 'thetaHomes'):
                self.thetaHomes = np.zeros(self.nCobras)
                self.thetaAngles = np.zeros(self.nCobras)
            self.thetaHomes[idx] = homeAngles
            self.thetaAngles[idx] = 0
        homeAngles = self.thetaHomes[idx]
        lastAngles = self.thetaAngles[idx]

        targetAngles = np.deg2rad(angle)
        if globalAngles:
            targetAngles = (targetAngles - homeAngles) % (np.pi*2)

        phiAngles = targetAngles*0
        ntries = 1
        notDone = targetAngles != 0
        left = targetAngles - lastAngles

        moves = moves0.copy()
        moveList.append(moves)
        for i in range(len(cobras)):
            cobraNum = cobras[i].cobraNum
            moves['iteration'][i] = 0
            moves['cobra'][i] = cobraNum
            moves['target'][i] = targetAngles[i]
            moves['position'][i] = lastAngles[i]
            moves['left'][i] = left[i]
            moves['done'][i] = not notDone[i]

        with np.printoptions(precision=2, suppress=True):
            self.logger.info("to: %s", np.rad2deg(targetAngles)[notDone])
            self.logger.info("at: %s", np.rad2deg(lastAngles)[notDone])
        while True:
            with np.printoptions(precision=2, suppress=True):
                self.logger.debug("to: %s", np.rad2deg(targetAngles)[notDone])
                self.logger.debug("at: %s", np.rad2deg(lastAngles)[notDone])
                self.logger.debug("left try %d/%d, %d/%d: %s",
                                  ntries, maxTries,
                                  notDone.sum(), len(cobras),
                                  np.rad2deg(left)[notDone])
                self.logger.info("left try %d/%d, %d/%d",
                                 ntries, maxTries,
                                 notDone.sum(), len(cobras))
            thetaSteps, _ = self.pfi.moveThetaPhi(cobras[notDone],
                                                  left[notDone],
                                                  phiAngles[notDone],
                                                  thetaFroms=lastAngles[notDone],
                                                  thetaFast=(doFast and ntries==1))
            allThetaSteps = np.zeros(len(cobras), dtype='i4')
            allThetaSteps[notDone] = thetaSteps

            # extract sources and fiber identification
            curPos = self.exposeAndExtractPositions(tolerance=0.2)[_idx]

            # Get our angle w.r.t. home.
            atAngles = unwrappedPosition(curPos, thetaCenters, homeAngles,
                                         lastAngles, targetAngles)
            left = targetAngles - atAngles

            lastNotDone = notDone.copy()
            tooCloseToMove = (allThetaSteps == 0)
            notDone[tooCloseToMove] = False

            # check position errors
            closeEnough = np.abs(left) <= tolerance
            notDone[closeEnough] = False

            moves = moves0.copy()
            for i in range(len(cobras)):
                cobraNum = cobras[i].cobraNum
                moves['iteration'][i] = ntries
                moves['cobra'][i] = cobraNum
                moves['target'][i] = targetAngles[i]
                moves['position'][i] = atAngles[i]
                moves['left'][i] = left[i]
                moves['done'][i] = not notDone[i]
            moveList[-1]['steps'][lastNotDone] = thetaSteps
            moveList.append(moves)

            if not np.any(notDone):
                self.logger.info(f'Convergence sequence done after {ntries} iterations')
                break

            tryDist = targetAngles - lastAngles
            gotDist = atAngles - lastAngles
            for c_i in np.where(notDone)[0]:
                rawScale = np.abs(tryDist[c_i]/gotDist[c_i])
                if abs(tryDist[c_i]) > np.deg2rad(2) and (rawScale < 0.9 or rawScale > 1.1):
                    direction = 'ccw' if tryDist[c_i] < 0 else 'cw'

                    if rawScale > 1:
                        scale = 1 + (rawScale - 1)/scaleFactor
                    else:
                        scale = 1/(1 + (1/rawScale - 1)/scaleFactor)

                    if scale <= 0.75 or scale >= 1.25:
                        logCall = self.logger.info
                    else:
                        logCall = self.logger.debug

                    logCall(f'{c_i+1} at={np.rad2deg(atAngles[c_i]):0.2f} '
                            f'try={np.rad2deg(tryDist[c_i]):0.2f} '
                            f'got={np.rad2deg(gotDist[c_i]):0.2f} '
                            f'rawScale={rawScale:0.2f} scale={scale:0.2f}')
                    self.pfi.scaleMotorOntime(cobras[c_i], 'theta', direction, scale)

            lastAngles = atAngles
            self.thetaAngles[idx] = atAngles
            if ntries >= maxTries:
                self.logger.warn(f'Reached max {maxTries} tries, {notDone.sum()} cobras left')
                self.logger.warn(f'   cobras: {[c.cobraNum for c in cobras[np.where(notDone)]]}')
                self.logger.warn(f'   left: {np.round(np.rad2deg(left)[notDone], 2)}')

                thetaSteps, _ = self.pfi.moveThetaPhi(cobras[notDone],
                                                      left[notDone],
                                                      phiAngles[notDone],
                                                      thetaFroms=lastAngles[notDone],
                                                      thetaFast=(doFast and ntries==1),
                                                      doRun=False)
                self.logger.warn(f'   steps: {thetaSteps}')
                break
            ntries += 1

        moves = np.concatenate(moveList)
        movesPath = self.runManager.outputDir / 'thetaConvergence.npy'
        np.save(movesPath, moves)

        return self.runManager.runDir

    def moveToXYfromHome(self, idx, targets, ccwLimit=True, threshold=2.0, maxTries=8):
        """ function to move cobras to target positions """

        if idx is None:
            idx = self.goodIdx
        _idx = self.getIndexInGoodCobras(idx)
        cobras = self.allCobras[idx]
        if len(targets) != len(idx):
            if len(targets) == self.nCobras:
                targets = targets[idx]
            else:
                raise RuntimeError('number of targets must match idx')

        if ccwLimit:
            self.pfi.moveXYfromHome(cobras, targets, ccwLimit=ccwLimit)
        else:
            self.pfi.moveXYfromHomeSafe(cobras, targets, ccwLimit=ccwLimit)

        ntries = 1
        keepMoving = np.where(targets != 0)
        guess = self.pfi.calibModel.centers[self.visibleIdx]
        guess[_idx] = targets

        while True:
            # extract sources and fiber identification
            curPos = self.exposeAndExtractPositions(tolerance=0.2, guess=guess)
            curPos = curPos[_idx]
            # check position errors
            with np.printoptions(precision=2, suppress=True):
                self.logger.info("to: %s", targets[keepMoving])
                self.logger.info("at: %s", curPos[keepMoving])

            notDone = np.abs(curPos - targets) > threshold
            if not np.any(notDone):
                self.logger.info('Convergence sequence done')
                break
            if ntries > maxTries:
                self.logger.info(f'Reach max {maxTries} tries, gave up on {idx[notDone]}')
                break
            with np.printoptions(precision=2, suppress=True):
                self.logger.info("left (%d/%d): %s", len(keepMoving[0]), len(targets),
                                 targets[keepMoving] - curPos[keepMoving])

            ntries += 1
            keepMoving = np.where(notDone)
            self.pfi.moveXY(cobras[keepMoving], curPos[keepMoving], targets[keepMoving])

    def moveToThetaPhiFromHome(self, idx, theta, phi, globalAngles=False, ccwLimit=True, threshold=3.0, maxTries=8, scaleFactor=2.0):
        """ move positioners to given theta, phi angles.
        """

        if idx is None:
            idx = self.goodIdx
        _idx = self.getIndexInGoodCobras(idx)
        cobras = self.allCobras[idx]

        if np.isscalar(theta):
            thetaAngles = np.full(len(cobras), theta, dtype='f4')
        elif len(theta) == len(cobras):
            thetaAngles = theta
        elif len(theta) == self.nCobras:
            thetaAngles = theta[idx]
        else:
            raise RuntimeError('number of thetas must match number of cobras')

        if np.isscalar(phi):
            phiAngles = np.full(len(cobras), phi, dtype='f4')
        elif len(phi) == len(cobras):
            phiAngles = phi
        elif len(phi) == self.nCobras:
            phiAngles = phi[idx]
        else:
            raise RuntimeError('number of phis must match number of cobras')

        if globalAngles:
            thetaAngles = self.pfi.thetaToLocal(cobras, thetaAngles)

        gapTht = (self.pfi.calibModel.tht1[idx] - self.pfi.calibModel.tht0[idx] + np.pi) % (2*np.pi) - np.pi
        deltaPhi = phiAngles
        if ccwLimit:
            deltaTht = thetaAngles
            self.pfi.moveThetaPhi(cobras, deltaTht, deltaPhi, thetaFast=True, phiFast=True, ccwLimit=True)
        else:
            # use safe moves, 2x phi speed
            deltaTht = thetaAngles - (gapTht + np.pi*2)
            self.pfi.moveThetaPhi(cobras, deltaTht, deltaPhi, thetaFast=False, phiFast=True, ccwLimit=False)
        with np.printoptions(precision=2, suppress=True):
            self.logger.debug(f"moving: {np.stack((deltaTht, deltaPhi))}")

        ntries = 1
        targets = self.pfi.anglesToPositions(cobras, thetaAngles, phiAngles)
        guess = self.pfi.calibModel.centers[self.visibleIdx]
        guess[_idx] = targets
        keepMoving = np.where(targets != 0)
        delta = np.deg2rad(20)
        lastTht = None
        lastPhi = None
        scale = np.full((len(cobras),2), 1.0)

        while True:
            # extract sources and fiber identification
            curPos = self.exposeAndExtractPositions(tolerance=0.2, guess=guess)
            curPos = curPos[_idx]
            # check position errors
            with np.printoptions(precision=2, suppress=True):
                self.logger.debug("to: %s", targets[keepMoving])
                self.logger.debug("at: %s", curPos[keepMoving])

            notDone = np.abs(curPos - targets) > threshold
            if not np.any(notDone):
                self.logger.info('Convergence sequence done')
                break
            if ntries > maxTries:
                self.logger.info(f'Reach max {maxTries} tries, gave up on {idx[notDone]}')
                break
            keepMoving = np.where(notDone)
            with np.printoptions(precision=2, suppress=True):
                self.logger.info("left (%d/%d): %s", len(keepMoving[0]), len(targets),
                                 targets[notDone] - curPos[notDone])

            ntries += 1
            curTht, curPhi, _ = self.pfi.positionsToAngles(cobras, curPos)
            for c_i in keepMoving[0]:
                if thetaAngles[c_i] > np.pi and curTht[c_i,0] < gapTht[c_i] + delta:
                    curTht[c_i,0] += 2*np.pi
                elif thetaAngles[c_i] < np.pi and curTht[c_i,0] > np.pi*2 - delta:
                    curTht[c_i,0] -= 2*np.pi
                if lastTht is not None:
                    lastDeltaTht = thetaAngles[c_i] - lastTht[c_i]
                    if abs(lastDeltaTht) > delta:
                        rawScale = lastDeltaTht / (curTht[c_i,0] - lastTht[c_i])
                        if rawScale > 0:
                            engageScale = (rawScale - 1) / scaleFactor + 1
                            direction = 'cw' if lastDeltaTht>0 else 'ccw'
                            scale[c_i,0] = self.pfi.scaleMotorOntimeBySpeed(cobras[c_i], 'theta', direction, False, engageScale)
                if lastPhi is not None:
                    lastDeltaPhi = phiAngles[c_i] - lastPhi[c_i]
                    if abs(lastDeltaPhi) > delta:
                        rawScale = lastDeltaPhi / (curPhi[c_i,0] - lastPhi[c_i])
                        if rawScale > 0:
                            engageScale = (rawScale - 1) / scaleFactor + 1
                            direction = 'cw' if lastDeltaPhi>0 else 'ccw'
                            scale[c_i,1] = self.pfi.scaleMotorOntimeBySpeed(cobras[c_i], 'phi', direction, False, engageScale)
            self.logger.debug(f'Scaling factor: {np.round(scale.T, 2)}')
            lastTht, lastPhi = curTht[:,0], curPhi[:,0]

            _curTht = curTht[notDone,0]
            _curPhi = curPhi[notDone,0]
            deltaTht = thetaAngles[notDone] - _curTht
            deltaPhi = phiAngles[notDone] - _curPhi
            with np.printoptions(precision=2, suppress=True):
                self.logger.debug(f"moving: {np.stack((deltaTht, deltaPhi))}")
            self.pfi.moveThetaPhi(cobras[notDone], deltaTht, deltaPhi, _curTht, _curPhi, thetaFast=False, phiFast=False)

    def moveBadCobrasOut(self):
        """ move bad cobras to point outwards """
        if len(self.invisibleIdx) <= 0:
            return
        self._connect()

        # Calculate up/down(outward) angles
        oddMoves = self.pfi.thetaToLocal(self.oddCobras, [np.deg2rad(270)]*len(self.oddCobras))
        evenMoves = self.pfi.thetaToLocal(self.evenCobras, [np.deg2rad(90)]*len(self.evenCobras))

        allMoves = np.zeros(self.nCobras)
        allMoves[::2] = oddMoves
        allMoves[1::2] = evenMoves
        allMoves[allMoves>1.9*np.pi] = 0

        zeros = np.zeros(self.nCobras)
        allSteps, _ = self.pfi.calculateSteps(zeros, allMoves, zeros, zeros)

        # Home
        self.pfi.moveAllSteps(self.invisibleCobras, -10000, -5000)

        # Move the bad cobras to up/down positions
        self.pfi.moveSteps(self.invisibleCobras, allSteps[self.invisibleIdx], np.zeros(len(self.invisibleIdx)))

    def moveGoodCobrasOut(self, threshold=2.0, maxTries=8, phiAngle=60, phiToHome=True):
        """ move visible positioners to outwards positions, phi arms are moved out for 60 degrees
            (outTargets) so we can measure the arm angles
        """
        self._connect()

        thetas = np.empty(self.nCobras, dtype=float)
        thetas[::2] = self.pfi.thetaToLocal(self.oddCobras, np.full(len(self.oddCobras), np.deg2rad(270)))
        thetas[1::2] = self.pfi.thetaToLocal(self.evenCobras, np.full(len(self.evenCobras), np.deg2rad(90)))
        phis = np.full(self.nCobras, np.deg2rad(phiAngle))

        # Home the good cobras
        self.logger.info(f'theta/phi move 10000/-5000 steps to limit')
        self.pfi.moveAllSteps(self.goodCobras, 10000, -5000)

        # move to safe angles
        self.moveToThetaPhiFromHome(self.goodIdx, thetas[self.goodIdx], phis[self.goodIdx], ccwLimit=False, threshold=threshold, maxTries=maxTries)
#        targets = self.pfi.anglesToPositions(self.allCobras, thetas, phis)
#        self.moveToXYfromHome(self.goodIdx, targets[self.goodIdx], ccwLimit=False, threshold=threshold, maxTries=maxTries)

        if phiToHome:
            # move phi arms in
            self.pfi.moveAllSteps(self.goodCobras, 0, -5000)

    def makePhiMotorMap(
            self,
            newXml,
            repeat=3,
            steps=100,
            totalSteps=5000,
            fast=False,
            phiOnTime=None,
            updateGeometry=False,
            limitOnTime=0.08,
            limitSteps=5000,
            resetScaling=True,
            delta=0.1,
            fromHome=False
        ):
        """ generate phi motor maps, it accepts custom phiOnTIme parameter.
            it assumes that theta arms have been move to up/down positions to avoid collision
            if phiOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime

            Example:
                makePhiMotorMap(xml, path, fast=True)             // update fast motor maps
                makePhiMotorMap(xml, path, fast=False)            // update slow motor maps
                makePhiMotorMap(xml, path, phiOnTime=0.06)        // motor maps for on-time=0.06
        """
        self._connect()
        defaultOnTimeFast = deepcopy([self.pfi.calibModel.motorOntimeFwd2,
                                      self.pfi.calibModel.motorOntimeRev2])
        defaultOnTimeSlow = deepcopy([self.pfi.calibModel.motorOntimeSlowFwd2,
                                      self.pfi.calibModel.motorOntimeSlowRev2])

        # set fast on-time to a large value so it can move over whole range, set slow on-time to the test value.
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

        # update ontimes for test
        self.pfi.calibModel.updateOntimes(phiFwd=fastOnTime[0], phiRev=fastOnTime[1], fast=True)
        self.pfi.calibModel.updateOntimes(phiFwd=slowOnTime[0], phiRev=slowOnTime[1], fast=False)

        # variable declaration for position measurement
        iteration = totalSteps // steps
        phiFW = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)
        phiRV = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)

        if resetScaling:
            self.pfi.resetMotorScaling(cobras=None, motor='phi')

        # record the phi movements
        dataPath = self.runManager.dataDir
        self.logger.info(f'phi home {-limitSteps} steps')
        self.pfi.moveAllSteps(self.goodCobras, 0, -limitSteps)  # default is fast
        for n in range(repeat):
            self.cam.resetStack(f'phiForwardStack{n}.fits')

            # forward phi motor maps
            phiFW[self.visibleIdx, n, 0] = self.exposeAndExtractPositions(f'phiBegin{n}.fits')

            notdoneMask = np.zeros(len(phiFW), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} phi forward to {(k+1)*steps}')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, (k+1)*steps, phiFast=False)
                else:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, steps, phiFast=False)
                phiFW[self.visibleIdx, n, k+1] = self.exposeAndExtractPositions(f'phiForward{n}N{k}.fits',
                                                                             guess=phiFW[self.visibleIdx, n, k])
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, -(k+1)*steps)

                doneMask, lastAngles = self.phiFWDone(phiFW, n, k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    phiFW[self.visibleIdx, n, k+2:] = phiFW[self.visibleIdx, n, k+1][:,None]
                    break
            if doneMask is not None and np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} cobras did not reach phi CW limit:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    with np.printoptions(precision=2, suppress=True):
                        self.logger.warn(f'  {str(c)}: {d}')

            # make sure it goes to the limit
            self.logger.info(f'{n+1}/{repeat} phi forward {limitSteps} to limit')
            self.pfi.moveAllSteps(self.goodCobras, 0, limitSteps)  # fast to limit

            # reverse phi motor maps
            self.cam.resetStack(f'phiReverseStack{n}.fits')
            phiRV[self.visibleIdx, n, 0] = self.exposeAndExtractPositions(f'phiEnd{n}.fits',
                                                                       guess=phiFW[self.visibleIdx, n, iteration])
            notdoneMask = np.zeros(len(phiRV), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} phi backward to {(k+1)*steps}')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, -(k+1)*steps, phiFast=False)
                else:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, -steps, phiFast=False)
                phiRV[self.visibleIdx, n, k+1] = self.exposeAndExtractPositions(f'phiReverse{n}N{k}.fits',
                                                                             guess=phiRV[self.visibleIdx, n, k])
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, (k+1)*steps)

                doneMask, lastAngles = self.phiRVDone(phiRV, n, k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    phiRV[self.visibleIdx, n, k+2:] = phiRV[self.visibleIdx, n, k+1][:,None]
                    break

            if doneMask is not None and np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} did not reach phi CCW limit:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    with np.printoptions(precision=2, suppress=True):
                        self.logger.warn(f'  {str(c)}: {d}')

            # At the end, make sure the cobra back to the hard stop
            self.logger.info(f'{n+1}/{repeat} phi reverse {-limitSteps} steps to limit')
            self.pfi.moveAllSteps(self.goodCobras, 0, -limitSteps)  # fast to limit
        self.cam.resetStack()

        # restore ontimes after test
        self.pfi.calibModel.updateOntimes(phiFwd=defaultOnTimeFast[0], phiRev=defaultOnTimeFast[1], fast=True)
        self.pfi.calibModel.updateOntimes(phiFwd=defaultOnTimeSlow[0], phiRev=defaultOnTimeSlow[1], fast=False)

        # save calculation result
        np.save(dataPath / 'phiFW', phiFW)
        np.save(dataPath / 'phiRV', phiRV)

        # calculate centers and phi angles
        phiCenter, phiRadius, phiAngFW, phiAngRV, badRange = self.cal.phiCenterAngles(phiFW, phiRV)
        for short in badRange:
            if short in self.badIdx:
                self.logger.warn(f"phi range for {short+1:-2d} is short, but that was expected")
            else:
                self.logger.warn(f'phi range for {short+1:-2d} is short: '
                                 f'out={np.rad2deg(phiAngRV[short,0,0]):-6.2f} '
                                 f'back={np.rad2deg(phiAngRV[short,0,-1]):-6.2f}')
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
        if fromHome:
            phiMMFW, phiMMRV, bad = self.cal.motorMaps2(phiAngFW, phiAngRV, steps, delta)
        else:
            phiMMFW, phiMMRV, bad = self.cal.motorMaps(phiAngFW, phiAngRV, steps, delta)
        bad[badRange] = True
        np.save(dataPath / 'phiMMFW', phiMMFW)
        np.save(dataPath / 'phiMMRV', phiMMRV)
        np.save(dataPath / 'bad', np.where(bad)[0])

        # update XML file, using Johannes weighting
        slow = not fast
        self.cal.updatePhiMotorMaps(phiMMFW, phiMMRV, bad, slow)
        if phiOnTime is not None:
            if np.isscalar(phiOnTime):
                onTime = np.full(self.nCobras, phiOnTime)
                self.cal.calibModel.updateOntimes(phiFwd=onTime, phiRev=onTime, fast=fast)
            else:
                self.cal.calibModel.updateOntimes(phiFwd=phiOnTime[0], phiRev=phiOnTime[1], fast=fast)
        if updateGeometry:
            self.pfi.calibModel.updateGeometry(centers=phiCenter, phiArms=phiRadius)
            # These are not really correct, since the inner limit is pinned at 0. But it gives the range.
            # self.cal.updatePhiHardStops(ccw=phiAngFW[:,0,0], cw=phiAngFW[:,0,-1])
        self.pfi.calibModel.createCalibrationFile(self.runManager.outputDir / newXml, name='phiModel')

        # restore default setting ( really? why? CPL )
        # self.cal.restoreConfig()
        # self.pfi.loadModel(self.xml)

        self.setPhiGeometryFromRun(self.runManager.runDir, onlyIfClear=True)

        bad[self.badIdx] = False
        return self.runManager.runDir, np.where(bad)[0]

    def _mapDone(self, centers, points, limits, n, k,
                 needAtEnd=4, closeEnough=np.deg2rad(1),
                 limitTolerance=np.deg2rad(2)):
        """ Return a mask of the cobras which we deem at the axis limit.

        See thetaFWDone.
        """

        if centers is None or limits is None or k+1 < needAtEnd:
            return None, None

        lastAngles = np.angle(points[:,n,k-needAtEnd+1:k+1] - centers[:,None])
        atEnd = np.abs(lastAngles[:,-1] - limits) <= limitTolerance
        endDiff = np.abs(np.diff(lastAngles, axis=1))
        stable = np.all(endDiff <= closeEnough, axis=1)

        # Diagnostic: return the needAtEnd distances from the limit.
        anglesFromEnd = lastAngles - limits[:,None]

        return atEnd & stable, anglesFromEnd

    def thetaFWDone(self, thetas, n, k, needAtEnd=4,
                    closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(2)):
        """ Return a mask of the cobras which we deem at the FW theta limit.

        Args
        ----
        thetas : `np.array` of `complex`
          2 or 3d array of measured positions.
          0th axis is cobra, last axis is iteration
        n, k : integer
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

        return self._mapDone(self.thetaCenter, thetas, self.thetaCWHome, n, k,
                             needAtEnd=needAtEnd, closeEnough=closeEnough,
                             limitTolerance=limitTolerance)

    def thetaRVDone(self, thetas, n, k, needAtEnd=4, closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(2)):
        """ Return a mask of the cobras which we deem at the RV theta limit.

        See `thetaFWDone`
        """
        return self._mapDone(self.thetaCenter, thetas, self.thetaCCWHome, n, k,
                             needAtEnd=needAtEnd, closeEnough=closeEnough,
                             limitTolerance=limitTolerance)

    def phiFWDone(self, phis, n, k, needAtEnd=4, closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(2)):
        """ Return a mask of the cobras which we deem at the FW phi limit.

        See `thetaFWDone`
        """
        return self._mapDone(self.phiCenter, phis, self.phiCWHome, n, k,
                             needAtEnd=needAtEnd, closeEnough=closeEnough,
                             limitTolerance=limitTolerance)

    def phiRVDone(self, phis, n, k, needAtEnd=4, closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(2)):
        """ Return a mask of the cobras which we deem at the RV phi limit.

        See `thetaFWDone`
        """
        return self._mapDone(self.phiCenter, phis, self.phiCCWHome, n, k,
                             needAtEnd=needAtEnd, closeEnough=closeEnough,
                             limitTolerance=limitTolerance)

    def acquireThetaMotorMap(self,
                             steps=100,
                             repeat=1,
                             totalSteps=10000,
                             fast=False,
                             thetaOnTime=None,
                             limitOnTime=0.08,
                             limitSteps=10000,
                             resetScaling=True,
                             fromHome=False):
        """ """
        self._connect()
        defaultOnTimeFast = deepcopy([self.pfi.calibModel.motorOntimeFwd1,
                                      self.pfi.calibModel.motorOntimeRev1])
        defaultOnTimeSlow = deepcopy([self.pfi.calibModel.motorOntimeSlowFwd1,
                                      self.pfi.calibModel.motorOntimeSlowRev1])

        # set fast on-time to a large value so it can move over whole range, set slow on-time to the test value.
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

        # update ontimes for test
        self.pfi.calibModel.updateOntimes(thetaFwd=fastOnTime[0], thetaRev=fastOnTime[1], fast=True)
        self.pfi.calibModel.updateOntimes(thetaFwd=slowOnTime[0], thetaRev=slowOnTime[1], fast=False)

        # variable declaration for position measurement
        iteration = totalSteps // steps
        thetaFW = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)
        thetaRV = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)

        if resetScaling:
            self.pfi.resetMotorScaling(cobras=None, motor='theta')

        # record the theta movements
        self.logger.info(f'theta home {-limitSteps} steps')
        self.pfi.moveAllSteps(self.goodCobras, -limitSteps, 0)
        for n in range(repeat):
            self.cam.resetStack(f'thetaForwardStack{n}.fits')

            # forward theta motor maps
            thetaFW[self.visibleIdx, n, 0] = self.exposeAndExtractPositions(f'thetaBegin{n}.fits')

            notdoneMask = np.zeros(len(thetaFW), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} theta forward to {(k+1)*steps}')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], (k+1)*steps, 0, thetaFast=False)
                else:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], steps, 0, thetaFast=False)
                thetaFW[self.visibleIdx, n, k+1] = self.exposeAndExtractPositions(f'thetaForward{n}N{k}.fits')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], -(k+1)*steps, 0)

                doneMask, lastAngles = self.thetaFWDone(thetaFW, n, k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    thetaFW[self.visibleIdx, n, k+2:] = thetaFW[self.visibleIdx, n, k+1][:,None]
                    break

            if doneMask is not None and np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} did not reach theta CW limit:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    with np.printoptions(precision=2, suppress=True):
                        self.logger.warn(f'  {str(c)}: {d}')

            # make sure it goes to the limit
            self.logger.info(f'{n+1}/{repeat} theta forward {limitSteps} to limit')
            self.pfi.moveAllSteps(self.goodCobras, limitSteps, 0)

            # reverse theta motor maps
            self.cam.resetStack(f'thetaReverseStack{n}.fits')
            thetaRV[self.visibleIdx, n, 0] = self.exposeAndExtractPositions(f'thetaEnd{n}.fits')

            notdoneMask = np.zeros(len(thetaFW), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} theta backward to {(k+1)*steps}')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], -(k+1)*steps, 0, thetaFast=False)
                else:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], -steps, 0, thetaFast=False)
                thetaRV[self.visibleIdx, n, k+1] = self.exposeAndExtractPositions(f'thetaReverse{n}N{k}.fits')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], (k+1)*steps, 0)

                doneMask, lastAngles = self.thetaRVDone(thetaRV, n, k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    thetaRV[self.visibleIdx, n, k+1:] = thetaRV[self.visibleIdx, n, k+1][:,None]
                    break

            if doneMask is not None and np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} did not reach theta CCW limit:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    with np.printoptions(precision=2, suppress=True):
                        self.logger.warn(f'  {str(c)}: {d}')

            # At the end, make sure the cobra back to the hard stop
            self.logger.info(f'{n+1}/{repeat} theta reverse {-limitSteps} steps to limit')
            self.pfi.moveAllSteps(self.goodCobras, -limitSteps, 0)
        self.cam.resetStack()

        # restore ontimes after test
        self.pfi.calibModel.updateOntimes(thetaFwd=defaultOnTimeFast[0], thetaRev=defaultOnTimeFast[1], fast=True)
        self.pfi.calibModel.updateOntimes(thetaFwd=defaultOnTimeSlow[0], thetaRev=defaultOnTimeSlow[1], fast=False)

        # save calculation result
        dataPath = self.runManager.dataDir
        np.save(dataPath / 'thetaFW', thetaFW)
        np.save(dataPath / 'thetaRV', thetaRV)

        return self.runManager.runDir, thetaFW, thetaRV

    def reduceThetaMotorMap(self, newXml, runDir, steps,
                            thetaOnTime=None,
                            delta=None, fast=False,
                            phiRunDir=None,
                            updateGeometry=False,
                            fromHome=False):
        dataPath = runDir / 'data'

        # load calculation result
        thetaFW = np.load(dataPath / 'thetaFW.npy')
        thetaRV = np.load(dataPath / 'thetaRV.npy')

        # calculate centers and theta angles
        thetaCenter, thetaRadius, thetaAngFW, thetaAngRV, badRange = self.cal.thetaCenterAngles(thetaFW,
                                                                                                thetaRV)
        for short in badRange:
            self.logger.warn(f'theta range for {short+1:-2d} is short: '
                             f'out={np.rad2deg(thetaAngRV[short,0,0]):-6.2f} '
                             f'back={np.rad2deg(thetaAngRV[short,0,-1]):-6.2f}')
        np.save(dataPath / 'thetaCenter', thetaCenter)
        np.save(dataPath / 'thetaRadius', thetaRadius)
        np.save(dataPath / 'thetaAngFW', thetaAngFW)
        np.save(dataPath / 'thetaAngRV', thetaAngRV)
        np.save(dataPath / 'badRange', badRange)

        self.thetaCenter = thetaCenter
        self.thetaCCWHome = thetaAngFW[:,0,0]
        self.thetaCCWHome = thetaAngRV[:,0,0]

        # calculate average speeds
        thetaSpeedFW, thetaSpeedRV = self.cal.speed(thetaAngFW, thetaAngRV, steps, delta)
        np.save(dataPath / 'thetaSpeedFW', thetaSpeedFW)
        np.save(dataPath / 'thetaSpeedRV', thetaSpeedRV)

        # calculate motor maps in Johannes weighting
        if fromHome:
            thetaMMFW, thetaMMRV, bad = self.cal.motorMaps2(thetaAngFW, thetaAngRV, steps, delta)
        else:
            thetaMMFW, thetaMMRV, bad = self.cal.motorMaps(thetaAngFW, thetaAngRV, steps, delta)
        for bad_i in np.where(bad)[0]:
            self.logger.warn(f'theta map for {bad_i+1} is bad')
        bad[badRange] = True
        np.save(dataPath / 'thetaMMFW', thetaMMFW)
        np.save(dataPath / 'thetaMMRV', thetaMMRV)
        np.save(dataPath / 'bad', np.where(bad)[0])

        # update XML file, using Johannes weighting
        slow = not fast
        self.cal.updateThetaMotorMaps(thetaMMFW, thetaMMRV, bad, slow)
        if thetaOnTime is not None:
            if np.isscalar(thetaOnTime):
                onTime = np.full(self.nCobras, thetaOnTime)
                self.cal.calibModel.updateOntimes(thetaFwd=onTime, thetaRev=onTime, fast=fast)
            else:
                self.cal.calibModel.updateOntimes(thetaFwd=thetaOnTime[0], thetaRev=thetaOnTime[1], fast=fast)
        if updateGeometry:
            phiCenter = np.load(phiRunDir / 'data' / 'phiCenter.npy')
            phiRadius = np.load(phiRunDir / 'data' / 'phiRadius.npy')
            phiFW = np.load(phiRunDir / 'data' / 'phiFW.npy')
            phiRV = np.load(phiRunDir / 'data' / 'phiRV.npy')

            thetaL, phiL, thetaCCW, thetaCW, phiCCW, phiCW = self.cal.geometry(thetaCenter, thetaRadius,
                                                                               thetaFW, thetaRV,
                                                                               phiCenter, phiRadius,
                                                                               phiFW, phiRV)
            self.pfi.calibModel.updateGeometry(thetaCenter, thetaL, phiL)
            self.pfi.calibModel.updateThetaHardStops(thetaCCW, thetaCW)
            self.pfi.calibModel.updatePhiHardStops(phiCCW, phiCW)

            self.setThetaGeometryFromRun(runDir)

        self.pfi.calibModel.createCalibrationFile(self.runManager.outputDir / newXml)

        bad[self.badIdx] = False
        return self.runManager.runDir, np.where(bad)[0]

    def makeThetaMotorMap(self, newXml,
                          repeat=3,
                          steps=100,
                          totalSteps=10000,
                          fast=False,
                          thetaOnTime=None,
                          updateGeometry=False,
                          phiRunDir=None,
                          limitOnTime=0.08,
                          limitSteps=10000,
                          resetScaling=True,
                          delta=np.deg2rad(5.0),
                          fromHome=False):

        if updateGeometry and phiRunDir is None:
            raise RuntimeError('To write geometry, need to be told the phiRunDir')

        runDir, thetaFW, thetaRV = self.acquireThetaMotorMap(steps=steps, repeat=repeat, totalSteps=totalSteps,
                                                             fast=fast, thetaOnTime=thetaOnTime,
                                                             limitOnTime=limitOnTime, limitSteps=limitSteps,
                                                             resetScaling=resetScaling, fromHome=fromHome)
        runDir, duds = self.reduceThetaMotorMap(newXml, runDir, steps,
                                                thetaOnTime=thetaOnTime,
                                                delta=delta, fast=fast,
                                                phiRunDir=phiRunDir,
                                                updateGeometry=updateGeometry,
                                                fromHome=fromHome)
        return runDir, duds

    def acquireThetaMotorMap2(self,
                             steps=100,
                             repeat=1,
                             totalSteps=10000,
                             fast=False,
                             thetaOnTime=None,
                             limitOnTime=0.08,
                             limitSteps=10000,
                             resetScaling=True,
                             fromHome=False):
        # generate theta motor maps, it accepts custom thetaOnTIme parameter.
        # it assumes that theta arms already point to the outward direction and phi arms inwards,
        # also there is good geometry measurement and motor maps in the XML file.
        # cobras are divided into three non-intefere groups so phi arms can be moved all way out
        # if thetaOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime
        # Example:
        #     makethetaMotorMap(xml, path, fast=True)               // update fast motor maps
        #     makethetaMotorMap(xml, path, fast=False)              // update slow motor maps
        #     makethetaMotorMap(xml, path, thetaOnTime=0.06)        // motor maps for on-time=0.06
        self._connect()

        defaultOnTimeFast = deepcopy([self.pfi.calibModel.motorOntimeFwd1,
                                      self.pfi.calibModel.motorOntimeRev1])
        defaultOnTimeSlow = deepcopy([self.pfi.calibModel.motorOntimeSlowFwd1,
                                      self.pfi.calibModel.motorOntimeSlowRev1])

        # set fast on-time to a large value so it can move over whole range, set slow on-time to the test value.
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

        # update ontimes for test
        self.pfi.calibModel.updateOntimes(thetaFwd=fastOnTime[0], thetaRev=fastOnTime[1], fast=True)
        self.pfi.calibModel.updateOntimes(thetaFwd=slowOnTime[0], thetaRev=slowOnTime[1], fast=False)

        # variable declaration for position measurement
        iteration = totalSteps // steps
        thetaFW = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)
        thetaRV = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)
        extracted = np.zeros(self.nCobras, dtype=complex)
        safeAngle = np.empty(self.nCobras)
        safeAngle[::2] = 270
        safeAngle[1::2] = 90
        phiLimitSteps = 5000

        if resetMotorScaling:
            self.pfi.resetMotorScaling(cobras=None, motor='theta')

        # record theta movement
        for g in range(3):
            gIdx = self.goodIdx[self.goodIdx%3==g]
            self.pfi.moveAllSteps(self.allCobras[gIdx], -limitSteps, phiLimitSteps)

            for n in range(repeat):
                self.cam.resetStack(f'thetaForwardStack{n}G{g}.fits')

                # forward theta motor maps
                extracted[self.visibleIdx] = self.exposeAndExtractPositions(f'thetaBegin{n}G{g}.fits')
                thetaFW[gIdx, n, 0] = extracted[gIdx]

                notdoneMask = np.zeros(len(thetaFW), 'bool')
                notdoneMask[gIdx] = True
                for k in range(iteration):
                    self.logger.info(f'Group{g+1} {n+1}/{repeat} theta forward to {(k+1)*steps}')
                    if fromHome:
                        self.pfi.moveAllSteps(self.allCobras[notdoneMask], (k+1)*steps, 0, thetaFast=False)
                    else:
                        self.pfi.moveAllSteps(self.allCobras[notdoneMask], steps, 0, thetaFast=False)
                    extracted[self.visibleIdx] = self.exposeAndExtractPositions(f'thetaForward{n}N{k}G{g}.fits')
                    thetaFW[gIdx, n, k+1] = extracted[gIdx]
                    if fromHome:
                        self.pfi.moveAllSteps(self.allCobras[notdoneMask], -(k+1)*steps, 0)

                    doneMask, lastAngles = self.thetaFWDone(thetaFW, n, k)
                    if doneMask is not None:
                        newlyDone = doneMask & notdoneMask
                        if np.any(newlyDone):
                            notdoneMask &= ~doneMask
                            self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                    if not np.any(notdoneMask):
                        thetaFW[gIdx, n, k+2:] = thetaFW[gIdx, n, k+1][:,None]
                        break

                if doneMask is not None and np.any(notdoneMask):
                    self.logger.warn(f'{(notdoneMask == True).sum()} did not finish:')
                    for c_i in np.where(notdoneMask)[0]:
                        c = self.allCobras[c_i]
                        d = np.rad2deg(lastAngles[c_i])
                        self.logger.warn(f'  {str(c)}: {np.round(d, 2)}')

                # make sure it goes to the limit
                self.logger.info(f'{n+1}/{repeat} theta forward {limitSteps} to limit')
                self.pfi.moveAllSteps(self.allCobras[gIdx], limitSteps, 0)

                # reverse theta motor maps
                self.cam.resetStack(f'thetaReverseStack{n}G{g}.fits')
                extracted[self.visibleIdx] = self.exposeAndExtractPositions(f'thetaEnd{n}G{g}.fits')
                thetaRV[gIdx, n, 0] = extracted[gIdx]

                notdoneMask = np.zeros(len(thetaFW), 'bool')
                notdoneMask[gIdx] = True
                for k in range(iteration):
                    self.logger.info(f'Group{g+1} {n+1}/{repeat} theta backward to {(k+1)*steps}')
                    if fromHome:
                        self.pfi.moveAllSteps(self.allCobras[notdoneMask], -(k+1)*steps, 0, thetaFast=False)
                    else:
                        self.pfi.moveAllSteps(self.allCobras[notdoneMask], -steps, 0, thetaFast=False)
                    extracted[self.visibleIdx] = self.exposeAndExtractPositions(f'thetaReverse{n}N{k}G{g}.fits')
                    thetaRV[gIdx, n, k+1] = extracted[gIdx]
                    if fromHome:
                        self.pfi.moveAllSteps(self.allCobras[notdoneMask], (k+1)*steps, 0)

                    doneMask, lastAngles = self.thetaRVDone(thetaRV, n, k)
                    if doneMask is not None:
                        newlyDone = doneMask & notdoneMask
                        if np.any(newlyDone):
                            notdoneMask &= ~doneMask
                            self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                    if not np.any(notdoneMask):
                        thetaRV[gIdx, n, k+2:] = thetaRV[gIdx, n, k+1][:,None]
                        break

                if doneMask is not None and np.any(notdoneMask):
                    self.logger.warn(f'{(notdoneMask == True).sum()} did not finish:')
                    for c_i in np.where(notdoneMask)[0]:
                        c = self.allCobras[c_i]
                        d = np.rad2deg(lastAngles[c_i])
                        self.logger.warn(f'  {str(c)}: {np.round(d, 2)}')

                # At the end, make sure the cobra back to the hard stop
                self.logger.info(f'G{g} {n+1}/{repeat} theta reverse {-limitSteps} steps to limit')
                self.pfi.moveAllSteps(self.allCobras[gIdx], -limitSteps, 0)

            # move theta arms to safe positions
            self.moveToThetaAngle(gIdx, angle=safeAngle[gIdx], globalAngles=True, maxTries=10)
            self.pfi.moveAllSteps(self.allCobras[gIdx], 0, -phiLimitSteps)
        self.cam.resetStack()

        # restore ontimes after test
        self.pfi.calibModel.updateOntimes(thetaFwd=defaultOnTimeFast[0], thetaRev=defaultOnTimeFast[1], fast=True)
        self.pfi.calibModel.updateOntimes(thetaFwd=defaultOnTimeSlow[0], thetaRev=defaultOnTimeSlow[1], fast=False)

        # save calculation result
        dataPath = self.runManager.dataDir
        np.save(dataPath / 'thetaFW', thetaFW)
        np.save(dataPath / 'thetaRV', thetaRV)

        return self.runManager.runDir, thetaFW, thetaRV

    def makeThetaMotorMap2(
            self,
            newXml,
            repeat=3,
            steps=100,
            totalSteps=10000,
            fast=False,
            thetaOnTime=None,
            limitOnTime=0.08,
            limitSteps=10000,
            delta=np.deg2rad(5.0),
            fromHome=False
        ):
        # generate theta motor maps, it accepts custom thetaOnTIme parameter.
        # it assumes that theta arms already point to the outward direction and phi arms inwards,
        # also there is good geometry measurement and motor maps in the XML file.
        # cobras are divided into three non-intefere groups so phi arms can be moved all way out
        # if thetaOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime
        # Example:
        #     makethetaMotorMap(xml, path, fast=True)               // update fast motor maps
        #     makethetaMotorMap(xml, path, fast=False)              // update slow motor maps
        #     makethetaMotorMap(xml, path, thetaOnTime=0.06)        // motor maps for on-time=0.06

        if updateGeometry and phiRunDir is None:
            raise RuntimeError('To write geometry, need to be told the phiRunDir')

        runDir, thetaFW, thetaRV = self.acquireThetaMotorMap2(steps=steps, repeat=repeat, totalSteps=totalSteps,
                                                             fast=fast, thetaOnTime=thetaOnTime,
                                                             limitOnTime=limitOnTime, limitSteps=limitSteps,
                                                             resetScaling=resetScaling, fromHome=fromHome)

        runDir, duds = self.reduceThetaMotorMap(newXml, self.runManager.runDir, steps,
                                                thetaOnTime=thetaOnTime,
                                                delta=delta, fast=fast,
                                                phiRunDir=phiRunDir,
                                                updateGeometry=updateGeometry,
                                                fromHome=fromHome)
        return runDir, duds

    def phiConvergenceTest(self, margin=15.0, runs=50, tries=8, fast=False, finalAngle=None, tolerance=0.2, scaleFactor=1.0):
        self._connect()
        dataPath = self.runManager.dataDir

        if (self.phiCenter is None or self.phiCWHome is None or self.phiCCWHome is None):
            self.logger.info('Get phi grometry first!!!')

            # variable declaration for center measurement
            steps = 200
            iteration = 4000 // steps
            phiFW = np.zeros((self.nCobras, iteration+1), dtype=complex)
            phiRV = np.zeros((self.nCobras, iteration+1), dtype=complex)

            #record the phi movements
            self.cam.resetStack('phiForwardStack.fits')
            self.pfi.resetMotorScaling(self.goodCobras, 'phi')
            self.pfi.moveAllSteps(self.goodCobras, 0, -5000, phiFast=True)
            phiFW[self.visibleIdx, 0] = self.exposeAndExtractPositions()

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, 0, steps, phiFast=False)
                phiFW[self.visibleIdx, k+1] = self.exposeAndExtractPositions(guess=phiFW[self.visibleIdx, k])

            # make sure it goes to the limit
            self.pfi.moveAllSteps(self.goodCobras, 0, 5000, phiFast=True)

            # reverse phi motors
            self.cam.resetStack('phiReverseStack.fits')
            phiRV[self.visibleIdx, 0] = self.exposeAndExtractPositions(guess=phiFW[self.visibleIdx, iteration])

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, 0, -steps, phiFast=False)
                phiRV[self.visibleIdx, k+1] = self.exposeAndExtractPositions(guess=phiRV[self.visibleIdx, k])
            self.cam.resetStack()

            # At the end, make sure the cobra back to the hard stop
            self.pfi.moveAllSteps(self.goodCobras, 0, -5000, phiFast=True)

            # save calculation result
            np.save(dataPath / 'phiFW', phiFW)
            np.save(dataPath / 'phiRV', phiRV)

            # variable declaration
            phiCenter = np.zeros(self.nCobras, dtype=complex)
            phiRadius = np.zeros(self.nCobras, dtype=float)
            phiCCWHome = np.zeros(self.nCobras, dtype=float)
            phiCWHome = np.zeros(self.nCobras, dtype=float)

            # measure centers
            for c in self.goodIdx:
                data = np.concatenate((phiFW[c].flatten(), phiRV[c].flatten()))
                x, y, r = calculation.circle_fitting(data)
                phiCenter[c] = x + y*(1j)
                phiRadius[c] = r

            # measure phi hard stops
            for c in self.goodIdx:
                phiCCWHome[c] = np.angle(phiFW[c, 0] - phiCenter[c])
                phiCWHome[c] = np.angle(phiRV[c, 0] - phiCenter[c])

            # save calculation result
            np.save(dataPath / 'phiCenter', phiCenter)
            np.save(dataPath / 'phiRadius', phiRadius)
            np.save(dataPath / 'phiCCWHome', phiCCWHome)
            np.save(dataPath / 'phiCWHome', phiCWHome)

            self.logger.info('Save phi geometry setting')
            centers = phiCenter[self.goodIdx]
            homes = phiCCWHome[self.goodIdx]
            self.phiCenter = phiCenter
            self.phiCCWHome = phiCCWHome
            self.phiCWHome = phiCWHome

        else:
            self.logger.info('Use current phi geometry setting!!!')
            centers = self.phiCenter[self.goodIdx]
            homes = self.phiCCWHome[self.goodIdx]

        # convergence test
        phiData = np.zeros((self.nCobras, runs, tries, 4))
        zeros = np.zeros(len(self.goodIdx))
        notdoneMask = np.zeros(self.nCobras, 'bool')
        nowDone = np.zeros(self.nCobras, 'bool')
        tolerance = np.deg2rad(tolerance)

        for i in range(runs):
            self.cam.resetStack(f'phiConvergenceTest{i}.fits')
            if runs > 1:
                angle = np.deg2rad(margin + (180 - 2 * margin) * i / (runs - 1))
            else:
                angle = np.deg2rad(90)
            notdoneMask[self.goodIdx] = True
            self.logger.info(f'Run {i+1}: angle={np.rad2deg(angle):.2f} degree')
            self.pfi.resetMotorScaling(self.goodCobras, 'phi')
            self.pfi.moveThetaPhi(self.goodCobras, zeros, zeros + angle, phiFast=fast)
            cAngles, cPositions = self.measureAngles(centers, homes)
            phiData[self.goodIdx, i, 0, 0] = cAngles
            phiData[self.goodIdx, i, 0, 1] = np.real(cPositions)
            phiData[self.goodIdx, i, 0, 2] = np.imag(cPositions)
            phiData[self.goodIdx, i, 0, 3] = 1.0

            scale = np.full(len(self.goodIdx), 1.0)
            for j in range(tries - 1):
                nm = notdoneMask[self.goodIdx]
                self.pfi.moveThetaPhi(self.allCobras[notdoneMask], zeros[nm], (angle - cAngles)[nm],
                                      phiFroms=cAngles[nm], phiFast=fast)
                lastAngle = cAngles
                cAngles, cPositions = self.measureAngles(centers, homes)
                cAngles[cAngles>np.pi*(3/2)] -= np.pi*2
                nowDone[:] = False
                nowDone[self.goodIdx[abs(cAngles - angle) < tolerance]] = True
                newlyDone = nowDone & notdoneMask
                if np.any(newlyDone):
                    notdoneMask &= ~newlyDone
                    self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                for k in range(len(self.goodIdx)):
                    if abs(cAngles[k] - lastAngle[k]) > self.minScalingAngle:
                        rawScale = abs((angle - lastAngle[k]) / (cAngles[k] - lastAngle[k]))
                        engageScale = (rawScale - 1) / scaleFactor + 1
                        direction = 'ccw' if angle < lastAngle[k] else 'cw'
                        scale[k] = self.pfi.scaleMotorOntimeBySpeed(self.goodCobras[k], 'phi', direction, fast, engageScale)

                phiData[self.goodIdx, i, j+1, 0] = cAngles
                phiData[self.goodIdx, i, j+1, 1] = np.real(cPositions)
                phiData[self.goodIdx, i, j+1, 2] = np.imag(cPositions)
                phiData[self.goodIdx, i, j+1, 3] = scale
                self.logger.debug(f'Scaling factor: {np.round(scale, 2)}')
                if not np.any(notdoneMask):
                    phiData[self.goodIdx, i, j+2:, 0] = cAngles[..., np.newaxis]
                    phiData[self.goodIdx, i, j+2:, 1] = np.real(cPositions)[..., np.newaxis]
                    phiData[self.goodIdx, i, j+2:, 2] = np.imag(cPositions)[..., np.newaxis]
                    phiData[self.goodIdx, i, j+2:, 3] = scale[..., np.newaxis]
                    break

            if np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} cobras did not finish: '
                                 f'{np.where(notdoneMask)[0]}, '
                                 f'{np.round(np.rad2deg(cAngles)[notdoneMask[self.goodIdx]], 2)}')

            # home phi
            self.pfi.moveAllSteps(self.goodCobras, 0, -5000, phiFast=True)
            self.cam.resetStack()

        # save calculation result
        np.save(dataPath / 'phiData', phiData)
        self.pfi.resetMotorScaling(self.goodCobras, 'phi')

        if finalAngle is not None:
            angle = np.deg2rad(finalAngle)
            self.pfi.moveThetaPhi(self.goodCobras, zeros, zeros + angle, phiFast=fast)
            cAngles, cPositions = self.measureAngles(centers, homes)

            for j in range(tries - 1):
                self.pfi.moveThetaPhi(self.goodCobras, zeros, angle - cAngles, phiFroms=cAngles, phiFast=fast)
                lastAngle = cAngles
                cAngles, cPositions = self.measureAngles(centers, homes)
                cAngles[cAngles>np.pi*(3/2)] -= np.pi*2
                for k in range(len(self.goodIdx)):
                    if abs(angle - lastAngle[k]) > self.minScalingAngle:
                        rawScale = abs((angle - lastAngle[k]) / (cAngles[k] - lastAngle[k]))
                        if angle < lastAngle[k]:
                            scale[k] = 1 + (rawScale - 1) / (scaleFactor * ratioRv[k])
                            self.pfi.scaleMotorOntime(self.goodCobras[k], 'phi', 'ccw', scale[k])
                        else:
                            scale[k] = 1 + (rawScale - 1) / (scaleFactor * ratioFw[k])
                            self.pfi.scaleMotorOntime(self.goodCobras[k], 'phi', 'cw', scale[k])
            self.logger.info(f'Final angles: {np.round(np.rad2deg(cAngles), 2)}')
            self.pfi.resetMotorScaling(self.goodCobras, 'phi')
        return self.runManager.runDir

    def thetaConvergenceTest(self, margin=15.0, runs=50, tries=8, fast=False, tolerance=0.2, scaleFactor=1.0):
        self._connect()
        dataPath = self.runManager.dataDir

        if (self.thetaCenter is None or self.thetaCWHome is None or self.thetaCCWHome is None):
            self.logger.info('Get theta grometry first!!!')

            # variable declaration for center measurement
            steps = 300
            iteration = 6000 // steps
            thetaFW = np.zeros((self.nCobras, iteration+1), dtype=complex)
            thetaRV = np.zeros((self.nCobras, iteration+1), dtype=complex)

            #record the theta movements
            self.cam.resetStack('thetaForwardStack.fits')
            self.pfi.resetMotorScaling(self.goodCobras, 'theta')
            self.pfi.moveAllSteps(self.goodCobras, -10000, 0, thetaFast=True)
            thetaFW[self.visibleIdx, 0] = self.exposeAndExtractPositions()

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, steps, 0, thetaFast=False)
                thetaFW[self.visibleIdx, 0] = self.exposeAndExtractPositions()

            # make sure it goes to the limit
            self.pfi.moveAllSteps(self.goodCobras, 10000, 0, thetaFast=True)

            # reverse theta motors
            self.cam.resetStack('thetaReverseStack.fits')
            thetaRV[self.visibleIdx, 0] = self.exposeAndExtractPositions()

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, -steps, 0, thetaFast=False)
                thetaRV[self.visibleIdx, k+1] = self.exposeAndExtractPositions()
            self.cam.resetStack()

            # At the end, make sure the cobra back to the hard stop
            self.pfi.moveAllSteps(self.goodCobras, -10000, 0, thetaFast=True)

            # save calculation result
            np.save(dataPath / 'thetaFW', thetaFW)
            np.save(dataPath / 'thetaRV', thetaRV)

            # variable declaration
            thetaCenter = np.zeros(self.nCobras, dtype=complex)
            thetaRadius = np.zeros(self.nCobras, dtype=float)
            thetaCCWHome = np.zeros(self.nCobras, dtype=float)
            thetaCWHome = np.zeros(self.nCobras, dtype=float)

            # measure centers
            for c in self.goodIdx:
                data = np.concatenate((thetaFW[c].flatten(), thetaRV[c].flatten()))
                x, y, r = calculation.circle_fitting(data)
                thetaCenter[c] = x + y*(1j)
                thetaRadius[c] = r

            # measure theta hard stops
            for c in self.goodIdx:
                thetaCCWHome[c] = np.angle(thetaFW[c, 0] - thetaCenter[c])
                thetaCWHome[c] = np.angle(thetaRV[c, 0] - thetaCenter[c])

            # save calculation result
            np.save(dataPath / 'thetaCenter', thetaCenter)
            np.save(dataPath / 'thetaRadius', thetaRadius)
            np.save(dataPath / 'thetaCCWHome', thetaCCWHome)
            np.save(dataPath / 'thetaCWHome', thetaCWHome)

            self.logger.info('Save theta geometry setting')
            centers = thetaCenter[self.goodIdx]
            homes = thetaCCWHome[self.goodIdx]
            self.thetaCenter = thetaCenter
            self.thetaCCWHome = thetaCCWHome
            self.thetaCWHome = thetaCWHome

        else:
            self.logger.info('Use current theta geometry setting!!!')
            centers = self.thetaCenter[self.goodIdx]
            homes = self.thetaCCWHome[self.goodIdx]

        # convergence test
        thetaData = np.zeros((self.nCobras, runs, tries, 4))
        zeros = np.zeros(len(self.goodIdx))
        tGaps = ((self.pfi.calibModel.tht1 - self.pfi.calibModel.tht0) % (np.pi*2))[self.goodIdx]
        notdoneMask = np.zeros(self.nCobras, 'bool')
        nowDone = np.zeros(self.nCobras, 'bool')
        tolerance = np.deg2rad(tolerance)

        for i in range(runs):
            self.cam.resetStack(f'thetaConvergenceTest{i}.fits')
            if runs > 1:
                angle = np.deg2rad(margin + (360 - 2*margin) * i / (runs - 1))
            else:
                angle = np.deg2rad(180)
            notdoneMask[self.goodIdx] = True
            self.logger.info(f'Run {i+1}: angle={np.rad2deg(angle):.2f} degree')
            self.pfi.resetMotorScaling(self.goodCobras, 'theta')
            self.pfi.moveThetaPhi(self.goodCobras, zeros + angle, zeros, thetaFast=fast)
            cAngles, cPositions = self.measureAngles(centers, homes)
            for k in range(len(self.goodIdx)):
                if angle > np.pi and cAngles[k] < tGaps[k] + 0.1:
                    cAngles[k] += np.pi*2
            thetaData[self.goodIdx, i, 0, 0] = cAngles
            thetaData[self.goodIdx, i, 0, 1] = np.real(cPositions)
            thetaData[self.goodIdx, i, 0, 2] = np.imag(cPositions)
            thetaData[self.goodIdx, i, 0, 3] = 1.0

            scale = np.full(len(self.goodIdx), 1.0)
            for j in range(tries - 1):
                dirs = angle > cAngles
                lastAngle = cAngles
                nm = notdoneMask[self.goodIdx]
                self.pfi.moveThetaPhi(self.allCobras[notdoneMask], (angle - cAngles)[nm],
                                      zeros[nm], thetaFroms=cAngles[nm], thetaFast=fast)
                cAngles, cPositions = self.measureAngles(centers, homes)
                nowDone[:] = False
                nowDone[self.goodIdx[abs((cAngles - angle + np.pi) % (np.pi*2) - np.pi) < tolerance]] = True
                newlyDone = nowDone & notdoneMask
                if np.any(newlyDone):
                    notdoneMask &= ~newlyDone
                    self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                for k in range(len(self.goodIdx)):
                    if angle > np.pi and cAngles[k] < tGaps[k] + 0.1:
                        cAngles[k] += np.pi*2
                    elif angle < np.pi and cAngles[k] > np.pi*2 - 0.1:
                        cAngles[k] -= np.pi*2
                    if abs(cAngles[k] - lastAngle[k]) > self.minScalingAngle:
                        rawScale = abs((angle - lastAngle[k]) / (cAngles[k] - lastAngle[k]))
                        engageScale = (rawScale - 1) / scaleFactor + 1
                        direction = 'cw' if dirs[k] else 'ccw'
                        scale[k] = self.pfi.scaleMotorOntimeBySpeed(self.goodCobras[k], 'theta', direction, fast, engageScale)

                thetaData[self.goodIdx, i, j+1, 0] = cAngles
                thetaData[self.goodIdx, i, j+1, 1] = np.real(cPositions)
                thetaData[self.goodIdx, i, j+1, 2] = np.imag(cPositions)
                thetaData[self.goodIdx, i, j+1, 3] = scale
                self.logger.debug(f'Scaling factor: {np.round(scale, 2)}')
                if not np.any(notdoneMask):
                    thetaData[self.goodIdx, i, j+2:, 0] = cAngles[..., np.newaxis]
                    thetaData[self.goodIdx, i, j+2:, 1] = np.real(cPositions)[..., np.newaxis]
                    thetaData[self.goodIdx, i, j+2:, 2] = np.imag(cPositions)[..., np.newaxis]
                    thetaData[self.goodIdx, i, j+2:, 3] = scale[..., np.newaxis]
                    break

            if np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} cobras did not finish: '
                                 f'{np.where(notdoneMask)[0]}, '
                                 f'{np.round(np.rad2deg(cAngles)[notdoneMask[self.goodIdx]], 2)}')

            # home theta
            self.pfi.moveAllSteps(self.goodCobras, -10000, 0, thetaFast=True)
            self.cam.resetStack()

        # save calculation result
        np.save(dataPath / 'thetaData', thetaData)
        self.pfi.resetMotorScaling(self.goodCobras, 'theta')
        return self.runManager.runDir

    def measureAngles(self, centers, homes):
        """ measure positions and angles for good cobras """

        guess = self.pfi.calibModel.centers
        guess[self.goodIdx] = centers
        curPos = np.zeros(self.nCobras, dtype='complex')
        curPos[self.visibleIdx] = self.exposeAndExtractPositions(guess=guess[self.visibleIdx])
        angles = (np.angle(curPos[self.goodIdx] - centers) - homes) % (np.pi*2)

        return angles, curPos[self.goodIdx]

    def convertXML(self, newXml):
        """ convert old XML to a new coordinate by taking 'phi homed' images
            assuming the cobra module is in horizontal setup
        """
        idx = self.visibleIdx
        idx1 = idx[idx <= self.camSplit]
        idx2 = idx[idx > self.camSplit]
        oldPos = self.cal.calibModel.centers
        newPos = np.zeros(self.nCobras, dtype=complex)

        # go home and measure new positions
#        self.pfi.moveAllSteps(self.goodCobras, 0, -5000)
        data, filename, bkgd = self.cam.expose()
        centroids = np.flip(np.sort_complex(data['x']+data['y']*1j))
        newPos[idx1] = centroids[:len(idx1)]
        newPos[idx2] = centroids[-len(idx2):]

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
#        self.cal.restoreConfig()

    def convertXML1(self, newXml):
        """ convert old XML to a new coordinate by taking 'phi homed' images
            assuming the cobra module is in horizontal setup
        """
        idx = self.visibleIdx
        oldPos = self.cal.calibModel.centers
        newPos = np.zeros(self.nCobras, dtype=complex)

        # go home and measure new positions
        self.pfi.moveAllSteps(self.visibleCobras, 0, -5000)
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

    def getIndexInGoodCobras(self, idx=None):
        # map an index for all cobras to an index for only the visible cobras
        if idx is None:
            return np.arange(len(self.visibleCobras))
        else:
            if len(set(idx) & set(self.badIdx)) > 0:
                raise RuntimeError('should not include bad cobras')
            _idx = np.zeros(self.nCobras, 'bool')
            _idx[idx] = True
            return np.where(_idx[self.visibleIdx])[0]

    def thetaOnTimeSearch(self, newXml, speeds=[0.06,0.12], steps=[1000,500], iteration=3, repeat=1, b=0.088):
        """ search the on time parameters for a specified motor speed """
        onTimeHigh = 0.08
        onTimeLow = 0.015
        onTimeHighSteps = 200

        if iteration < 3:
            self.logger.warn(f'Change iteration parameter from {iteration} to 3!')
            iteration = 3
        if np.isscalar(speeds) or len(speeds) != 2:
            raise ValueError(f'speeds parameter should be a two value tuples: {speeds}')
        if speeds[0] > speeds[1]:
            speeds = speeds[1], speeds[0]
        speeds = np.deg2rad(speeds)
        if np.isscalar(steps) or len(steps) != 2:
            raise ValueError(f'steps parameter should be a two value tuples: {steps}')
        if steps[0] < steps[1]:
            steps = steps[1], steps[0]

        slopeF = np.zeros(self.nCobras)
        slopeR = np.zeros(self.nCobras)
        ontF = np.zeros(self.nCobras)
        ontR = np.zeros(self.nCobras)
        _ontF = []
        _ontR = []
        _spdF = []
        _spdR = []

        # get the average speeds for onTimeHigh, small step size since it's fast
        self.logger.info(f'Initial run, onTime = {onTimeHigh}')
        runDir, duds = self.makeThetaMotorMap(newXml, repeat=repeat, steps=onTimeHighSteps, thetaOnTime=onTimeHigh, fast=True)
        spdF = np.load(runDir / 'data' / 'thetaSpeedFW.npy')
        spdR = np.load(runDir / 'data' / 'thetaSpeedRV.npy')

        # assume a typical value for bad cobras, sticky??
        limitSpeed = np.deg2rad(0.02)
        spdF[spdF<limitSpeed] = limitSpeed
        spdR[spdR<limitSpeed] = limitSpeed

        _ontF.append(np.full(self.nCobras, onTimeHigh))
        _ontR.append(np.full(self.nCobras, onTimeHigh))
        _spdF.append(spdF.copy())
        _spdR.append(spdR.copy())

        # rough estimation for on time
        for (fast, speed, step) in zip([False, True], speeds, steps):
            # calculate on time
            for c_i in self.goodIdx:
                ontF[c_i] = self.thetaModel.getOntimeFromData(speed, _spdF[0][c_i], onTimeHigh)
                ontR[c_i] = self.thetaModel.getOntimeFromData(speed, _spdR[0][c_i], onTimeHigh)

            for n in range(iteration):
                ontF[ontF>onTimeHigh] = onTimeHigh
                ontR[ontR>onTimeHigh] = onTimeHigh
                ontF[ontF<onTimeLow] = onTimeLow
                ontR[ontR<onTimeLow] = onTimeLow
                self.logger.info(f'Run {n+1}/{iteration}, onTime = {np.round([ontF, ontR],4)}')
                runDir, duds = self.makeThetaMotorMap(newXml, repeat=repeat, steps=step, thetaOnTime=[ontF, ontR], fast=fast)
                spdF = np.load(runDir / 'data' / 'thetaSpeedFW.npy')
                spdR = np.load(runDir / 'data' / 'thetaSpeedRV.npy')
                _ontF.append(ontF.copy())
                _ontR.append(ontR.copy())
                _spdF.append(spdF.copy())
                _spdR.append(spdR.copy())

                # try the same on-time again for bad measuement
                spdF[spdF<=0.0] = speed
                spdR[spdR<=0.0] = speed

                # calculate on time
                for c_i in self.goodIdx:
                    ontF[c_i] = self.thetaModel.getOntimeFromData(speed, spdF[c_i], ontF[c_i])
                    ontR[c_i] = self.thetaModel.getOntimeFromData(speed, spdR[c_i], ontR[c_i])

        # try to find best on time, maybe.....
        ontF = self.searchOnTime(speeds[0], np.array(_spdF), np.array(_ontF))
        ontR = self.searchOnTime(speeds[0], np.array(_spdR), np.array(_ontR))
        ontF[ontF>onTimeHigh] = onTimeHigh
        ontR[ontR>onTimeHigh] = onTimeHigh

        # build motor maps
        self.logger.info(f'Build motor maps, best onTime = {np.round([ontF, ontR],4)}')
        runDir, duds = self.makeThetaMotorMap(newXml, repeat=repeat, steps=steps[0], thetaOnTime=[ontF, ontR], fast=False)
        self.xml = runDir / 'output' / newXml
        self.pfi.loadModel(self.xml)

        # for fast on time
        ontF = self.searchOnTime(speeds[1], np.array(_spdF), np.array(_ontF))
        ontR = self.searchOnTime(speeds[1], np.array(_spdR), np.array(_ontR))
        ontF[ontF>onTimeHigh] = onTimeHigh
        ontR[ontR>onTimeHigh] = onTimeHigh

        # build motor maps
        self.logger.info(f'Build motor maps, best onTime = {np.round([ontF, ontR],4)}')
        runDir, duds = self.makeThetaMotorMap(newXml, repeat=repeat, steps=steps[1], thetaOnTime=[ontF, ontR], fast=True)
        self.xml = runDir / 'output' / newXml
        self.pfi.loadModel(self.xml)

        return self.xml

    def phiOnTimeSearch(self, newXml, speeds=(0.06,0.12), steps=(500,250), iteration=3, repeat=1, b=0.07):
        """ search the on time parameters for a specified motor speed """
        onTimeHigh = 0.05
        onTimeLow = 0.01
        onTimeHighSteps = 100

        if iteration < 3:
            self.logger.warn(f'Change iteration parameter from {iteration} to 3!')
            iteration = 3
        if np.isscalar(speeds) or len(speeds) != 2:
            raise ValueError(f'speeds parameter should be a two value tuples: {speeds}')
        if speeds[0] > speeds[1]:
            speeds = speeds[1], speeds[0]
        speeds = np.deg2rad(speeds)
        if np.isscalar(steps) or len(steps) != 2:
            raise ValueError(f'steps parameter should be a two value tuples: {steps}')
        if steps[0] < steps[1]:
            steps = steps[1], steps[0]

        slopeF = np.zeros(self.nCobras)
        slopeR = np.zeros(self.nCobras)
        ontF = np.zeros(self.nCobras)
        ontR = np.zeros(self.nCobras)
        _ontF = []
        _ontR = []
        _spdF = []
        _spdR = []

        # get the average speeds for onTimeHigh, small step size since it's fast
        self.logger.info(f'Initial run, onTime = {onTimeHigh}')
        runDir, duds = self.makePhiMotorMap(newXml, repeat=repeat, steps=onTimeHighSteps, phiOnTime=onTimeHigh, fast=True)
        spdF = np.load(runDir / 'data' / 'phiSpeedFW.npy')
        spdR = np.load(runDir / 'data' / 'phiSpeedRV.npy')

        # assume a typical value for bad cobras, sticky??
        limitSpeed = np.deg2rad(0.02)
        spdF[spdF<limitSpeed] = limitSpeed
        spdR[spdR<limitSpeed] = limitSpeed

        _ontF.append(np.full(self.nCobras, onTimeHigh))
        _ontR.append(np.full(self.nCobras, onTimeHigh))
        _spdF.append(spdF.copy())
        _spdR.append(spdR.copy())

        for (fast, speed, step) in zip([False, True], speeds, steps):
            # calculate on time
            self.logger.info(f'Run for best {"Fast" if fast else "Slow"} motor maps')
            for c_i in self.goodIdx:
                ontF[c_i] = self.phiModel.getOntimeFromData(speed, _spdF[0][c_i], onTimeHigh)
                ontR[c_i] = self.phiModel.getOntimeFromData(speed, _spdR[0][c_i], onTimeHigh)

            for n in range(iteration):
                ontF[ontF>onTimeHigh] = onTimeHigh
                ontR[ontR>onTimeHigh] = onTimeHigh
                ontF[ontF<onTimeLow] = onTimeLow
                ontR[ontR<onTimeLow] = onTimeLow
                self.logger.info(f'Run {n+1}/{iteration}, onTime = {np.round([ontF, ontR],4)}')
                runDir, duds = self.makePhiMotorMap(newXml, repeat=repeat, steps=step, phiOnTime=[ontF, ontR], fast=fast)
                spdF = np.load(runDir / 'data' / 'phiSpeedFW.npy')
                spdR = np.load(runDir / 'data' / 'phiSpeedRV.npy')
                _ontF.append(ontF.copy())
                _ontR.append(ontR.copy())
                _spdF.append(spdF.copy())
                _spdR.append(spdR.copy())

                # try the same on-time again for bad measuement
                spdF[spdF<=0.0] = speed
                spdR[spdR<=0.0] = speed

                # calculate on time
                for c_i in self.goodIdx:
                    ontF[c_i] = self.thetaModel.getOntimeFromData(speed, spdF[c_i], ontF[c_i])
                    ontR[c_i] = self.thetaModel.getOntimeFromData(speed, spdR[c_i], ontR[c_i])

        # try to find best on time, maybe.....
        ontF = self.searchOnTime(speeds[0], np.array(_spdF), np.array(_ontF))
        ontR = self.searchOnTime(speeds[0], np.array(_spdR), np.array(_ontR))
        ontF[ontF>onTimeHigh] = onTimeHigh
        ontR[ontR>onTimeHigh] = onTimeHigh

        # build motor maps
        self.logger.info(f'Build motor maps, best onTime = {np.round([ontF, ontR],4)}')
        runDir, duds = self.makePhiMotorMap(newXml, repeat=repeat, steps=steps[0], phiOnTime=[ontF, ontR], fast=False)
        self.xml = runDir / 'output' / newXml
        self.pfi.loadModel(self.xml)

        # for fast motor maps
        ontF = self.searchOnTime(speeds[1], np.array(_spdF), np.array(_ontF))
        ontR = self.searchOnTime(speeds[1], np.array(_spdR), np.array(_ontR))
        ontF[ontF>onTimeHigh] = onTimeHigh
        ontR[ontR>onTimeHigh] = onTimeHigh

        # build motor maps
        self.logger.info(f'Build motor maps, best onTime = {np.round([ontF, ontR],4)}')
        runDir, duds = self.makePhiMotorMap(newXml, repeat=repeat, steps=steps[1], phiOnTime=[ontF, ontR], fast=True)
        self.xml = runDir / 'output' / newXml
        self.pfi.loadModel(self.xml)

        return self.xml

    def searchOnTime(self, speed, sData, tData):
        """ There should be some better ways to do!!! """
        onTime = np.zeros(self.nCobras)

        for c in self.goodIdx:
            s = sData[:,c]
            t = tData[:,c]
            model = SpeedModel()
            err = model.buildModel(s, t)

            if err:
                self.logger.warn(f'Building model failed #{c+1}, set to max value')
                onTime[c] = np.max(t)
            else:
                onTime[c] = model.toOntime(speed)
                if not np.isfinite(onTime[c]):
                    self.logger.warn(f'Curve fitting failed #{c+1}, set to median value')
                    onTime[c] = np.median(t)

        return onTime
