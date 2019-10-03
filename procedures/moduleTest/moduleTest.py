from importlib import reload
import logging
import numpy as np
from astropy.io import fits
import sep
from copy import deepcopy

from procedures.moduleTest import calculation
reload(calculation)

from procedures.moduleTest.mcs import camera
reload(camera)
from ics.cobraCharmer import pfi as pfiControl
from ics.cobraCharmer import pfiDesign
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
    def __init__(self, fpgaHost, xml, brokens=None, cam1Id=1, cam2Id=2, camSplit=28, logLevel=logging.INFO):

        self.logger = logging.getLogger('moduleTest')
        self.logger.setLevel(logLevel)

        self.runManager = butler.RunTree(doCreate=False)

        """ Init module 1 cobras """

        # NO, not 1!! Pass in moduleName, etc. -- CPL
        reload(pfiControl)
        self.allCobras = np.array(pfiControl.PFI.allocateCobraModule(1))
        self.fpgaHost = fpgaHost
        self.xml = xml
        self.brokens = brokens
        self.camSplit = camSplit

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

        self.setBrokenCobras(self.brokens)

    def _connect(self):
        self.runManager.newRun()
        # Initializing COBRA module
        self.pfi = pfiControl.PFI(fpgaHost=self.fpgaHost,
                                  doLoadModel=False,
                                  logDir=self.runManager.logDir)
        self.pfi.loadModel(self.xml)
        self.pfi.setFreq()

        # initialize cameras
        self.cam = camera.cameraFactory(doClear=True, runManager=self.runManager)

        # init calculation library
        self.cal = calculation.Calculation(self.xml, None, None)

        # define the broken/good cobras
        self.setBrokenCobras(self.brokens)

    def setBrokenCobras(self, brokens=None):
        """ define the broken/good cobras """
        if brokens is None:
            brokens = []
        visibles = [e for e in range(1, 58) if e not in brokens]
        self.badIdx = np.array(brokens) - 1
        self.goodIdx = np.array(visibles) - 1
        self.badCobras = np.array(self.getCobras(self.badIdx))
        self.goodCobras = np.array(self.getCobras(self.goodIdx))

        if hasattr(self, 'cal'):
            self.cal.setBrokenCobras(brokens)

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
        if len(positions) != len(self.goodCobras):
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
        outputDir = self.makePhiMotorMap('quickPhiScan.xml',
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

        FW = np.load(geometryRun / 'data' / 'phiFW.npy')
        RV = np.load(geometryRun / 'data' / 'phiRV.npy')
        self.phiCCWHome = np.angle(FW[:,0,0] - self.phiCenter)
        self.phiCWHome = np.angle(RV[:,0,0] - self.phiCenter)
        dAng = self.phiCWHome - self.phiCCWHome
        dAng[dAng<0] += 2*np.pi
        stopped = np.where(dAng < np.deg2rad(182.0))[0]
        if len(stopped) > 0:
            self.logger.error(f"phi ranges for cobras {stopped+1} are too small: "
                              f"CW={np.rad2deg(self.phiCWHome[stopped])} "
                              f"CCW={np.rad2deg(self.phiCCWHome[stopped])}")
            self.logger.error(f"     {np.round(np.rad2deg(dAng[stopped]), 2)}")

    def setThetaCentersFromRun(self, geometryRun):
        self.thetaCenter = np.load(geometryRun / 'data' / 'thetaCenter.npy')

    def setThetaGeometryFromRun(self, geometryRun, onlyIfClear=True):
        if (onlyIfClear and (self.thetaCenter is not None
                             and self.thetaCWHome is not None
                             and self.thetaCCWHome is not None)):
            return

        self.setThetaCentersFromRun(geometryRun)

        FW = np.load(geometryRun / 'data' / 'thetaFW.npy')
        RV = np.load(geometryRun / 'data' / 'thetaRV.npy')
        self.thetaCCWHome = np.angle(FW[:,0,0] - self.thetaCenter)
        self.thetaCWHome = np.angle(RV[:,0,0] - self.thetaCenter)

        dAng = self.thetaCWHome - self.thetaCCWHome
        dAng[dAng<np.pi] += 2*np.pi
        dAng[dAng<np.pi] += 2*np.pi
        stopped = np.where(dAng < np.deg2rad(375.0))[0]
        if len(stopped) > 0:
            self.logger.error(f"theta ranges for cobras {stopped+1} are too small: "
                              f"CW={np.rad2deg(self.thetaCWHome[stopped])} "
                              f"CCW={np.rad2deg(self.thetaCCWHome[stopped])}")
            self.logger.error(f"     {np.round(np.rad2deg(dAng[stopped]), 2)}")

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
        doFast : bool
          For the first move, use the fast map?
        """

        dtype = np.dtype(dict(names=['iteration', 'cobra', 'target', 'position', 'left', 'done'],
                              formats=['i2', 'i2', 'f4', 'f4', 'f4', 'i1']))

        # We do want a new stack of these images.
        self._connect()
        self.cam.resetStack(doStack=True)

        if idx is None:
            idx = np.arange(len(self.goodCobras))
        else:
            _idx = np.zeros(57, 'bool')
            _idx[idx] = True
            idx = _idx[self.goodIdx]

        cobras = np.array(self.goodCobras)
        cobras = cobras[idx]

        moveList = []
        moves0 = np.zeros(len(cobras), dtype=dtype)

        try:
            phiCenters = self.phiCenter[self.goodIdx]
        except AttributeError:
            raise RuntimeError("moduleTest needs to have been to told the phi Centers")

        tolerance = np.deg2rad(tolerance)

        # extract sources and fiber identification
        curPos = self.exposeAndExtractPositions(tolerance=0.2)
        if idx is not None:
            curPos = curPos[idx]
            phiCenters = phiCenters[idx]
        if keepExistingPosition:
            homeAngles = self.phiCCWHome[self.goodIdx]
            if idx is not None:
                homeAngles = homeAngles[idx]
            curAngles = self._fullAngle(curPos, phiCenters)
            lastAngles = self.dPhiAngle(curAngles, homeAngles, doAbs=True)
        else:
            homeAngles = self._fullAngle(curPos, phiCenters)
            curAngles = homeAngles
            lastAngles = np.zeros(len(homeAngles))
            self.phiHomes = homeAngles

        targetAngles = np.full(len(homeAngles), np.deg2rad(angle))
        thetaAngles = targetAngles * 0
        ntries = 1
        notDone = targetAngles != 0
        left = self.dPhiAngle(targetAngles, lastAngles, doWrap=True)

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
            left = self.dPhiAngle(targetAngles, atAngles, doWrap=True)

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

                    if scale <= 0.75 or scale >= 1.25:
                        logCall = self.logger.info
                    else:
                        logCall = self.logger.debug

                    logCall(f'{c_i+1} try={np.rad2deg(tryDist):0.2f} '
                            f'at={np.rad2deg(atAngles[c_i]):0.2f} '
                            f'got={np.rad2deg(gotDist):0.2f} '
                            f'rawScale={rawScale:0.2f} scale={scale:0.2f}')
                    self.pfi.scaleMotorOntime(cobras[c_i], 'phi', direction, scale)

            lastAngles = atAngles
            if ntries >= maxTries:
                self.logger.warn(f'Reached max {maxTries} tries, '
                                 f'left: {[str(c) for c in cobras[np.where(notDone)]]}: '
                                 f'{np.rad2deg(left)[notDone]}')
                break
            ntries += 1

        self.pfi.resetMotorScaling(self.goodCobras, 'phi')
        moves = np.stack(moveList)
        movesPath = self.runManager.outputDir / 'phiConvergence.npy'
        np.save(movesPath, moves)

        return self.runManager.runDir

    def gotoSafeFromPhi60(self, phiAngle=60.0, tolerance=1.0):
        """ Move cobras to nominal safe position: thetas OUT, phis in.
        Assumes phi is at 60deg and that we know thetaPositions.

        """
        if not hasattr(self, 'thetaHomes'):
            keepExisting = False
        else:
            keepExisting = True

        angle = (180.0 - phiAngle) / 2.0
        thetaAngle = np.zeros(len(self.goodIdx))
        thetaAngle[self.goodIdx%2==0] = 270 - angle
        thetaAngle[self.goodIdx%2!=0] = 90 - angle

        run = self.moveToThetaAngle(angle=thetaAngle, tolerance=tolerance,
                                    keepExistingPosition=keepExisting, globalAngles=True)
        return run

    def gotoShippingFromPhi60(self, phiAngle=60.0, tolerance=1.0):
        """ Move cobras to nominal safe shipping position: thetas IN, phis in.
        Assumes phi is at 60deg and that we know thetaPositions.

        """
        if not hasattr(self, 'thetaHomes'):
            keepExisting = False
        else:
            keepExisting = True

        angle = (180.0 - phiAngle) / 2.0
        thetaAngle = np.zeros(len(self.goodIdx))
        thetaAngle[self.goodIdx%2==0] = 90 - angle
        thetaAngle[self.goodIdx%2!=0] = 270 - angle

        run = self.moveToThetaAngle(angle=thetaAngle, tolerance=tolerance,
                                     keepExistingPosition=keepExisting, globalAngles=True)
        return run

    def moveToThetaAngle(self, idx=None, angle=60.0,
                         keepExistingPosition=False,
                         tolerance=1.0, maxTries=7, scaleFactor=10,
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

        dtype = np.dtype(dict(names=['iteration', 'cobra', 'target', 'position', 'left', 'done'],
                              formats=['i2', 'i2', 'f4', 'f4', 'f4', 'i1']))

        # We do want a new stack of these images.
        self._connect()
        self.cam.resetStack(doStack=True)

        if np.isscalar(angle):
            angle = np.full(len(self.goodCobras), angle)

        if idx is None:
            idx = np.arange(len(self.goodCobras))
        else:
            _idx = np.zeros(57, 'bool')
            _idx[idx] = True
            idx = _idx[self.goodIdx]

        cobras = np.array(self.goodCobras)
        cobras = cobras[idx]
        angle = angle[idx]

        moveList = []
        moves0 = np.zeros(len(cobras), dtype=dtype)

        thetaCenters = self.thetaCenter
        if thetaCenters is None:
            thetaCenters = self.pfi.calibModel.centers

        tolerance = np.deg2rad(tolerance)

        if not keepExistingPosition or not hasattr(self, 'thetaHomes'):
            # extract sources and fiber identification
            self.logger.info(f'theta backward -10000 steps to limit')
            self.pfi.moveAllSteps(self.goodCobras, -10000, 0)
            allCurPos = self.exposeAndExtractPositions(tolerance=0.2)

            homeAngles = self._fullAngle(allCurPos, thetaCenters[self.goodIdx])
            lastAngles = np.zeros(len(homeAngles))
            self.thetaHomes = homeAngles
            self.thetaAngles = lastAngles

        homeAngles = self.thetaHomes[idx]
        lastAngles = self.thetaAngles[idx]
        thetaCenters = thetaCenters[self.goodIdx[idx]]

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
            moves['target'][i] = angle[i]
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
            self.pfi.moveThetaPhi(cobras[notDone],
                                  left[notDone],
                                  phiAngles[notDone],
                                  thetaFroms=lastAngles[notDone],
                                  thetaFast=(doFast and ntries==1))

            # extract sources and fiber identification
            curPos = self.exposeAndExtractPositions(tolerance=0.2)
            curPos = curPos[idx]

            # Get our angle w.r.t. home.
            atAngles = unwrappedPosition(curPos, thetaCenters, homeAngles,
                                         lastAngles, targetAngles)
            left = targetAngles - atAngles

            # check position errors
            notDone = np.abs(left) > tolerance

            moves = moves0.copy()
            moveList.append(moves)
            for i in range(len(cobras)):
                cobraNum = cobras[i].cobraNum
                moves['iteration'][i] = ntries
                moves['cobra'][i] = cobraNum
                moves['target'][i] = angle[i]
                moves['position'][i] = atAngles[i]
                moves['left'][i] = left[i]
                moves['done'][i] = not notDone[i]

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

                    logCall(f'{c_i+1} try={np.rad2deg(tryDist[c_i]):0.2f} '
                            f'at={np.rad2deg(atAngles[c_i]):0.2f} '
                            f'got={np.rad2deg(gotDist[c_i]):0.2f} '
                            f'rawScale={rawScale:0.2f} scale={scale:0.2f}')
                    self.pfi.scaleMotorOntime(cobras[c_i], 'theta', direction, scale)

            lastAngles = atAngles
            self.thetaAngles[idx] = atAngles
            if ntries >= maxTries:
                self.logger.warn(f'Reached max {maxTries} tries, '
                                 f'left: {[str(c) for c in cobras[np.where(notDone)]]}: '
                                 f'{np.rad2deg(left)[notDone]}')
                break
            ntries += 1

        self.pfi.resetMotorScaling(self.goodCobras, 'theta')
        moves = np.concatenate(moveList)
        movesPath = self.runManager.outputDir / 'thetaConvergence.npy'
        np.save(movesPath, moves)

        return self.runManager.runDir

    def moveToXYfromHome(self, idx, targets, threshold=3.0, maxTries=8):
        """ function to move cobras to target positions """

        if idx is None:
            idx = np.arange(len(self.allCobras))
        cobras = self.getCobras(idx)
        if len(targets) != len(idx):
            raise RuntimeError('number of targets must match idx')
        if len(set(idx) & set(self.badIdx)) > 0:
            raise RuntimeError('should not include bad/broken cobras')

        arr = np.full(len(self.allCobras), False)
        arr[idx] = True
        _idx = arr[self.goodIdx]

        self.pfi.moveXYfromHome(cobras, targets, thetaThreshold=threshold, phiThreshold=threshold)

        ntries = 1
        keepMoving = np.where(targets != 0)
        while True:
            # extract sources and fiber identification
            curPos = self.exposeAndExtractPositions(tolerance=0.2)
            curPos = curPos[_idx]
            # check position errors
            self.logger.info("to: %s", targets[keepMoving])
            self.logger.info("at: %s", curPos[keepMoving])

            notDone = np.abs(curPos - targets) > threshold
            if not np.any(notDone):
                print('Convergence sequence done')
                break
            if ntries > maxTries:
                print(f'Reach max {maxTries} tries, gave up, gave up on {np.where(notDone)}')
                break
            self.logger.info("left (%d/%d): %s", len(keepMoving[0]), len(targets),
                             targets[keepMoving] - curPos[keepMoving])

            ntries += 1

            keepMoving = np.where(notDone)
            self.pfi.moveXY(cobras[keepMoving], curPos[keepMoving], targets[keepMoving],
                            thetaThreshold=threshold, phiThreshold=threshold)

    def moveToXY(self, idx, theta, phi, threshold=3.0, maxTries=8):
        """ move positioners to given theta, phi angles.
        """

        cobras = self.getCobras(idx)
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
        self.moveToXYfromHome(idx, outTargets[idx], threshold=threshold, maxTries=maxTries)

    def moveToThetaPhi(self, idx, theta, phi, threshold=3.0, maxTries=8):
        """ move positioners to given theta, phi angles.
        """

        if idx is None:
            idx = np.arange(len(self.allCobras))

        cobras = self.getCobras(None)
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
        outTargets = self.pfi.anglesToPositions(self.allCobras, thetaAngles, phiAngles)

        # move to outTargets
        self.moveToXYfromHome(idx, outTargets[idx], threshold=threshold, maxTries=maxTries)

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
            fast=False,
            phiOnTime=None,
            updateGeometry=False,
            limitOnTime=0.08,
            resetScaling=True,
            delta=np.deg2rad(5.0),
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
        fastOnTime = [np.full(57, limitOnTime)] * 2
        if phiOnTime is not None:
            if np.isscalar(phiOnTime):
                slowOnTime = [np.full(57, phiOnTime)] * 2
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
        phiFW = np.zeros((57, repeat, iteration+1), dtype=complex)
        phiRV = np.zeros((57, repeat, iteration+1), dtype=complex)

        if resetScaling:
            self.pfi.resetMotorScaling(cobras=None, motor='phi')

        # record the phi movements
        dataPath = self.runManager.dataDir
        self.logger.info(f'phi home {-totalSteps} steps')
        self.pfi.moveAllSteps(self.goodCobras, 0, -totalSteps)  # default is fast
        for n in range(repeat):
            self.cam.resetStack(f'phiForwardStack{n}.fits')

            # forward phi motor maps
            phiFW[self.goodIdx, n, 0] = self.exposeAndExtractPositions(f'phiBegin{n}.fits')

            notdoneMask = np.zeros(len(phiFW), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} phi forward to {(k+1)*steps}')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, (k+1)*steps, phiFast=False)
                else:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, steps, phiFast=False)
                phiFW[self.goodIdx, n, k+1] = self.exposeAndExtractPositions(f'phiForward{n}N{k}.fits',
                                                                             guess=phiFW[self.goodIdx, n, k])
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, -(k+1)*steps)

                doneMask, lastAngles = self.phiFWDone(phiFW, n, k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    phiFW[self.goodIdx, n, k+2:] = phiFW[self.goodIdx, n, k+1][:,None]
                    break
            if doneMask is not None and np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} cobras did not finish:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    self.logger.warn(f'  {str(c)}: {np.round(d, 2)}')

            # make sure it goes to the limit
            self.logger.info(f'{n+1}/{repeat} phi forward {totalSteps} to limit')
            self.pfi.moveAllSteps(self.goodCobras, 0, totalSteps)  # fast to limit

            # reverse phi motor maps
            self.cam.resetStack(f'phiReverseStack{n}.fits')
            phiRV[self.goodIdx, n, 0] = self.exposeAndExtractPositions(f'phiEnd{n}.fits',
                                                                       guess=phiFW[self.goodIdx, n, iteration])
            notdoneMask = np.zeros(len(phiRV), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} phi backward to {(k+1)*steps}')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, -(k+1)*steps, phiFast=False)
                else:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, -steps, phiFast=False)
                phiRV[self.goodIdx, n, k+1] = self.exposeAndExtractPositions(f'phiReverse{n}N{k}.fits',
                                                                             guess=phiRV[self.goodIdx, n, k])
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, (k+1)*steps)
                doneMask, lastAngles = self.phiRVDone(phiRV, n, k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    phiRV[self.goodIdx, n, k+2:] = phiRV[self.goodIdx, n, k+1][:,None]
                    break

            if doneMask is not None and np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} did not finish:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    self.logger.warn(f'  {str(c)}: {np.round(d, 2)}')

            # At the end, make sure the cobra back to the hard stop
            self.logger.info(f'{n+1}/{repeat} phi reverse {-totalSteps} steps to limit')
            self.pfi.moveAllSteps(self.goodCobras, 0, -totalSteps)  # fast to limit
        self.cam.resetStack()

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
        if fromHome:
            phiMMFW, phiMMRV, bad = self.cal.motorMaps2(phiAngFW, phiAngRV, steps, delta)
        else:
            phiMMFW, phiMMRV, bad = self.cal.motorMaps(phiAngFW, phiAngRV, steps, delta)
        bad[badRange] = True
        np.save(dataPath / 'phiMMFW', phiMMFW)
        np.save(dataPath / 'phiMMRV', phiMMRV)
        np.save(dataPath / 'bad', np.where(bad)[0])

        # calculate motor maps by average speeds
        #phiMMFW2, phiMMRV2, bad2 = self.cal.motorMaps2(phiAngFW, phiAngRV, steps, delta)
        #bad2[badRange] = True
        #np.save(dataPath / 'phiMMFW2', phiMMFW2)
        #np.save(dataPath / 'phiMMRV2', phiMMRV2)
        #np.save(dataPath / 'bad2', np.where(bad2)[0])

        # update XML file, using Johannes weighting
        slow = not fast
        self.cal.updatePhiMotorMaps(phiMMFW, phiMMRV, bad, slow)
        if phiOnTime is not None:
            if np.isscalar(phiOnTime):
                onTime = np.full(57, phiOnTime)
                self.cal.calibModel.updateOntimes(phiFwd=onTime, phiRev=onTime, fast=fast)
            else:
                self.cal.calibModel.updateOntimes(phiFwd=phiOnTime[0], phiRev=phiOnTime[1], fast=fast)
        if updateGeometry:
            self.cal.calibModel.updateGeometry(centers=phiCenter, phiArms=phiRadius)
        self.cal.calibModel.createCalibrationFile(self.runManager.outputDir / newXml, name='phiModel')

        # restore default setting ( really? why? CPL )
        # self.cal.restoreConfig()
        # self.pfi.loadModel(self.xml)

        self.setPhiGeometryFromRun(self.runManager.runDir, onlyIfClear=True)
        return self.runManager.runDir

    def thetaFWDone(self, thetas, n, k, needAtEnd=4,
                    closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(2)):
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

        if self.thetaCenter is None or self.thetaCWHome is None or k+1 < needAtEnd:
            return None,  None

        lastAngles = np.angle(thetas[:,n,k-needAtEnd+1:k+1] - self.thetaCenter[:,None])
        atEnd = np.abs(self.dAngle(lastAngles[:,-1] - self.thetaCWHome)) <= limitTolerance
        endDiff = np.abs(self.dAngle(np.diff(lastAngles, axis=1)))
        stable = np.all(endDiff <= closeEnough, axis=1)

        return atEnd & stable, endDiff

    def thetaRVDone(self, thetas, n, k, needAtEnd=4, closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(2)):
        """ Return a mask of the cobras which we deem at the RV theta limit.

        See `thetaFWDone`
        """
        if self.thetaCenter is None or self.thetaCCWHome is None or k+1 < needAtEnd:
            return None, None

        lastAngles = np.angle(thetas[:,n,k-needAtEnd+1:k+1] - self.thetaCenter[:,None])
        atEnd = np.abs(self.dAngle(lastAngles[:,-1] - self.thetaCCWHome)) <= limitTolerance
        endDiff = np.abs(self.dAngle(np.diff(lastAngles, axis=1)))
        stable = np.all(endDiff <= closeEnough, axis=1)

        return atEnd & stable, endDiff

    def phiFWDone(self, phis, n, k, needAtEnd=4, closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(2)):
        """ Return a mask of the cobras which we deem at the FW phi limit.

        See `thetaFWDone`
        """
        if self.phiCenter is None or self.phiCWHome is None or k+1 < needAtEnd:
            return None, None

        lastAngles = np.angle(phis[:,n,k-needAtEnd+1:k+1] - self.phiCenter[:,None])
        atEnd = np.abs(self.dAngle(lastAngles[:,-1] - self.phiCWHome)) <= limitTolerance
        endDiff = np.abs(self.dAngle(np.diff(lastAngles, axis=1)))
        stable = np.all(endDiff <= closeEnough, axis=1)

        return atEnd & stable, endDiff

    def phiRVDone(self, phis, n, k, needAtEnd=4, closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(2)):
        """ Return a mask of the cobras which we deem at the RV phi limit.

        See `thetaFWDone`
        """
        if self.phiCenter is None or self.phiCCWHome is None or k+1 < needAtEnd:
            return None, None

        lastAngles = np.angle(phis[:,n,k-needAtEnd+1:k+1] - self.phiCenter[:,None])
        atEnd = np.abs(self.dAngle(lastAngles[:,-1] - self.phiCCWHome)) <= limitTolerance
        endDiff = np.abs(self.dAngle(np.diff(lastAngles, axis=1)))
        stable = np.all(endDiff <= closeEnough, axis=1)

        return atEnd & stable, endDiff

    def makeThetaMotorMap(
            self,
            newXml,
            repeat=3,
            steps=100,
            totalSteps=10000,
            fast=False,
            thetaOnTime=None,
            updateGeometry=False,
            phiRunDir=None,
            limitOnTime=0.08,
            resetScaling=True,
            delta=np.deg2rad(5.0),
            fromHome=False
        ):
        # generate theta motor maps, it accepts custom thetaOnTIme parameter.
        # it assumes that phi arms have been move to ~60 degrees out to avoid collision
        # if thetaOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime
        # Example:
        #     makethetaMotorMap(xml, path, fast=True)               // update fast motor maps
        #     makethetaMotorMap(xml, path, fast=False)              // update slow motor maps
        #     makethetaMotorMap(xml, path, thetaOnTime=0.06)        // motor maps for on-time=0.06
        self._connect()
        if updateGeometry and phiRunDir is None:
            raise RuntimeError('To write geometry, need to be told the phiRunDir')

        defaultOnTimeFast = deepcopy([self.pfi.calibModel.motorOntimeFwd1,
                                      self.pfi.calibModel.motorOntimeRev1])
        defaultOnTimeSlow = deepcopy([self.pfi.calibModel.motorOntimeSlowFwd1,
                                      self.pfi.calibModel.motorOntimeSlowRev1])

        # set fast on-time to a large value so it can move over whole range, set slow on-time to the test value.
        fastOnTime = [np.full(57, limitOnTime)] * 2
        if thetaOnTime is not None:
            if np.isscalar(thetaOnTime):
                slowOnTime = [np.full(57, thetaOnTime)] * 2
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
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], (k+1)*steps, 0, thetaFast=False)
                else:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], steps, 0, thetaFast=False)
                thetaFW[self.goodIdx, n, k+1] = self.exposeAndExtractPositions(f'thetaForward{n}N{k}.fits')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], -(k+1)*steps, 0)

                doneMask, lastAngles = self.thetaFWDone(thetaFW, n, k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    thetaFW[self.goodIdx, n, k+2:] = thetaFW[self.goodIdx, n, k+1][:,None]
                    break

            if doneMask is not None and np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} did not finish:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    self.logger.warn(f'  {str(c)}: {np.round(d, 2)}')

            # make sure it goes to the limit
            self.logger.info(f'{n+1}/{repeat} theta forward {totalSteps} to limit')
            self.pfi.moveAllSteps(self.goodCobras, totalSteps, 0)

            # reverse theta motor maps
            self.cam.resetStack(f'thetaReverseStack{n}.fits')
            thetaRV[self.goodIdx, n, 0] = self.exposeAndExtractPositions(f'thetaEnd{n}.fits')

            notdoneMask = np.zeros(len(thetaFW), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} theta backward to {(k+1)*steps}')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], -(k+1)*steps, 0, thetaFast=False)
                else:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], -steps, 0, thetaFast=False)
                thetaRV[self.goodIdx, n, k+1] = self.exposeAndExtractPositions(f'thetaReverse{n}N{k}.fits')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], (k+1)*steps, 0)

                doneMask, lastAngles = self.thetaRVDone(thetaRV, n, k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    thetaRV[self.goodIdx, n, k+2:] = thetaRV[self.goodIdx, n, k+1][:,None]
                    break

            if doneMask is not None and np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} did not finish:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    self.logger.warn(f'  {str(c)}: {np.round(d, 2)}')

            # At the end, make sure the cobra back to the hard stop
            self.logger.info(f'{n+1}/{repeat} theta reverse {-totalSteps} steps to limit')
            self.pfi.moveAllSteps(self.goodCobras, -totalSteps, 0)
        self.cam.resetStack()

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

        self.thetaCenter = thetaCenter
        self.thetaCCWHome = np.angle(thetaFW[:,0,0] - thetaCenter)
        self.thetaCWHome = np.angle(thetaRV[:,0,0] - thetaCenter)

        # calculate average speeds
        thetaSpeedFW, thetaSpeedRV = self.cal.speed(thetaAngFW, thetaAngRV, steps, delta)
        np.save(dataPath / 'thetaSpeedFW', thetaSpeedFW)
        np.save(dataPath / 'thetaSpeedRV', thetaSpeedRV)

        # calculate motor maps in Johannes weighting
        if fromHome:
            thetaMMFW, thetaMMRV, bad = self.cal.motorMaps2(thetaAngFW, thetaAngRV, steps, delta)
        else:
            thetaMMFW, thetaMMRV, bad = self.cal.motorMaps(thetaAngFW, thetaAngRV, steps, delta)
        bad[badRange] = True
        np.save(dataPath / 'thetaMMFW', thetaMMFW)
        np.save(dataPath / 'thetaMMRV', thetaMMRV)
        np.save(dataPath / 'bad', np.where(bad)[0])

        # calculate motor maps by average speeds
        #thetaMMFW2, thetaMMRV2, bad2 = self.cal.motorMaps2(thetaAngFW, thetaAngRV, steps, delta)
        #bad2[badRange] = True
        #np.save(dataPath / 'thetaMMFW2', thetaMMFW2)
        #np.save(dataPath / 'thetaMMRV2', thetaMMRV2)
        #np.save(dataPath / 'bad2', np.where(bad2)[0])

        # update XML file, using Johannes weighting
        slow = not fast
        self.cal.updateThetaMotorMaps(thetaMMFW, thetaMMRV, bad, slow)
        if thetaOnTime is not None:
            if np.isscalar(thetaOnTime):
                onTime = np.full(57, thetaOnTime)
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
            self.cal.calibModel.updateGeometry(thetaCenter, thetaL, phiL)
            self.cal.calibModel.updateThetaHardStops(thetaCCW, thetaCW)
            self.cal.calibModel.updatePhiHardStops(phiCCW, phiCW)

        self.cal.calibModel.createCalibrationFile(self.runManager.outputDir / newXml)

        # restore default setting
        # self.cal.restoreConfig()
        # self.pfi.loadModel(self.xml)

        self.setThetaGeometryFromRun(self.runManager.runDir, onlyIfClear=True)
        return self.runManager.runDir

    def phiConvergenceTest(self, margin=15.0, runs=50, tries=8, fast=False, finalAngle=None, scaleFactor=8.0, tolerance=0.2):
        self._connect()
        dataPath = self.runManager.dataDir

        if (self.phiCenter is None or self.phiCWHome is None or self.phiCCWHome is None):
            self.logger.info('Get phi grometry first!!!')

            # variable declaration for center measurement
            steps = 200
            iteration = 4000 // steps
            phiFW = np.zeros((57, iteration+1), dtype=complex)
            phiRV = np.zeros((57, iteration+1), dtype=complex)

            #record the phi movements
            self.cam.resetStack('phiForwardStack.fits')
            self.pfi.moveAllSteps(self.goodCobras, 0, -5000, phiFast=True)
            phiFW[self.goodIdx, 0] = self.exposeAndExtractPositions()

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, 0, steps, phiFast=False)
                phiFW[self.goodIdx, k+1] = self.exposeAndExtractPositions(guess=phiFW[self.goodIdx, k])

            # make sure it goes to the limit
            self.pfi.moveAllSteps(self.goodCobras, 0, 5000, phiFast=True)

            # reverse phi motors
            self.cam.resetStack('phiReverseStack.fits')
            phiRV[self.goodIdx, 0] = self.exposeAndExtractPositions(guess=phiFW[self.goodIdx, iteration])

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, 0, -steps, phiFast=False)
                phiRV[self.goodIdx, k+1] = self.exposeAndExtractPositions(guess=phiRV[self.goodIdx, k])
            self.cam.resetStack()

            # At the end, make sure the cobra back to the hard stop
            self.pfi.moveAllSteps(self.goodCobras, 0, -5000, phiFast=True)

            # save calculation result
            np.save(dataPath / 'phiFW', phiFW)
            np.save(dataPath / 'phiRV', phiRV)

            # variable declaration
            phiCenter = np.zeros(57, dtype=complex)
            phiRadius = np.zeros(57, dtype=float)
            phiCCWHome = np.zeros(57, dtype=float)
            phiCWHome = np.zeros(57, dtype=float)

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
        phiData = np.zeros((57, runs, tries, 4))
        zeros = np.zeros(len(self.goodIdx))
        notdoneMask = np.zeros(57, 'bool')
        nowDone = np.zeros(57, 'bool')
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
                scale = np.full(len(self.goodIdx), 1.0)
                for k in range(len(self.goodIdx)):
                    if abs(cAngles[k] - lastAngle[k]) > np.deg2rad(0.5):
                        rawScale = abs((angle - lastAngle[k]) / (cAngles[k] - lastAngle[k]))
                        scale[k] = 1 + (rawScale - 1) / scaleFactor
                        direction = 'ccw' if angle < lastAngle[k] else 'cw'
                        self.pfi.scaleMotorOntime(self.goodCobras[k], 'phi', direction, scale[k])
                phiData[self.goodIdx, i, j+1, 0] = cAngles
                phiData[self.goodIdx, i, j+1, 1] = np.real(cPositions)
                phiData[self.goodIdx, i, j+1, 2] = np.imag(cPositions)
                phiData[self.goodIdx, i, j+1, 3] = scale
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
                    if abs(angle - lastAngle[k]) > np.deg2rad(2.0):
                        rawScale = abs((angle - lastAngle[k]) / (cAngles[k] - lastAngle[k]))
                        scale = 1 + (rawScale - 1) / scaleFactor
                        direction = 'ccw' if angle < lastAngle[k] else 'cw'
                        self.pfi.scaleMotorOntime(self.goodCobras[k], 'phi', direction, scale) 
            self.logger.info(f'Final angles: {np.round(np.rad2deg(cAngles), 2)}')
            self.pfi.resetMotorScaling(self.goodCobras, 'phi')
        return self.runManager.runDir

    def thetaConvergenceTest(self, margin=15.0, runs=50, tries=8, fast=False, scaleFactor=8.0, tolerance=0.2):
        self._connect()
        dataPath = self.runManager.dataDir

        if (self.thetaCenter is None or self.thetaCWHome is None or self.thetaCCWHome is None):
            self.logger.info('Get theta grometry first!!!')

            # variable declaration for center measurement
            steps = 300
            iteration = 6000 // steps
            thetaFW = np.zeros((57, iteration+1), dtype=complex)
            thetaRV = np.zeros((57, iteration+1), dtype=complex)

            #record the theta movements
            self.cam.resetStack('thetaForwardStack.fits')
            self.pfi.moveAllSteps(self.goodCobras, -10000, 0, thetaFast=True)
            thetaFW[self.goodIdx, 0] = self.exposeAndExtractPositions()

<<<<<<< HEAD
<<<<<<< HEAD
            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, steps, 0, thetaFast=False)
                thetaFW[self.goodIdx, 0] = self.exposeAndExtractPositions()
=======
    def thetaConvergenceTest(self, dataPath, margin=15.0, runs=50, tries=8, fastFirstMove=True):
=======
    def thetaConvergenceTest(self, dataPath, margin=15.0, runs=50, tries=8, 
        thetaThreshold = 250, fastFirstMove=True):
        
>>>>>>> Adding step threshold for switching between fast and slow maps
        # variable declaration for center measurement
        steps = 300
        iteration = 6000 // steps
        thetaFW = np.zeros((57, iteration+1), dtype=complex)
        thetaRV = np.zeros((57, iteration+1), dtype=complex)
>>>>>>> Use fast motor map at first move.

            # make sure it goes to the limit
            self.pfi.moveAllSteps(self.goodCobras, 10000, 0, thetaFast=True)

            # reverse theta motors
            self.cam.resetStack('thetaReverseStack.fits')
            thetaRV[self.goodIdx, 0] = self.exposeAndExtractPositions()

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, -steps, 0, thetaFast=False)
                thetaRV[self.goodIdx, k+1] = self.exposeAndExtractPositions()
            self.cam.resetStack()

            # At the end, make sure the cobra back to the hard stop
            self.pfi.moveAllSteps(self.goodCobras, -10000, 0, thetaFast=True)

            # save calculation result
            np.save(dataPath / 'thetaFW', thetaFW)
            np.save(dataPath / 'thetaRV', thetaRV)

            # variable declaration
            thetaCenter = np.zeros(57, dtype=complex)
            thetaRadius = np.zeros(57, dtype=float)
            thetaCCWHome = np.zeros(57, dtype=float)
            thetaCWHome = np.zeros(57, dtype=float)

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
        thetaData = np.zeros((57, runs, tries, 4))
        zeros = np.zeros(len(self.goodIdx))
        tGaps = ((self.pfi.calibModel.tht1 - self.pfi.calibModel.tht0) % (np.pi*2))[self.goodIdx]
        notdoneMask = np.zeros(57, 'bool')
        nowDone = np.zeros(57, 'bool')
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

            for j in range(tries - 1):
                dirs = angle > cAngles
<<<<<<< HEAD
                lastAngle = cAngles
                nm = notdoneMask[self.goodIdx]
                self.pfi.moveThetaPhi(self.allCobras[notdoneMask], (angle - cAngles)[nm],
                                      zeros[nm], thetaFroms=cAngles[nm], thetaFast=fast)
=======
                
                self.pfi.moveThetaPhi(self.goodCobras, angle - cAngles, zeros, thetaFroms=cAngles, 
                    thetaFast=False, thetaThreshold=thetaThreshold)
                
>>>>>>> Adding step threshold for switching between fast and slow maps
                cAngles, cPositions = self.measureAngles(centers, homes)
                nowDone[:] = False
                nowDone[self.goodIdx[abs((cAngles - angle + np.pi) % (np.pi*2) - np.pi) < tolerance]] = True
                newlyDone = nowDone & notdoneMask
                if np.any(newlyDone):
                    notdoneMask &= ~newlyDone
                    self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                scale = np.full(len(self.goodIdx), 1.0)
                for k in range(len(self.goodIdx)):
                    if dirs[k] and cAngles[k] < lastAngle[k] - 0.01 and cAngles[k] < tGaps[k] + 0.1:
                        cAngles[k] += np.pi*2
                    elif not dirs[k] and cAngles[k] > lastAngle[k] + 0.01 and cAngles[k] > np.pi*2 - 0.1:
                        cAngles[k] -= np.pi*2
                    if abs(cAngles[k] - lastAngle[k]) > np.deg2rad(0.5):
                        rawScale = abs((angle - lastAngle[k]) / (cAngles[k] - lastAngle[k]))
                        scale[k] = 1 + (rawScale - 1) / scaleFactor
                        direction = 'ccw' if not dirs[k] else 'cw'
                        self.pfi.scaleMotorOntime(self.goodCobras[k], 'theta', direction, scale[k])
                thetaData[self.goodIdx, i, j+1, 0] = cAngles
                thetaData[self.goodIdx, i, j+1, 1] = np.real(cPositions)
                thetaData[self.goodIdx, i, j+1, 2] = np.imag(cPositions)
                thetaData[self.goodIdx, i, j+1, 3] = scale
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

        curPos = self.exposeAndExtractPositions(guess=centers)
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

    def getCobras(self, cobs):
        # cobs is 0-indexed list
        if cobs is None:
            cobs = np.arange(len(self.allCobras))

        # assumes module == 1 XXX
        return np.array(pfiControl.PFI.allocateCobraList(zip(np.full(len(cobs), 1), np.array(cobs) + 1)))

    def thetaOnTimeSearch(self, newXml, speed=0.06, onTimeHigh=0.07, onTimeIntercept=0.027, steps=1000, iteration=6, repeat=1):
        """ search the on time parameters for a specified motor speed """
        speed = np.deg2rad(speed)
        if iteration < 3:
            self.logger.warn(f'Change iteration parameter from {iteration} to 3!')
            iteration = 3
        idx = self.goodIdx
        slopeF = np.zeros(57)
        slopeR = np.zeros(57)
        ontF = np.full(57, onTimeIntercept)
        ontR = np.full(57, onTimeIntercept)
        onTimeLow = onTimeIntercept / 2.0
        _ontF = []
        _ontR = []
        _spdF = []
        _spdR = []

        # get the average speeds for onTimeHigh, smaller step size since it's fast
        self.logger.info(f'Initial run, onTime = {onTimeHigh}')
        dataDir = self.makeThetaMotorMap(newXml, repeat=repeat, steps=400, thetaOnTime=onTimeHigh)
        spdF = np.load(dataDir / 'data' / 'thetaSpeedFW.npy')
        spdR = np.load(dataDir / 'data' / 'thetaSpeedRV.npy')

        # assume a typical value for bad cobras, sticky??
        spdF[spdF<np.deg2rad(0.02)] = np.deg2rad(0.2)
        spdR[spdR<np.deg2rad(0.02)] = np.deg2rad(0.2)

        # rough estimation for on time
        slopeF[idx] = (onTimeHigh - onTimeIntercept) / spdF[idx]
        slopeR[idx] = (onTimeHigh - onTimeIntercept) / spdR[idx]
        ontF[idx] += slopeF[idx] * speed
        ontR[idx] += slopeR[idx] * speed
        self.logger.info(f'Slope = {np.round(np.deg2rad([slopeF, slopeR]),4)}')

        for n in range(iteration):
            ontF[ontF>onTimeHigh] = onTimeHigh
            ontR[ontR>onTimeHigh] = onTimeHigh
            ontF[ontF<onTimeLow] = onTimeLow
            ontR[ontR<onTimeLow] = onTimeLow
            self.logger.info(f'Run {n+1}/{iteration}, onTime = {np.round([ontF, ontR],4)}')
            dataDir = self.makeThetaMotorMap(newXml, repeat=repeat, steps=steps, thetaOnTime=[ontF, ontR])
            spdF = np.load(dataDir / 'data' / 'thetaSpeedFW.npy')
            spdR = np.load(dataDir / 'data' / 'thetaSpeedRV.npy')
            _ontF.append(ontF.copy())
            _ontR.append(ontR.copy())
            _spdF.append(spdF.copy())
            _spdR.append(spdR.copy())

            # try the same on-time again for bad measuement
            spdF[spdF<=0.0] = speed
            spdR[spdR<=0.0] = speed

            # calculate on time
            ontF[idx] += slopeF[idx] * (speed - spdF[idx])
            ontR[idx] += slopeR[idx] * (speed - spdR[idx])

        # try to find best on time, maybe.....
        ontF = self.searchOnTime(speed, np.array(_spdF), np.array(_ontF))
        ontR = self.searchOnTime(speed, np.array(_spdR), np.array(_ontR))

        # build motor maps
        self.logger.info(f'Build motor maps, best onTime = {np.round([ontF, ontR],4)}')
        self.makeThetaMotorMap(newXml, repeat=repeat, steps=steps, thetaOnTime=[ontF, ontR])

        return ontF, ontR

    def phiOnTimeSearch(self, newXml, speed=0.06, onTimeHigh=0.05, onTimeIntercept=0.01, steps=500, iteration=6, repeat=1):
        """ search the on time parameters for a specified motor speed """
        speed = np.deg2rad(speed)
        if iteration < 3:
            self.logger.warn(f'Change iteration parameter from {iteration} to 3!')
            iteration = 3
        idx = self.goodIdx
        slopeF = np.zeros(57)
        slopeR = np.zeros(57)
        ontF = np.full(57, onTimeIntercept)
        ontR = np.full(57, onTimeIntercept)
        _ontF = []
        _ontR = []
        _spdF = []
        _spdR = []

        # get the average speeds for onTimeHigh, smaller step size since it's fast
        self.logger.info(f'Initial run, onTime = {onTimeHigh}')
        dataDir = self.makePhiMotorMap(newXml, repeat=repeat, steps=200, phiOnTime=onTimeHigh)
        spdF = np.load(dataDir / 'data' / 'phiSpeedFW.npy')
        spdR = np.load(dataDir / 'data' / 'phiSpeedRV.npy')

        # assume a typical value for bad cobras, sticky??
        spdF[spdF<np.deg2rad(0.02)] = np.deg2rad(0.2)
        spdR[spdR<np.deg2rad(0.02)] = np.deg2rad(0.2)

        # rough estimation for on time
        slopeF[idx] = (onTimeHigh - onTimeIntercept) / spdF[idx]
        slopeR[idx] = (onTimeHigh - onTimeIntercept) / spdR[idx]
        ontF[idx] += slopeF[idx] * speed
        ontR[idx] += slopeR[idx] * speed
        self.logger.info(f'Slope = {np.round(np.deg2rad([slopeF, slopeR]),4)}')

        for n in range(iteration):
            ontF[ontF>onTimeHigh] = onTimeHigh
            ontR[ontR>onTimeHigh] = onTimeHigh
            ontF[ontF<onTimeIntercept] = onTimeIntercept
            ontR[ontR<onTimeIntercept] = onTimeIntercept
            self.logger.info(f'Run {n+1}/{iteration}, onTime = {np.round([ontF, ontR],4)}')
            dataDir = self.makePhiMotorMap(newXml, repeat=repeat, steps=steps, phiOnTime=[ontF, ontR])
            spdF = np.load(dataDir / 'data' / 'phiSpeedFW.npy')
            spdR = np.load(dataDir / 'data' / 'phiSpeedRV.npy')
            _ontF.append(ontF.copy())
            _ontR.append(ontR.copy())
            _spdF.append(spdF.copy())
            _spdR.append(spdR.copy())

            # try the same on-time again for bad measuement
            spdF[spdF<=0.0] = speed
            spdR[spdR<=0.0] = speed

            # calculate on time
            ontF[idx] += slopeF[idx] * (speed - spdF[idx])
            ontR[idx] += slopeR[idx] * (speed - spdR[idx])

        # try to find best on time, maybe.....
        ontF = self.searchOnTime(speed, np.array(_spdF), np.array(_ontF))
        ontR = self.searchOnTime(speed, np.array(_spdR), np.array(_ontR))

        # build motor maps
        self.logger.info(f'Build motor maps, best onTime = {np.round([ontF, ontR],4)}')
        self.makePhiMotorMap(newXml, repeat=repeat, steps=steps, phiOnTime=[ontF, ontR])

        return ontF, ontR

    def searchOnTime(self, speed, sData, tData):
        """ There should be some better ways to do!!! """

        onTime = np.zeros(57)

        for c in self.goodIdx:
            s = sData[:,c]
            t = tData[:,c]
            t0 = np.min(t)
            t1 = np.max(t)
            s0 = np.min(s)
            s1 = np.max(s)
            onTime[c] = t0 + (t1-t0)/(s1-s0)*(speed-s0)

        return onTime

def combineFastSlowMotorMap(inputXML, newXML, arm='phi', brokens=None, fastPath=None, slowPath=None):
    binSize = np.deg2rad(3.6)
    model = pfiDesign.PFIDesign(inputXML)

    if fastPath is not None:
        fastFwdMM = np.load(f'{fastPath}{arm}MMFW.npy')
        fastRevMM = np.load(f'{fastPath}{arm}MMRV.npy')
    
    
    if slowPath is not None:
        slowFwdMM = np.load(f'{slowPath}{arm}MMFW.npy')
        slowRevMM = np.load(f'{slowPath}{arm}MMRV.npy')

 
    if brokens is None:
        brokens = []

    visibles = [e for e in range(1, 58) if e not in brokens]
    goodIdx = np.array(visibles) - 1

    new = model

        
    slowFW = binSize / new.S1Pm
    slowRV = binSize / new.S1Nm

    fastFW = binSize / new.F1Pm
    fastRV = binSize / new.F1Nm
    
    
    fastFW[goodIdx] = fastFwdMM[goodIdx]
    fastRV[goodIdx] = fastRevMM[goodIdx]
    slowFW[goodIdx] = slowFwdMM[goodIdx]
    slowRV[goodIdx] = slowRevMM[goodIdx]

    if arm is 'phi':
        new.updateMotorMaps(phiFwd=slowFW, phiRev=slowRV, useSlowMaps=True)
        new.updateMotorMaps(phiFwd=fastFW, phiRev=fastRV, useSlowMaps=False)

    else:
        new.updateMotorMaps(thtFwd=slowFW, thtRev=slowRV, useSlowMaps=True)
        new.updateMotorMaps(thtFwd=fastFW, thtRev=fastRV, useSlowMaps=False)

    new.createCalibrationFile(newXML)


def runMotorMap():
    module = 'Science29'
    arm = 'theta'

    dataPath = '/data/SC29/20190930/'

    stepList = ['50','400']
    speedList = ['','Fast']
    brokens = [57]
    xml = '/data/SC29/20190930/science29_theta_20190930.xml'

    for s in speedList:
        for f in stepList:
            path= dataPath+f'{arm}{f}Step{s}/'
            figpath = dataPath+f'{arm}{f}Step{s}MotorMap/'
            if not (os.path.exists(path)):
                    os.mkdir(path)
            if not (os.path.exists(figpath)):
                    os.mkdir(figpath)        
            mt = ModuleTest('128.149.77.24', xml,brokens=brokens,camSplit=28)
            vis = visDianosticPlot.VisDianosticPlot(path, brokens=brokens, camSplit=28)
            pfi = mt.pfi
            
            if arm is 'phi':
                pfi.moveAllSteps(mt.allCobras, 0, -5000)
                pfi.moveAllSteps(mt.allCobras, 0, -1000)
            else:
                pfi.moveAllSteps(mt.allCobras, -10000, 0)
                pfi.moveAllSteps(mt.allCobras, -2000, 0)
            
            if s is 'Fast':
                if arm is 'phi':
                    mt.makePhiMotorMap(f'{module}_{arm}_{f}Step{s}.xml', path, 
                        repeat = 3, steps = int(f), fast=True,totalSteps = 6000)
                else:
                    mt.makeThetaMotorMap(f'{module}_{arm}_{f}Step{s}.xml', path, 
                        repeat = 3, steps = int(f), fast=True,totalSteps = 12000)
            else:
                if arm is 'phi':
                    mt.makePhiMotorMap(f'{module}_{arm}_{f}Step{s}.xml', path, 
                        repeat = 3, steps = int(f), fast=False,totalSteps = 6000)
                else:
                    mt.makeThetaMotorMap(f'{module}_{arm}_{f}Step{s}.xml', path, 
                        repeat = 3, steps = int(f), fast=False,totalSteps = 12000)
                
            if arm is 'phi':
                vis.visCobraMotorMap(stepsize=int(f), figpath=figpath, arm='phi')    
            else:
                vis.visCobraMotorMap(stepsize=int(f), figpath=figpath, arm='theta')
                
            print(path)
            del(mt)
            del(vis)