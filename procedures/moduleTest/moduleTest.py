from importlib import reload
import logging
import numpy as np
from astropy.io import fits
import sep
from copy import deepcopy
import calculation
reload(calculation)

from mcs import camera
from ics.cobraCharmer import pfi as pfiControl
from ics.cobraCharmer.utils import butler
from ics.cobraCharmer.fpgaState import fpgaState

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
    def __init__(self, fpgaHost, xml, brokens=None, cam1Id=1, cam2Id=2, camSplit=26):

        self.logger = logging.getLogger('moduleTest')
        self.logger.setLevel(logging.DEBUG)

        self.runManager = butler.RunTree(doCreate=False)

        """ Init module 1 cobras """

        # NO, not 1!! Pass in moduleName, etc. -- CPL
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
        self.badCobras = getCobras(self.badIdx)
        self.goodCobras = getCobras(self.goodIdx)

        if hasattr(self, 'cal'):
            self.cal.setBrokenCobras(brokens)

    movesDtype = np.dtype(dict(names=['expId', 'spotId',
                                      'module', 'cobra',
                                      'phiSteps', 'phiOntime',
                                      'thetaSteps','thetaOntime'],
                               formats=['U12', 'i4',
                                        'i2', 'i2',
                                        'f4', 'f4',
                                        'f4', 'f4']))

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
            moveInfo = fpgaState.cobraLastMove(cobraInfo)

            moveTable['spotId'][pos_i] = indexMap[pos_i]
            moveTable['module'][pos_i] = cobraInfo.module
            moveTable['cobra'][pos_i] = cobraInfo.cobraNum
            for field in ('phiSteps', 'phiOntime',
                          'thetaSteps', 'thetaOntime'):
                moveTable[field][pos_i] = moveInfo[field]

        movesPath = self.runManager.outputDir / "moves.npz"
        self.logger.info(f'saving {len(moveTable)} moves to {movesPath}')
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

    def moveToXYfromHome(self, idx, targets, threshold=3.0, maxTries=8):
        """ function to move cobras to target positions """
        cobras = getCobras(idx)
        self.pfi.moveXYfromHome(cobras, targets, thetaThreshold=threshold, phiThreshold=threshold)

        ntries = 1
        while True:
            # extract sources and fiber identification
            curPos = self.exposeAndExtractPositions(tolerance=0.2)
            print(curPos)

            # check position errors
            done = np.abs(curPos - targets) <= threshold
            if np.all(done):
                print('Convergence sequence done')
                break
            if ntries > maxTries:
                print(f'Reach max {maxTries} tries, gave up')
                break
            ntries += 1

            # move again, skip bad center measurement
            good = (curPos != self.pfi.calibModel.centers[idx])
            self.pfi.moveXY(cobras[good], curPos[good], targets[good], thetaThreshold=threshold, phiThreshold=threshold)

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
            limitOnTime=0.06,
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

        #record the phi movements
        dataPath = self.runManager.dataDir
        self.pfi.moveAllSteps(self.goodCobras, 0, -5000)
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
        self.cal.calibModel.createCalibrationFile(self.runManager.outputDir / newXml, name='phiModel')
        self.cal.restoreConfig()

        # restore default setting
        self.pfi.loadModel(self.xml)

        return self.runManager.runDir

    def makeThetaMotorMap(
            self,
            newXml,
            dataPath,
            repeat=3,
            steps=200,
            totalSteps=10000,
            fast=True,
            thetaOnTime=None,
            limitOnTime=0.06,
            delta=np.deg2rad(5.0)
        ):
        # generate theta motor maps, it accepts custom thetaOnTIme parameter.
        # it assumes that phi arms have been move to ~60 degrees out to avoid collision
        # if thetaOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime
        # Example:
        #     makethetaMotorMap(xml, path, fast=True)               // update fast motor maps
        #     makethetaMotorMap(xml, path, fast=False)              // update slow motor maps
        #     makethetaMotorMap(xml, path, thetaOnTime=0.06)        // motor maps for on-time=0.06
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

        #record the theta movements
        self.cam1.reload()
        self.cam2.reload()
        self.pfi.moveAllSteps(self.goodCobras, -10000, 0)
        for n in range(repeat):
            # forward theta motor maps
            data1 = self.cam1.expose(dataPath + f'/theta1Begin{n}.fits.gz')
            data2 = self.cam2.expose(dataPath + f'/theta2Begin{n}.fits.gz')
            thetaFW[self.goodIdx, n, 0] = self.extractPositions(data1, data2)
            stack_image1 = data1
            stack_image2 = data2

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, steps, 0, thetaFast=False)
                data1 = self.cam1.expose(dataPath + f'/theta1Forward{n}N{k}.fits.gz')
                data2 = self.cam2.expose(dataPath + f'/theta2Forward{n}N{k}.fits.gz')
                thetaFW[self.goodIdx, n, k+1] = self.extractPositions(data1, data2, guess=thetaFW[self.goodIdx, n, k])
                stack_image1 += data1
                stack_image2 += data2
            fits.writeto(dataPath + f'/theta1ForwardStack{n}.fits.gz', stack_image1, overwrite=True)
            fits.writeto(dataPath + f'/theta2ForwardStack{n}.fits.gz', stack_image2, overwrite=True)

            # make sure it goes to the limit
            self.pfi.moveAllSteps(self.goodCobras, 10000, 0)

            # reverse theta motor maps
            data1 = self.cam1.expose(dataPath + f'/theta1End{n}.fits.gz')
            data2 = self.cam2.expose(dataPath + f'/theta2End{n}.fits.gz')
            thetaRV[self.goodIdx, n, 0] = self.extractPositions(data1, data2, guess=thetaFW[self.goodIdx, n, iteration])
            stack_image1 = data1
            stack_image2 = data2

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, -steps, 0, thetaFast=False)
                data1 = self.cam1.expose(dataPath + f'/theta1Reverse{n}N{k}.fits.gz')
                data2 = self.cam2.expose(dataPath + f'/theta2Reverse{n}N{k}.fits.gz')
                thetaRV[self.goodIdx, n, k+1] = self.extractPositions(data1, data2, guess=thetaRV[self.goodIdx, n, k])
                stack_image1 += data1
                stack_image2 += data2
            fits.writeto(dataPath + f'/theta1ReverseStack{n}.fits.gz', stack_image1, overwrite=True)
            fits.writeto(dataPath + f'/theta2ReverseStack{n}.fits.gz', stack_image2, overwrite=True)

            # At the end, make sure the cobra back to the hard stop
            self.pfi.moveAllSteps(self.goodCobras, -10000, 0)

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
            self.cal.calibModel.updateOntimes(thtFwd=onTime, thtRev=onTime, fast=fast)
        self.cal.calibModel.createCalibrationFile(dataPath / '' + newXml)
        self.cal.restoreConfig()

        # restore default setting
        self.pfi.loadModel(self.xml)

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

