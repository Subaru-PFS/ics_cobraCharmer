import sys
import os
import numpy as np
from astropy.io import fits
import sep
from copy import deepcopy
from idsCamera import idsCamera
from ics.cobraCharmer import pfi as pfiControl

class Camera():
    def __init__(self, devId):
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
        # Define module 1 cobras
        self.allCobras = pfiControl.PFI.allocateCobraRange(range(1, 2))

        # partition module 1 cobras into odd and even sets
        moduleCobras = {}
        for group in 1, 2:
            cm = range(group, 58, 2)
            mod = [1]*len(cm)
            moduleCobras[group] = pfiControl.PFI.allocateCobraList(zip(mod, cm))
        self.oddCobras = moduleCobras[1]
        self.evenCobras = moduleCobras[2]

        # Initializing COBRA module
        self.pfi = pfiControl.PFI(fpgaHost=fpgaHost, doLoadModel=False)
        if not os.path.exists(xml):
            print(f"Error: {xml} is not presented!")
            sys.exit()
        self.xml = xml
        self.pfi.loadModel(xml)
        self.pfi.setFreq(self.allCobras)

        # define the broken/good cobras
        self.setBrokenCobras(brokens)

        # initialize cameras
        self.cam1 = Camera(cam1Id)
        self.cam2 = Camera(cam2Id)
        self.camSplit = camSplit

    def setBrokenCobras(self, brokens=None):
        # define the broken/good cobras
        if brokens is None:
            brokens = []
        visibles = [e for e in range(1, 58) if e not in brokens]
        self.badIdx = np.array(brokens) - 1
        self.goodIdx = np.array(visibles) - 1
        self.badCobras = getCobras(self.badIdx)
        self.goodCobras = getCobras(self.goodIdx)

    def extractPositions(self, data1, data2, guess=None, tolerance=None):
        idx = self.goodIdx
        idx1 = idx[idx <= self.camSplit]
        idx2 = idx[idx > self.camSplit]
        if tolerance is not None:
            radii = (self.pfi.calibModel.L1 + self.pfi.calibModel.L2) * (1 + tolerance)
            radii1 = radii[idx1]
            radii2 = radii[idx2]
        else:
            radii1 = None
            radii2 = None

        if guess is None:
            center1 = self.pfi.calibModel.centers[idx1]
            center2 = self.pfi.calibModel.centers[idx2]
        else:
            center1 = guess[:len(idx1)]
            center2 = guess[len(idx1):]

        ext1 = sep.extract(data1.astype(float), 200)
        pos1 = np.array(ext1['x'] + ext1['y']*(1j))
        target1 = lazyIdentification(center1, pos1, radii=radii1)
        ext2 = sep.extract(data2.astype(float), 200)
        pos2 = np.array(ext2['x'] + ext2['y']*(1j))
        target2 = lazyIdentification(center2, pos2, radii=radii2)

        pos = np.zeros(len(idx), dtype=complex)
        for n, k in enumerate(target1):
            if k < 0:
                pos[n] = self.pfi.calibModel.centers[idx[n]]
            else:
                pos[n] = pos1[k]
        for n, k in enumerate(target2):
            m = n + len(target1)
            if k < 0:
                pos[m] = self.pfi.calibModel.centers[idx[m]]
            else:
                pos[m] = pos2[k]
        return pos

    # function to move cobras to target positions
    def moveToXYfromHome(self, idx, targets, threshold=3.0, maxTries=8):
        cobras = getCobras(idx)
        self.pfi.moveXYfromHome(cobras, targets, thetaThreshold=threshold, phiThreshold=threshold)

        ntries = 1
        while True:
            # check current positions
            data1 = self.cam1.expose()
            data2 = self.cam2.expose()

            # extract sources and fiber identification
            curPos = self.extractPositions(data1, data2, 0.01)
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
        # move visible positioners to outwards positions, phi arms are moved out for 60 degrees
        # (outTargets) so we can measure the arm angles
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
            dataPath,
            repeat=3,
            steps=200,
            totalSteps=5000,
            fast=True,
            phiOnTime=None,
            limitOnTime=0.08,
            delta=np.deg2rad(5.0)
        ):
        # generate phi motor maps, it accepts custom phiOnTIme parameter.
        # it assumes that theta arms have been move to up/down positions to avoid collision
        # if phiOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime
        #
        # Example:
        #     makePhiMotorMap(xml, path, fast=True)             // update fast motor maps
        #     makePhiMotorMap(xml, path, fast=False)            // update slow motor maps
        #     makePhiMotorMap(xml, path, phiOnTime=0.06)        // motor maps for on-time=0.06
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
        self.cam1.reload()
        self.cam2.reload()
        self.pfi.moveAllSteps(self.goodCobras, 0, -5000)
        for n in range(repeat):
            # forward phi motor maps
            data1 = self.cam1.expose(dataPath + f'/phi1Begin{n}.fits.gz')
            data2 = self.cam2.expose(dataPath + f'/phi2Begin{n}.fits.gz')
            phiFW[self.goodIdx, n, 0] = self.extractPositions(data1, data2)
            stack_image1 = data1
            stack_image2 = data2

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, 0, steps, phiFast=False)
                data1 = self.cam1.expose(dataPath + f'/phi1Forward{n}N{k}.fits.gz')
                data2 = self.cam2.expose(dataPath + f'/phi2Forward{n}N{k}.fits.gz')
                phiFW[self.goodIdx, n, k+1] = self.extractPositions(data1, data2, guess=phiFW[self.goodIdx, n, k])
                stack_image1 += data1
                stack_image2 += data2
            fits.writeto(dataPath + f'/phi1ForwardStack{n}.fits.gz', stack_image1, overwrite=True)
            fits.writeto(dataPath + f'/phi2ForwardStack{n}.fits.gz', stack_image2, overwrite=True)

            # make sure it goes to the limit
            self.pfi.moveAllSteps(self.goodCobras, 0, 5000)

            # reverse phi motor maps
            data1 = self.cam1.expose(dataPath + f'/phi1End{n}.fits.gz')
            data2 = self.cam2.expose(dataPath + f'/phi2End{n}.fits.gz')
            phiRV[self.goodIdx, n, 0] = self.extractPositions(data1, data2, guess=phiFW[self.goodIdx, n, iteration])
            stack_image1 = data1
            stack_image2 = data2

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, 0, -steps, phiFast=False)
                data1 = self.cam1.expose(dataPath + f'/phi1Reverse{n}N{k}.fits.gz')
                data2 = self.cam2.expose(dataPath + f'/phi2Reverse{n}N{k}.fits.gz')
                phiRV[self.goodIdx, n, k+1] = self.extractPositions(data1, data2, guess=phiRV[self.goodIdx, n, k])
                stack_image1 += data1
                stack_image2 += data2
            fits.writeto(dataPath + f'/phi1ReverseStack{n}.fits.gz', stack_image1, overwrite=True)
            fits.writeto(dataPath + f'/phi2ReverseStack{n}.fits.gz', stack_image2, overwrite=True)

            # At the end, make sure the cobra back to the hard stop
            self.pfi.moveAllSteps(self.goodCobras, 0, -5000)

        # save calculation result
        np.save(dataPath + '/phiFW', phiFW)
        np.save(dataPath + '/phiRV', phiRV)

        # variable declaration for phi angles
        phiCenter = np.zeros(57, dtype=complex)
        phiRadius = np.zeros(57, dtype=float)
        phiAngFW = np.zeros((57, repeat, iteration+1), dtype=float)
        phiAngRV = np.zeros((57, repeat, iteration+1), dtype=float)

        # measure centers
        for c in self.goodIdx:
            data = np.concatenate((phiFW[c].flatten(), phiRV[c].flatten()))
            x, y, r = circle_fitting(data)
            phiCenter[c] = x + y*(1j)
            phiRadius[c] = r

        # measure phi angles
        for c in self.goodIdx:
            for n in range(repeat):
                for k in range(iteration+1):
                    phiAngFW[c, n, k] = np.angle(phiFW[c, n, k] - phiCenter[c])
                    phiAngRV[c, n, k] = np.angle(phiRV[c, n, k] - phiCenter[c])
                home = phiAngFW[c, n, 0]
                phiAngFW[c, n] = (phiAngFW[c, n] - home + np.pi/2) % (np.pi*2) - np.pi/2
                phiAngRV[c, n] = (phiAngRV[c, n] - home + np.pi/2) % (np.pi*2) - np.pi/2

        # save calculation result
        np.save(dataPath + '/phiCenter', phiCenter)
        np.save(dataPath + '/phiRadius', phiRadius)
        np.save(dataPath + '/phiAngFW', phiAngFW)
        np.save(dataPath + '/phiAngRV', phiAngRV)

        # use both Johannes way and average steps for motor maps
        binSize = np.deg2rad(3.6)
        regions = 112
        phiMMFW = np.zeros((57, regions), dtype=float)
        phiMMRV = np.zeros((57, regions), dtype=float)
        phiMMFW2 = np.zeros((57, regions), dtype=float)
        phiMMRV2 = np.zeros((57, regions), dtype=float)
        phiSpeedFW = np.zeros(57, dtype=float)
        phiSpeedRV = np.zeros(57, dtype=float)
        phiStops = phiAngRV[:, :, 0] - delta
        cnt = np.zeros(regions)
        bad = np.full(57, False)

        # calculate phi motor maps
        for c in self.goodIdx:
            for b in range(regions):
                binMin = binSize * b
                binMax = binMin + binSize

                # forward motor maps
                fracSum = 0
                valueSum = 0
                for n in range(repeat):
                    for k in range(iteration):
                        if phiAngFW[c, n, k+1] < phiAngFW[c, n, k] or phiAngFW[c, n, k+1] > phiStops[c, n]:
                            # hit hard stop or somethings went wrong, then skip it
                            continue
                        if phiAngFW[c, n, k] < binMax and phiAngFW[c, n, k+1] > binMin:
                            moveSizeInBin = np.min([phiAngFW[c, n, k+1], binMax]) - np.max([phiAngFW[c, n, k], binMin])
                            entireMoveSize = phiAngFW[c, n, k+1] - phiAngFW[c, n, k]
                            fraction = moveSizeInBin * moveSizeInBin / entireMoveSize
                            fracSum += fraction
                            valueSum += fraction * entireMoveSize / steps
                if fracSum > 0:
                    phiMMFW[c, b] = valueSum / fracSum
                else:
                    phiMMFW[c, b] = 0

                # reverse motor maps
                fracSum = 0
                valueSum = 0
                for n in range(repeat):
                    for k in range(iteration):
                        if phiAngRV[c, n, k+1] > phiAngRV[c, n, k] or phiAngRV[c, n, k+1] < delta:
                            # hit hard stop or somethings went wrong, then skip it
                            continue
                        if phiAngRV[c, n, k] > binMin and phiAngRV[c, n, k+1] < binMax:
                            moveSizeInBin = np.min([phiAngRV[c, n, k], binMax]) - np.max([phiAngRV[c, n, k+1], binMin])
                            entireMoveSize = phiAngRV[c, n, k] - phiAngRV[c, n, k+1]
                            fraction = moveSizeInBin * moveSizeInBin / entireMoveSize
                            fracSum += fraction
                            valueSum += fraction * entireMoveSize / steps
                if fracSum > 0:
                    phiMMRV[c, b] = valueSum / fracSum
                else:
                    phiMMRV[c, b] = 0

            # fill the zeros closed to hard stops
            nz = np.nonzero(phiMMFW[c])[0]
            if nz.size > 0:
                phiMMFW[c, :nz[0]] = phiMMFW[c, nz[0]]
                phiMMFW[c, nz[-1]+1:] = phiMMFW[c, nz[-1]]
            else:
                bad[c] = True

            nz = np.nonzero(phiMMRV[c])[0]
            if nz.size > 0:
                phiMMRV[c, :nz[0]] = phiMMRV[c, nz[0]]
                phiMMRV[c, nz[-1]+1:] = phiMMRV[c, nz[-1]]
            else:
                bad[c] = True

            # calculate average speed
            mSteps = 0
            mAngle = 0
            for n in range(repeat):
                for k in range(iteration):
                    if phiAngFW[c, n, k+1] > phiStops[c, n]:
                        break
                mAngle += phiAngFW[c, n, k] - phiAngFW[c, n, 0]
                mSteps += k * steps
            phiSpeedFW[c] = mAngle / mSteps

            mSteps = 0
            mAngle = 0
            for n in range(repeat):
                for k in range(iteration):
                    if phiAngRV[c, n, k+1] < delta:
                        break
                mAngle += phiAngRV[c, n, 0] - phiAngRV[c, n, k]
                mSteps += k * steps
            phiSpeedRV[c] = mAngle / mSteps

            # calculate motor maps based on average step counts
            cnt[:] = 0
            for n in range(repeat):
                for k in range(iteration):
                    if phiAngFW[c, n, k+1] < phiAngFW[c, n, k] or phiAngFW[c, n, k+1] > phiStops[c, n]:
                        # hit hard stop or somethings went wrong, stop here
                        break
                x = np.arange(regions+1) * binSize
                xp = phiAngFW[c, n, :k+1]
                fp = np.arange(k+1) * steps
                mm = np.interp(x, xp, fp)
                diff = mm[1:] - mm[:-1]
                nz = np.nonzero(diff)[0]
                if nz.size > 0:
                    phiMMFW2[c] += diff
                    cnt[nz[:-1]] += 1
                    if phiAngFW[c, n, k] % binSize != 0:
                        cnt[nz[-1]] += (phiAngFW[c, n, k] % binSize) / binSize
                    else:
                        cnt[nz[-1]] += 1
            nz = np.nonzero(cnt)[0]
            if nz.size > 0:
                phiMMFW2[c, nz] = binSize / (phiMMFW2[c, nz] / cnt[nz])
                phiMMFW2[c, nz[-1]+1:] = phiMMFW2[c, nz[-1]]
            else:
                bad[c] = True

            cnt[:] = 0
            for n in range(repeat):
                for k in range(iteration):
                    if phiAngRV[c, n, k+1] - phiAngRV[c, n, k] > 0 or phiAngRV[c, n, k+1] < delta:
                        # hit hard stop or somethings went wrong, stop here
                        break
                x = np.arange(regions+1) * binSize
                xp = np.flip(phiAngRV[c, n, :k+1], 0)
                fp = np.arange(k+1) * steps
                mm = np.interp(x, xp, fp)
                diff = mm[1:] - mm[:-1]
                nz = np.nonzero(diff)[0]
                if nz.size > 0:
                    phiMMRV2[c] += diff
                    cnt[nz[1:-1]] += 1
                    cnt[nz[0]] += 1 - (phiAngRV[c, n, k] % binSize) / binSize
                    if phiAngRV[c, n, 0] % binSize != 0:
                        cnt[nz[-1]] += (phiAngRV[c, n, 0] % binSize) / binSize
                    else:
                        cnt[nz[-1]] += 1
            nz = np.nonzero(cnt)[0]
            if nz.size > 0:
                phiMMRV2[c, nz] = binSize / (phiMMRV2[c, nz] / cnt[nz])
                phiMMRV2[c, :nz[0]] = phiMMRV2[c, nz[0]]
                phiMMRV2[c, nz[-1]+1:] = phiMMRV2[c, nz[-1]]
            else:
                bad[c] = True

        # save calculation result
        np.save(dataPath + '/phiMMFW', phiMMFW)
        np.save(dataPath + '/phiMMRV', phiMMRV)
        np.save(dataPath + '/phiMMFW2', phiMMFW2)
        np.save(dataPath + '/phiMMRV2', phiMMRV2)
        np.save(dataPath + '/phiSpeedFW', phiSpeedFW)
        np.save(dataPath + '/phiSpeedRV', phiSpeedRV)
        np.save(dataPath + '/bad', np.where(bad)[0])

        # update XML configuration
        new = self.pfi.calibModel
        idx = np.array([c for c in self.goodIdx if not bad[c]])

        sPhiFW = binSize / new.S2Pm
        sPhiRV = binSize / new.S2Nm
        fPhiFW = binSize / new.F2Pm
        fPhiRV = binSize / new.F2Nm

        if phiOnTime is not None:
            # update motor maps, fast: Johannes, slow: simple
            fPhiFW[idx] = phiMMFW[idx]
            fPhiRV[idx] = phiMMRV[idx]
            new.updateMotorMaps(phiFwd=fPhiFW, phiRev=fPhiRV, useSlowMaps=False)

            sPhiFW[idx] = phiMMFW2[idx]
            sPhiRV[idx] = phiMMRV2[idx]
            new.updateMotorMaps(phiFwd=sPhiFW, phiRev=sPhiRV, useSlowMaps=True)

            # set fast on-time
            self.pfi.calibModel.updateOntimes(*(defaultOnTime[:2] + slowOnTime[2:]), fast=True)

        elif fast:
            # update fast motor maps, Johannes weighting
            fPhiFW[idx] = phiMMFW[idx]
            fPhiRV[idx] = phiMMRV[idx]
            new.updateMotorMaps(phiFwd=fPhiFW, phiRev=fPhiRV, useSlowMaps=False)

            # restore on-time
            self.pfi.calibModel.updateOntimes(*defaultOnTime, fast=True)
            self.pfi.calibModel.updateOntimes(*defaultOnTimeSlow, fast=False)

        else:
            # update slow motor maps, Johanees weighting
            sPhiFW[idx] = phiMMFW[idx]
            sPhiRV[idx] = phiMMRV[idx]
            new.updateMotorMaps(phiFwd=sPhiFW, phiRev=sPhiRV, useSlowMaps=True)

            # restore on-time
            self.pfi.calibModel.updateOntimes(*defaultOnTime, fast=True)

        # create a new XML file
        new.createCalibrationFile(dataPath + '/' + newXml)

        # restore default setting
        self.pfi.loadModel(self.xml)

    def makeThetaMotorMap(
            self,
            newXml,
            dataPath,
            repeat=3,
            steps=200,
            totalSteps=10000,
            fast=True,
            thetaOnTime=None,
            limitOnTime=0.08,
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
        np.save(dataPath + '/thetaFW', thetaFW)
        np.save(dataPath + '/thetaRV', thetaRV)

        # variable declaration for theta angles
        thetaCenter = np.zeros(57, dtype=complex)
        thetaRadius = np.zeros(57, dtype=float)
        thetaAngFW = np.zeros((57, repeat, iteration+1), dtype=float)
        thetaAngRV = np.zeros((57, repeat, iteration+1), dtype=float)

        # measure centers
        for c in self.goodIdx:
            data = np.concatenate((thetaFW[c].flatten(), thetaRV[c].flatten()))
            x, y, r = circle_fitting(data)
            thetaCenter[c] = x + y*(1j)
            thetaRadius[c] = r

        # measure theta angles
        for c in self.goodIdx:
            for n in range(repeat):
                for k in range(iteration+1):
                    thetaAngFW[c, n, k] = np.angle(thetaFW[c, n, k] - thetaCenter[c])
                    thetaAngRV[c, n, k] = np.angle(thetaRV[c, n, k] - thetaCenter[c])
                home1 = thetaAngFW[c, n, 0]
                home2 = thetaAngRV[c, n, -1]
                thetaAngFW[c, n] = (thetaAngFW[c, n] - home1 + 0.1) % (np.pi*2)
                thetaAngRV[c, n] = (thetaAngRV[c, n] - home2 + 0.1) % (np.pi*2)

                # fix over 2*pi angle issue
                diff = thetaAngFW[c, n, 1:] - thetaAngFW[c, n, :-1]
                t = np.where(diff < -np.pi/2)
                if t[0].size != 0:
                    thetaAngFW[c, n, t[0][0]+1:] += np.pi*2
                thetaAngFW[c, n] -= 0.1

                diff = thetaAngRV[c, n, 1:] - thetaAngRV[c, n, :-1]
                t = np.where(diff > np.pi/2)
                if t[0].size != 0:
                    thetaAngRV[c, n, :t[0][0]+1] += np.pi*2
                thetaAngRV[c, n] += (home2 - home1 + 0.1) % (np.pi*2) - 0.2

        # mark bad cobras by checking hard stops
        badRange = np.where(np.any(thetaAngRV[:, :, 0] < np.pi*2, axis=1))[0]

        # save calculation result
        np.save(dataPath + '/thetaCenter', thetaCenter)
        np.save(dataPath + '/thetaRadius', thetaRadius)
        np.save(dataPath + '/thetaAngFW', thetaAngFW)
        np.save(dataPath + '/thetaAngRV', thetaAngRV)
        np.save(dataPath + '/badRange', badRange)

        # use both Johannes way and average steps for motor maps
        binSize = np.deg2rad(3.6)
        regions = 112
        thetaMMFW = np.zeros((57, regions), dtype=float)
        thetaMMRV = np.zeros((57, regions), dtype=float)
        thetaMMFW2 = np.zeros((57, regions), dtype=float)
        thetaMMRV2 = np.zeros((57, regions), dtype=float)
        thetaSpeedFW = np.zeros(57, dtype=float)
        thetaSpeedRV = np.zeros(57, dtype=float)
        thetaStops = thetaAngRV[:, :, 0] - delta
        cnt = np.zeros(regions)
        bad = np.full(57, False)

        for c in self.goodIdx:
            # calculate theta motor maps in Jonhannes way
            for b in range(regions):
                binMin = binSize * b
                binMax = binMin + binSize

                # forward motor maps
                fracSum = 0
                valueSum = 0
                for n in range(repeat):
                    for k in range(iteration):
                        if thetaAngFW[c, n, k+1] < thetaAngFW[c, n, k] or thetaAngFW[c, n, k+1] > thetaStops[c, n]:
                            # hit hard stop or somethings went wrong, then skip it
                            continue
                        if thetaAngFW[c, n, k] < binMax and thetaAngFW[c, n, k+1] > binMin:
                            moveSizeInBin = np.min([thetaAngFW[c, n, k+1], binMax]) - np.max([thetaAngFW[c, n, k], binMin])
                            entireMoveSize = thetaAngFW[c, n, k+1] - thetaAngFW[c, n, k]
                            fraction = moveSizeInBin * moveSizeInBin / entireMoveSize
                            fracSum += fraction
                            valueSum += fraction * entireMoveSize / steps
                if fracSum > 0:
                    thetaMMFW[c, b] = valueSum / fracSum
                else:
                    thetaMMFW[c, b] = 0

                # reverse motor maps
                fracSum = 0
                valueSum = 0
                for n in range(repeat):
                    for k in range(iteration):
                        if thetaAngRV[c, n, k+1] > thetaAngRV[c, n, k] or thetaAngRV[c, n, k+1] < delta:
                            # hit hard stop or somethings went wrong, then skip it
                            continue
                        if thetaAngRV[c, n, k] > binMin and thetaAngRV[c, n, k+1] < binMax:
                            moveSizeInBin = np.min([thetaAngRV[c, n, k], binMax]) - np.max([thetaAngRV[c, n, k+1], binMin])
                            entireMoveSize = thetaAngRV[c, n, k] - thetaAngRV[c, n, k+1]
                            fraction = moveSizeInBin * moveSizeInBin / entireMoveSize
                            fracSum += fraction
                            valueSum += fraction * entireMoveSize / steps
                if fracSum > 0:
                    thetaMMRV[c, b] = valueSum / fracSum
                else:
                    thetaMMRV[c, b] = 0

            # fill the zeros closed to hard stops
            nz = np.nonzero(thetaMMFW[c])[0]
            if nz.size > 0:
                thetaMMFW[c, :nz[0]] = thetaMMFW[c, nz[0]]
                thetaMMFW[c, nz[-1]+1:] = thetaMMFW[c, nz[-1]]
            else:
                bad[c] = True

            nz = np.nonzero(thetaMMRV[c])[0]
            if nz.size > 0:
                thetaMMRV[c, :nz[0]] = thetaMMRV[c, nz[0]]
                thetaMMRV[c, nz[-1]+1:] = thetaMMRV[c, nz[-1]]
            else:
                bad[c] = True

            # calculate average speed
            mSteps = 0
            mAngle = 0
            for n in range(repeat):
                for k in range(iteration):
                    if thetaAngFW[c, n, k+1] > thetaStops[c, n]:
                        break
                mAngle += thetaAngFW[c, n, k] - thetaAngFW[c, n, 0]
                mSteps += k * steps
            thetaSpeedFW[c] = mAngle / mSteps

            mSteps = 0
            mAngle = 0
            for n in range(repeat):
                for k in range(iteration):
                    if thetaAngRV[c, n, k+1] < delta:
                        break
                mAngle += thetaAngRV[c, n, 0] - thetaAngRV[c, n, k]
                mSteps += k * steps
            thetaSpeedRV[c] = mAngle / mSteps

            # calculate motor maps based on average step counts
            cnt[:] = 0
            for n in range(repeat):
                for k in range(iteration):
                    if thetaAngFW[c, n, k+1] < thetaAngFW[c, n, k] or thetaAngFW[c, n, k+1] > thetaStops[c, n]:
                        # hit hard stop or somethings went wrong, stop here
                        break
                x = np.arange(regions+1) * binSize
                xp = thetaAngFW[c, n, :k+1]
                fp = np.arange(k+1) * steps
                mm = np.interp(x, xp, fp)
                diff = mm[1:] - mm[:-1]
                nz = np.nonzero(diff)[0]
                if nz.size > 0:
                    thetaMMFW2[c] += diff
                    cnt[nz[:-1]] += 1
                    if thetaAngFW[c, n, k] % binSize != 0:
                        cnt[nz[-1]] += (thetaAngFW[c, n, k] % binSize) / binSize
                    else:
                        cnt[nz[-1]] += 1
            nz = np.nonzero(cnt)[0]
            if nz.size > 0:
                thetaMMFW2[c, nz] = binSize / (thetaMMFW2[c, nz] / cnt[nz])
                thetaMMFW2[c, nz[-1]+1:] = thetaMMFW2[c, nz[-1]]
            else:
                bad[c] = True

            cnt[:] = 0
            for n in range(repeat):
                for k in range(iteration):
                    if thetaAngRV[c, n, k+1] > thetaAngRV[c, n, k] or thetaAngRV[c, n, k+1] < delta:
                        # hit hard stop or somethings went wrong, stop here
                        break
                x = np.arange(regions+1) * binSize
                xp = np.flip(thetaAngRV[c, n, :k+1], 0)
                fp = np.arange(k+1) * steps
                mm = np.interp(x, xp, fp)
                diff = mm[1:] - mm[:-1]
                nz = np.nonzero(diff)[0]
                if nz.size > 0:
                    thetaMMRV2[c] += diff
                    cnt[nz[1:-1]] += 1
                    cnt[nz[0]] += 1 - (thetaAngRV[c, n, k] % binSize) / binSize
                    if thetaAngRV[c, n, 0] % binSize != 0:
                        cnt[nz[-1]] += (thetaAngRV[c, n, 0] % binSize) / binSize
                    else:
                        cnt[nz[-1]] += 1
            nz = np.nonzero(cnt)[0]
            if nz.size > 0:
                thetaMMRV2[c, nz] = binSize / (thetaMMRV2[c, nz] / cnt[nz])
                thetaMMRV2[c, :nz[0]] = thetaMMRV2[c, nz[0]]
                thetaMMRV2[c, nz[-1]+1:] = thetaMMRV2[c, nz[-1]]
            else:
                bad[c] = True

        # save calculation result
        np.save(dataPath + '/thetaMMFW', thetaMMFW)
        np.save(dataPath + '/thetaMMRV', thetaMMRV)
        np.save(dataPath + '/thetaMMFW2', thetaMMFW2)
        np.save(dataPath + '/thetaMMRV2', thetaMMRV2)
        np.save(dataPath + '/thetaSpeedFW', thetaSpeedFW)
        np.save(dataPath + '/thetaSpeedRV', thetaSpeedRV)
        np.save(dataPath + '/bad', np.where(bad)[0])

        # update XML configuration
        new = self.pfi.calibModel
        idx = np.array([c for c in self.goodIdx if not bad[c]])

        sThetaFW = binSize / new.S2Pm
        sThetaRV = binSize / new.S2Nm
        fThetaFW = binSize / new.F2Pm
        fThetaRV = binSize / new.F2Nm

        if thetaOnTime is not None:
            # update motor maps, fast: Johannes, slow: simple
            fThetaFW[idx] = thetaMMFW[idx]
            fThetaRV[idx] = thetaMMRV[idx]
            new.updateMotorMaps(thtFwd=fThetaFW, thtRev=fThetaRV, useSlowMaps=False)

            sThetaFW[idx] = thetaMMFW2[idx]
            sThetaRV[idx] = thetaMMRV2[idx]
            new.updateMotorMaps(thtFwd=sThetaFW, thtRev=sThetaRV, useSlowMaps=True)

            # set fast on-time
            self.pfi.calibModel.updateOntimes(*(defaultOnTime[:2] + slowOnTime[2:]), fast=True)

        elif fast:
            # update fast motor maps, Johannes weighting
            fThetaFW[idx] = thetaMMFW[idx]
            fThetaRV[idx] = thetaMMRV[idx]
            new.updateMotorMaps(thtFwd=fTHetaFW, thtRev=fThetaRV, useSlowMaps=False)

            # restore on-time
            self.pfi.calibModel.updateOntimes(*defaultOnTime, fast=True)
            self.pfi.calibModel.updateOntimes(*defaultOnTimeSlow, fast=False)

        else:
            # update slow motor maps, Johannes weighting
            sThetaFW[idx] = thetaMMFW[idx]
            sThetaRV[idx] = thetaMMRV[idx]
            new.updateMotorMaps(thtFwd=sThetaFW, thtRev=sThetaRV, useSlowMaps=True)

            # restore on-time
            self.pfi.calibModel.updateOntimes(*defaultOnTime, fast=True)

        # create a new XML file
        new.createCalibrationFile(dataPath + '/' + newXml)

        # restore default setting
        self.pfi.loadModel(self.xml)

    def phiConvergenceTest(self, dataPath, margin=15.0, runs=50, tries=8, fast=True):
        # variable declaration for center measurement
        steps = 200
        iteration = 4000 // steps
        phiFW = np.zeros((57, iteration+1), dtype=complex)
        phiRV = np.zeros((57, iteration+1), dtype=complex)

        #record the phi movements
        self.pfi.moveAllSteps(self.goodCobras, 0, -5000)
        data1 = self.cam1.expose()
        data2 = self.cam2.expose()
        phiFW[self.goodIdx, 0] = self.extractPositions(data1, data2)
        stack_image1 = data1
        stack_image2 = data2

        for k in range(iteration):
            self.pfi.moveAllSteps(self.goodCobras, 0, steps)
            data1 = self.cam1.expose()
            data2 = self.cam2.expose()
            phiFW[self.goodIdx, k+1] = self.extractPositions(data1, data2, guess=phiFW[self.goodIdx, k])
            stack_image1 += data1
            stack_image2 += data2
        fits.writeto(dataPath + f'/phi1ForwardStack.fits.gz', stack_image1, overwrite=True)
        fits.writeto(dataPath + f'/phi2ForwardStack.fits.gz', stack_image2, overwrite=True)

        # make sure it goes to the limit
        self.pfi.moveAllSteps(self.goodCobras, 0, 5000)

        # reverse phi motors
        data1 = self.cam1.expose()
        data2 = self.cam2.expose()
        phiRV[self.goodIdx, 0] = self.extractPositions(data1, data2, guess=phiFW[self.goodIdx, iteration])
        stack_image1 = data1
        stack_image2 = data2

        for k in range(iteration):
            self.pfi.moveAllSteps(self.goodCobras, 0, -steps)
            data1 = self.cam1.expose()
            data2 = self.cam2.expose()
            phiRV[self.goodIdx, k+1] = self.extractPositions(data1, data2, guess=phiRV[self.goodIdx, k])
            stack_image1 += data1
            stack_image2 += data2
        fits.writeto(dataPath + f'/phi1ReverseStack.fits.gz', stack_image1, overwrite=True)
        fits.writeto(dataPath + f'/phi2ReverseStack.fits.gz', stack_image2, overwrite=True)

        # At the end, make sure the cobra back to the hard stop
        self.pfi.moveAllSteps(self.goodCobras, 0, -5000)

        # save calculation result
        np.save(dataPath + '/phiFW', phiFW)
        np.save(dataPath + '/phiRV', phiRV)

        # variable declaration
        phiCenter = np.zeros(57, dtype=complex)
        phiRadius = np.zeros(57, dtype=float)
        phiHS = np.zeros(57, dtype=float)

        # measure centers
        for c in self.goodIdx:
            data = np.concatenate((phiFW[c].flatten(), phiRV[c].flatten()))
            x, y, r = circle_fitting(data)
            phiCenter[c] = x + y*(1j)
            phiRadius[c] = r

        # measure phi hard stops
        for c in self.goodIdx:
            phiHS[c] = np.angle(phiFW[c, 0] - phiCenter[c])

        # save calculation result
        np.save(dataPath + '/phiCenter', phiCenter)
        np.save(dataPath + '/phiRadius', phiRadius)
        np.save(dataPath + '/phiHS', phiHS)

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
            self.moveThetaPhi(self.goodCobras, zeros, zeros + angle, phiFast=fast)
            cAngles, cPositions = self.measureAngles(centers, homes)
            phiData[goodIdx, i, 0, 0] = cAngles
            phiData[goodIdx, i, 0, 1] = np.real(cPositions)
            phiData[goodIdx, i, 0, 2] = np.imag(cPositions)

            for j in range(tries - 1):
                self.moveThetaPhi(self.goodCobras, zeros, angle - cAngles, phiFroms=cAngles, phiFast=fast)
                cAngles, cPositions = self.measureAngles(centers, homes)
                cAngles[cAngles>np.pi*(3/2)] -= np.pi*2
                phiData[goodIdx, i, j+1, 0] = cAngles
                phiData[goodIdx, i, j+1, 1] = np.real(cPositions)
                phiData[goodIdx, i, j+1, 2] = np.imag(cPositions)

            # home phi
            self.pfi.moveAllSteps(self.goodCobras, 0, -5000)

        # save calculation result
        np.save(dataPath + '/phiData', phiData)

    def thetaConvergenceTest(self, dataPath, margin=15.0, runs=50, tries=8, fast=True):
        # variable declaration for center measurement
        steps = 300
        iteration = 6000 // steps
        thetaFW = np.zeros((57, iteration+1), dtype=complex)
        thetaRV = np.zeros((57, iteration+1), dtype=complex)

        #record the theta movements
        self.pfi.moveAllSteps(self.goodCobras, -10000, 0)
        data1 = self.cam1.expose()
        data2 = self.cam2.expose()
        thetaFW[self.goodIdx, 0] = self.extractPositions(data1, data2)
        stack_image1 = data1
        stack_image2 = data2

        for k in range(iteration):
            self.pfi.moveAllSteps(self.goodCobras, steps, 0)
            data1 = self.cam1.expose()
            data2 = self.cam2.expose()
            thetaFW[self.goodIdx, k+1] = self.extractPositions(data1, data2)
            stack_image1 += data1
            stack_image2 += data2
        fits.writeto(dataPath + f'/theta1ForwardStack.fits.gz', stack_image1, overwrite=True)
        fits.writeto(dataPath + f'/theta2ForwardStack.fits.gz', stack_image2, overwrite=True)

        # make sure it goes to the limit
        self.pfi.moveAllSteps(self.goodCobras, 10000, 0)

        # reverse theta motors
        data1 = self.cam1.expose()
        data2 = self.cam2.expose()
        thetaRV[self.goodIdx, 0] = self.extractPositions(data1, data2)
        stack_image1 = data1
        stack_image2 = data2

        for k in range(iteration):
            self.pfi.moveAllSteps(self.goodCobras, -steps, 0)
            data1 = self.cam1.expose()
            data2 = self.cam2.expose()
            thetaRV[self.goodIdx, k+1] = self.extractPositions(data1, data2)
            stack_image1 += data1
            stack_image2 += data2
        fits.writeto(dataPath + f'/theta1ReverseStack.fits.gz', stack_image1, overwrite=True)
        fits.writeto(dataPath + f'/theta2ReverseStack.fits.gz', stack_image2, overwrite=True)

        # At the end, make sure the cobra back to the hard stop
        self.pfi.moveAllSteps(self.goodCobras, -10000, 0)

        # save calculation result
        np.save(dataPath + '/thetaFW', thetaFW)
        np.save(dataPath + '/thetaRV', thetaRV)

        # variable declaration
        thetaCenter = np.zeros(57, dtype=complex)
        thetaRadius = np.zeros(57, dtype=float)
        thetaHS = np.zeros(57, dtype=float)

        # measure centers
        for c in self.goodIdx:
            data = np.concatenate((thetaFW[c].flatten(), thetaRV[c].flatten()))
            x, y, r = circle_fitting(data)
            thetaCenter[c] = x + y*(1j)
            thetaRadius[c] = r

        # measure theta hard stops
        for c in self.goodIdx:
            thetaHS[c] = np.angle(thetaFW[c, 0] - thetaCenter[c])

        # save calculation result
        np.save(dataPath + '/thetaCenter', thetaCenter)
        np.save(dataPath + '/thetaRadius', thetaRadius)
        np.save(dataPath + '/thetaHS', thetaHS)

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
            self.moveThetaPhi(self.goodCobras, zeros + angle, zeros, thetaFast=fast)
            cAngles, cPositions = self.measureAngles(centers, homes)
            for k in range(len(goodIdx)):
                if angle > np.pi + tGaps[k] and cAngles[k] < tGaps[k] + 0.1:
                    cAngles[k] += np.pi*2
            thetaData[goodIdx, i, 0, 0] = cAngles
            thetaData[goodIdx, i, 0, 1] = np.real(cPositions)
            thetaData[goodIdx, i, 0, 2] = np.imag(cPositions)

            for j in range(tries - 1):
                dirs = angle > cAngles
                self.moveThetaPhi(self.goodCobras, angle - cAngles, zeros, thetaFroms=cAngles, thetaFast=fast)
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
            self.pfi.moveAllSteps(self.goodCobras, -10000, 0)

        # save calculation result
        np.save(dataPath + '/thetaData', thetaData)

    def measureAngles(self, centers, homes):
        """ measure positions and angles for good cobras """

        data1 = self.cam1.expose()
        data2 = self.cam2.expose()
        curPos = self.extractPositions(data1, data2)
        angles = (np.angle(curPos - centers) - homes) % (np.pi*2)
        return angles, curPos


def getCobras(cobs):
    # cobs is 0-indexed list
    return pfiControl.PFI.allocateCobraList(zip(np.full(len(cobs), 1), np.array(cobs) + 1))

def lazyIdentification(centers, spots, radii=None):
    n = len(centers)
    if radii is not None and len(radii) != n:
        raise RuntimeError("number of centers must match number of radii")
    ans = np.empty(n, dtype=int)
    for i in range(n):
        dist = np.absolute(spots - centers[i])
        j = np.argmin(dist)
        if radii is not None and np.absolute(centers[i] - spots[j]) > radii[i]:
            ans[i] = -1
        else:
            ans[i] = j
    return ans

def circle_fitting(p):
    x = np.real(p)
    y = np.imag(p)
    m = np.vstack([x, y, np.ones(len(p))]).T
    n = np.array(x*x + y*y)
    a, b, c = np.linalg.lstsq(m, n, rcond=None)[0]
    return a/2, b/2, np.sqrt(c+(a*a+b*b)/4)
