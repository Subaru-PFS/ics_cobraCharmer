import os
import sys
import numpy as np
from astropy.io import fits
import sep
from copy import deepcopy
from ics.cobraCharmer import pfiDesign

class moduleAnalyze():
    def __init__(self, xml, brokens=None, camSplit=26):
        if not os.path.exists(xml):
            print(f"Error: {xml} is not presented!")
            sys.exit()
        self.xml = xml
        self.calibModel = pfiDesign.PFIDesign(xml)
        self.setBrokenCobras(brokens)
        self.camSplit = camSplit

    def setBrokenCobras(self, brokens=None):
        # define the broken/good cobras
        if brokens is None:
            brokens = []
        visibles = [e for e in range(1, 58) if e not in brokens]
        self.badIdx = np.array(brokens, dtype=int) - 1
        self.goodIdx = np.array(visibles, dtype=int) - 1

    def extractPositions(self, data1, data2, guess=None, tolerance=None):
        idx = self.goodIdx
        idx1 = idx[idx <= self.camSplit]
        idx2 = idx[idx > self.camSplit]
        if tolerance is not None:
            radii = (self.calibModel.L1 + self.calibModel.L2) * (1 + tolerance)
            radii1 = radii[idx1]
            radii2 = radii[idx2]
        else:
            radii1 = None
            radii2 = None

        if guess is None:
            center1 = self.calibModel.centers[idx1]
            center2 = self.calibModel.centers[idx2]
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
                pos[n] = self.calibModel.centers[idx[n]]
            else:
                pos[n] = pos1[k]
        for n, k in enumerate(target2):
            m = n + len(target1)
            if k < 0:
                pos[m] = self.calibModel.centers[idx[m]]
            else:
                pos[m] = pos2[k]
        return pos

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
        defaultOnTime = deepcopy([self.calibModel.motorOntimeFwd1,
                                  self.calibModel.motorOntimeRev1,
                                  self.calibModel.motorOntimeFwd2,
                                  self.calibModel.motorOntimeRev2])
        defaultOnTimeSlow = deepcopy([self.calibModel.motorOntimeSlowFwd1,
                                      self.calibModel.motorOntimeSlowRev1,
                                      self.calibModel.motorOntimeSlowFwd2,
                                      self.calibModel.motorOntimeSlowRev2])

        # set fast on-time to a large value so it can move over whole range, set slow on-time to the test value.
        fastOnTime = [np.full(57, limitOnTime)] * 4
        if phiOnTime is not None:
            slowOnTime = defaultOnTimeSlow[:2] + [np.full(57, phiOnTime)] * 2
        elif fast:
            slowOnTime = defaultOnTimeSlow[:2] + defaultOnTime[2:]
        else:
            slowOnTime = defaultOnTimeSlow

        # update one-time for test
        self.calibModel.updateOntimes(*fastOnTime, fast=True)
        self.calibModel.updateOntimes(*slowOnTime, fast=False)

        # variable declaration for position measurement
        iteration = totalSteps // steps
        phiFW = np.zeros((57, repeat, iteration+1), dtype=complex)
        phiRV = np.zeros((57, repeat, iteration+1), dtype=complex)

        #record the phi movements
        for n in range(repeat):
            # forward phi motor maps
            data1 = fits.getdata(dataPath + f'/phi1Begin{n}.fits.gz')
            data2 = fits.getdata(dataPath + f'/phi2Begin{n}.fits.gz')
            phiFW[self.goodIdx, n, 0] = self.extractPositions(data1, data2)

            for k in range(iteration):
                data1 = fits.getdata(dataPath + f'/phi1Forward{n}N{k}.fits.gz')
                data2 = fits.getdata(dataPath + f'/phi2Forward{n}N{k}.fits.gz')
                phiFW[self.goodIdx, n, k+1] = self.extractPositions(data1, data2, guess=phiFW[self.goodIdx, n, k])

            # reverse phi motor maps
            data1 = fits.getdata(dataPath + f'/phi1End{n}.fits.gz')
            data2 = fits.getdata(dataPath + f'/phi2End{n}.fits.gz')
            phiRV[self.goodIdx, n, 0] = self.extractPositions(data1, data2, guess=phiFW[self.goodIdx, n, iteration])

            for k in range(iteration):
                data1 = fits.getdata(dataPath + f'/phi1Reverse{n}N{k}.fits.gz')
                data2 = fits.getdata(dataPath + f'/phi2Reverse{n}N{k}.fits.gz')
                phiRV[self.goodIdx, n, k+1] = self.extractPositions(data1, data2, guess=phiRV[self.goodIdx, n, k])

        # save calculation result
        np.save(dataPath + '/phiFW_A', phiFW)
        np.save(dataPath + '/phiRV_A', phiRV)

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
        np.save(dataPath + '/phiCenter_A', phiCenter)
        np.save(dataPath + '/phiRadius_A', phiRadius)
        np.save(dataPath + '/phiAngFW_A', phiAngFW)
        np.save(dataPath + '/phiAngRV_A', phiAngRV)

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

        for c in self.goodIdx:
            # calculate phi motor maps in Johannes way
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
                        if phiAngRV[c, n, k+1] - phiAngRV[c, n, k] > 0 or phiAngRV[c, n, k+1] < delta:
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
            phiMMFW[c, :nz[0]] = phiMMFW[c, nz[0]]
            phiMMFW[c, nz[-1]+1:] = phiMMFW[c, nz[-1]]

            nz = np.nonzero(phiMMRV[c])[0]
            phiMMRV[c, :nz[0]] = phiMMRV[c, nz[0]]
            phiMMRV[c, nz[-1]+1:] = phiMMRV[c, nz[-1]]

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
                phiMMFW2[c] += diff
                cnt[nz[:-1]] += 1
                if phiAngFW[c, n, k] % binSize != 0:
                    cnt[nz[-1]] += (phiAngFW[c, n, k] % binSize) / binSize
                else:
                    cnt[nz[-1]] += 1
            nz = np.nonzero(cnt)[0]
            phiMMFW2[c, nz] = binSize / (phiMMFW2[c, nz] / cnt[nz])
            phiMMFW2[c, nz[-1]+1:] = phiMMFW2[c, nz[-1]]

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
                phiMMRV2[c] += diff
                cnt[nz[1:-1]] += 1
                cnt[nz[0]] += 1 - (phiAngRV[c, n, k] % binSize) / binSize
                if phiAngRV[c, n, 0] % binSize != 0:
                    cnt[nz[-1]] += (phiAngRV[c, n, 0] % binSize) / binSize
                else:
                    cnt[nz[-1]] += 1
            nz = np.nonzero(cnt)[0]
            phiMMRV2[c, nz] = binSize / (phiMMRV2[c, nz] / cnt[nz])
            phiMMRV2[c, :nz[0]] = phiMMRV2[c, nz[0]]
            phiMMRV2[c, nz[-1]+1:] = phiMMRV2[c, nz[-1]]

        # save calculation result
        np.save(dataPath + '/phiMMFW_A', phiMMFW)
        np.save(dataPath + '/phiMMRV_A', phiMMRV)
        np.save(dataPath + '/phiMMFW2_A', phiMMFW2)
        np.save(dataPath + '/phiMMRV2_A', phiMMRV2)
        np.save(dataPath + '/phiSpeedFW_A', phiSpeedFW)
        np.save(dataPath + '/phiSpeedRV_A', phiSpeedRV)

        # update XML configuration
        new = self.calibModel
        idx = self.goodIdx

        sPhiFW = binSize / new.S2Pm
        sPhiRV = binSize / new.S2Nm
        fPhiFW = binSize / new.F2Pm
        fPhiRV = binSize / new.F2Nm

        if phiOnTime is not None:
            # update motor maps, fast: Johannes, slow: average step counts
            fPhiFW[idx] = phiMMFW[idx]
            fPhiRV[idx] = phiMMRV[idx]
            new.updateMotorMaps(phiFwd=fPhiFW, phiRev=fPhiRV, useSlowMaps=False)

            sPhiFW[idx] = phiMMFW2[idx]
            sPhiRV[idx] = phiMMRV2[idx]
            new.updateMotorMaps(phiFwd=sPhiFW, phiRev=sPhiRV, useSlowMaps=True)

            # set fast on-time
            self.calibModel.updateOntimes(*(defaultOnTime[:2] + slowOnTime[2:]), fast=True)

        elif fast:
            # update fast motor maps, Johannes weighting
            fPhiFW[idx] = phiMMFW[idx]
            fPhiRV[idx] = phiMMRV[idx]
            new.updateMotorMaps(phiFwd=fPhiFW, phiRev=fPhiRV, useSlowMaps=False)

            # restore on-time
            self.calibModel.updateOntimes(*defaultOnTime, fast=True)
            self.calibModel.updateOntimes(*defaultOnTimeSlow, fast=False)

        else:
            # update slow motor maps, Johanees weighting
            sPhiFW[idx] = phiMMFW[idx]
            sPhiRV[idx] = phiMMRV[idx]
            new.updateMotorMaps(phiFwd=sPhiFW, phiRev=sPhiRV, useSlowMaps=True)

            # restore on-time
            self.calibModel.updateOntimes(*defaultOnTime, fast=True)

        # create a new XML file
        new.createCalibrationFile(dataPath + '/' + newXml)

        # restore default setting
        self.calibModel = pfiDesign.PFIDesign(self.xml)

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
        defaultOnTime = deepcopy([self.calibModel.motorOntimeFwd1,
                                  self.calibModel.motorOntimeRev1,
                                  self.calibModel.motorOntimeFwd2,
                                  self.calibModel.motorOntimeRev2])
        defaultOnTimeSlow = deepcopy([self.calibModel.motorOntimeSlowFwd1,
                                      self.calibModel.motorOntimeSlowRev1,
                                      self.calibModel.motorOntimeSlowFwd2,
                                      self.calibModel.motorOntimeSlowRev2])

        # set fast on-time to a large value so it can move over whole range, set slow on-time to the test value.
        fastOnTime = [np.full(57, limitOnTime)] * 4
        if thetaOnTime is not None:
            slowOnTime = [np.full(57, thetaOnTime)] * 2 + defaultOnTimeSlow[2:]
        elif fast:
            slowOnTime = defaultOnTime[:2] + defaultOnTimeSlow[2:]
        else:
            slowOnTime = defaultOnTimeSlow

        # update one-time for test
        self.calibModel.updateOntimes(*fastOnTime, fast=True)
        self.calibModel.updateOntimes(*slowOnTime, fast=False)

        # variable declaration for position measurement
        iteration = totalSteps // steps
        thetaFW = np.zeros((57, repeat, iteration+1), dtype=complex)
        thetaRV = np.zeros((57, repeat, iteration+1), dtype=complex)

        #record the theta movements
        for n in range(repeat):
            # forward theta motor maps
            data1 = fits.getdata(dataPath + f'/theta1Begin{n}.fits.gz')
            data2 = fits.getdata(dataPath + f'/theta2Begin{n}.fits.gz')
            thetaFW[self.goodIdx, n, 0] = self.extractPositions(data1, data2)

            for k in range(iteration):
                data1 = fits.getdata(dataPath + f'/theta1Forward{n}N{k}.fits.gz')
                data2 = fits.getdata(dataPath + f'/theta2Forward{n}N{k}.fits.gz')
                thetaFW[self.goodIdx, n, k+1] = self.extractPositions(data1, data2, guess=thetaFW[self.goodIdx, n, k])

            # reverse theta motor maps
            data1 = fits.getdata(dataPath + f'/theta1End{n}.fits.gz')
            data2 = fits.getdata(dataPath + f'/theta2End{n}.fits.gz')
            thetaRV[self.goodIdx, n, 0] = self.extractPositions(data1, data2, guess=thetaFW[self.goodIdx, n, iteration])

            for k in range(iteration):
                data1 = fits.getdata(dataPath + f'/theta1Reverse{n}N{k}.fits.gz')
                data2 = fits.getdata(dataPath + f'/theta2Reverse{n}N{k}.fits.gz')
                thetaRV[self.goodIdx, n, k+1] = self.extractPositions(data1, data2, guess=thetaRV[self.goodIdx, n, k])

        # save calculation result
        np.save(dataPath + '/thetaFW_A', thetaFW)
        np.save(dataPath + '/thetaRV_A', thetaRV)

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
        bad = np.where(np.any(thetaAngRV[:, :, 0] < np.pi*2, axis=1))[0]

        # save calculation result
        np.save(dataPath + '/thetaCenter_A', thetaCenter)
        np.save(dataPath + '/thetaRadius_A', thetaRadius)
        np.save(dataPath + '/thetaAngFW_A', thetaAngFW)
        np.save(dataPath + '/thetaAngRV_A', thetaAngRV)
        np.save(dataPath + '/bad_A', bad)

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
            thetaMMFW[c, :nz[0]] = thetaMMFW[c, nz[0]]
            thetaMMFW[c, nz[-1]+1:] = thetaMMFW[c, nz[-1]]

            nz = np.nonzero(thetaMMRV[c])[0]
            thetaMMRV[c, :nz[0]] = thetaMMRV[c, nz[0]]
            thetaMMRV[c, nz[-1]+1:] = thetaMMRV[c, nz[-1]]

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
                thetaMMFW2[c] += diff
                cnt[nz[:-1]] += 1
                if thetaAngFW[c, n, k] % binSize != 0:
                    cnt[nz[-1]] += (thetaAngFW[c, n, k] % binSize) / binSize
                else:
                    cnt[nz[-1]] += 1
            nz = np.nonzero(cnt)[0]
            thetaMMFW2[c, nz] = binSize / (thetaMMFW2[c, nz] / cnt[nz])
            thetaMMFW2[c, nz[-1]+1:] = thetaMMFW2[c, nz[-1]]

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
                thetaMMRV2[c] += diff
                cnt[nz[1:-1]] += 1
                cnt[nz[0]] += 1 - (thetaAngRV[c, n, k] % binSize) / binSize
                if thetaAngRV[c, n, 0] % binSize != 0:
                    cnt[nz[-1]] += (thetaAngRV[c, n, 0] % binSize) / binSize
                else:
                    cnt[nz[-1]] += 1
            nz = np.nonzero(cnt)[0]
            thetaMMRV2[c, nz] = binSize / (thetaMMRV2[c, nz] / cnt[nz])
            thetaMMRV2[c, :nz[0]] = thetaMMRV2[c, nz[0]]
            thetaMMRV2[c, nz[-1]+1:] = thetaMMRV2[c, nz[-1]]

        # save calculation result
        np.save(dataPath + '/thetaMMFW_A', thetaMMFW)
        np.save(dataPath + '/thetaMMRV_A', thetaMMRV)
        np.save(dataPath + '/thetaMMFW2_A', thetaMMFW2)
        np.save(dataPath + '/thetaMMRV2_A', thetaMMRV2)
        np.save(dataPath + '/thetaSpeedFW_A', thetaSpeedFW)
        np.save(dataPath + '/thetaSpeedRV_A', thetaSpeedRV)

        # update XML configuration
        new = self.calibModel
        idx = self.goodIdx

        sThetaFW = binSize / new.S2Pm
        sThetaRV = binSize / new.S2Nm
        fThetaFW = binSize / new.F2Pm
        fThetaRV = binSize / new.F2Nm

        if thetaOnTime is not None:
            # update motor maps, fast: Johannes, slow: average step counts
            fThetaFW[idx] = thetaMMFW[idx]
            fThetaRV[idx] = thetaMMRV[idx]
            new.updateMotorMaps(thtFwd=fThetaFW, thtRev=fThetaRV, useSlowMaps=False)

            sThetaFW[idx] = thetaMMFW2[idx]
            sThetaRV[idx] = thetaMMRV2[idx]
            new.updateMotorMaps(thtFwd=sThetaFW, thtRev=sThetaRV, useSlowMaps=True)

            # set fast on-time
            self.calibModel.updateOntimes(*(defaultOnTime[:2] + slowOnTime[2:]), fast=True)

        elif fast:
            # update fast motor maps, Johannes weighting
            fThetaFW[idx] = thetaMMFW[idx]
            fThetaRV[idx] = thetaMMRV[idx]
            new.updateMotorMaps(thtFwd=fTHetaFW, thtRev=fThetaRV, useSlowMaps=False)

            # restore on-time
            self.calibModel.updateOntimes(*defaultOnTime, fast=True)
            self.calibModel.updateOntimes(*defaultOnTimeSlow, fast=False)

        else:
            # update slow motor maps, Johannes weighting
            sThetaFW[idx] = thetaMMFW[idx]
            sThetaRV[idx] = thetaMMRV[idx]
            new.updateMotorMaps(thtFwd=sThetaFW, thtRev=sThetaRV, useSlowMaps=True)

            # restore on-time
            self.calibModel.updateOntimes(*defaultOnTime, fast=True)

        # create a new XML file
        new.createCalibrationFile(dataPath + '/' + newXml)

        # restore default setting
        self.calibModel = pfiDesign.PFIDesign(self.xml)

    def calculateGeometry(self, newXml, dataPath, thetaPath, phiPath):
        """ Update xml file for cobra geometry """

        if os.path.isfile(thetaPath + '/thetaCenter_A.npy'):
            thetaC = np.load(thetaPath + '/thetaCenter_A.npy')
            thetaR = np.load(thetaPath + '/thetaRadius_A.npy')
            thetaFW = np.load(thetaPath + '/thetaFW_A.npy')
            thetaRV = np.load(thetaPath + '/thetaRV_A.npy')
        else:
            thetaC = np.load(thetaPath + '/thetaCenter.npy')
            thetaR = np.load(thetaPath + '/thetaRadius.npy')
            thetaFW = np.load(thetaPath + '/thetaFW.npy')
            thetaRV = np.load(thetaPath + '/thetaRV.npy')

        if os.path.isfile(phiPath + '/phiCenter_A.npy'):
            phiC = np.load(phiPath + '/phiCenter_A.npy')
            phiR = np.load(phiPath + '/phiRadius_A.npy')
            phiFW = np.load(phiPath + '/phiFW_A.npy')
            phiRV = np.load(phiPath + '/phiRV_A.npy')
        else:
            phiC = np.load(phiPath + '/phiCenter.npy')
            phiR = np.load(phiPath + '/phiRadius.npy')
            phiFW = np.load(phiPath + '/phiFW.npy')
            phiRV = np.load(phiPath + '/phiRV.npy')
        nCobra = phiC.shape[0]
        iteration = phiFW.shape[1]

        # calculate arm legnths
        thetaL = np.absolute(phiC - thetaC)
        phiL = phiR

        # calculate phi hard stops
        phiCCW = np.full(nCobra, np.pi)
        phiCW = np.zeros(nCobra)

        s = np.angle(thetaC - phiC)
        for n in range(iteration):
            # CCW hard stops for phi arms
            t = (np.angle(phiFW[:, n, 0] - phiC) - s + (np.pi/2)) % (np.pi*2) - (np.pi/2)
            p = np.where(t < phiCCW)[0]
            phiCCW[p] = t[p]
            # CW hard stops for phi arms
            t = (np.angle(phiRV[:, n, 0] - phiC) - s + (np.pi/2)) % (np.pi*2) - (np.pi/2)
            p = np.where(t > phiCW)[0]
            phiCW[p] = t[p]

        # calculate theta hard stops
        thetaCCW = np.zeros(nCobra)
        thetaCW = np.zeros(nCobra)
        y = self.goodIdx

        for n in range(iteration):
            # CCW hard stops for theta arms
            a = np.absolute(thetaFW[y, n, 0] - thetaC[y])
            s = np.arccos((thetaL[y]*thetaL[y] + a*a - phiL[y]*phiL[y]) / (2*a*thetaL[y]))
            t = (np.angle(thetaFW[y, n, 0] - thetaC[y]) + s) % (np.pi*2)
            if n == 0:
                thetaCCW[y] = t
            else:
                q = (t - thetaCCW[y] + np.pi) % (np.pi*2) - np.pi
                p = np.where(q < 0)[0]
                thetaCCW[y[p]] = t[y[p]]

            # CW hard stops for theta arms
            a = np.absolute(thetaRV[y, n, 0] - thetaC[y])
            s = np.arccos((thetaL[y]*thetaL[y] + a*a - phiL[y]*phiL[y]) / (2*a*thetaL[y]))
            t = (np.angle(thetaRV[y, n, 0] - thetaC[y]) + s) % (np.pi*2)
            if n == 0:
                thetaCW[y] = t
            else:
                q = (t - thetaCW[y] + np.pi) % (np.pi*2) - np.pi
                p = np.where(q > 0)[0]
                thetaCW[y[p]] = t[y[p]]

        # save calculation result
        np.save(dataPath + '/center_A', thetaC)
        np.save(dataPath + '/thetaL_A', thetaL)
        np.save(dataPath + '/phiL_A', phiL)
        np.save(dataPath + '/thetaCCW_A', thetaCCW)
        np.save(dataPath + '/thetaCW_A', thetaCW)
        np.save(dataPath + '/phiCCW_A', phiCCW)
        np.save(dataPath + '/phiCW_A', phiCW)

        # update XML configuration
        new = self.calibModel

        # keep bad cobra configuration
        z = self.badIdx
        thetaC[z] = new.centers[z]
        thetaL[z] = new.L1[z]
        phiL[z] = new.L2[z]
        thetaCCW[z] = new.tht0[z]
        thetaCW[z] = new.tht1[z]
        phiCCW[z] = new.phiIn[z] + np.pi
        phiCW[z] = new.phiOut[z] + np.pi

        # create a new XML file
        new.updateGeometry(thetaC, thetaL, phiL)
        new.updateThetaHardStops(thetaCCW, thetaCW)
        new.updatePhiHardStops(phiCCW, phiCW)
        new.createCalibrationFile(dataPath + '/' + newXml)

        # restore default setting
        self.calibModel = pfiDesign.PFIDesign(self.xml)


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
