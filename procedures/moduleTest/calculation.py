import numpy as np
import sep
from ics.cobraCharmer import pfiDesign
import os
import sys

binSize = np.deg2rad(3.6)
regions = 112

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

def transform(origPoints, newPoints):
    """ return the tranformation parameters and a function that can convert origPoints to newPoints """
    origCenter = np.mean(origPoints)
    newCenter = np.mean(newPoints)
    origVectors = origPoints - origCenter
    newVectors = newPoints - newCenter
    scale = np.sum(np.abs(newVectors)) / np.sum(np.abs(origVectors))
    diffAngles = ((np.angle(newVectors) - np.angle(origVectors)) + np.pi) % (2*np.pi) - np.pi
    tilt = np.sum(diffAngles * np.abs(origVectors)) / np.sum(np.abs(origVectors))
    offset = -origCenter * scale * np.exp(tilt * (1j)) + newCenter
    def tr(x):
        return x * scale * np.exp(tilt * (1j)) + offset
    return offset, scale, tilt, tr

class Calculation():
    def __init__(self, xml, brokens, camSplit):
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

    def extractPositions(self, data1, data2=None, guess=None, tolerance=None):
        if data2 is None:
            return self.extractPositions1(data1, guess=guess, tolerance=tolerance)

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

    def matchPositions(self, objects, guess=None, tolerance=None):
        """ Given a set of measured spots, return the measured positions of our cobras.

        Args
        ----
        objects : `ndarray`, which includes an x and a y column.
           The _measured_ positions, from the camera.

        guess : `ndarray` of complex coordinates.
           Close to where we expect the spots to be. Uses the the cobra center if None

        tolerance : `float`
           A expansion factor to apply to the cobra geometry, for matching.

        Returns
        -------
        pos : `ndarray` of complex coordinates
           The measured positions of the cobras.
           Note that (hmm), the cobra center is returned of there is not matching spot.

        indexMap : `ndarray` of ints
           Indices from our cobra array to the matching spots.
           -1 if there is no matching spot.
        """

        idx = self.goodIdx
        if tolerance is not None:
            radii = (self.calibModel.L1 + self.calibModel.L2) * (1 + tolerance)
        else:
            radii = None

        if guess is None:
            centers = self.calibModel.centers[idx]
        else:
            centers = guess[:len(idx)]

        measPos = np.array(objects['x'] + objects['y']*(1j))
        target = lazyIdentification(centers, measPos, radii=radii)

        pos = np.zeros(len(idx), dtype=complex)
        for n, k in enumerate(target):
            if k < 0:
                pos[n] = self.calibModel.centers[idx[n]]
            else:
                pos[n] = measPos[k]

        return pos, target

    def extractPositions1(self, data, guess=None, tolerance=None):
        idx = self.goodIdx
        if tolerance is not None:
            radii = (self.calibModel.L1 + self.calibModel.L2) * (1 + tolerance)
        else:
            radii = None

        if guess is None:
            centers = self.calibModel.centers[idx]
        else:
            centers = guess[:len(idx)]

        ext = sep.extract(data.astype(float), 200)
        pos = np.array(ext['x'] + ext['y']*(1j))
        target = lazyIdentification(centers, pos, radii=radii)

        pos = np.zeros(len(idx), dtype=complex)
        for n, k in enumerate(target):
            if k < 0:
                pos[n] = self.calibModel.centers[idx[n]]
            else:
                pos[n] = pos[k]

        return pos

    def phiCenterAngles(self, phiFW, phiRV):
        # variable declaration for phi angles
        phiCenter = np.zeros(57, dtype=complex)
        phiRadius = np.zeros(57, dtype=float)
        phiAngFW = np.zeros(phiFW.shape, dtype=float)
        phiAngRV = np.zeros(phiFW.shape, dtype=float)

        # measure centers
        for c in self.goodIdx:
            data = np.concatenate((phiFW[c].flatten(), phiRV[c].flatten()))
            x, y, r = circle_fitting(data)
            phiCenter[c] = x + y*(1j)
            phiRadius[c] = r

        # measure phi angles
        for c in self.goodIdx:
            phiAngFW[c] = np.angle(phiFW[c] - phiCenter[c])
            phiAngRV[c] = np.angle(phiRV[c] - phiCenter[c])
            home = np.copy(phiAngFW[c, :, 0, np.newaxis])
            phiAngFW[c] = (phiAngFW[c] - home + np.pi/2) % (np.pi*2) - np.pi/2
            phiAngRV[c] = (phiAngRV[c] - home + np.pi/2) % (np.pi*2) - np.pi/2

        # mark bad cobras by checking hard stops
        bad = np.any(phiAngRV[:, :, 0] < np.pi*0.8, axis=1)
        bad[np.std(phiAngRV[:, :, 0], axis=1) > 0.1] = True
        badRange = np.where(bad)[0]

        return phiCenter, phiRadius, phiAngFW, phiAngRV, badRange

    def thetaCenterAngles(self, thetaFW, thetaRV):
        # variable declaration for theta angles
        thetaCenter = np.zeros(57, dtype=complex)
        thetaRadius = np.zeros(57, dtype=float)
        thetaAngFW = np.zeros(thetaFW.shape, dtype=float)
        thetaAngRV = np.zeros(thetaFW.shape, dtype=float)

        # measure centers
        for c in self.goodIdx:
            data = np.concatenate((thetaFW[c].flatten(), thetaRV[c].flatten()))
            x, y, r = circle_fitting(data)
            thetaCenter[c] = x + y*(1j)
            thetaRadius[c] = r

        # measure theta angles
        for c in self.goodIdx:
            for n in range(thetaFW.shape[1]):
                thetaAngFW[c, n] = np.angle(thetaFW[c, n] - thetaCenter[c])
                thetaAngRV[c, n] = np.angle(thetaRV[c, n] - thetaCenter[c])
                home1 = thetaAngFW[c, n, 0]
                home2 = thetaAngRV[c, n, -1]
                thetaAngFW[c, n] = (thetaAngFW[c, n] - home1 + 0.1) % (np.pi*2)
                thetaAngRV[c, n] = (thetaAngRV[c, n] - home2 + 0.1) % (np.pi*2)

                # fix over 2*pi angle issue
                diff = thetaAngFW[c, n, 1:] - thetaAngFW[c, n, :-1]
                t = np.where(diff < -np.pi/2)
                if t[0].size > 0:
                    thetaAngFW[c, n, t[0][0]+1:] += np.pi*2
                thetaAngFW[c, n] -= 0.1

                diff = thetaAngRV[c, n, 1:] - thetaAngRV[c, n, :-1]
                t = np.where(diff > np.pi/2)
                if t[0].size > 0:
                    thetaAngRV[c, n, :t[0][0]+1] += np.pi*2
                thetaAngRV[c, n] += (home2 - home1 + 0.1) % (np.pi*2) - 0.2

                # in case only travel in overlapping region
                if thetaAngRV[c, n, 0] - thetaAngFW[c, n, -1] < -0.1:
                    thetaAngRV[c, n] += np.pi*2

        # mark bad cobras by checking hard stops
        bad = np.any(thetaAngRV[:, :, 0] < np.pi*2, axis=1)
        bad[np.std(thetaAngRV[:, :, 0], axis=1) > 0.1] = True
        badRange = np.where(bad)[0]

        return thetaCenter, thetaRadius, thetaAngFW, thetaAngRV, badRange

    def motorMaps(self, angFW, angRV, steps, delta=0.1):
        """ use Johannes weighting for motor maps, delta is the margin for detecting hard stops """
        mmFW = np.zeros((57, regions), dtype=float)
        mmRV = np.zeros((57, regions), dtype=float)
        bad = np.full(57, False)
        repeat = angFW.shape[1]
        iteration = angFW.shape[2] - 1

        for c in self.goodIdx:
            # calculate motor maps in Johannes way
            for b in range(regions):
                binMin = binSize * b
                binMax = binMin + binSize

                # forward motor maps
                fracSum = 0
                valueSum = 0
                for n in range(repeat):
                    for k in range(iteration):
                        if angFW[c, n, k+1] < angFW[c, n, k] or angRV[c, n, 0] - angFW[c, n, k+1] < delta:
                            # hit hard stop or somethings went wrong, then skip it
                            continue
                        if angFW[c, n, k] < binMax and angFW[c, n, k+1] > binMin:
                            moveSizeInBin = np.min([angFW[c, n, k+1], binMax]) - np.max([angFW[c, n, k], binMin])
                            entireMoveSize = angFW[c, n, k+1] - angFW[c, n, k]
                            fraction = moveSizeInBin * moveSizeInBin / entireMoveSize
                            fracSum += fraction
                            valueSum += fraction * entireMoveSize / steps
                if fracSum > 0:
                    mmFW[c, b] = valueSum / fracSum
                else:
                    mmFW[c, b] = 0

                # reverse motor maps
                fracSum = 0
                valueSum = 0
                for n in range(repeat):
                    for k in range(iteration):
                        if angRV[c, n, k+1] > angRV[c, n, k] or angRV[c, n, k+1] < delta:
                            # hit hard stop or somethings went wrong, then skip it
                            continue
                        if angRV[c, n, k] > binMin and angRV[c, n, k+1] < binMax:
                            moveSizeInBin = np.min([angRV[c, n, k], binMax]) - np.max([angRV[c, n, k+1], binMin])
                            entireMoveSize = angRV[c, n, k] - angRV[c, n, k+1]
                            fraction = moveSizeInBin * moveSizeInBin / entireMoveSize
                            fracSum += fraction
                            valueSum += fraction * entireMoveSize / steps
                if fracSum > 0:
                    mmRV[c, b] = valueSum / fracSum
                else:
                    mmRV[c, b] = 0

            # fill the zeros closed to hard stops
            nz = np.nonzero(mmFW[c])[0]
            if nz.size > 0:
                mmFW[c, :nz[0]] = mmFW[c, nz[0]]
                mmFW[c, nz[-1]+1:] = mmFW[c, nz[-1]]
            else:
                bad[c] = True

            nz = np.nonzero(mmRV[c])[0]
            if nz.size > 0:
                mmRV[c, :nz[0]] = mmRV[c, nz[0]]
                mmRV[c, nz[-1]+1:] = mmRV[c, nz[-1]]
            else:
                bad[c] = True

        return mmFW, mmRV, bad

    def motorMaps2(self, angFW, angRV, steps, delta=0.1):
        """ calculate motor maps based on average step counts """
        mmFW = np.zeros((57, regions), dtype=float)
        mmRV = np.zeros((57, regions), dtype=float)
        bad = np.full(57, False)
        cnt = np.zeros(regions)
        repeat = angFW.shape[1]
        iteration = angFW.shape[2] - 1

        for c in self.goodIdx:
            cnt[:] = 0
            for n in range(repeat):
                nz = np.nonzero(np.all([angFW[c, n, 1:] > angFW[c, n, :-1], angRV[c, n, 0] - angFW[c, n, 1:] > delta], axis=0))[0]
                if nz.size <= 0:
                    continue
                k = nz[-1]
                x = np.arange(regions+1) * binSize
                xp = angFW[c, n, :k+2]
                fp = np.arange(k+2) * steps
                mm = np.interp(x, xp, fp)
                diff = mm[1:] - mm[:-1]
                nz = np.nonzero(diff)[0]
                if nz.size > 0:
                    mmFW[c] += diff
                    cnt[nz[:-1]] += 1
                    if angFW[c, n, k+1] % binSize != 0:
                        cnt[nz[-1]] += (angFW[c, n, k+1] % binSize) / binSize
                    else:
                        cnt[nz[-1]] += 1
            nz = np.nonzero(cnt)[0]
            if nz.size > 0:
                mmFW[c, nz] = binSize / (mmFW[c, nz] / cnt[nz])
                mmFW[c, nz[-1]+1:] = mmFW[c, nz[-1]]
            else:
                bad[c] = True

            cnt[:] = 0
            for n in range(repeat):
                # avoid sticky problem at hard stops
                if angRV[c, n, 1] > angRV[c, n, 0]:
                    first = 1
                else:
                    first = 0
                nz = np.nonzero(np.all([angRV[c, n, 1:] < angRV[c, n, :-1], angRV[c, n, 1:] > delta], axis=0))[0]
                if nz.size <= 0:
                    continue
                k = nz[-1]
                x = np.arange(regions+1) * binSize
                xp = np.flip(angRV[c, n, first:k+2], 0)
                fp = np.arange(k+2-first) * steps
                mm = np.interp(x, xp, fp)
                diff = mm[1:] - mm[:-1]
                nz = np.nonzero(diff)[0]
                if nz.size > 0:
                    mmRV[c] += diff
                    cnt[nz[1:-1]] += 1
                    cnt[nz[0]] += 1 - (angRV[c, n, k+1] % binSize) / binSize
                    if angRV[c, n, first] % binSize != 0:
                        cnt[nz[-1]] += (angRV[c, n, first] % binSize) / binSize
                    else:
                        cnt[nz[-1]] += 1
            nz = np.nonzero(cnt)[0]
            if nz.size > 0:
                mmRV[c, nz] = binSize / (mmRV[c, nz] / cnt[nz])
                mmRV[c, :nz[0]] = mmRV[c, nz[0]]
                mmRV[c, nz[-1]+1:] = mmRV[c, nz[-1]]
            else:
                bad[c] = True

        return mmFW, mmRV, bad

    def speed(self, angFW, angRV, steps, delta=0.1):
        # calculate average speed
        speedFW = np.zeros(57, dtype=float)
        speedRV = np.zeros(57, dtype=float)
        repeat = angFW.shape[1]
        iteration = angFW.shape[2] - 1

        for c in self.goodIdx:
            fSteps = 0
            fAngle = 0
            rSteps = 0
            rAngle = 0
            for n in range(repeat):
                nz = np.nonzero(angFW[c, n] < (angRV[c, n, 0] - delta))[0]
                if nz.size > 0:
                    k = nz[-1]
                    fAngle += angFW[c, n, k] - angFW[c, n, 0]
                    fSteps += k * steps

                # check if stuck at hard stops
                if angRV[c, n, 0] < angRV[c, n, 1]:
                    start = 1
                else:
                    start = 0

                nz = np.nonzero(angRV[c, n] > delta)[0]
                if nz.size > 0:
                    k = nz[-1]
                    rAngle += angRV[c, n, start] - angRV[c, n, k]
                    rSteps += (k - start) * steps

            if fSteps > 0:
                speedFW[c] = fAngle / fSteps
            if rSteps > 0:
                speedRV[c] = rAngle / rSteps

        return speedFW, speedRV

    def updateThetaMotorMaps(self, thetaMMFW, thetaMMRV, bad=None, slow=True):
        # update XML configuration
        if bad is None:
            bad = []
        idx = np.array([c for c in self.goodIdx if not bad[c]])
        new = self.calibModel

        if slow:
            mmFW = binSize / new.S1Pm
            mmRV = binSize / new.S1Nm
        else:
            mmFW = binSize / new.F1Pm
            mmRV = binSize / new.F1Nm
        mmFW[idx] = thetaMMFW[idx]
        mmRV[idx] = thetaMMRV[idx]

        new.updateMotorMaps(thtFwd=mmFW, thtRev=mmRV, useSlowMaps=slow)

    def updatePhiMotorMaps(self, phiMMFW, phiMMRV, bad=None, slow=True):
        # update XML configuration
        if bad is None:
            bad = []
        idx = np.array([c for c in self.goodIdx if not bad[c]])
        new = self.calibModel

        if slow:
            mmFW = binSize / new.S2Pm
            mmRV = binSize / new.S2Nm
        else:
            mmFW = binSize / new.F2Pm
            mmRV = binSize / new.F2Nm
        mmFW[idx] = phiMMFW[idx]
        mmRV[idx] = phiMMRV[idx]

        new.updateMotorMaps(phiFwd=mmFW, phiRev=mmRV, useSlowMaps=slow)

    def restoreConfig(self):
        # restore default setting
        self.calibModel = pfiDesign.PFIDesign(self.xml)

    def geometry(self, thetaC, thetaR, thetaFW, thetaRV, phiC, phiR, phiFW, phiRV):
        """ calculate geometry from theta and phi motor maps process """
        nCobra = phiC.shape[0]

        # calculate arm legnths
        thetaL = np.absolute(phiC - thetaC)
        phiL = phiR

        # calculate phi hard stops
        phiCCW = np.full(nCobra, np.pi)
        phiCW = np.zeros(nCobra)

        s = np.angle(thetaC - phiC)
        for n in range(phiFW.shape[1]):
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

        for n in range(thetaFW.shape[1]):
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

        return thetaL, phiL, thetaCCW, thetaCW, phiCCW, phiCW
