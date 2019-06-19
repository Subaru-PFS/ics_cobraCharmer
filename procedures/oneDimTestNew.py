import sys, os
from importlib import reload
import numpy as np
import time
import datetime
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import glob
from copy import deepcopy
from ics.cobraCharmer import pfi as pfiControl
import idsCamera

class COBRA():
    def __init__(self, IPstring, XML, dataPath, badlist=None, cam_split=26):
        if badlist is not None:
            self.badIdx = np.array(badlist) - 1
            self.goodIdx = np.array([e for e in range(57) if e not in self.badIdx])
        else:
            self.badIdx = np.array([])
            self.goodIdx = np.array(range(57))

        self.goodCobras = getCobras(self.goodIdx)
        self.badCobras = getCobras(self.badIdx)
        self.group1Idx = self.goodIdx[self.goodIdx <= cam_split]
        self.group2Idx = self.goodIdx[self.goodIdx > cam_split]
        self.cam_split = cam_split
        self.datapath = dataPath

         # Define the cobra range.
        mod1Cobras = pfiControl.PFI.allocateCobraRange(range(1, 2))
        self.allCobras = mod1Cobras

        # partition module 1 cobras into odd and even sets
        moduleCobras2 = {}
        for group in 1, 2:
            cm = range(group, 58, 2)
            mod = [1]*len(cm)
            moduleCobras2[group] = pfiControl.PFI.allocateCobraList(zip(mod, cm))
        self.oddCobras = moduleCobras2[1]
        self.evenCobras = moduleCobras2[2]

        '''
            Make connection to module
        '''
        # Initializing COBRA module
        self.pfi = pfiControl.PFI(fpgaHost=IPstring, doLoadModel=False) #'fpga' for real device.

        if not os.path.exists(XML):
            print(f"Error: {XML} not presented!")
            sys.exit()
        self.pfi.loadModel(XML)
        self.pfi.setFreq(self.allCobras)

        # Prepare the data path for the work
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
        self.datapath = dataPath

        # init cameras
        self.cam1 = idsCamera.idsCamera(1)
        self.cam2 = idsCamera.idsCamera(2)
        self.cam1.setExpoureTime(20)
        self.cam2.setExpoureTime(20)

    def find_points(self, cIdx, fn):
        centers = self.pfi.calibModel.centers[cIdx]
        if type(fn) is str:
            data = fits.getdata(fn)
        else:
            data = fn
        cs = sep.extract(data.astype(float), 50)
        spots = np.array([(c['x']+c['y']*(1j)) for c in cs])
        idx = lazyIdentification(centers, spots)
        return spots[idx]

    def expose_and_measure(self, idx):
        # expose
        data1 = self.cam1.getCurrentFrame()
        data2 = self.cam2.getCurrentFrame()

        # extract sources and fiber identification
        curPos1 = self.find_points(idx[idx <= self.cam_split], data1)
        curPos2 = self.find_points(idx[idx > self.cam_split], data2)
        return np.append(curPos1, curPos2)

    def find_circles(self, pos):
        circles = np.zeros((pos.shape[0], 3))

        # find centers
        for i in range(pos.shape[0]):
            x0, y0, r0 = circle_fitting(pos[i])
            circles[i] = x0, y0, r0
        return circles

    def moveToXYfromHome(self, idx, targets, threshold=3.0, maxTries=8, thetaThreshold=0.1, phiThreshold=0.1):
        cobras = getCobras(idx)
        self.pfi.moveXYfromHome(cobras, targets)

        ntries = 1
        posArray = []
        while True:
            # check current positions
            curPos = self.expose_and_measure(idx)
            posArray.append(curPos)

            # check position errors
            done = np.abs(curPos - targets) <= threshold
            if np.all(done):
                print('Convergence sequence done')
                break
            if ntries > maxTries:
                print(f'Reach max {maxTries} tries, gave up')
                break
            ntries += 1

            # move again
            self.pfi.moveXY(cobras, curPos, targets, thetaThreshold, phiThreshold)

        return posArray

    def moveCobra(self, c, theta, phi):
        self.pfi.moveSteps([self.allCobras[c-1]], np.zeros(1)+theta, np.zeros(1)+phi)

    def moveCobras(self, cs, theta, phi):
        cobs = []
        for c in cs:
            cobs.append(self.allCobras[c-1])
        self.pfi.moveSteps(cobs, np.array(theta), np.array(phi))

    def moveThetaPhi(self, cobras, thetaMoves, phiMoves, thetaFroms=None, phiFroms=None, thetaFast=True, phiFast=True):
        if len(cobras) != len(thetaMoves):
            raise RuntimeError("number of theta moves must match number of cobras")
        if len(cobras) != len(phiMoves):
            raise RuntimeError("number of phi moves must match number of cobras")
        if thetaFroms is not None and len(cobras) != len(thetaFroms):
            raise RuntimeError("number of theta froms must match number of cobras")
        if phiFroms is not None and len(cobras) != len(phiFroms):
            raise RuntimeError("number of phi froms must match number of cobras")
        nCobras = self.pfi.calibModel.nCobras

        _phiMoves = np.zeros(nCobras)
        _thetaMoves = np.zeros(nCobras)
        _phiFroms = np.zeros(nCobras)
        _thetaFroms = np.zeros(nCobras)

        cIdx = [self.pfi._mapCobraIndex(c) for c in cobras]
        _phiMoves[cIdx] = phiMoves
        _thetaMoves[cIdx] = thetaMoves
        if phiFroms is not None:
            _phiFroms[cIdx] = phiFroms
        if thetaFroms is not None:
            _thetaFroms[cIdx] = thetaFroms

        if type(thetaFast) is bool:
            _thetaFast = thetaFast
        elif len(thetaFast) == len(cobras):
            _thetaFast = np.full(nCobras, True)
            _thetaFast[cIdx] = thetaFast
        else:
            raise RuntimeError("number of thetaFast must match number of cobras")

        if type(phiFast) is bool:
            _phiFast = phiFast
        elif len(phiFast) == len(cobras):
            _phiFast = np.full(nCobras, True)
            _phiFast[cIdx] = phiFast
        else:
            raise RuntimeError("number of phiFast must match number of cobras")

        thetaSteps, phiSteps = self.pfi.calculateSteps(_thetaFroms, _thetaMoves, _phiFroms, _phiMoves, _thetaFast, _phiFast)

        for idx in range(nCobras):
            if thetaSteps[idx] > 10000 or thetaSteps[idx] < -10000:
                print(f'wrong theta steps: {idx}, {_thetaFroms[idx]}, {_thetaMoves[idx]}, {thetaSteps[idx]}')
                thetaSteps[idx] = 0
            if phiSteps[idx] > 10000 or phiSteps[idx] < -10000:
                print(f'wrong phi steps: {idx}, {_phiFroms[idx]}, {_phiMoves[idx]}, {phiSteps[idx]}')
                phiSteps[idx] = 0

        cThetaSteps = thetaSteps[cIdx]
        cPhiSteps = phiSteps[cIdx]

        self.pfi.logger.info(f'steps: {list(zip(cThetaSteps, cPhiSteps))}')
        self.pfi.moveSteps(cobras, cThetaSteps, cPhiSteps)

    def measureAngles(self, idx, centers, homes):
        curPos = self.expose_and_measure(idx)

        # calculate angles
        return (np.angle(curPos - centers) - homes) % (2 * np.pi), curPos

    def moveThetaOut(self, phiAngle, maxTries=16):
        thetas = np.empty(57, dtype=float)
        thetas[::2] = self.pfi.thetaToLocal(self.oddCobras, np.full(len(self.oddCobras), np.deg2rad(270)))
        thetas[1::2] = self.pfi.thetaToLocal(self.evenCobras, np.full(len(self.evenCobras), np.deg2rad(90)))
        phis = np.full(57, np.deg2rad(phiAngle))
        outTargets = self.pfi.anglesToPositions(self.allCobras, thetas, phis)

        self.pfi.moveAllSteps(self.goodCobras, -10000, -5000)
        self.moveToXYfromHome(self.goodIdx, outTargets[self.goodIdx], maxTries=maxTries)

    def moveBadThetaOut(self, phiAngle):
        thetas = np.empty(57, dtype=float)
        thetas[::2] = self.pfi.thetaToLocal(self.oddCobras, np.full(len(self.oddCobras), np.deg2rad(270)))
        thetas[1::2] = self.pfi.thetaToLocal(self.evenCobras, np.full(len(self.evenCobras), np.deg2rad(90)))
        phis = np.zeros(57)

        self.pfi.moveAllSteps(self.badCobras, -10000, -5000)
        self.pfi.moveSteps(self.badCobras, thetas[self.badIdx], phis[self.badIdx])

    def findPhiCenters(self, showPlot=False, iterations=20, steps=250):
        # find phi centers for all good cobras
        myIdx = self.goodIdx
        myCobras = self.goodCobras
        phiCircles = np.zeros((57, 3), dtype=float)
        points = np.zeros(57, dtype=complex)
        circles = np.zeros((57, iterations), dtype=complex)

        # take one image at limit
        self.pfi.moveAllSteps(myCobras, 0, -5000)
        points[myIdx] = self.expose_and_measure(myIdx)

        # move phi out and record the positions
        for i in range(iterations):
            self.pfi.moveAllSteps(myCobras, 0, steps)
            circles[myIdx, i] = self.expose_and_measure(myIdx)
        phiCircles[myIdx] = self.find_circles(circles[myIdx])
        self.pfi.moveAllSteps(myCobras, 0, -5000)

        # phi centers, radius and homes
        phiCenters = phiCircles[:, 0] + phiCircles[:, 1]*(1j)
        phiRadius = phiCircles[:, 2]
        phiHomes = np.angle(points - phiCenters)

        if showPlot:
            group1 = self.group1Idx
            group2 = self.group2Idx

            plt.figure(1)
            plt.clf()

            plt.subplot(211)
            ax = plt.gca()
            ax.plot(phiCircles[group1, 0], phiCircles[group1, 1], 'mo')
            for idx in group1:
                c = plt.Circle((phiCircles[idx, 0], phiCircles[idx, 1]), phiCircles[idx, 2], color='b', fill=False)
                ax.add_artist(c)
            ax.set_title(f'1st camera: phi')

            plt.subplot(212)
            ax = plt.gca()
            ax.plot(phiCircles[group2, 0], phiCircles[group2, 1], 'mo')
            for idx in group2:
                c = plt.Circle((phiCircles[idx, 0], phiCircles[idx, 1]), phiCircles[idx, 2], color='b', fill=False)
                ax.add_artist(c)
            ax.set_title(f'2nd camera: phi')

            plt.show()

        return phiCenters, phiRadius, phiHomes

    def phiTest(self, phiCenters, phiHomes, margin=15.0, runs=50, tries=8, phi60=False, fast=True):
        phiData = np.zeros((57, runs, tries, 3))
        goodIdx = self.goodIdx
        goodCobras = self.goodCobras
        zeros = np.zeros(len(goodCobras))
        centers = phiCenters[goodIdx]
        homes = phiHomes[goodIdx]

        self.pfi.moveAllSteps(goodCobras, 0, -5000)
        for i in range(runs):
            if runs > 1:
                angle = np.deg2rad(margin + (180 - 2 * margin) * i / (runs - 1))
            else:
                angle = np.deg2rad(90)
            self.moveThetaPhi(goodCobras, zeros, zeros + angle, phiFast=fast)
            time.sleep(1.0)
            cAngles, cPositions = self.measureAngles(goodIdx, centers, homes)
            phiData[goodIdx, i, 0, 0] = cAngles
            phiData[goodIdx, i, 0, 1] = np.real(cPositions)
            phiData[goodIdx, i, 0, 2] = np.imag(cPositions)

            for j in range(tries - 1):
                self.moveThetaPhi(goodCobras, zeros, angle - cAngles, phiFroms=cAngles, phiFast=fast)
                time.sleep(1.0)
                cAngles, cPositions = self.measureAngles(goodIdx, centers, homes)
                for k in range(len(goodIdx)):
                    if cAngles[k] > np.pi*(3/2):
                        cAngles[k] = 0
                phiData[goodIdx, i, j+1, 0] = cAngles
                phiData[goodIdx, i, j+1, 1] = np.real(cPositions)
                phiData[goodIdx, i, j+1, 2] = np.imag(cPositions)

            # home phi
            self.pfi.moveAllSteps(goodCobras, 0, -5000)

        if phi60:
            # move phi to 60 degress for theta test
            angle = np.deg2rad(60)
            self.moveThetaPhi(goodCobras, zeros, zeros + angle, phiFast=fast)
            time.sleep(1.0)
            cAngles, cPositions = self.measureAngles(goodIdx, centers, homes)
            for j in range(tries - 1):
                self.moveThetaPhi(goodCobras, zeros, angle - cAngles, phiFroms=cAngles, phiFast=fast)
                time.sleep(1.0)
                cAngles, cPositions = self.measureAngles(goodIdx, centers, homes)
                for k in range(len(goodIdx)):
                    if cAngles[k] > np.pi*(3/2):
                        cAngles[k] = 0

        return phiData

    def findThetaCenters(self, showPlot=False, iterations=20, steps=400):
        # find theta centers for all good cobras
        myIdx = self.goodIdx
        myCobras = self.goodCobras
        thetaCircles = np.zeros((57, 3), dtype=float)
        points = np.zeros(57, dtype=complex)
        circles = np.zeros((57, iterations), dtype=complex)

        # take one image at limit
        self.pfi.moveAllSteps(myCobras, -10000, 0)
        points[myIdx] = self.expose_and_measure(myIdx)

        # move theta out and record the positions
        for i in range(iterations):
            self.pfi.moveAllSteps(myCobras, steps, 0)
            circles[myIdx, i] = self.expose_and_measure(myIdx)
        thetaCircles[myIdx] = self.find_circles(circles[myIdx])
        self.pfi.moveAllSteps(myCobras, -10000, 0)

        # theta centers, radius and homes
        thetaCenters = thetaCircles[:, 0] + thetaCircles[:, 1]*(1j)
        thetaRadius = thetaCircles[:, 2]
        thetaHomes = np.angle(points - thetaCenters)

        if showPlot:
            group1 = self.group1Idx
            group2 = self.group2Idx

            plt.figure(2)
            plt.clf()

            plt.subplot(211)
            ax = plt.gca()
            ax.plot(thetaCircles[group1, 0], thetaCircles[group1, 1], 'mo')
            for idx in group1:
                c = plt.Circle((thetaCircles[idx, 0], thetaCircles[idx, 1]), thetaCircles[idx, 2], color='b', fill=False)
                ax.add_artist(c)
            ax.set_title(f'1st camera: theta')

            plt.subplot(212)
            ax = plt.gca()
            ax.plot(thetaCircles[group2, 0], thetaCircles[group2, 1], 'mo')
            for idx in group2:
                c = plt.Circle((thetaCircles[idx, 0], thetaCircles[idx, 1]), thetaCircles[idx, 2], color='b', fill=False)
                ax.add_artist(c)
            ax.set_title(f'2nd camera: theta')

            plt.show()

        return thetaCenters, thetaRadius, thetaHomes

    def thetaTest(self, thetaCenters, thetaHomes, margin=15.0, runs=50, tries=8, fast=True):
        thetaData = np.zeros((57, runs, tries, 3))
        goodIdx = self.goodIdx
        goodCobras = self.goodCobras
        zeros = np.zeros(len(goodCobras))
        centers = thetaCenters[goodIdx]
        homes = thetaHomes[goodIdx]

        self.pfi.moveAllSteps(goodCobras, -10000, 0)
        for i in range(runs):
            if runs > 1:
                angle = np.deg2rad(margin + (360 - 2 * margin) * i / (runs - 1))
            else:
                angle = np.deg2rad(180)
            self.moveThetaPhi(goodCobras, zeros + angle, zeros, thetaFast=fast)
            time.sleep(1.0)
            cAngles, cPositions = self.measureAngles(goodIdx, centers, homes)
            thetaData[goodIdx, i, 0, 0] = cAngles
            thetaData[goodIdx, i, 0, 1] = np.real(cPositions)
            thetaData[goodIdx, i, 0, 2] = np.imag(cPositions)

            for j in range(tries - 1):
                time.sleep(1.0)
                self.moveThetaPhi(goodCobras, angle - cAngles, zeros, thetaFroms=cAngles, thetaFast=fast)
                time.sleep(1.0)
                cAngles, cPositions = self.measureAngles(goodIdx, centers, homes)
                for k in range(len(goodIdx)):
                    if cAngles[k] < np.pi*(1/4) and angle > np.pi*(3/2):
                        cAngles[k] += np.pi*2
                    elif cAngles[k] > np.pi*(15/8) and angle < np.pi*(1/2):
                        cAngles[k] = 0
                thetaData[goodIdx, i, j+1, 0] = cAngles
                thetaData[goodIdx, i, j+1, 1] = np.real(cPositions)
                thetaData[goodIdx, i, j+1, 2] = np.imag(cPositions)

            # home theta
            self.pfi.moveAllSteps(goodCobras, -10000, 0)

        return thetaData


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


def getCobras(cobs):
    # cobs is 0-indexed list
    return pfiControl.PFI.allocateCobraList(zip(np.full(len(cobs), 1), np.array(cobs) + 1))


def circle_fitting(p):
    x = np.real(p)
    y = np.imag(p)
    m = np.vstack([x, y, np.ones(len(p))]).T
    n = np.array(x*x + y*y)
    a, b, c = np.linalg.lstsq(m, n, rcond=None)[0]
    return a/2, b/2, np.sqrt(c+(a*a+b*b)/4)


def main():
    # You should manually align all the bad/broken cobras
    # to the outward direction before running this script.

    cobraCharmerPath = '/home/pfs/mhs/devel/ics_cobraCharmer/'
    xml = cobraCharmerPath + 'xml/motormap_20190312.xml'

    datetoday = datetime.datetime.now().strftime("%Y%m%d")
    dataPath = '/data/pfs/Converge_' + datetoday

    IP = '128.149.77.24'

    brokens = [1, 39, 43, 54]
    cobra = COBRA(IP, xml, dataPath, badlist=brokens, cam_split=26)

    cobra.moveThetaOut(60, maxTries=10)
    cobra.pfi.moveAllSteps(cobra.goodCobras, 0, -5000)
    # you may want to do some inspection here
    #cobra.moveCobra(23, -200, 0)

    phiCenters, phiRadius, phiHomes = cobra.findPhiCenters()
    phiData = cobra.phiTest(phiCenters, phiHomes, runs=5, tries=8, margin=15.0, phi60=True, fast=True)
    np.save(dataPath + '/phiData', phiData)

    #cobra.moveThetaOut(60)
    thetaCenters, thetaRadius, thetaHomes = cobra.findThetaCenters()
    thetaData = cobra.thetaTest(thetaCenters, thetaHomes, runs=5, tries=8, margin=15.0, fast=True)
    np.save(dataPath + '/thetaData', thetaData)


if __name__ == '__main__':
    main()
