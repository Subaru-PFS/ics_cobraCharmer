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

class COBRA():

    def __init__(self, IPstring, XML, dataPath, badlist=None, cam_split=26):
        if badlist is not None:
            self.badIdx = np.array(badlist) - 1
            self.goodIdx = np.array([e for e in range(57) if e not in self.badIdx])
        else:
            self.badIdx = np.array([])
            self.goodIdx =  np.array(range(57))

        self.goodCobras = getCobras(self.goodIdx)
        self.badCobras = getCobras(self.badIdx)
        self.group1Idx = self.goodIdx[self.goodIdx <= cam_split]
        self.group2Idx = self.goodIdx[self.goodIdx > cam_split]
        self.cam_split = cam_split
        self.datapath = dataPath

         # Define the cobra range.
        mod1Cobras = pfiControl.PFI.allocateCobraRange(range(1,2))
        self.allCobras = mod1Cobras

        # partition module 1 cobras into odd and even sets
        moduleCobras2 = {}
        for group in 1,2:
            cm = range(group,58,2)
            mod = [1]*len(cm)
            moduleCobras2[group] = pfiControl.PFI.allocateCobraList(zip(mod,cm))
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
        if not (os.path.exists(dataPath)):
            os.makedirs(dataPath)
        self.datapath = dataPath

    def find_points(self, cIdx, fn):
        centers = self.pfi.calibModel.centers[cIdx]
        data = fits.getdata(fn)
        cs = sep.extract(data.astype(float), 50)
        spots = np.array([(c['x']+c['y']*(1j)) for c in cs])
        idx = lazyIdentification(centers, spots)
        return spots[idx]

    def find_circles(self, cIdx, fn):
        cnt = len(glob.glob(f'{fn}*')) - 1
        pos = np.zeros((len(cIdx), cnt), dtype=complex)
        circles = np.zeros((len(cIdx), 3))

        for i in range(cnt):
            pos[:,i] = self.find_points(cIdx, f'{fn}{i+1:04d}.fits')

        # find centers
        for i in range(len(cIdx)):
            x0, y0, r0 = circle_fitting(pos[i])
            circles[i] = x0, y0, r0
        return circles

    def moveToXYfromHome(self, idx, targets, threshold=3.0, maxTries=8):
        cobras = getCobras(idx)
        self.pfi.moveXYfromHome(cobras, targets)

        ntries = 1
        posArray = []
        while True:
            # check current positions, first exposing
            p1 = Popen(["/home/pfs/IDSControl/idsexposure", '-d', '1', '-e', '18', '-f', self.datapath+'/cam1_'], stdout=PIPE)
            p1.communicate()
            p2 = Popen(["/home/pfs/IDSControl/idsexposure", '-d', '2', '-e', '18', '-f', self.datapath+'/cam2_'], stdout=PIPE)
            p2.communicate()

            # extract sources and fiber identification
            curPos1 = self.find_points(idx[idx <= self.cam_split], self.datapath+'/cam1_0001.fits')
            curPos2 = self.find_points(idx[idx > self.cam_split], self.datapath+'/cam2_0001.fits')
            curPos = np.append(curPos1, curPos2)
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
            self.pfi.moveXY(cobras, curPos, targets)

        return posArray

    def moveCobra(self, c, theta, phi):
        self.pfi.moveSteps([self.allCobras[c-1]], np.zeros(1)+theta, np.zeros(1)+phi)

    def moveCobras(self, cs, theta, phi):
        cobs = []
        for c in cs:
            cobs.append(self.allCobras[c-1])
        self.pfi.moveSteps(cobs, np.array(theta), np.array(phi))

    def moveThetaPhi(self, cobras, thetaMoves, phiMoves, thetaFroms=None, phiFroms=None):
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
        for i, c in enumerate(cIdx):
            _phiMoves[c] = phiMoves[i]
            _thetaMoves[c] = thetaMoves[i]
            if phiFroms is not None:
                _phiFroms[c] = phiFroms[i]
            if thetaFroms is not None:
                _thetaFroms[c] = thetaFroms[i]

        thetaSteps, phiSteps = self.pfi.calculateSteps(_thetaFroms, _thetaMoves, _phiFroms, _phiMoves)
        thetaSteps[thetaSteps > 10000] = 0
        phiSteps[phiSteps > 10000] = 0

        cThetaSteps = thetaSteps[cIdx]
        cPhiSteps = phiSteps[cIdx]

        self.pfi.logger.info(f'steps: {list(zip(cThetaSteps, cPhiSteps))}')
        self.pfi.moveSteps(cobras, cThetaSteps, cPhiSteps)

    def measureAngles(self, idx, centers, homes):
        p1 = Popen(["/home/pfs/IDSControl/idsexposure", '-d', '1', '-e', '18', '-f', self.datapath+'/cam1_'], stdout=PIPE)
        p1.communicate()
        p2 = Popen(["/home/pfs/IDSControl/idsexposure", '-d', '2', '-e', '18', '-f', self.datapath+'/cam2_'], stdout=PIPE)
        p2.communicate()
        time.sleep(1.0)

        # extract sources and fiber identification
        curPos1 = self.find_points(idx[idx <= self.cam_split], self.datapath+'/cam1_0001.fits')
        curPos2 = self.find_points(idx[idx > self.cam_split], self.datapath+'/cam2_0001.fits')
        curPos = np.append(curPos1, curPos2)

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

    def findPhiCenters(self, showPlot=False):
        # find phi centers for all good cobras
        myIdx = self.goodIdx
        myCobras = self.goodCobras

        # take one image at limit
        self.pfi.moveAllSteps(myCobras, 0, -5000)
        p1 = Popen(['/home/pfs/IDSControl/idsexposure', '-d', '1', '-e', '18', '-f', self.datapath+'/cam1P1_'], stdout=PIPE)
        p1.communicate()
        p2 = Popen(['/home/pfs/IDSControl/idsexposure', '-d', '2', '-e', '18', '-f', self.datapath+'/cam2P1_'], stdout=PIPE)
        p2.communicate()
        time.sleep(2.0)

        # move phi out and capture the video
        p1 = Popen(['/home/pfs/IDSControl/idsexposure', '-d', '1', '-e', '18', '-i', '100', '-l', '9999', '-f', self.datapath+'/cam1phi_'], stdout=PIPE)
        p2 = Popen(['/home/pfs/IDSControl/idsexposure', '-d', '2', '-e', '18', '-i', '100', '-l', '9999', '-f', self.datapath+'/cam2phi_'], stdout=PIPE)
        time.sleep(5.0)
        self.pfi.moveAllSteps(myCobras, 0, 5000)
        time.sleep(0.5)
        p1.kill()
        p2.kill()
        p1.communicate()
        p2.communicate()
        self.pfi.moveAllSteps(myCobras, 0, 5000)

        # take one image at limit
        p1 = Popen(['/home/pfs/IDSControl/idsexposure', '-d', '1', '-e', '18', '-f', self.datapath+'/cam1P2_'], stdout=PIPE)
        p1.communicate()
        p2 = Popen(['/home/pfs/IDSControl/idsexposure', '-d', '2', '-e', '18', '-f', self.datapath+'/cam2P2_'], stdout=PIPE)
        p2.communicate()
        time.sleep(2.0)
        self.pfi.moveAllSteps(myCobras, 0, -10000)

        # variable declaration
        phiCircles = np.zeros((57, 3), dtype=float)
        points = np.zeros(57, dtype=complex)

        # first camera
        myIdx = self.group1Idx
        phiCircles[myIdx] = self.find_circles(myIdx, self.datapath+'/cam1phi_')
        points[myIdx] = self.find_points(myIdx, self.datapath+'/cam1P1_0001.fits')

        # second camera
        myIdx = self.group2Idx
        phiCircles[myIdx] = self.find_circles(myIdx, self.datapath+'/cam2phi_')
        points[myIdx] = self.find_points(myIdx, self.datapath+'/cam2P1_0001.fits')

        # phi centers, radius and homes
        phiCenters = phiCircles[:,0] + phiCircles[:,1]*(1j)
        phiRadius = phiCircles[:,2]
        phiHomes = np.angle(points - phiCenters)

        if showPlot:
            group1 = self.group1Idx
            group2 = self.group2Idx

            plt.figure(1)
            plt.clf()

            plt.subplot(211)
            ax = plt.gca()
            ax.plot(phiCircles[group1,0], phiCircles[group1,1], 'mo')
            for idx in group1:
                c = plt.Circle((phiCircles[idx,0], phiCircles[idx,1]), phiCircles[idx,2], color='b', fill=False)
                ax.add_artist(c)
            ax.set_title(f'1st camera: phi')

            plt.subplot(212)
            ax = plt.gca()
            ax.plot(phiCircles[group2,0], phiCircles[group2,1], 'mo')
            for idx in group2:
                c = plt.Circle((phiCircles[idx,0], phiCircles[idx,1]), phiCircles[idx,2], color='b', fill=False)
                ax.add_artist(c)
            ax.set_title(f'2nd camera: phi')

            plt.show()

        return phiCenters, phiRadius, phiHomes

    def phiTest(self, phiCenters, phiHomes, margin=15.0, runs=50, tries=8, phi60=False):
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
            self.moveThetaPhi(goodCobras, zeros, zeros + angle)
            time.sleep(1.0)
            cAngles, cPositions = self.measureAngles(goodIdx, centers, homes)
            phiData[goodIdx, i, 0, 0] = cAngles
            phiData[goodIdx, i, 0, 1] = np.real(cPositions)
            phiData[goodIdx, i, 0, 2] = np.imag(cPositions)

            for j in range(tries - 1):
                self.moveThetaPhi(goodCobras, zeros, angle - cAngles, phiFroms=cAngles)
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
            self.moveThetaPhi(goodCobras, zeros, zeros + angle)
            time.sleep(1.0)
            cAngles, cPositions = self.measureAngles(goodIdx, centers, homes)
            for j in range(tries - 1):
                self.moveThetaPhi(goodCobras, zeros, angle - cAngles, phiFroms=cAngles)
                time.sleep(1.0)
                cAngles, cPositions = self.measureAngles(goodIdx, centers, homes)
                for k in range(len(goodIdx)):
                    if cAngles[k] > np.pi*(3/2):
                        cAngles[k] = 0

        return phiData

    def findThetaCenters(self, showPlot=False):
        # find phi centers for all good cobras
        myIdx = self.goodIdx
        myCobras = self.goodCobras

        # take one image at limit
        self.pfi.moveAllSteps(myCobras, -10000, 0)
        p1 = Popen(["/home/pfs/IDSControl/idsexposure", '-d', '1', '-e', '18', '-f', self.datapath+'/cam1P3_'], stdout=PIPE)
        p1.communicate()
        p2 = Popen(["/home/pfs/IDSControl/idsexposure", '-d', '2', '-e', '18', '-f', self.datapath+'/cam2P3_'], stdout=PIPE)
        p2.communicate()
        time.sleep(1.0)

        # move phi out and capture the video
        p1 = Popen(["/home/pfs/IDSControl/idsexposure", '-d', '1', '-e', '18', '-i', '100', '-l', '9999', '-f', self.datapath+'/cam1theta_'], stdout=PIPE)
        p2 = Popen(["/home/pfs/IDSControl/idsexposure", '-d', '2', '-e', '18', '-i', '100', '-l', '9999', '-f', self.datapath+'/cam2theta_'], stdout=PIPE)
        time.sleep(5.0)
        self.pfi.moveAllSteps(myCobras, 10000, 0)
        time.sleep(0.5)
        p1.kill()
        p2.kill()
        p1.communicate()
        p2.communicate()
        self.pfi.moveAllSteps(myCobras, 10000, 0)

        # take one image at limit
        p1 = Popen(["/home/pfs/IDSControl/idsexposure", '-d', '1', '-e', '18', '-f', self.datapath+'/cam1P4_'], stdout=PIPE)
        p1.communicate()
        p2 = Popen(["/home/pfs/IDSControl/idsexposure", '-d', '2', '-e', '18', '-f', self.datapath+'/cam2P4_'], stdout=PIPE)
        p2.communicate()
        time.sleep(1.0)
        self.pfi.moveAllSteps(myCobras, -10000, 0)

        # variable declaration
        thetaCircles = np.zeros((57, 3), dtype=float)
        points = np.zeros(57, dtype=complex)

        # first camera
        myIdx = self.group1Idx
        thetaCircles[myIdx] = self.find_circles(myIdx, self.datapath+'/cam1theta_')
        points[myIdx] = self.find_points(myIdx, self.datapath+'/cam1P3_0001.fits')

        # second camera
        myIdx = self.group2Idx
        thetaCircles[myIdx] = self.find_circles(myIdx, self.datapath+'/cam2theta_')
        points[myIdx] = self.find_points(myIdx, self.datapath+'/cam2P3_0001.fits')

        # theta centers, radius and homes
        thetaCenters = thetaCircles[:,0] + thetaCircles[:,1]*(1j)
        thetaRadius = thetaCircles[:,2]
        thetaHomes = np.angle(points - thetaCenters)

        if showPlot:
            group1 = self.group1Idx
            group2 = self.group2Idx

            plt.figure(2)
            plt.clf()

            plt.subplot(211)
            ax = plt.gca()
            ax.plot(thetaCircles[group1,0], thetaCircles[group1,1], 'mo')
            for idx in group1:
                c = plt.Circle((thetaCircles[idx,0], thetaCircles[idx,1]), thetaCircles[idx,2], color='b', fill=False)
                ax.add_artist(c)
            ax.set_title(f'1st camera: theta')

            plt.subplot(212)
            ax = plt.gca()
            ax.plot(thetaCircles[group2,0], thetaCircles[group2,1], 'mo')
            for idx in group2:
                c = plt.Circle((thetaCircles[idx,0], thetaCircles[idx,1]), thetaCircles[idx,2], color='b', fill=False)
                ax.add_artist(c)
            ax.set_title(f'2nd camera: theta')

            plt.show()

        return thetaCenters, thetaRadius, thetaHomes

    def thetaTest(self, thetaCenters, thetaHomes, margin=30.0, runs=50, tries=8):
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
            self.moveThetaPhi(goodCobras, zeros + angle, zeros)
            time.sleep(1.0)
            cAngles, cPositions = self.measureAngles(goodIdx, centers, homes)
            thetaData[goodIdx, i, 0, 0] = cAngles
            thetaData[goodIdx, i, 0, 1] = np.real(cPositions)
            thetaData[goodIdx, i, 0, 2] = np.imag(cPositions)

            for j in range(tries - 1):
                time.sleep(1.0)
                self.moveThetaPhi(goodCobras, angle - cAngles, zeros, thetaFroms=cAngles)
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

    cobraCharmerPath = '/home/pfs/mhs/devel/ics_cobraCharmer/'
    xml = cobraCharmerPath + 'xml/motormap_20190312.xml'

    datetoday=datetime.datetime.now().strftime("%Y%m%d")
    dataPath = '/data/pfs/Converge_' + datetoday

    IP = '128.149.77.24'

    brokens = [1, 39, 43, 54]
    cobra = COBRA(IP, xml, dataPath, badlist=brokens, cam_split=26)

    cobra.moveThetaOut(60, maxTries=10)
    cobra.pfi.moveAllSteps(cobra.goodCobras, 0, -5000)
    
    # you may want to do some inspection here
    cobra.moveCobra(3, -600, 0)

    # phiCenters, phiRadius, phiHomes = cobra.findPhiCenters()
    # phiData = cobra.phiTest(phiCenters, phiHomes, runs=50, tries=8, phi60=True)

    # #cobra.moveThetaOut(60)
    # thetaCenters, thetaRadius, thetaHomes = cobra.findThetaCenters()
    # thetaData = cobra.thetaTest(thetaCenters, thetaHomes, runs=50, tries=8, margin=30.0)

    # np.save(dataPath + '/phiData', phiData)
    # np.save(dataPath + '/thetaData', thetaData)


if __name__ == '__main__':
    main()
