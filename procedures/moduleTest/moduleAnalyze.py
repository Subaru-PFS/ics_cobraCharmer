import os
import sys
import numpy as np
from astropy.io import fits
import sep
from copy import deepcopy
import calculation
import pandas as pd

class moduleAnalyze():
    def __init__(self, xml, brokens=None, camSplit=26):
        if not os.path.exists(xml):
            print(f"Error: {xml} is not presented!")
            sys.exit()
        self.cal = calculation.Calculation(xml, brokens, camSplit)

    def loadPhiData(self, dataPath, goodIdx, iterations, repeats=1, reCenter=False):
        if not reCenter:
            phiFW = np.load(dataPath + '/phiFW')
            phiRV = np.load(dataPath + '/phiRV')
            nCobras, nRepeats, nIterations = phiFW.shape
            if nCobras != len(goodIdx) or nRepeats != repeats or nIterations != iterations:
                raise RuntimeError("saved data shape is not expected shape")

            return phiFW, phiRV

        # calculate the phi movements
        phiFW = np.zeros((57, repeats, iterations+1), dtype=complex)
        phiRV = np.zeros((57, repeats, iterations+1), dtype=complex)
        for n in range(repeats):
            # forward phi motor maps
            data1 = fits.getdata(dataPath + f'/phi1Begin{n}.fits.gz')
            data2 = fits.getdata(dataPath + f'/phi2Begin{n}.fits.gz')
            phiFW[goodIdx, n, 0] = self.cal.extractPositions(data1, data2)

            for k in range(iterations):
                data1 = fits.getdata(dataPath + f'/phi1Forward{n}N{k}.fits.gz')
                data2 = fits.getdata(dataPath + f'/phi2Forward{n}N{k}.fits.gz')
                phiFW[goodIdx, n, k+1] = self.cal.extractPositions(data1, data2,
                                                                   guess=phiFW[goodIdx, n, k])

            # reverse phi motor maps
            data1 = fits.getdata(dataPath + f'/phi1End{n}.fits.gz')
            data2 = fits.getdata(dataPath + f'/phi2End{n}.fits.gz')
            phiRV[goodIdx, n, 0] = self.cal.extractPositions(data1, data2,
                                                             guess=phiFW[goodIdx, n, iterations])

            for k in range(iterations):
                data1 = fits.getdata(dataPath + f'/phi1Reverse{n}N{k}.fits.gz')
                data2 = fits.getdata(dataPath + f'/phi2Reverse{n}N{k}.fits.gz')
                phiRV[goodIdx, n, k+1] = self.cal.extractPositions(data1, data2,
                                                                   guess=phiRV[goodIdx, n, k])

        return phiFW, phiRV

    def makePhiMotorMap(
            self,
            newXml,
            dataPath,
            repeat=3,
            steps=200,
            totalSteps=5000,
            fast=True,
            phiOnTime=None,
            delta=0.1,
            reCenter=False):

        # generate phi motor maps, it accepts custom phiOnTIme parameter.
        # it assumes that theta arms have been move to up/down positions to avoid collision
        # if phiOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime
        #
        # Example:
        #     makePhiMotorMap(xml, path, fast=True)             // update fast motor maps
        #     makePhiMotorMap(xml, path, fast=False)            // update slow motor maps
        #     makePhiMotorMap(xml, path, phiOnTime=0.06)        // motor maps for on-time=0.06

        # variable declaration for position measurement
        iterations = totalSteps // steps
        goodIdx = self.cal.goodIdx

        phiFW, phiRV = self.loadPhiData(dataPath, repeat, goodIdx, iterations, recenter=reCenter)

        # save calculation result
        np.save(dataPath + '/phiFW_A', phiFW)
        np.save(dataPath + '/phiRV_A', phiRV)

        # calculate centers and phi angles
        phiCenter, phiRadius, phiAngFW, phiAngRV, badRange = self.cal.phiCenterAngles(phiFW, phiRV)
        np.save(dataPath + '/phiCenter_A', phiCenter)
        np.save(dataPath + '/phiRadius_A', phiRadius)
        np.save(dataPath + '/phiAngFW_A', phiAngFW)
        np.save(dataPath + '/phiAngRV_A', phiAngRV)
        np.save(dataPath + '/badRange_A', badRange)

        # calculate average speeds
        phiSpeedFW, phiSpeedRV = self.cal.speed(phiAngFW, phiAngRV, steps, delta)
        np.save(dataPath + '/phiSpeedFW_A', phiSpeedFW)
        np.save(dataPath + '/phiSpeedRV_A', phiSpeedRV)

        # calculate motor maps in Johannes weighting
        phiMMFW, phiMMRV, bad = self.cal.motorMaps(phiAngFW, phiAngRV, steps, delta)
        bad[badRange] = True
        np.save(dataPath + '/phiMMFW_A', phiMMFW)
        np.save(dataPath + '/phiMMRV_A', phiMMRV)
        np.save(dataPath + '/bad_A', np.where(bad)[0])

        # calculate motor maps by average speeds
        phiMMFW2, phiMMRV2, bad2 = self.cal.motorMaps2(phiAngFW, phiAngRV, steps, delta)
        bad2[badRange] = True
        np.save(dataPath + '/phiMMFW2_A', phiMMFW2)
        np.save(dataPath + '/phiMMRV2_A', phiMMRV2)
        np.save(dataPath + '/bad2_A', np.where(bad2)[0])

        # update XML file, using Johannes weighting
        slow = not fast
        self.cal.updatePhiMotorMaps(phiMMFW, phiMMRV, bad, slow)
        if phiOnTime is not None:
            onTime = np.full(57, phiOnTime)
            self.cal.calibModel.updateOntimes(phiFwd=onTime, phiRev=onTime, fast=fast)
        self.cal.calibModel.createCalibrationFile(dataPath + '/' + newXml)
        self.cal.restoreConfig()

    def makeThetaMotorMap(
            self,
            newXml,
            dataPath,
            repeat=3,
            steps=200,
            totalSteps=10000,
            fast=True,
            thetaOnTime=None,
            delta=0.1
        ):
        # generate theta motor maps, it accepts custom thetaOnTIme parameter.
        # it assumes that phi arms have been move to ~60 degrees out to avoid collision
        # if thetaOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime
        # Example:
        #     makethetaMotorMap(xml, path, fast=True)               // update fast motor maps
        #     makethetaMotorMap(xml, path, fast=False)              // update slow motor maps
        #     makethetaMotorMap(xml, path, thetaOnTime=0.06)        // motor maps for on-time=0.06

        # variable declaration for position measurement
        iteration = totalSteps // steps
        thetaFW = np.zeros((57, repeat, iteration+1), dtype=complex)
        thetaRV = np.zeros((57, repeat, iteration+1), dtype=complex)
        goodIdx = self.cal.goodIdx

        #record the theta movements
        for n in range(repeat):
            # forward theta motor maps
            data1 = fits.getdata(dataPath + f'/thetaBegin{n}.fits')
            thetaFW[goodIdx, n, 0] = self.cal.extractPositions(data1)

            for k in range(iteration):
                data1 = fits.getdata(dataPath + f'/thetaForward{n}N{k}.fits')
                thetaFW[goodIdx, n, k+1] = self.cal.extractPositions(data1, guess=thetaFW[goodIdx, n, k])

            # reverse theta motor maps
            data1 = fits.getdata(dataPath + f'/thetaEnd{n}.fits')
            thetaRV[goodIdx, n, 0] = self.cal.extractPositions(data1, guess=thetaFW[goodIdx, n, iteration])

            for k in range(iteration):
                data1 = fits.getdata(dataPath + f'/thetaReverse{n}N{k}.fits')
                thetaRV[goodIdx, n, k+1] = self.cal.extractPositions(data1, guess=thetaRV[goodIdx, n, k])

        # save calculation result
        np.save(dataPath + '/thetaFW_A', thetaFW)
        np.save(dataPath + '/thetaRV_A', thetaRV)

        # calculate centers and theta angles
        thetaCenter, thetaRadius, thetaAngFW, thetaAngRV, badRange = self.cal.thetaCenterAngles(thetaFW, thetaRV)
        np.save(dataPath + '/thetaCenter_A', thetaCenter)
        np.save(dataPath + '/thetaRadius_A', thetaRadius)
        np.save(dataPath + '/thetaAngFW_A', thetaAngFW)
        np.save(dataPath + '/thetaAngRV_A', thetaAngRV)
        np.save(dataPath + '/badRange_A', badRange)

        # calculate average speeds
        thetaSpeedFW, thetaSpeedRV = self.cal.speed(thetaAngFW, thetaAngRV, steps, delta)
        np.save(dataPath + '/thetaSpeedFW_A', thetaSpeedFW)
        np.save(dataPath + '/thetaSpeedRV_A', thetaSpeedRV)

        # calculate motor maps in Johannes weighting
        thetaMMFW, thetaMMRV, bad = self.cal.motorMaps(thetaAngFW, thetaAngRV, steps, delta)
        bad[badRange] = True
        np.save(dataPath + '/thetaMMFW_A', thetaMMFW)
        np.save(dataPath + '/thetaMMRV_A', thetaMMRV)
        np.save(dataPath + '/bad_A', np.where(bad)[0])

        # calculate motor maps by average speeds
        thetaMMFW2, thetaMMRV2, bad2 = self.cal.motorMaps2(thetaAngFW, thetaAngRV, steps, delta)
        bad2[badRange] = True
        np.save(dataPath + '/thetaMMFW2_A', thetaMMFW2)
        np.save(dataPath + '/thetaMMRV2_A', thetaMMRV2)
        np.save(dataPath + '/bad2_A', np.where(bad2)[0])

        # update XML file, using Johannes weighting
        slow = not fast
        self.cal.updateThetaMotorMaps(thetaMMFW, thetaMMRV, bad, slow)
        if thetaOnTime is not None:
            onTime = np.full(57, thetaOnTime)
            self.cal.calibModel.updateOntimes(thtFwd=onTime, thtRev=onTime, fast=fast)
        self.cal.calibModel.createCalibrationFile(dataPath + '/' + newXml)
        self.cal.restoreConfig()

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

        thetaL, phiL, thetaCCW, thetaCW, phiCCW, phiCW = self.cal.geometry(thetaC, thetaR, thetaFW, thetaRV, phiC, phiR, phiFW, phiRV)

        # save calculation result
        np.save(dataPath + '/center_A', thetaC)
        np.save(dataPath + '/thetaL_A', thetaL)
        np.save(dataPath + '/phiL_A', phiL)
        np.save(dataPath + '/thetaCCW_A', thetaCCW)
        np.save(dataPath + '/thetaCW_A', thetaCW)
        np.save(dataPath + '/phiCCW_A', phiCCW)
        np.save(dataPath + '/phiCW_A', phiCW)

        # update XML configuration
        new = self.cal.calibModel

        # keep bad cobra configuration
        z = self.cal.badIdx
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
        self.cal.restoreConfig()

    def makeGeometryTable(self, dataPath):

        center = np.load(dataPath + 'center_A.npy')
        phi_arm = np.load(dataPath + 'phiL_A.npy')
        theta_arm = np.load(dataPath + 'thetaL_A.npy')
        d={'Center X': center.real,
            'Center Y': center.imag,
            'Phi Arm Length': phi_arm,
            'Theta Arm Length':theta_arm}

        dataframe = pd.DataFrame(d)   

        dataframe.to_csv(dataPath+'geometry.csv')

        phiCCW = np.load(dataPath + 'phiCCW_A.npy')
        phiCW = np.load(dataPath + 'phiCW_A.npy')
        thetaCCW = np.load(dataPath + 'thetaCCW_A.npy')
        thetaCW = np.load(dataPath + 'thetaCW_A.npy')

        d={'Theta CCW Stop': np.rad2deg(thetaCCW),
        'Theta CW Stop': np.rad2deg(thetaCW),
        'Phi CCW Stop': np.rad2deg(phiCCW),
        'Phi CW Stop': np.rad2deg(phiCW)}
        dataframe = pd.DataFrame(d)        
        dataframe.to_csv(dataPath+'hardstop.csv')

        
    def convertXML(self, newXml, dataPath, image1=None, image2=None):
        """ convert old XML to a new coordinate by using the 'phi homed' images
            assuming the cobra module is in horizontal setup
            One can use the path for generating phi motor maps
        """
        idx = self.cal.goodIdx
        idx1 = idx[idx <= self.cal.camSplit]
        idx2 = idx[idx > self.cal.camSplit]
        oldPos = self.cal.calibModel.centers
        newPos = np.zeros(57, dtype=complex)

        # read data and measure new positions
        if image1 is None:
            src1 = fits.getdata(dataPath + '/phi1Begin0.fits.gz')
        else:
            src1 = fits.getdata(dataPath + '/' + image1)
        if image2 is None:
            src2 = fits.getdata(dataPath + '/phi2Begin0.fits.gz')
        else:
            src2 = fits.getdata(dataPath + '/' + image2)
        data1 = sep.extract(src1.astype(float), 200)
        data2 = sep.extract(src2.astype(float), 200)
        home1 = np.array(sorted([(c['x'], c['y']) for c in data1], key=lambda t: t[0], reverse=True))
        home2 = np.array(sorted([(c['x'], c['y']) for c in data2], key=lambda t: t[0], reverse=True))
        newPos[idx1] = home1[:len(idx1), 0] + home1[:len(idx1), 1] * (1j)
        newPos[idx2] = home2[-len(idx2):, 0] + home2[-len(idx2):, 1] * (1j)

        # calculation tranformation
        offset1, scale1, tilt1, convert1 = calculation.transform(oldPos[idx1], newPos[idx1])
        offset2, scale2, tilt2, convert2 = calculation.transform(oldPos[idx2], newPos[idx2])

        split = self.cal.camSplit + 1
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
        old.createCalibrationFile(dataPath + '/' + newXml)
        self.cal.restoreConfig()
