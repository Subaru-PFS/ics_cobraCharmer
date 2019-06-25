import os
import sys
import numpy as np
from astropy.io import fits
import calculation

class moduleAnalyze():
    def __init__(self, xml, brokens=None, camSplit=26):
        if not os.path.exists(xml):
            print(f"Error: {xml} is not presented!")
            sys.exit()
        self.cal = calculation.Calculation(xml, brokens, camSplit)

    def makePhiMotorMap(
            self,
            newXml,
            dataPath,
            repeat=3,
            steps=200,
            totalSteps=5000,
            fast=True,
            phiOnTime=None,
            delta=0.1
        ):
        # generate phi motor maps, it accepts custom phiOnTIme parameter.
        # it assumes that theta arms have been move to up/down positions to avoid collision
        # if phiOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime
        #
        # Example:
        #     makePhiMotorMap(xml, path, fast=True)             // update fast motor maps
        #     makePhiMotorMap(xml, path, fast=False)            // update slow motor maps
        #     makePhiMotorMap(xml, path, phiOnTime=0.06)        // motor maps for on-time=0.06

        # variable declaration for position measurement
        iteration = totalSteps // steps
        phiFW = np.zeros((57, repeat, iteration+1), dtype=complex)
        phiRV = np.zeros((57, repeat, iteration+1), dtype=complex)
        goodIdx = self.cal.goodIdx

        # calculate the phi movements
        for n in range(repeat):
            # forward phi motor maps
            data1 = fits.getdata(dataPath + f'/phi1Begin{n}.fits.gz')
            data2 = fits.getdata(dataPath + f'/phi2Begin{n}.fits.gz')
            phiFW[goodIdx, n, 0] = self.cal.extractPositions(data1, data2)

            for k in range(iteration):
                data1 = fits.getdata(dataPath + f'/phi1Forward{n}N{k}.fits.gz')
                data2 = fits.getdata(dataPath + f'/phi2Forward{n}N{k}.fits.gz')
                phiFW[goodIdx, n, k+1] = self.cal.extractPositions(data1, data2, guess=phiFW[goodIdx, n, k])

            # reverse phi motor maps
            data1 = fits.getdata(dataPath + f'/phi1End{n}.fits.gz')
            data2 = fits.getdata(dataPath + f'/phi2End{n}.fits.gz')
            phiRV[goodIdx, n, 0] = self.cal.extractPositions(data1, data2, guess=phiFW[goodIdx, n, iteration])

            for k in range(iteration):
                data1 = fits.getdata(dataPath + f'/phi1Reverse{n}N{k}.fits.gz')
                data2 = fits.getdata(dataPath + f'/phi2Reverse{n}N{k}.fits.gz')
                phiRV[goodIdx, n, k+1] = self.cal.extractPositions(data1, data2, guess=phiRV[goodIdx, n, k])

        # save calculation result
        np.save(dataPath + '/phiFW_A', phiFW)
        np.save(dataPath + '/phiRV_A', phiRV)

        # calculate centers and phi angles
        phiCenter, phiRadius, phiAngFW, phiAngRV = self.cal.phiCenterAngles(phiFW, phiRV)
        np.save(dataPath + '/phiCenter_A', phiCenter)
        np.save(dataPath + '/phiRadius_A', phiRadius)
        np.save(dataPath + '/phiAngFW_A', phiAngFW)
        np.save(dataPath + '/phiAngRV_A', phiAngRV)

        # calculate average speeds
        phiSpeedFW, phiSpeedRV = self.cal.speed(phiAngFW, phiAngRV, steps, delta)
        np.save(dataPath + '/phiSpeedFW_A', phiSpeedFW)
        np.save(dataPath + '/phiSpeedRV_A', phiSpeedRV)

        # calculate motor maps in Johannes weighting
        phiMMFW, phiMMRV, bad = self.cal.motorMaps(phiAngFW, phiAngRV, steps, delta)
        np.save(dataPath + '/phiMMFW_A', phiMMFW)
        np.save(dataPath + '/phiMMRV_A', phiMMRV)
        np.save(dataPath + '/bad_A', np.where(bad)[0])

        # calculate motor maps by average speeds
        phiMMFW2, phiMMRV2, bad2 = self.cal.motorMaps2(phiAngFW, phiAngRV, steps, delta)
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
            data1 = fits.getdata(dataPath + f'/theta1Begin{n}.fits.gz')
            data2 = fits.getdata(dataPath + f'/theta2Begin{n}.fits.gz')
            thetaFW[goodIdx, n, 0] = self.cal.extractPositions(data1, data2)

            for k in range(iteration):
                data1 = fits.getdata(dataPath + f'/theta1Forward{n}N{k}.fits.gz')
                data2 = fits.getdata(dataPath + f'/theta2Forward{n}N{k}.fits.gz')
                thetaFW[goodIdx, n, k+1] = self.cal.extractPositions(data1, data2, guess=thetaFW[goodIdx, n, k])

            # reverse theta motor maps
            data1 = fits.getdata(dataPath + f'/theta1End{n}.fits.gz')
            data2 = fits.getdata(dataPath + f'/theta2End{n}.fits.gz')
            thetaRV[goodIdx, n, 0] = self.cal.extractPositions(data1, data2, guess=thetaFW[goodIdx, n, iteration])

            for k in range(iteration):
                data1 = fits.getdata(dataPath + f'/theta1Reverse{n}N{k}.fits.gz')
                data2 = fits.getdata(dataPath + f'/theta2Reverse{n}N{k}.fits.gz')
                thetaRV[goodIdx, n, k+1] = self.cal.extractPositions(data1, data2, guess=thetaRV[goodIdx, n, k])

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
        np.save(dataPath + '/thetaMMFW_A', thetaMMFW)
        np.save(dataPath + '/thetaMMRV_A', thetaMMRV)
        np.save(dataPath + '/bad_A', np.where(bad)[0])

        # calculate motor maps by average speeds
        thetaMMFW2, thetaMMRV2, bad2 = self.cal.motorMaps2(thetaAngFW, thetaAngRV, steps, delta)
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
