from importlib import reload
import logging
import numpy as np
import sys
import os

from ics.cobraCharmer import ethernet
from ics.cobraCharmer import func
from ics.cobraCharmer.log import Logger

class PFI(object):
    nCobrasPerModule = 57
    nModules = 42

    def __init__(self, fpgaHost='localhost', doConnect=True, doLoadModel=True, debug=False):
        """ Initialize a PFI class
        Args:
           fpgaHost    - fpga device
           doConnect   - do connection or not
           doLoadModel - load data model or not
        """
        self.logger = Logger.getLogger('fpga', debug)
        self.ioLogger = Logger.getLogger('fpgaIO', debug)

        self.fpgaHost = fpgaHost
        if doConnect:
            self.connect()
        if doLoadModel:
            self.loadModel()

    def connect(self, fpgaHost=None):
        """ Connect to COBRA fpga device """
        if fpgaHost is not None:
            self.fpgaHost = fpgaHost
        ethernet.sock.connect(self.fpgaHost, 4001)
        self.ioLogger.info(f'FPGA connection to {self.fpgaHost}')

    def disconnect(self):
        """ Disconnect from COBRA fpga device """
        ethernet.sock.close()
        ethernet.sock = ethernet.Sock()
        self.ioLogger.info(f'FPGA connection closed')

    def loadModel(self, filename=None):
        """ Load a motormap XML file. """
        import ics.cobraCharmer.pfiDesign as pfiDesign
        reload(pfiDesign)

        if filename is None:
            filename = os.path.dirname(sys.modules[__name__].__file__)
            filename += '../../../xml/updatedLinksAndMaps.xml'
        self.calibModel = pfiDesign.PFIDesign(filename)
        self.logger.info(f'load cobra model from {filename}')

    def _freqToPeriod(self, freq):
        """ Convert frequency to 60ns ticks """
        return int(round(16e3/freq))

    def _periodToFreq(self, freq):
        """ Convert 60ns ticks to a frequency """
        return (16e3/freq) if (freq >= 1) else 0

    def _mapCobraIndex(self, cobra):
        """ Convert our module + cobra to global cobra index for the calibration product. """

        return ((cobra.module - 1)*self.nCobrasPerModule + cobra.cobraNum-1)

    def reset(self, sectors=0x3f):
        """ Reset COBRA fpga device """
        err = func.RST(sectors)
        if err:
            self.logger.error(f'send RST command failed')
        else:
            self.logger.info(f'send RST command succeeded')

    def power(self, sectors=0x3f):
        """ Set COBRA PSU on/off """
        err = func.POW(sectors)
        if err:
            self.logger.error(f'send POW command failed')
        else:
            self.logger.info(f'send POW command succeeded')

    def setFreq(self, cobras, thetaFreq=None, phiFreq=None):
        """ Set COBRA motor frequency """
        if len(cobras) != len(thetaFreq):
            raise RuntimeError("number of theta frquencies must match number of cobras")
        if len(cobras) != len(phiFreq):
            raise RuntimeError("number of phi frquencies must match number of cobras")

        for c in cobras:
            cobraIdx = self._mapCobraIndex(c)
            if thetaFreq is None:
                thetaPer = self._freqToPeriod(self.calibModel.motorFreq1[cobraIdx]/1000)
            else:
                thetaPer = self._freqToPeriod(np.array(thetaFreq))
            if phiFreq is None:
                phiPer = self._freqToPeriod(self.calibModel.motorFreq2[cobraIdx]/1000)
            else:
                phiPer = self._freqToPeriod(np.array(phiFreq))

            # print(f'set {c.board},{c.cobra} to {thetaPer},{phiPer} {self.calibModel.motorFreq1[c.cobra]}')
            c.p = func.SetParams(p0=thetaPer, p1=phiPer, en=(True, True))
        err = func.SET(cobras)
        if err:
            self.logger.error(f'send SET command failed')
        else:
            self.logger.info(f'send SET command succeeded')

    def calibrateFreq(self, cobras, thetaLow=60.4, thetaHigh=70.3, phiLow=94.4, phiHigh=108.2, clockwise=True):
        """ Calibrate COBRA motor frequency """
        spin = func.CW_DIR if clockwise else func.CCW_DIR
        m0 = (self._freqToPeriod(thetaHigh), self._freqToPeriod(thetaLow))
        m1 = (self._freqToPeriod(phiHigh), self._freqToPeriod(phiLow))
        for c in cobras:
            c.p = func.CalParams(m0=m0, m1=m1, en=(True,True), dir=spin)
        err = func.CAL(cobras, timeout=65535)
        if err:
            self.logger.error(f'send Calibrate command failed')
        else:
            self.logger.info(f'send Calibrate command succeeded')

    def houseKeeping(self, modules=None, m0=(0,1000), m1=(0,1000), temps=(16.0,31.0), cur=(0.25,1.2), volt=(9.5,10.5)):
        """ HK command """

        if modules is None:
            modules = [1]
        elif np.isscalar(modules):
            modules = [modules]

        for m in modules:
            self.logger.info(f'HK command for Cobra module #{m}')
            for board in [1, 2]:
                # two boards in one module
                cobra_num = np.arange(board, self.nCobrasPerModule+1, 2)
                cobras = self.allocateCobraRange(m, cobra_num)
                for c in cobras:
                    c.p = func.HkParams(m0, m1, temps, cur, volt)
                err = func.HK(cobras)
                if err:
                    self.logger.error(f'send HK command failed')
                else:
                    self.logger.info(f'send HK command succeeded')

    def moveAllThetaPhiFromHome(self, cobras, thetaMove, phiMove, thetaFast=True, phiFast=True):
        """ Move all cobras by theta and phi angles from home

            thetaMove ,phiMove: the angle to move away, positive/negative values mean moving away from CCW/CW limits
            thetaFast: a boolean value for fast/slow theta movement
            phiFast: a boolean value for fast/slow phi movement
        """

        nCobras = self.calibModel.nCobras

        if thetaMove < 0:
            thetaHomes = (self.calibModel.tht1 - self.calibModel.tht0) % (2*np.pi) + 2*np.pi
            thetaMoves = np.zeros(nCobras) - np.abs(thetaMove)
        else:
            thetaHomes = np.zeros(nCobras)
            thetaMoves = np.zeros(nCobras) + np.abs(thetaMove)
        if phiMove < 0:
            phiHomes = (self.calibModel.phiOut - self.calibModel.phiIn) % (2*np.pi)
            phiMoves = np.zeros(nCobras) - np.abs(phiMove)
        else:
            phiHomes = np.zeros(nCobras)
            phiMoves = np.zeros(nCobras) + np.abs(phiMove)

        thetaSteps, phiSteps = self.calculateSteps(thetaHomes, thetaMoves, phiHomes, phiMoves, thetaFast, phiFast)

        cIdx = [self._mapCobraIndex(c) for c in cobras]
        cThetaSteps = thetaSteps[cIdx]
        cPhiSteps = phiSteps[cIdx]
        self.logger.info(f'steps: {list(zip(cThetaSteps, cPhiSteps))}')
        self.moveSteps(cobras, cThetaSteps, cPhiSteps, thetaFast=thetaFast, phiFast=phiFast)

    def moveAllSteps(self, cobras, thetaSteps, phiSteps, thetaFast=True, phiFast=True):
        """ Move all cobras for some theta and phi steps """
        thetaAllSteps = np.zeros(len(cobras)) + thetaSteps
        phiAllSteps = np.zeros(len(cobras)) + phiSteps

        self.moveSteps(cobras, thetaAllSteps, phiAllSteps, thetaFast=thetaFast, phiFast=phiFast)

    def moveThetaPhi(self, cobras, thetaMoves, phiMoves, thetaFroms=None, phiFroms=None, thetaFast=True, phiFast=True):
        """ Move cobras with theta and phi angles, angles are measured from CCW hard stops

            thetaMoves: A numpy array with theta angles to go
            phiMoves: A numpy array with phi angles to go
            thetaFroms: A numpy array with starting theta positions
            phiFroms: A numpy array with starting phi positions
            thetaFast: a boolean value for fast/slow theta movement
            phiFast: a boolean value for fast/slow phi movement

        """

        if len(cobras) != len(thetaMoves):
            raise RuntimeError("number of theta moves must match number of cobras")
        if len(cobras) != len(phiMoves):
            raise RuntimeError("number of phi moves must match number of cobras")
        if thetaFroms is not None and len(cobras) != len(thetaFroms):
            raise RuntimeError("number of theta froms must match number of cobras")
        if phiFroms is not None and len(cobras) != len(phiFroms):
            raise RuntimeError("number of phi froms must match number of cobras")
        nCobras = self.calibModel.nCobras

        _phiMoves = np.zeros(nCobras)
        _thetaMoves = np.zeros(nCobras)
        _phiFroms = np.zeros(nCobras)
        _thetaFroms = np.zeros(nCobras)

        cIdx = [self._mapCobraIndex(c) for c in cobras]
        _phiMoves[cIdx] = phiMoves
        _thetaMoves[cIdx] = thetaMoves
        if phiFroms is not None:
            _phiFroms[cIdx] = phiFroms
        if thetaFroms is not None:
            _thetaFroms[cIdx] = thetaFroms

        if isinstance(thetaFast, bool):
            _thetaFast = thetaFast
        elif len(thetaFast) == len(cobras):
            _thetaFast = np.full(self.calibModel.nCobras, True)
            _thetaFast[cIdx] = thetaFast
        else:
            raise RuntimeError("number of thetaFast must match number of cobras")

        if isinstance(phiFast, bool):
            _phiFast = phiFast
        elif len(phiFast) == len(cobras):
            _phiFast = np.full(self.calibModel.nCobras, True)
            _phiFast[cIdx] = phiFast
        else:
            raise RuntimeError("number of phiFast must match number of cobras")

        thetaSteps, phiSteps = self.calculateSteps(_thetaFroms, _thetaMoves, _phiFroms, _phiMoves, _thetaFast, _phiFast)

        cThetaSteps = thetaSteps[cIdx]
        cPhiSteps = phiSteps[cIdx]

        self.logger.info(f'steps: {list(zip(cThetaSteps, cPhiSteps))}')
        self.moveSteps(cobras, cThetaSteps, cPhiSteps, thetaFast=thetaFast, phiFast=phiFast)

    def thetaToGlobal(self, cobras, thetaLocals):
        """ Convert theta angles from relative to hard stops to global coordinate

            thetaLocals: the angle from home, positive/negative values mean away from CCW/CW limits

            returns: the angles in global coordinate
        """

        if len(cobras) != len(thetaLocals):
            raise RuntimeError("number of theta angles must match number of cobras")

        thetaGlobals = np.zeros(len(cobras))
        cIdx = [self._mapCobraIndex(c) for c in cobras]
        for i, c in enumerate(cIdx):
            if thetaLocals[i] < 0:
                thetaGlobals[i] = (self.calibModel.tht1[c] + thetaLocals[i]) % (2 * np.pi)
            else:
                thetaGlobals[i] = (self.calibModel.tht0[c] + thetaLocals[i]) % (2 * np.pi)
        return thetaGlobals

    def thetaToLocal(self, cobras, thetaGlobals):
        """ Convert theta angles from global coordinate to relative to CCW hard stops
            Be careful of the overlapping region between two hard stops

            cobras: a list of cobras
            thetaGlobals: the angles in global coordinate

            returns: the angle from CCW hard stops
        """

        if len(cobras) != len(thetaGlobals):
            raise RuntimeError("number of theta angles must match number of cobras")

        cIdx = [self._mapCobraIndex(c) for c in cobras]
        thetaLocals = (thetaGlobals - self.calibModel.tht0[cIdx]) % (2 * np.pi)
        return thetaLocals

    def moveSteps(self, cobras, thetaSteps, phiSteps, waitThetaSteps=None, waitPhiSteps=None, interval=2.5, thetaFast=True, phiFast=True):
        """ Move cobras with theta and phi steps

            thetaSteps: A numpy array with theta steps to go
            phiSteps: A numpy array with phi steps to go
            waitThetaSteps: A numpy array for theta delay
            waitPhiSteps: A numpy array with phi delay
            interval: the FPGA interval parameter for RUN command
            thetaFast: a boolean value for using fast/slow theta movement
            phiFast: a boolean value for using fast/slow phi movement

        """

        if len(cobras) != len(thetaSteps):
            raise RuntimeError("number of theta steps must match number of cobras")
        if len(cobras) != len(phiSteps):
            raise RuntimeError("number of phi steps must match number of cobras")
        if waitThetaSteps is not None and len(cobras) != len(waitThetaSteps):
            raise RuntimeError("number of waitThetaSteps must match number of cobras")
        if waitPhiSteps is not None and len(cobras) != len(waitPhiSteps):
            raise RuntimeError("number of waitPhiSteps must match number of cobras")
        if isinstance(thetaFast, bool):
            _thetaFast = np.full(len(cobras), thetaFast)
        elif len(cobras) != len(thetaFast):
            raise RuntimeError("number of thetaFast must match number of cobras")
        else:
            _thetaFast = thetaFast
        if isinstance(phiFast, bool):
            _phiFast = np.full(len(cobras), phiFast)
        elif len(cobras) != len(phiFast):
            raise RuntimeError("number of phiFast must match number of cobras")
        else:
            _phiFast = phiFast

        for c_i, c in enumerate(cobras):
            steps1 = int(np.abs(thetaSteps[c_i])), int(np.abs(phiSteps[c_i]))
            dirs1 = ['cw', 'cw']

            if thetaSteps[c_i] < 0:
                dirs1[0] = 'ccw'
            if phiSteps[c_i] < 0:
                dirs1[1] = 'ccw'

            en = (steps1[0] != 0, steps1[1] != 0)
            cobraId = self._mapCobraIndex(c)

            if _thetaFast[c_i]:
                if dirs1[0] == 'cw':
                    ontime1 = self.calibModel.motorOntimeFwd1[cobraId]
                else:
                    ontime1 = self.calibModel.motorOntimeRev1[cobraId]
            else:
                if dirs1[0] == 'cw':
                    ontime1 = self.calibModel.motorOntimeSlowFwd1[cobraId]
                else:
                    ontime1 = self.calibModel.motorOntimeSlowRev1[cobraId]

            if _phiFast[c_i]:
                if dirs1[1] == 'cw':
                    ontime2 = self.calibModel.motorOntimeFwd2[cobraId]
                else:
                    ontime2 = self.calibModel.motorOntimeRev2[cobraId]
            else:
                if dirs1[1] == 'cw':
                    ontime2 = self.calibModel.motorOntimeSlowFwd2[cobraId]
                else:
                    ontime2 = self.calibModel.motorOntimeSlowRev2[cobraId]

            # For early-late offsets.
            if waitThetaSteps is not None:
                offtime1 = waitThetaSteps[c_i]
            else:
                offtime1 = 0

            if waitPhiSteps is not None:
                offtime2 = waitPhiSteps[c_i]
            else:
                offtime2 = 0

            c.p = func.RunParams(pu=(int(1000*ontime1), int(1000*ontime2)),
                                 st=(steps1),
                                 sl=(int(offtime1), int(offtime2)),
                                 en=en,
                                 dir=dirs1)
        # temperarily fix for interval and timeout
        err = func.RUN(cobras, inter=int(interval*1000/16), timeout=65535)
        if err:
            self.logger.error(f'send RUN command failed')
        else:
            self.logger.info(f'send RUN command succeeded')

    def homePhi(self, cobras, nsteps=5000, fast=True):
        # positive/negative steps for CCW/CW limit stops

        thetaSteps = np.zeros(len(cobras))
        phiSteps = np.zeros(len(cobras)) - nsteps
        self.moveSteps(cobras, thetaSteps, phiSteps, phiFast=fast)

    def homePhiSafe(self, cobras, nsteps=5000, iterations=10, fast=True):
        # positive/negative steps for CCW/CW limit stops

        thetaSteps = (np.zeros(len(cobras)) + nsteps) * (0.5 / iterations)
        phiSteps = (np.zeros(len(cobras)) + nsteps) * (-1.0 / iterations)
        for _ in range(iterations):
            self.moveSteps(cobras, thetaSteps, phiSteps, thetaFast=fast, phiFast=fast)

    def homeTheta(self, cobras, nsteps=10000, fast=True):
        # positive/negative steps for CCW/CW limit stops

        thetaSteps = np.zeros(len(cobras)) - nsteps
        phiSteps = np.zeros(len(cobras))
        self.moveSteps(cobras, thetaSteps, phiSteps, thetaFast=fast)

    def cobraBySerial(self, serial):
        """ Find a cobra from its serial number. """
        idx = np.where(self.calibModel.serialIds == serial)
        if len(idx) == 0:
            return None
        return func.Cobra(self.calibModel.moduleIds[idx],
                          self.calibModel.positionerIds[idx])

    def calculateSteps(self, startTht, deltaTht, startPhi, deltaPhi, thetaFast=True, phiFast=True):
        """ Modified from ics_cobraOps MotorMapGroup.py
        Calculates the total number of motor steps required to move the
        cobra fibers the given theta and phi delta angles from
        CCW hard stops

        Parameters
        ----------
        startTht: object
            A numpy array with the starting theta angle position.
        deltaTht: object
            A numpy array with the theta delta offsets relative to the starting
            theta positions.
        startPhi: object
            A numpy array with the starting phi angle positions.
        deltaPhi: object
            A numpy array with the phi delta offsets relative to the starting
            phi positions.
        thetaFast: object
            A boolean value or array for fast/slow theta motor movement.
        phiFast: object
            A boolean value or array for fast/slow phi motor movement.

        Returns
        -------
        tuple
            A python tuple with the total number of motor steps for the theta
            and phi angles.

        """

        nCobras = self.calibModel.nCobras
        if len(startTht) != nCobras:
            raise RuntimeError("number of startTht must match total number of cobras")
        if len(deltaTht) != nCobras:
            raise RuntimeError("number of deltaTht must match total number of cobras")
        if len(startPhi) != nCobras:
            raise RuntimeError("number of startPhi must match total number of cobras")
        if len(deltaPhi) != nCobras:
            raise RuntimeError("number of deltaPhi must match total number of cobras")

        # check thetaFast/phiFast parameters
        if isinstance(thetaFast, bool):
            _thetaFast = np.full(nCobras, thetaFast)
        elif len(thetaFast) != nCobras:
            raise RuntimeError("the length of thetaFast must match total number of cobras")
        else:
            _thetaFast = thetaFast
        if isinstance(phiFast, bool):
            _phiFast = np.full(nCobras, phiFast)
        elif len(phiFast) != nCobras:
            raise RuntimeError("the length of phiFast must match total number of cobras")
        else:
            _phiFast = phiFast

        # Calculate the total number of motor steps for each angle
        nThtSteps = np.empty(nCobras)
        nPhiSteps = np.empty(nCobras)

        for c in range(nCobras):
            # Get the integrated step maps for the theta angle
            if deltaTht[c] >= 0:
                if _thetaFast[c]:
                    thtSteps = self.calibModel.posThtSteps[c]
                else:
                    thtSteps = self.calibModel.posThtSlowSteps[c]
            else:
                if _thetaFast[c]:
                    thtSteps = self.calibModel.negThtSteps[c]
                else:
                    thtSteps = self.calibModel.negThtSlowSteps[c]

            # Get the integrated step maps for the phi angle
            if deltaPhi[c] >= 0:
                if _phiFast[c]:
                    phiSteps = self.calibModel.posPhiSteps[c]
                else:
                    phiSteps = self.calibModel.posPhiSlowSteps[c]
            else:
                if _phiFast[c]:
                    phiSteps = self.calibModel.negPhiSteps[c]
                else:
                    phiSteps = self.calibModel.negPhiSlowSteps[c]

            # Calculate the total number of motor steps for the theta movement
            stepsRange = np.interp([startTht[c], startTht[c] + deltaTht[c]], self.calibModel.thtOffsets[c], thtSteps)
            nThtSteps[c] = stepsRange[1] - stepsRange[0]

            # Calculate the total number of motor steps for the phi movement
            stepsRange = np.interp([startPhi[c], startPhi[c] + deltaPhi[c]], self.calibModel.phiOffsets[c], phiSteps)
            nPhiSteps[c] = stepsRange[1] - stepsRange[0]

        return (nThtSteps, nPhiSteps)

    def anglesToPositions(self, cobras, thetaAngles, phiAngles):
        """Convert the theta, phi angles to fiber positions.

        Parameters
        ----------
        cobras: a list of cobras
        thetaAngles: object
            A numpy array with the theta angles from CCW limit.
        phiAngles: object
            A numpy array with the phi angles from CCW limit.

        Returns
        -------
        numpy array
            A complex numpy array with the fiber positions.

        """
        if len(cobras) != len(thetaAngles):
            raise RuntimeError("number of theta angles must match number of cobras")
        if len(cobras) != len(phiAngles):
            raise RuntimeError("number of phi angles must match number of cobras")

        cIdx = np.array([self._mapCobraIndex(c) for c in cobras])
        thtRange = (self.calibModel.tht1[cIdx] - self.calibModel.tht0[cIdx]) % (2*np.pi) + (2*np.pi)
        if np.any(np.zeros(len(cIdx)) > thetaAngles) or np.any(thtRange < thetaAngles):
            self.logger.error('Some theta angles are out of range')
        phiRange = self.calibModel.phiOut[cIdx] - self.calibModel.phiIn[cIdx]
        if np.any(np.zeros(len(cIdx)) > phiAngles) or np.any(phiRange < phiAngles):
            self.logger.error('Some phi angles are out of range')

        ang1 = self.calibModel.tht0[cIdx] + thetaAngles
        ang2 = ang1 + phiAngles + self.calibModel.phiIn[cIdx]
        return self.calibModel.centers[cIdx] + self.calibModel.L1[cIdx] * np.exp(1j * ang1) + self.calibModel.L2[cIdx] * np.exp(1j * ang2)

    def positionsToAngles(self, cobras, positions):
        """Convert the fiber positions to theta, phi angles from CCW limit.

        Parameters
        ----------
        cobras: a list of cobras
        positions: numpy array
            A complex numpy array with the fiber positions.

        Returns
        -------
        tuple
            A python tuples with all the possible angles (theta, phi, overlapping).
            Since there are possible 2 phi solutions so the dimensions of theta
            and phi are (len(cobras), 2), the value np.nan indicates
            there is no solution. overlapping is a boolean array of the same size,
            true indicatess the theta solution is within the two hard stops.

        """
        if len(cobras) != len(positions):
            raise RuntimeError("number of positions must match number of cobras")
        cIdx = np.array([self._mapCobraIndex(c) for c in cobras])

        # Calculate the cobras rotation angles applying the law of cosines
        relativePositions = positions - self.calibModel.centers[cIdx]
        distance = np.abs(relativePositions)
        L1 = self.calibModel.L1[cIdx]
        L2 = self.calibModel.L2[cIdx]
        distanceSq = distance ** 2
        L1Sq = L1 ** 2
        L2Sq = L2 ** 2
        phiIn = self.calibModel.phiIn[cIdx] + np.pi
        phiOut = self.calibModel.phiOut[cIdx] + np.pi
        tht0 = self.calibModel.tht0[cIdx]
        tht1 = self.calibModel.tht1[cIdx]
        phi = np.full((len(cobras), 2), np.nan)
        tht = np.full((len(cobras), 2), np.nan)
        overlapping = np.full((len(cobras), 2), False)

        for i in range(len(positions)):
            # check if the positions are reachable by cobras
            if distance[i] > L1[i] + L2[i] or distance[i] < np.abs(L1[i] - L2[i]):
                continue

            ang1 = np.arccos((L1Sq[i] + L2Sq[i] - distanceSq[i]) / (2 * L1[i] * L2[i]))
            ang2 = np.arccos((L1Sq[i] + distanceSq[i] - L2Sq[i]) / (2 * L1[i] * distance[i]))

            # the regular solutions, phi angle is between 0 and pi, no checking for phi hard stops
            #if ang1 > phiIn[i] and ang1 < phiOut[i]:
            phi[i][0] = ang1 - phiIn[i]
            tht[i][0] = (np.angle(relativePositions[i]) + ang2 - tht0[i]) % (2 * np.pi)
            # check if tht is within two theta hard stops
            if tht[i][0] <= (tht1[i] - tht0[i]) % (2 * np.pi):
                overlapping[i][0] = True

            # check if there are additional phi solutions
            if phiIn[i] <= -ang1 and ang1 > 0:
                # phiIn < 0
                phi[i][1] = -ang1 - phiIn[i]
                tht[i][1] = (np.angle(relativePositions[i]) - ang2 - tht0[i]) % (2 * np.pi)
                # check if tht is within two theta hard stops
                if tht[i][1] <= (tht1[i] - tht0[i]) % (2 * np.pi):
                    overlapping[i][1] = True
            elif phiOut[i] >= 2 * np.pi - ang1 and ang1 < np.pi:
                # phiOut > np.pi
                phi[i][1] = 2 * np.pi - ang1 - phiIn[i]
                tht[i][1] = (np.angle(relativePositions[i]) - ang2 - tht0[i]) % (2 * np.pi)
                # check if tht is within two theta hard stops
                if tht[i][1] <= (tht1[i] - tht0[i]) % (2 * np.pi):
                    overlapping[i][1] = True
        return (tht, phi, overlapping)

    def moveXY(self, cobras, startPositions, targetPositions, thetaThreshold=-1.0, phiThreshold=-1.0):
        """Move the Cobras in XY coordinate.

        Parameters
        ----------
        cobras: a list of cobras
        targetPositions: numpy array
            A complex numpy array with the target fiber positions.
        startPositions: numpy array
            A complex numpy array with the starting fiber positions.
        thetaThreshold: a double value
            The threshold value for using slow/fast theta motor maps, the default is fast.
        phiThreshold: a double value
            The threshold value for using slow/fast phi motor maps, the default is fast.

        If there are more than one possible convertion to theta/phi, this function picks the regular one.
        For better control, the caller should use positionsToAngles to determine which solution is the right one.
        """

        if len(cobras) != len(startPositions):
            raise RuntimeError("number of starting positions must match number of cobras")
        if len(cobras) != len(targetPositions):
            raise RuntimeError("number of target positions must match number of cobras")

        startTht, startPhi, _ = self.positionsToAngles(cobras, startPositions)
        targetTht, targetPhi, _ = self.positionsToAngles(cobras, targetPositions)
        deltaTht = targetTht[:,0] - startTht[:,0]
        deltaPhi = targetPhi[:,0] - startPhi[:,0]
        thetaFast = np.full(len(cobras), True)
        thetaFast[deltaTht < thetaThreshold] = False
        phiFast = np.full(len(cobras), True)
        phiFast[deltaPhi < phiThreshold] = False

        # check if there is a solution
        valids = np.all([np.isnan(startTht[:,0]) == False, np.isnan(targetTht[:,0]) == False], axis=0)
        valid_cobras = [c for i,c in enumerate(cobras) if valids[i]]
        if len(valid_cobras) <= 0:
            self.logger.error("no valid target positions are found")
            return
        elif not np.all(valids):
            self.logger.info("some target positions are invalid")

        # move bobras by angles
        self.logger.info(f"engaged cobras: {[(c.module,c.cobraNum) for c in valid_cobras]}")
        self.logger.info(f"move to: {list(zip(targetTht[valids,0], targetPhi[valids,0]))}")
        self.logger.info(f"move from: {list(zip(startTht[valids,0], startPhi[valids,0]))}")
        self.moveThetaPhi(valid_cobras, deltaTht[valids], deltaPhi[valids], startTht[valids,0], startPhi[valids,0], thetaFast[valids], phiFast[valids])

    def moveXYfromHome(self, cobras, targetPositions, ccwLimit=True, thetaThreshold=-1.0, phiThreshold=-1.0):
        """Move the Cobras in XY coordinate from hard stops.

        Parameters
        ----------
        cobras: a list of cobras
        targetPositions: numpy array
            A complex numpy array with the target fiber positions.
        thetaHome: 'ccw'(default) or 'cw' hard stop
        thetaThreshold: a double value
            The threshold value for using slow/fast theta motor maps, the default is fast.
        phiThreshold: a double value
            The threshold value for using slow/fast phi motor maps, the default is fast.

        If there are more than one possible convertion to theta/phi, this function picks the regular one.
        For better control, the caller should use positionsToAngles to determine which solution is the right one.
        """

        if len(cobras) != len(targetPositions):
            raise RuntimeError("number of target positions must match number of cobras")

        targetTht, targetPhi, _ = self.positionsToAngles(cobras, targetPositions)

        # check if there is a solution
        valids = np.isnan(targetTht[:,0]) == False
        valid_cobras = [c for i,c in enumerate(cobras) if valids[i]]
        if len(valid_cobras) <= 0:
            self.logger.error("no valid target positions are found")
            return
        elif not np.all(valids):
            self.logger.info("some target positions are invalid")

        # define home positions
        phiHomes = np.zeros(len(valid_cobras))
        if ccwLimit:
            thtHomes = phiHomes
        else:
            cIdx = np.array([self._mapCobraIndex(c) for c in valid_cobras])
            thtHomes = (self.calibModel.tht1[cIdx] - self.calibModel.tht0[cIdx]) % (2*np.pi) + (2*np.pi)
        self.logger.info(f"engaged cobras: {[(c.module,c.cobraNum) for c in valid_cobras]}")
        self.logger.info(f"move to: {list(zip(targetTht[valids,0], targetPhi[valids,0]))}")
        self.logger.info(f"move from: {list(zip(thtHomes, phiHomes))}")

        # move cobras by angles
        deltaTht = targetTht[valids,0] - thtHomes
        deltaPhi = targetPhi[valids,0] - phiHomes
        thetaFast = np.full(len(valid_cobras), True)
        thetaFast[deltaTht < thetaThreshold] = False
        phiFast = np.full(len(valid_cobras), True)
        phiFast[deltaPhi < phiThreshold] = False
        self.moveThetaPhi(valid_cobras, deltaTht, deltaPhi, thtHomes, phiHomes, thetaFast, phiFast)


    @classmethod
    def allocateAllCobras(cls):
        return cls.allocateCobraRange(range(1,cls.nModules))

    @classmethod
    def allocateCobraRange(cls, modules, cobraNums=None):
        """ Utility to allocate swaths of cobras:

        Args:
          modules (int array-like): a list of 1-indexed boards to allocate from.
          cobras  (int array-like): a list of 1-indexed cobras to allocate from.

        Return:
          cobras
        """
        cobras = []

        if np.isscalar(modules):
            modules = [modules]
        for m in modules:
            if m == 0:
                raise IndexError('module numbers are 1-indexed, grrr.')
            if cobraNums is None:
                _cobraNums = range(1,cls.nCobrasPerModule+1)
            else:
                _cobraNums = cobraNums

            for c in _cobraNums:
                if c == 0:
                    raise IndexError('cobra numbers are 1-indexed, grrr.')

                cobras.append(func.Cobra(m, c))

        return cobras

    @classmethod
    def allocateCobraList(cls, cobraIds):
        cobras = []
        for mc in cobraIds:
            m, c = mc
            if m == 0:
                raise IndexError('module numbers are 1-indexed, grrr.')
            if c == 0:
                raise IndexError('cobra numbers are 1-indexed, grrr.')

            cobras.append(func.Cobra(m, c))

        return cobras
