from importlib import reload
import logging
import numpy as np
import sys
import os

from ics.cobraCharmer import ethernet
from ics.cobraCharmer import func
from ics.cobraCharmer.log import Logger

import ics.cobraCharmer.fpgaLogger as fpgaLogger

class PFI(object):
    CW = 1
    CCW = -1
    DISABLE = 0

    nCobrasPerModule = 57
    nModules = 42

    "CW is the same as forward or positive direction, CCW means reverse or negative direction"
    dirIds = {'ccw':'ccw', 'cw':'cw', CCW:'ccw', CW:'cw', 'CCW':'ccw', 'CW':'cw'}

    def __init__(self, fpgaHost='localhost', doConnect=True, doLoadModel=True, debug=False):
        """ Initialize a PFI class
        Args:
           fpgaHost    - fpga device
           doConnect   - do connection or not
           doLoadModel - load data model or not
        """
        self.logger = Logger.getLogger('fpga', debug)
        self.ioLogger = Logger.getLogger('fpgaIO', debug)
        self.protoLogger = fpgaLogger.FPGAProtocolLogger()

        self.calibModel = None
        self.motorMap = None

        self.fpgaHost = fpgaHost
        if doConnect:
            self.connect()
        if doLoadModel:
            self.loadModel()

    def connect(self, fpgaHost=None):
        """ Connect to COBRA fpga device """
        if fpgaHost is not None:
            self.fpgaHost = fpgaHost
        ethernet.sock.connect(self.fpgaHost, 4001, protoLogger=self.protoLogger)
        self.ioLogger.info(f'FPGA connection to {self.fpgaHost}')

    def disconnect(self):
        """ Disconnect from COBRA fpga device """
        ethernet.sock.close()
        ethernet.sock = ethernet.Sock()
        self.ioLogger.info(f'FPGA connection closed')

    def loadModel(self, filename=None):
        """ Load a motormap XML file. """

        import ics.cobraOps.MotorMapGroup as cobraMotorMap
        import ics.cobraCharmer.pfiDesign as pfiDesign
        reload(pfiDesign)
        reload(cobraMotorMap)

        if filename is None:
            filename = os.path.dirname(sys.modules[__name__].__file__)
            filename += '../../../xml/updatedLinksAndMaps.xml'
        self.calibModel = pfiDesign.PFIDesign(filename)
        self.motorMap = cobraMotorMap.MotorMapGroup(self.calibModel.nCobras)

        self.motorMap.useCalibrationProduct(self.calibModel)
        self.logger.info(f'load cobra model from {filename}')

    def _freqToPeriod(self, freq):
        """ Convert frequency to 60ns ticks """
        return int(round(16e3/freq))

    def _periodToFreq(self, freq):
        """ Convert 60ns ticks to a frequency """
        return (16e3/freq) if (freq >= 1) else 0

    def _mapCobraIndex(self, cobra):
        """ Convert our module + cobra to global cobra index for the calibration product. """

        if self.calibModel is not None:
            return self.calibModel.findCobraByModuleAndPositioner(cobra.module,
                                                                  cobra.cobraNum)
        else:
            self.logger.warn('no calibModel, so we are guessing about the calibModel index for a cobra')
            return ((cobra.module - 1)*self.nCobrasPerModule + cobra.cobraNum-1)

    def reset(self, sectors=0x3f):
        """ Reset COBRA fpga device """
        err = func.RST(sectors)
        if err:
            self.logger.error(f'send RST command failed')
        else:
            self.logger.debug(f'send RST command succeeded')

    def diag(self):
        """ Get fpga board inventory"""
        err = func.DIA()
        if err:
            self.logger.error(f'send DIA command failed')
        else:
            self.logger.debug(f'send DIA command succeeded')

    def power(self, sectors=0x3f):
        """ Set COBRA PSU on/off """
        err = func.POW(sectors)
        if err:
            self.logger.error(f'send POW command failed')
        else:
            self.logger.debug(f'send POW command succeeded')

    def hk(self, module, board, updateModel=False):
        """ Fetch housekeeping info for a board.

        Note:

        The FPGA deals with _boards_, but the original code deals with _modules_. Wrap that.
        """
        cobras = self.allocateCobraBoard(module, board)
        err = func.HK(cobras, updateModel=(self.calibModel if updateModel else None))
        if err:
            self.logger.error(f'send HK command failed')
        else:
            self.logger.debug(f'send HK command succeeded')

    def setFreq(self, cobras=None):
        """ Set COBRA motor frequency """
        if cobras is None:
            cobras = self.getAllDefinedCobras()

        for c in cobras:
            cobraIdx = self._mapCobraIndex(c)
            thetaPer = self._freqToPeriod(self.calibModel.motorFreq1[cobraIdx]/1000)
            phiPer = self._freqToPeriod(self.calibModel.motorFreq2[cobraIdx]/1000)

            # print(f'set {c.board},{c.cobra} to {thetaPer},{phiPer} {self.calibModel.motorFreq1[c.cobra]}')
            c.p = func.SetParams(p0=thetaPer, p1=phiPer, en=(True, True))
        err = func.SET(cobras)
        if err:
            self.logger.error(f'send SET command failed')
        else:
            self.logger.debug(f'send SET command succeeded')

    def calibrate(self, cobras=None,
                  thetaLow=60.4, thetaHigh=70.3,
                  phiLow=94.4, phiHigh=108.2,
                  clockwise=True):
        """ calibrate a set of cobras.

        Args:
        thetaLow, thetaHigh -
        phiLow, phiHigh -

        """
        if cobras is None:
            cobras = self.getAllDefinedCobras()

        spin = ('cw','cw') if clockwise else ('ccw','ccw')
        for c in cobras:
            c.p = func.CalParams(m0=(self._freqToPeriod(thetaLow), self._freqToPeriod(thetaHigh)),
                                 m1=(self._freqToPeriod(phiLow), self._freqToPeriod(phiHigh)),
                                 en=(True,True), dir=spin)

        err = func.CAL(cobras)
        if err:
            raise RuntimeError("calibration failed")

    def moveAllThetaPhi(self, cobras, thetaMove, phiMove, thetaHome=None, phiHome=None):
        """ Move all cobras by theta and phi angles from home

            thetaHome, phiHome: current home position, 'ccw'(default) or 'cw'
            thetaMove ,phiMove: the angle to move away from home
        """
        nCobras = self.calibModel.nCobras

        thetaHome = self.dirIds.get(thetaHome, 'ccw')
        phiHome = self.dirIds.get(phiHome, 'ccw')
        if thetaHome != 'ccw':
            thetaHomes = (self.calibModel.tht1 - self.calibModel.tht0) % (2*np.pi)
            thetaMoves = np.zeros(nCobras) - np.abs(thetaMove)
        else:
            thetaHomes = np.zeros(nCobras)
            thetaMoves = np.zeros(nCobras) + np.abs(thetaMove)
        if phiHome != 'ccw':
            phiHomes = (self.calibModel.phiOut - self.calibModel.phiIn) % (2*np.pi)
            phiMoves = np.zeros(nCobras) - np.abs(phiMove)
        else:
            phiHomes = np.zeros(nCobras)
            phiMoves = np.zeros(nCobras) + np.abs(phiMove)

        thetaSteps, phiSteps = self.calculateSteps(thetaHomes, thetaMoves, phiHomes, phiMoves)

        cIdx = [self._mapCobraIndex(c) for c in cobras]
        cThetaSteps = thetaSteps[cIdx]
        cPhiSteps = phiSteps[cIdx]

        self.logger.info(f'thetaSteps: {cThetaSteps}')
        self.logger.info(f'phiSteps: {cPhiSteps}')

        self.moveSteps(cobras, cThetaSteps, cPhiSteps)

    def moveAllSteps(self, cobras, thetaSteps, phiSteps, dirs=None):
        """ Move all cobras for an unique theta and phi steps """
        thetaAllSteps = np.zeros(len(cobras)) + thetaSteps
        phiAllSteps = np.zeros(len(cobras)) + phiSteps
        if dirs is not None:
            allDirs = [dirs]*len(cobras)
        else:
            allDirs = None

        self.moveSteps(cobras, thetaAllSteps, phiAllSteps, allDirs)

    def moveThetaPhi(self, cobras, thetaMoves, phiMoves, thetaFroms=None, phiFroms=None):
        """ Move cobras with theta and phi angles, angles are measured from CCW hard stops

            thetaMoves: A numpy array with theta angles to go
            phiMoves: A numpy array with phi angles to go
            thetaFroms: A numpy array with starting theta positions
            phiFroms: A numpy array with starting phi positions

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
        for i, c in enumerate(cIdx):
            _phiMoves[c] = phiMoves[i]
            _thetaMoves[c] = thetaMoves[i]
            if phiFroms is not None:
                _phiFroms[c] = phiFroms[i]
            if thetaFroms is not None:
                _thetaFroms[c] = thetaFroms[i]

        thetaSteps, phiSteps = self.calculateSteps(_thetaFroms, _thetaMoves, _phiFroms, _phiMoves)

        cThetaSteps = thetaSteps[cIdx]
        cPhiSteps = phiSteps[cIdx]

        self.logger.info(f'steps: {list(zip(cThetaSteps, cPhiSteps))}')
        self.moveSteps(cobras, cThetaSteps, cPhiSteps)

    def thetaToGlobal(self, cobras, thetaLocals, thetaHome=None):
        """ Convert theta angles from relative to hard stops to global coordinate

            thetaLocals: the angle from CW or CCW hard stops
            thetaHome: 'ccw'(default) or 'cw'

            returns: the angles in global coordinate
        """

        thetaHome = self.dirIds.get(thetaHome, 'ccw')
        if len(cobras) != len(thetaLocals):
            raise RuntimeError("number of theta angles must match number of cobras")

        thetaGlobals = np.zeros(len(cobras))
        thetaLocals = np.abs(thetaLocals)
        cIdx = [self._mapCobraIndex(c) for c in cobras]
        if thetaHome != 'ccw':
            for i, c in enumerate(cIdx):
                thetaGlobals[i] = (self.calibModel.tht1[c] - thetaLocals[i]) % (2 * np.pi)
        else:
            for i, c in enumerate(cIdx):
                thetaGlobals[i] = (thetaLocals[i] + self.calibModel.tht0[c]) % (2 * np.pi)
        return thetaGlobals

    def thetaToLocal(self, cobras, thetaGlobals, thetaHome=None):
        """ Convert theta angles from global coordinate to relative to hard stops
            Be careful of the overlapping region between two hard stops

            cobras: a list of cobras
            thetaGlobals: the angles in global coordinate
            thetaHome: 'ccw'(default) or 'cw'

            returns: the angle from CW or CCW hard stops
        """

        if len(cobras) != len(thetaGlobals):
            raise RuntimeError("number of theta angles must match number of cobras")

        thetaHome = self.dirIds.get(thetaHome, 'ccw')
        thetaLocals = np.zeros(len(cobras))
        cIdx = [self._mapCobraIndex(c) for c in cobras]
        if thetaHome != 'ccw':
            for i, c in enumerate(cIdx):
                thetaLocals[i] = (self.calibModel.tht1[c] - thetaGlobals[i]) % (2 * np.pi)
        else:
            for i, c in enumerate(cIdx):
                thetaLocals[i] = (thetaGlobals[i] - self.calibModel.tht0[c]) % (2 * np.pi)
        return thetaLocals

    def moveSteps(self, cobras, thetaSteps, phiSteps, dirs=None, waitThetaSteps=None, waitPhiSteps=None, interval=2.5):
        """ Move cobras with theta and phi steps """

        if len(cobras) != len(thetaSteps):
            raise RuntimeError("number of theta steps must match number of cobras")
        if len(cobras) != len(phiSteps):
            raise RuntimeError("number of phi steps must match number of cobras")
        if dirs is not None and len(cobras) != len(dirs):
            raise RuntimeError("number of directions must match number of cobras")
        if waitThetaSteps is not None and len(cobras) != len(waitThetaSteps):
            raise RuntimeError("number of waitThetaSteps must match number of cobras")
        if waitPhiSteps is not None and len(cobras) != len(waitPhiSteps):
            raise RuntimeError("number of waitPhiSteps must match number of cobras")

        model = self.calibModel

        for c_i, c in enumerate(cobras):
            steps1 = int(np.abs(thetaSteps[c_i])), int(np.abs(phiSteps[c_i]))
            if dirs is not None:
                dirs1 = [self.dirIds.get(dirs[c_i][0], 'cw'), self.dirIds.get(dirs[c_i][1], 'cw')]
            else:
                dirs1 = ['cw', 'cw']

            if thetaSteps[c_i] < 0:
                if dirs1[0] == 'ccw':
                    dirs1[0] = 'cw'
                else:
                    dirs1[0] = 'ccw'
            if phiSteps[c_i] < 0:
                if dirs1[1] == 'ccw':
                    dirs1[1] = 'cw'
                else:
                    dirs1[1] = 'ccw'

            en = (steps1[0] != 0, steps1[1] != 0)
            cobraId = self._mapCobraIndex(c)

            if dirs1[0] == 'cw':
                ontime1 = model.motorOntimeFwd1[cobraId]
            elif dirs1[0] == 'ccw':
                ontime1 = model.motorOntimeRev1[cobraId]
            else:
                raise ValueError(f'invalid direction: {dirs1[0]}')

            if dirs1[1] == 'cw':
                ontime2 = model.motorOntimeFwd2[cobraId]
            elif dirs1[1] == 'ccw':
                ontime2 = model.motorOntimeRev2[cobraId]
            else:
                raise ValueError(f'invalid direction: {dirs1[1]}')

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

    def homePhi(self, cobras, nsteps=5000, dir='ccw'):
        thetaSteps = np.zeros(len(cobras))
        phiSteps = np.zeros(len(cobras)) + nsteps
        dirs = [(dir,dir)]*len(cobras)
        self.moveSteps(cobras, thetaSteps, phiSteps, dirs)

    def homePhiSafe(self, cobras, nsteps=5000, dir='ccw', iterations=20):
        thetaSteps = (np.zeros(len(cobras)) + nsteps) * (0.5 / iterations)
        phiSteps = (np.zeros(len(cobras)) + nsteps) * (-1.0 / iterations)
        if dir == 'cw':
            thetaSteps = -thetaSteps
            phiSteps = -phiSteps
        for i in range(iterations):
            self.moveSteps(cobras, thetaSteps, phiSteps)

    def homeTheta(self, cobras, nsteps=10000, dir='ccw'):
        thetaSteps = np.zeros(len(cobras)) + nsteps
        phiSteps = np.zeros(len(cobras))
        dirs = [(dir,dir)]*len(cobras)
        self.moveSteps(cobras, thetaSteps, phiSteps, dirs)

    def cobraBySerial(self, serial):
        """ Find a cobra from its serial number. """
        idx = np.where(self.calibModel.serialIds == serial)
        if len(idx) == 0:
            return None
        return func.Cobra(self.calibModel.moduleIds[idx],
                          self.calibModel.positionerIds[idx])

    def calculateSteps(self, startTht, deltaTht, startPhi, deltaPhi):
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

        Returns
        -------
        tuple
            A python tuple with the total number of motor steps for the theta
            and phi angles.

        """
        # Get the integrated step maps for the theta angle
        thtSteps = self.motorMap.negThtSteps.copy()
        thtSteps[deltaTht >= 0] = self.motorMap.posThtSteps[deltaTht >= 0]

        # Get the integrated step maps for the phi angle
        phiSteps = self.motorMap.negPhiSteps.copy()
        phiSteps[deltaPhi >= 0] = self.motorMap.posPhiSteps[deltaPhi >= 0]

        # Calculate the total number of motor steps for each angle
        nThtSteps = np.empty(self.motorMap.nMaps)
        nPhiSteps = np.empty(self.motorMap.nMaps)

        for c in range(self.motorMap.nMaps):
            # Calculate the total number of motor steps for the theta movement
            stepsRange = np.interp([startTht[c], startTht[c] + deltaTht[c]], self.motorMap.thtOffsets[c], thtSteps[c])
            nThtSteps[c] = stepsRange[1] - stepsRange[0]

            # Calculate the total number of motor steps for the phi movement
            stepsRange = np.interp([startPhi[c], startPhi[c] + deltaPhi[c]], self.motorMap.phiOffsets[c], phiSteps[c])
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

    def moveXY(self, cobras, startPositions, targetPositions):
        """Move the Cobras in XY coordinate.

        Parameters
        ----------
        cobras: a list of cobras
        targetPositions: numpy array
            A complex numpy array with the target fiber positions.
        startPositions: numpy array
            A complex numpy array with the starting fiber positions.

        If there are more than one possible convertion to theta/phi, this function picks the regular one.
        For better control, the caller should use positionsToAngles to determine which solution is the right one.
        """

        if len(cobras) != len(startPositions):
            raise RuntimeError("number of starting positions must match number of cobras")
        if len(cobras) != len(targetPositions):
            raise RuntimeError("number of target positions must match number of cobras")

        startTht, startPhi, _ = self.positionsToAngles(cobras, startPositions)
        targetTht, targetPhi, _ = self.positionsToAngles(cobras, targetPositions)
        deltaTht = targetTht - startTht
        deltaPhi = targetPhi - startPhi

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
        self.moveThetaPhi(valid_cobras, deltaTht[valids,0], deltaPhi[valids,0], startTht[valids,0], startPhi[valids,0])

    def moveXYfromHome(self, cobras, targetPositions, thetaHome='ccw'):
        """Move the Cobras in XY coordinate from hard stops.

        Parameters
        ----------
        cobras: a list of cobras
        targetPositions: numpy array
            A complex numpy array with the target fiber positions.
        thetaHome: 'ccw'(default) or 'cw' hard stop

        If there are more than one possible convertion to theta/phi, this function picks the regular one.
        For better control, the caller should use positionsToAngles to determine which solution is the right one.
        """

        if len(cobras) != len(targetPositions):
            raise RuntimeError("number of target positions must match number of cobras")
        home = self.dirIds.get(thetaHome, 'ccw')

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
        if home == 'ccw':
            thtHomes = phiHomes
        else:
            cIdx = np.array([self._mapCobraIndex(c) for i,c in enumerate(cobras) if valids[i]])
            thtHomes = (self.calibModel.tht1[cIdx] - self.calibModel.tht0[cIdx]) % (2*np.pi) + (2*np.pi)
        self.logger.info(f"engaged cobras: {[(c.module,c.cobraNum) for c in valid_cobras]}")
        self.logger.info(f"move to: {list(zip(targetTht[valids,0], targetPhi[valids,0]))}")
        self.logger.info(f"move from: {list(zip(thtHomes, phiHomes))}")

        # move cobras by angles
        deltaTht = targetTht[valids,0] - thtHomes
        delPhi = targetPhi[valids,0] - phiHomes
        self.moveThetaPhi(valid_cobras, deltaTht, delPhi, thtHomes, phiHomes)


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

    @classmethod
    def allocateCobraModule(cls, module):
        if module < 1 or module > cls.nModules:
            raise IndexError(f'module numbers are 1..{cls.nModules}')
        cobras = []
        for c in range(1,cls.nCobrasPerModule+1):
            cobras.append(func.Cobra(module, c))

        return cobras

    @classmethod
    def allocateCobraBoard(cls, module, board):
        if module < 1 or module > cls.nModules:
            raise IndexError(f'module numbers are 1..{cls.nModules}')
        if board not in (1,2):
            raise IndexError('board numbers are 1 or 2.')
        cobras = []
        for c in range(1,cls.nCobrasPerModule+1):
            if (c%2 == 1 and board == 1) or (c%2 == 0 and board == 2):
                cobras.append(func.Cobra(module, c))

        return cobras

    def getAllDefinedCobras(self):
        cobras = []
        for i in self.calibModel.findAllCobras():
            c = func.Cobra(self.calibModel.moduleIds[i],
                           self.calibModel.positionerIds[i])
            cobras.append(c)

        return cobras

    def getAllConnectedCobras(self):
        res = func.DIA()

        boards = 0
        for i in range(len(res)):
            boardsInSector = res[i]
            if boards%14 != 0 and boardsInSector != 0:
                raise RuntimeError("sectors are not left-packed with boards.")
            boards += boardsInSector

        cobras = []
        for b in range(1,boards+1):
            mod = (b-1)//2 + 1
            brd = (b-1)%2 + 1
            cobras.extend(self.allocateCobraBoard(mod, brd))

        return cobras
