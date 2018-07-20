from importlib import reload

import numpy as np

from ics.cobraCharmer import ethernet
from ics.cobraCharmer import func

class PFI(object):
    CW = 1
    CCW = -1
    DISABLE = 0

    nCobrasPerModule = 57
    nModules = 42

    def __init__(self, fpgaHost='localhost', doConnect=True, doLoadModel=True):
        """ Initialize a PFI class
        Args:
           fpgaHost    - fpga device
           doConnect   - do connection or not
           doLoadModel - load data model or not
        """

        self.fpgaHost = fpgaHost
        if doConnect:
            self.connect()
        if doLoadModel:
            self.loadModel()

    def connect(self):
        """ Connect to COBRA fpga device """
        ethernet.sock.connect(self.fpgaHost, 4001)

    def disconnect(self):
        """ Disconnect from COBRA fpga device """
        ethernet.sock.close()
        ethernet.sock = ethernet.Sock()

    def loadModel(self, filename=None):
        """ Load a motormap XML file. """

        import ics.cobraOps.CobrasCalibrationProduct as cobraCalib
        import ics.cobraOps.MotorMapGroup as cobraMotorMap
        reload(cobraCalib)
        reload(cobraMotorMap)


        if filename is None:
            filename = "../xml/updatedLinksAndMaps.xml"

        self.calibModel = cobraCalib.CobrasCalibrationProduct(filename)
        self.motorMap = cobraMotorMap.MotorMapGroup(self.calibModel.nCobras)

        self.motorMap.useCalibrationProduct(self.calibModel)

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

    def power(self, sectors=0x3f):
        """ Set COBRA PSU on/off """
        err = func.POW(sectors)

    def setFreq(self, cobras):
        """ Set COBRA motor frequency """
        for c in cobras:
            cobraIdx = self._mapCobraIndex(c)
            thetaPer = self._freqToPeriod(self.calibModel.motorFreq1[cobraIdx]/1000)
            phiPer = self._freqToPeriod(self.calibModel.motorFreq2[cobraIdx]/1000)

            # print(f'set {c.board},{c.cobra} to {thetaPer},{phiPer} {self.calibModel.motorFreq1[c.cobra]}')
            c.p = func.SetParams(p0=thetaPer, p1=phiPer, en=(True, True))
        err = func.SET(cobras)

    def moveAllThetaPhi(self, cobras, thetaMove, phiMove):
        """ Move all cobras by theta and phi angles from home """
        nCobras = self.calibModel.nCobras

        thetaHomes = np.zeros(nCobras)
        phiHomes = np.zeros(nCobras)
        phiMoves = np.zeros(nCobras) + phiMove
        thetaMoves = np.zeros(nCobras) + thetaMove

        thetaSteps, phiSteps = self.calculateSteps(thetaHomes, thetaMoves, phiHomes, phiMoves)

        cIdx = [self._mapCobraIndex(c) for c in cobras]
        cThetaSteps = thetaSteps[cIdx]
        cPhiSteps = phiSteps[cIdx]

        print('thetaSteps: ', cThetaSteps)
        print('phiSteps: ', cPhiSteps)

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
        """ Move cobras with theta and phi angles

            thetaMoves: A numpy array with theta angles
            phiMoves: A numpy array with phi angles
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
            if phiFroms != None:
                _phiFroms[c] = phiFroms[i]
            if thetaFroms != None:
                _thetaFroms[c] = thetaFroms[i]

        thetaSteps, phiSteps = self.calculateSteps(_thetaFroms, _thetaMoves, _phiFroms, _phiMoves)

        cThetaSteps = thetaSteps[cIdx]
        cPhiSteps = phiSteps[cIdx]

        print('thetaSteps: ', cThetaSteps)
        print('phiSteps: ', cPhiSteps)

        self.moveSteps(cobras, cThetaSteps, cPhiSteps)

    def thetaToGlobal(self, cobras, thetaLocals):
        """ Convert theta angles from relative to CCW hard stops to global coordinate """

        if len(cobras) != len(thetaLocals):
            raise RuntimeError("number of theta angles must match number of cobras")

        thetaGlobals = np.zeros(len(cobras))
        cIdx = [self._mapCobraIndex(c) for c in cobras]
        for i, c in enumerate(cIdx):
            thetaGlobals[i] = (thetaLocals[i] + self.calibModel.tht0[c]) % (2 * np.pi)
        return thetaGlobals

    def thetaToLocal(self, cobras, thetaGlobals):
        """ Convert theta angles from global coordinate to relative to CCW hard stop
            Be careful of the overlapping region between two hard stops
        """

        if len(cobras) != len(thetaGlobals):
            raise RuntimeError("number of theta angles must match number of cobras")

        thetaLocals = np.zeros(len(cobras))
        cIdx = [self._mapCobraIndex(c) for c in cobras]
        for i, c in enumerate(cIdx):
            thetaLocals[i] = (thetaGlobals[i] - self.calibModel.tht0[c]) % (2 * np.pi)
        return thetaLocals

    def moveSteps(self, cobras, thetaSteps, phiSteps, dirs=None, waitThetaSteps=None, waitPhiSteps=None):
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
                dirs1 = dirs[c_i]
            else:
                dirs1 = ['', '']
                if thetaSteps[c_i] >= 0:
                    dirs1[0] = 'cw'
                else:
                    dirs1[0] = 'ccw'
                if phiSteps[c_i] >= 0:
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
        err = func.RUN(cobras)

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
        cobra fibers the given theta and phi delta angles.

        Parameters
        ----------
        startTht: object
            A numpy array with the starting theta angle positions.
        deltaTht: object
            A numpy array with the theta delta offsets relative to the home
            positions.
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

        # Calculate the theta and phi offsets relative to the home positions
        thtOffset = startTht % 360.0
        phiOffset = startPhi % 360.0

        # Calculate the total number of motor steps for each angle
        nThtSteps = np.empty(self.motorMap.nMaps)
        nPhiSteps = np.empty(self.motorMap.nMaps)

        for c in range(self.motorMap.nMaps):
            # Calculate the total number of motor steps for the theta movement
            stepsRange = np.interp([thtOffset[c], thtOffset[c] + deltaTht[c]], self.motorMap.thtOffsets[c], thtSteps[c])
            nThtSteps[c] = stepsRange[1] - stepsRange[0]

            # Calculate the total number of motor steps for the phi movement
            stepsRange = np.interp([phiOffset[c], phiOffset[c] + deltaPhi[c]], self.motorMap.phiOffsets[c], phiSteps[c])
            nPhiSteps[c] = stepsRange[1] - stepsRange[0]

        return (nThtSteps, nPhiSteps)

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
