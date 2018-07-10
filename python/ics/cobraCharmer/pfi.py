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
        self.fpgaHost = fpgaHost
        if doConnect:
            self.connect()
        if doLoadModel:
            self.loadModel()

    def connect(self):
        ethernet.sock.connect(self.fpgaHost, 4001)

    def disconnect(self):
        ethernet.sock.close()
        ethernet.sock = ethernet.Sock()

    def loadModel(self, filename=None):
        """ Load a motormap XML file. """

        import ics.cobraOps.CobrasCalibrationProduct as cobraCalib
        import ics.cobraOps.MotorMapGroup as cobraMotorMap
        reload(cobraCalib)
        reload(cobraMotorMap)


        if filename is None:
            filename = "/Users/cloomis/Sumire/PFS/git/ics_cobraOps/python/ics/cobraOps/usedXMLFile.xml"

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
        err = func.RST(sectors)

    def powerCycle(self, sectors=0x3f):
        err = func.POW(sectors)

    def setFreq(self, cobras):
        for c in cobras:
            cobraIdx = self._mapCobraIndex(c)
            thetaPer = self._freqToPeriod(self.calibModel.motorFreq1[cobraIdx]/1000)
            phiPer = self._freqToPeriod(self.calibModel.motorFreq2[cobraIdx]/1000)

            # print(f'set {c.board},{c.cobra} to {thetaPer},{phiPer} {self.calibModel.motorFreq1[c.cobra]}')
            c.p = func.SetParams(p0=thetaPer, p1=phiPer, en=(True, True))
        err = func.SET(cobras)

    def moveAllThetaPhi(self, cobras, thetaMove, phiMove):
        nCobras = self.calibModel.nCobras

        phiHomes = np.zeros(nCobras)
        phiMoves = np.zeros(nCobras) + phiMove
        thetaMoves = np.zeros(nCobras) + thetaMove

        thetaSteps, phiSteps = self.motorMap.calculateSteps(thetaMoves, phiHomes, phiMoves)

        print('thetaSteps: ', thetaSteps)
        print('phiSteps: ', phiSteps)

        cIdx = [self._mapCobraIndex(c) for c in cobras]
        cThetaSteps = thetaSteps[cIdx]
        cPhiSteps = phiSteps[cIdx]

        self.moveSteps(cobras, cThetaSteps, cPhiSteps, [('cw', 'cw')]*len(cIdx))

    def moveAllSteps(self, cobras, thetaSteps, phiSteps, dirs):
        thetaAllSteps = np.zeros(len(cobras)) + thetaSteps
        phiAllSteps = np.zeros(len(cobras)) + phiSteps
        allDirs = [dirs]*len(cobras)

        self.moveSteps(cobras, thetaAllSteps, phiAllSteps, allDirs)

    def moveSteps(self, cobras, thetaSteps, phiSteps, dirs, waitThetaSteps=None, waitPhiSteps=None):

        if len(cobras) != len(thetaSteps):
            raise RuntimeError("number of theta steps must match number of cobras")
        if len(cobras) != len(phiSteps):
            raise RuntimeError("number of phi steps must match number of cobras")
        if len(cobras) != len(dirs):
            raise RuntimeError("number of directions must match number of cobras")
        if waitThetaSteps is not None and len(cobras) != len(waitThetaSteps):
            raise RuntimeError("number of waitThetaSteps must match number of cobras")
        if waitPhiSteps is not None and len(cobras) != len(waitPhiSteps):
            raise RuntimeError("number of waitPhiSteps must match number of cobras")

        model = self.calibModel

        for c_i, c in enumerate(cobras):
            steps1 = int(thetaSteps[c_i]), int(phiSteps[c_i])
            dirs1 = dirs[c_i]
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
