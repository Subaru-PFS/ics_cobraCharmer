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

    def moveAllThetaPhi(self, cobras, thetaMove, phiMove, phiHome='ccw'):
        nCobras = self.calibModel.nCobras

        phiHomes = np.zeros(nCobras)
        phiMoves = np.zeros(nCobras) + phiMove
        thetaMoves = np.zeros(nCobras) + thetaMove

        thetaSteps, phiSteps = self.motorMap.calculateSteps(thetaMoves, phiHomes, phiMoves)

        print('thetaSteps: ', thetaSteps)
        print('phiSteps: ', phiSteps)

        stepMoves = list(zip(thetaSteps.tolist(), phiSteps.tolist()))

        cIdx = [self._mapCobraIndex(c) for c in cobras]
        cSteps = [stepMoves[i] for i in cIdx]

        self.moveSteps(cobras, cSteps, [('cw', 'cw')]*len(cIdx))

    def moveAllSteps(self, cobras, steps, dirs):
        allSteps = [steps]*len(cobras)
        allDirs = [dirs]*len(cobras)

        self.moveSteps(cobras, allSteps, allDirs)

    def moveSteps(self, cobras, steps, dirs, waitTimes=None):

        if len(cobras) != len(steps):
            raise RuntimeError("number of steps must match number of cobras")
        if len(cobras) != len(dirs):
            raise RuntimeError("number of directions must match number of cobras")
        if waitTimes is not None and len(cobras) != len(waitTimes):
            raise RuntimeError("number of waitTimes must match number of cobras")

        model = self.calibModel

        for c_i, c in enumerate(cobras):
            steps1 = int(steps[c_i][0]), int(steps[c_i][1])
            dirs1 = dirs[c_i]
            en = (steps1[0] != 0, steps1[1] != 0)
            cobraId = self._mapCobraIndex(c)

            if dirs1[0] == 'cw':
                ontime1 = model.motorOntimeFwd1[cobraId]
                offtime1 = model.motorOfftimeFwd1[cobraId]
            elif dirs1[0] == 'ccw':
                ontime1 = model.motorOntimeRev1[cobraId]
                offtime1 = model.motorOfftimeRev1[cobraId]
            else:
                raise ValueError(f'invalid direction: {dirs1[0]}')

            if dirs1[1] == 'cw':
                ontime2 = model.motorOntimeFwd2[cobraId]
                offtime2 = model.motorOfftimeFwd2[cobraId]
            elif dirs1[1] == 'ccw':
                ontime2 = model.motorOntimeRev2[cobraId]
                offtime2 = model.motorOfftimeRev2[cobraId]
            else:
                raise ValueError(f'invalid direction: {dirs1[1]}')

            # For early-late offsets.
            if waitTimes is not None:
                offtime1 = waitTimes[c_i][0]
                offtime2 = waitTimes[c_i][1]
            else:
                offtime1 = offtime2 = 0

            c.p = func.RunParams(pu=(int(1000*ontime1), int(1000*ontime2)),
                                 st=(steps1),
                                 sl=(int(1000*offtime1), int(1000*offtime2)),
                                 en=en,
                                 dir=dirs1)
        err = func.RUN(cobras)

    def homePhi(self, cobras, nsteps=3000, dir='ccw'):
        steps = [(0,nsteps)]*len(cobras)
        dirs = [(dir,dir)]*len(cobras)
        self.moveSteps(cobras, steps, dirs)

    def homeTheta(self, cobras, nsteps=6000, dir='ccw'):
        steps = [(nsteps,0)]*len(cobras)
        dirs = [(dir,dir)]*len(cobras)
        self.moveSteps(cobras, steps, dirs)

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

