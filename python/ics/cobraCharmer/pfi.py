from importlib import reload
import logging
import numpy as np

from ics.cobraCharmer import ethernet
from ics.cobraCharmer import func
from ics.cobraCharmer import log
from ics.cobraCharmer.log import Logger

from ics.cobraCharmer import pfiDesign
from ics.cobraCharmer import fpgaLogger

from ics.cobraCharmer import cobraState

reload(pfiDesign)
reload(func)

class PFI(object):
    nCobrasPerModule = 57
    nModules = 42

    # bit array for (x,y) to (theta, phi) convertion
    SOLUTION_OK              = 0x0001  # 1 if the solution is valid
    IN_OVERLAPPING_REGION    = 0x0002  # 1 if the position in overlapping region
    PHI_NEGATIVE             = 0x0004  # 1 if phi angle is negative(phi CCW limit < 0)
    PHI_BEYOND_PI            = 0x0008  # 1 if phi angle is beyond PI(phi CW limit > PI)
    TOO_CLOSE_TO_CENTER      = 0x0016  # 1 if the position is too close to the center
    TOO_FAR_FROM_CENTER      = 0x0032  # 1 if the position is too far from the center

    # on time vs speed model parameters
    thetaParameter = 0.09
    phiParameter = 0.07

    def __init__(self, fpgaHost='localhost', logDir=None, doConnect=True, doLoadModel=False, debug=False):
        """ Initialize a PFI class
        Args:
           fpgaHost    - fpga device
           logDir      - directory name for logs
           doConnect   - do connection or not
           doLoadModel - load data model or not
        """
        self.logger = Logger.getLogger('fpga', debug)
        self.logger = logging.getLogger('pfi')
        self.logger.setLevel(logging.INFO)
        self.ioLogger = Logger.getLogger('fpgaIO', debug)
        if logDir is not None:
            log.setupLogPaths(logDir)

        self.protoLogger = fpgaLogger.FPGAProtocolLogger(logRoot=logDir)

        self.calibModel = None
        self.motorMap = None
        self.ontimeScales = cobraState.motorScales
        self.maxThetaOntime = 0.14
        self.maxPhiOntime = 0.14
        self.maxThetaSteps = 10000
        self.maxPhiSteps = 7000

        self.flipPowerPolarity = (fpgaHost == 'pfi')
        if fpgaHost in {'fpga', 'pfi'}:
            fpgaHost = '128.149.77.24'  # A JPL address which somehow got burned into the FPGAs. See INSTRM-464
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
        self.protoLogger.logger.info('FPGA connection to {self.fpgaHost}')

    def disconnect(self):
        """ Disconnect from COBRA fpga device """
        ethernet.sock.close()
        ethernet.sock = ethernet.Sock()
        self.ioLogger.info(f'FPGA connection closed')
        self.protoLogger.logger.info('FPGA connection closed')

    def loadModel(self, filename=None, version=None, moduleVersion=None):
        """ Load a motormap XML file. """
        import ics.cobraCharmer.pfiDesign as pfiDesign
        reload(pfiDesign)
        des = pfiDesign.PFIDesign()
        #des.loadModelFiles(filename)
        
        if filename is not None:
            des.loadModelFiles(filename)
            self.calibModel = des
            self.logger.info(f'load cobra model from {filename}')
        else:
            self.calibModel = pfiDesign.PFIDesign.loadPfi(version, moduleVersion)

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
        res = func.DIA()
        self.logger.info("Board Counts: %s" %(res) )
        return res
    
    def admin(self, debugLevel=0):
        """ Set debug level, get version and uptime """
        err, version, uptime = func.ADMIN(debugLevel=debugLevel)
        if err:
            self.logger.error(f'send ADMIN command failed')
        else:
            self.logger.debug(f'send ADMIN command succeeded')
        return version, uptime

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
        ret = func.HK(cobras, updateModel=(self.calibModel if updateModel else None),
                      feedback=True)
        err = ret[0]
        if err:
            self.logger.error(f'send HK command failed: {err}')
        else:
            self.logger.debug(f'send HK command succeeded')

        return ret

    def setFreq(self, cobras=None, thetaFreq=None, phiFreq=None):
        """ Set COBRA motor frequency """
        if cobras is None:
            cobras = self.getAllDefinedCobras()

        if thetaFreq is not None and len(cobras) != len(thetaFreq):
            raise RuntimeError("number of theta frquencies must match number of cobras")
        if phiFreq is not None and len(cobras) != len(phiFreq):
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
            self.logger.debug(f'send SET command succeeded')

    def calibrateFreq(self, cobras=None,
                      thetaLow=60.4, thetaHigh=70.3, phiLow=94.4, phiHigh=108.2,
                      clockwise=True,
                      enabled=(True,True)):
        """ Calibrate COBRA motor frequency """
        if cobras is None:
            cobras = self.getAllDefinedCobras()

        spin = func.CW_DIR if clockwise else func.CCW_DIR
        m0 = (self._freqToPeriod(thetaHigh), self._freqToPeriod(thetaLow))
        m1 = (self._freqToPeriod(phiHigh), self._freqToPeriod(phiLow))
        for c in cobras:
            c.p = func.CalParams(m0=m0, m1=m1, en=enabled, dir=spin)
        err = func.CAL(cobras, timeout=65535)
        if err:
            self.logger.error(f'send Calibrate command failed')
        else:
            self.logger.debug(f'send Calibrate command succeeded')

    def houseKeeping(self, modules=None, m0=(0,1000), m1=(0,1000), temps=(16.0,31.0), cur=(0.25,1.2), volt=(9.5,10.5)):
        """ HK command """

        if modules is None:
            modules = [1]
        elif np.isscalar(modules):
            modules = [modules]

        errors = np.full((len(modules), 2), False)
        temps = np.zeros((len(modules), 2, 2))
        voltages = np.zeros((len(modules), 2))
        freqs1 = np.zeros((len(modules), self.nCobrasPerModule))
        currents1 = np.zeros((len(modules), self.nCobrasPerModule))
        freqs2 = np.zeros((len(modules), self.nCobrasPerModule))
        currents2 = np.zeros((len(modules), self.nCobrasPerModule))

        for k, m in enumerate(modules):
            self.logger.debug(f'HK command for Cobra module #{m}')
            for board in range(2):
                # two boards in one module
                cobra_num = np.arange(board+1, self.nCobrasPerModule+1, 2)
                cobras = self.allocateCobraRange(m, cobra_num)
                for c in cobras:
                    c.p = func.HkParams(m0, m1, temps, cur, volt)
                err, t1, t2, v, f1, c1, f2, c2 = func.HK(cobras, feedback=True)
                errors[k, board] = err
                temps[k, board] = [t1, t2]
                voltages[k, board] = v
                freqs1[k, board::2] = f1
                currents1[k, board::2] = c1
                freqs2[k, board::2] = f2
                currents2[k, board::2] = c2
                if err:
                    self.logger.error(f'Module {m}:{board} send HK command failed')
                else:
                    self.logger.debug(f'Module {m}:{board} send HK command succeeded')
        return errors, temps, voltages, freqs1, currents1, freqs2, currents2

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
        self.logger.debug(f'steps: {list(zip(cThetaSteps, cPhiSteps))}')
        self.moveSteps(cobras, cThetaSteps, cPhiSteps, thetaFast=thetaFast, phiFast=phiFast)

    def moveAllSteps(self, cobras, thetaSteps, phiSteps, thetaFast=True, phiFast=True):
        """ Move all cobras for some theta and phi steps """

        if cobras is None:
            cobras = self.getAllDefinedCobras()
        thetaAllSteps = np.zeros(len(cobras)) + thetaSteps
        phiAllSteps = np.zeros(len(cobras)) + phiSteps

        self.moveSteps(cobras, thetaAllSteps, phiAllSteps, thetaFast=thetaFast, phiFast=phiFast)

    def moveThetaPhi(self, cobras, thetaMoves, phiMoves, thetaFroms=None, phiFroms=None,
                     thetaFast=True, phiFast=True, doRun=True, ccwLimit=True):
        """ Move cobras with theta and phi angles

            thetaMoves: A numpy array with theta angles to go
            phiMoves: A numpy array with phi angles to go
            thetaFroms: A numpy array with starting theta positions
            phiFroms: A numpy array with starting phi positions
            thetaFast: a boolean value for fast/slow theta movement
            phiFast: a boolean value for fast/slow phi movement
            ccwLimit: using theta CCW or CW hard stops if thetaFroms is None
        """

        nCobras = len(cobras)
        if np.ndim(thetaMoves) == 0:
            thetaMoves = np.zeros(nCobras) + thetaMoves
        if np.ndim(phiMoves) == 0:
            phiMoves = np.zeros(nCobras) + phiMoves

        if nCobras != len(thetaMoves):
            raise RuntimeError("number of theta moves must match number of cobras")
        if nCobras != len(phiMoves):
            raise RuntimeError("number of phi moves must match number of cobras")
        if thetaFroms is not None and nCobras != len(thetaFroms):
            raise RuntimeError("number of theta froms must match number of cobras")
        if phiFroms is not None and nCobras != len(phiFroms):
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
        elif not ccwLimit:
            _thetaFroms = (self.calibModel.tht1 - self.calibModel.tht0 + np.pi) % (2*np.pi) + np.pi

        if isinstance(thetaFast, bool):
            _thetaFast = thetaFast
        elif len(thetaFast) == len(cobras):
            _thetaFast = np.full(nCobras, True)
            _thetaFast[cIdx] = thetaFast
        else:
            raise RuntimeError("number of thetaFast must match number of cobras")

        if isinstance(phiFast, bool):
            _phiFast = phiFast
        elif len(phiFast) == len(cobras):
            _phiFast = np.full(nCobras, True)
            _phiFast[cIdx] = phiFast
        else:
            raise RuntimeError("number of phiFast must match number of cobras")

        thetaSteps, phiSteps = self.calculateSteps(_thetaFroms, _thetaMoves, _phiFroms, _phiMoves, _thetaFast, _phiFast)
        cThetaSteps = thetaSteps[cIdx]
        cPhiSteps = phiSteps[cIdx]

        """
        Looking for NaN values and put them as 0
        """
        thetaIndex = np.isnan(cThetaSteps)
        phiIndex = np.isnan(cPhiSteps)
        cThetaSteps[thetaIndex] = 0
        cPhiSteps[phiIndex] = 0

        self.logger.debug(f'steps (run={doRun}): {list(zip(cThetaSteps, cPhiSteps))}')
        if doRun:
            self.moveSteps(cobras, cThetaSteps, cPhiSteps, thetaFast=thetaFast, phiFast=phiFast)

        return cThetaSteps, cPhiSteps

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

    def resetMotorScaling(self, cobras=None, motor=None):
        """ Declare that we want the scaling for some cobras to be reset to 1.0.

        Args
        ----
        cobras : list of `Cobra`s
           The cobras to reset. All if None.
        motor : {`theta`, `phi`}
           The motor to reset. Both if None.
        """

        if cobras is None:
            cobras = self.getAllDefinedCobras()

        if motor is None:
            motors = ['theta', 'phi']
        else:
            motors = [motor]

        for c in cobras:
            for m in motors:
                self.scaleMotorOntime(c, m, 'cw', 1.0, doReset=True)
                self.scaleMotorOntime(c, m, 'ccw', 1.0, doReset=True)

    def scaleMotorOntime(self, cobra, motor, direction, scale, doReset=False):
        """ Declare that we want a given motor's ontime to be scaled after interpolation.

        If there is an existing scaling, the new scaling is applied
        _that_: we are expecting to be told that the last effective
        move neeeded adjustment.

        Args
        ----
        cobra : internal cobra object FIX -- CPL
           A single cobra.
        motor : {'theta', 'phi'}
           Which motor to adjust
        direction : {'ccw', 'cw'}
           Which motor map to adjust
        scale : `float`
           Scaling to apply to the theta motor's ontime
        doReset : `bool`
           Whether to replace the existing scaling or adjust it.
        """

        cobraId = self._mapCobraIndex(cobra)
        mapId = cobraState.mapId(cobraId, motor, direction)
        if not doReset:
            existingScale = cobraState.motorScales.get(mapId, 1.0)
        else:
            existingScale = 1.0

        newScale = existingScale * scale
        if newScale < 0.5:
            self.logger.warn(f'clipping scale adjustment from {newScale} to 0.5')
            newScale = 0.5
        if newScale > 2.0:
            self.logger.warn(f'clipping scale adjustment from {newScale} to 2.0')
            newScale = 2.0

        cobraState.motorScales[mapId] = newScale
        self.logger.debug(f'setadjust {mapId} {existingScale:0.2f} -> {newScale:0.2f}')

    def scaleMotorOntimeBySpeed(self, cobra, motor, direction, fast, scale):
        """ Declare that we want a given motor's ontime to be scaled after interpolation.

        If there is an existing scaling, the new scaling is applied
        _that_: we are expecting to be told that the last effective
        move neeeded adjustment.

        Args
        ----
        cobra : internal cobra object FIX -- CPL
           A single cobra.
        motor : {'theta', 'phi'}
           Which motor to adjust
        direction : {'ccw', 'cw'}
           Which motor map to adjust
        scale : `float`
           Scaling for the motor speed
        fast : `bool`
           Using fast or slow motor map
        """

        cobraId = self._mapCobraIndex(cobra)
        mapId = cobraState.mapId(cobraId, motor, direction)
        existingScale = cobraState.motorScales.get(mapId, 1.0)

        if scale <= 0:
            self.logger.warn(f'scale is negative, give up scaling: {scale}')
            return cobraState.motorScales[mapId]

        if motor == 'theta':
            b0 = self.thetaParameter
            if fast:
                if direction == 'cw':
                    ontime = self.calibModel.motorOntimeFwd1[cobraId]
                else:
                    ontime = self.calibModel.motorOntimeRev1[cobraId]
            else:
                if direction == 'cw':
                    ontime = self.calibModel.motorOntimeSlowFwd1[cobraId]
                else:
                    ontime = self.calibModel.motorOntimeSlowRev1[cobraId]
            nowOntime = existingScale * ontime
            if nowOntime > self.maxThetaOntime:
                nowOntime = self.maxThetaOntime
        else:
            b0 = self.phiParameter
            if fast:
                if direction == 'cw':
                    ontime = self.calibModel.motorOntimeFwd2[cobraId]
                else:
                    ontime = self.calibModel.motorOntimeRev2[cobraId]
            else:
                if direction == 'cw':
                    ontime = self.calibModel.motorOntimeSlowFwd2[cobraId]
                else:
                    ontime = self.calibModel.motorOntimeSlowRev2[cobraId]
            nowOntime = existingScale * ontime
            if nowOntime > self.maxPhiOntime:
                nowOntime = self.maxPhiOntime

        a0 = np.sqrt(nowOntime*nowOntime + b0*b0) - b0
        a1 = a0*scale + b0
        newOntime = np.sqrt(a1*a1 - b0*b0)
        if not np.isfinite(newOntime):
            self.logger.warn(f'invalid scaling adjustment, give up')
            return existingScale

        newScale = newOntime / ontime
        if newScale < 0.3:
            self.logger.warn(f'clipping scale adjustment from {newScale} to 0.3')
            newScale = 0.3
        if newScale > 3.0:
            self.logger.warn(f'clipping scale adjustment from {newScale} to 3.0')
            newScale = 3.0

        cobraState.motorScales[mapId] = newScale
        self.logger.debug(f'setadjust {mapId} {existingScale:0.2f} -> {newScale:0.2f}')

        return newScale

    def adjustThetaOnTime(self, cobraId, ontime, fast, direction):
        mapId = cobraState.mapId(cobraId, 'theta', direction)
        scale = cobraState.motorScales.get(mapId, 1.0)
        self.logger.debug(f'adjust {mapId} {scale:0.2f}')

        newOntime = ontime*scale
        if newOntime > self.maxThetaOntime:
            newOntime = self.maxThetaOntime
            cobraState.motorScales[mapId] = newOntime/ontime
            self.logger.warn(f'clipping {mapId} ontime to {newOntime} and '
                             f'scale {scale:0.2f} to {cobraState.motorScales[mapId]}')

        return newOntime

    def adjustPhiOnTime(self, cobraId, ontime, fast, direction):
        mapId = cobraState.mapId(cobraId, 'phi', direction)
        scale = cobraState.motorScales.get(mapId, 1.0)
        self.logger.debug(f'adjust {mapId} {scale:0.2f}')

        newOntime = ontime*scale
        if newOntime > self.maxPhiOntime:
            newOntime = self.maxPhiOntime
            cobraState.motorScales[mapId] = newOntime/ontime
            self.logger.warn(f'clipping {mapId} ontime to {newOntime} and '
                             f'scale {scale:0.2f} to {cobraState.motorScales[mapId]}')

        return newOntime

    def moveSteps(self, cobras, thetaSteps, phiSteps, waitThetaSteps=None, waitPhiSteps=None,
                  interval=2.5, thetaFast=True, phiFast=True, force=False):
        """ Move cobras with theta and phi steps

            thetaSteps: A numpy array with theta steps to go
            phiSteps: A numpy array with phi steps to go
            waitThetaSteps: A numpy array for theta delay
            waitPhiSteps: A numpy array with phi delay
            interval: the FPGA interval parameter for RUN command
            thetaFast: a boolean value for using fast/slow theta movement
            phiFast: a boolean value for using fast/slow phi movement
            force: ignore disabled cobra status

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

        if len(cobras) == 0:
            self.logger.debug(f'skipping RUN command: no cobras')
            return

        maxStep = int(np.amax(np.abs((thetaSteps, phiSteps))))
        for c_i, c in enumerate(cobras):
            cobraId = self._mapCobraIndex(c)
            steps1 = [int(np.abs(thetaSteps[c_i])), int(np.abs(phiSteps[c_i]))]
            if steps1[0] > self.maxThetaSteps:
                self.logger.warn(f'clipping #{c_i+1} theta steps from {steps1[0]} to {self.maxThetaSteps}')
                steps1[0] = self.maxThetaSteps
            if steps1[1] > self.maxPhiSteps:
                self.logger.warn(f'clipping #{c_i+1} phi steps from {steps1[1]} to {self.maxPhiSteps}')
                steps1[1] = self.maxPhiSteps

            dirs1 = ['cw', 'cw']

            if thetaSteps[c_i] < 0:
                dirs1[0] = 'ccw'
            if phiSteps[c_i] < 0:
                dirs1[1] = 'ccw'

            en = (steps1[0] != 0, steps1[1] != 0)
            isBad = self.calibModel.cobraIsBad(c.cobraNum, c.module)
            if isBad and not force:
                en = (False, False)

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
                ontime1 = self.adjustThetaOnTime(cobraId, ontime1, fast=_thetaFast[c_i], direction=dirs1[0])

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
                ontime2 = self.adjustPhiOnTime(cobraId, ontime2, fast=_phiFast[c_i], direction=dirs1[1])

            offtime1 = 0
            offtime2 = 0
            if waitThetaSteps is None and waitPhiSteps is None:
                # set delay parameters for safer operation
                if phiSteps[c_i] > 0 and thetaSteps[c_i] < 0:
                    offtime1 = maxStep + thetaSteps[c_i]
                    offtime2 = maxStep - phiSteps[c_i]
                elif phiSteps[c_i] > 0 and thetaSteps[c_i] > 0:
                    offtime2 = maxStep - phiSteps[c_i]
                elif phiSteps[c_i] < 0 and thetaSteps[c_i] < 0:
                    offtime1 = maxStep + thetaSteps[c_i]
            else:
                # For early-late offsets.
                if waitThetaSteps is not None:
                    offtime1 = waitThetaSteps[c_i]

                if waitPhiSteps is not None:
                    offtime2 = waitPhiSteps[c_i]

            #self.logger.info(f'{c_i} {cobraId} {ontime1} {ontime2}')
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
            self.logger.debug(f'send RUN command succeeded')

    def homePhi(self, cobras, nsteps=-5000, fast=True):
        # go to the hard stops for phi arms

        thetaSteps = np.zeros(len(cobras))
        phiSteps = np.zeros(len(cobras)) + nsteps
        self.moveSteps(cobras, thetaSteps, phiSteps, phiFast=fast)

    def homeThetaPhi(self, cobras, thetaSteps=10000, phiSteps=-5000, thetaFast=True, phiFast=True):
        """ go to the hard stops fir both theta and phi arms
            Default: theta(CW) and phi(CCW) arms moves in oppsosite directions.
        """

        self.moveAllSteps(cobras, thetaSteps, phiSteps, thetaFast=thetaFast, phiFast=phiFast)

    def homeTheta(self, cobras, nsteps=-10000, fast=True):
        # go to the hard stops for theta arms

        thetaSteps = np.zeros(len(cobras)) + nsteps
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
            stepsRange = np.interp([startTht[c], startTht[c] + deltaTht[c]], self.calibModel.thtOffsets[c],
                                   thtSteps)
            if not np.all(np.isfinite(stepsRange)):
                raise ValueError(f"theta angle to step interpolation out of range: "
                                 f"{c} {startTht[c]} {startTht[c] + deltaTht[c]}")
            nThtSteps[c] = np.rint(stepsRange[1] - stepsRange[0]).astype('i4')

            # Calculate the total number of motor steps for the phi movement
            stepsRange = np.interp([startPhi[c], startPhi[c] + deltaPhi[c]], self.calibModel.phiOffsets[c],
                                   phiSteps)
            if not np.all(np.isfinite(stepsRange)):
                raise ValueError(f"phi angle to step interpolation out of range: "
                                 f"{startTht[c]} {startTht[c] + deltaTht[c]}")
            nPhiSteps[c] = np.rint(stepsRange[1] - stepsRange[0]).astype('i4')

        self.logger.debug(f'start={startPhi[:3]}, delta={deltaPhi[:3]} move={nPhiSteps[:3]}')
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
        thtRange = (self.calibModel.tht1[cIdx] - self.calibModel.tht0[cIdx] + np.pi) % (2*np.pi) + np.pi
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
            A python tuples with all the possible angles (theta, phi, flags).
            Since there are possible 2 phi solutions (since phi CCW<0 and CW>PI)
            so the dimensions of theta and phi are (len(cobras), 2), the value
            np.nan indicates there is no solution. flags is a bit map.

            There are several different cases:
            - No solution: This means the distance from the given position to
              the center and two arm lengths(theta, phi) can't form a triangle.
              In this case, either TOO_CLOSE_TO_CENTER or TOO_FAR_FROM_CENTER
              is set. For TOO_CLOSE_TO_CENTER case, phi is set to 0 and for
              TOO_FAR_FROM_CENTER case, phi is set to PI, theta is set to
              the angle from the center to the given position for both cases.
            - Two phi solutions: This happens because the range of phi arms can
              be negative and beyond PI. When the measured phi angle is small or
              close to PI, this case may happen. The second solution is also
              calculated and returned. The bit PHI_NEGATIVE or PHI_BEYOND_PI is
              set in this situation. If this solution is within the hard stops,
              the bit SOLUTION_OK is set.
            - Theta overlapping region: Since theta arms can move beyond PI*2,
              so in the overlapping region(between two hard stops) we have two
              possible theta solutions. The bit IN_OVERLAPPING_REGION is set.
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
        flags = np.full((len(cobras), 2), 0, dtype='u2')

        for i in range(len(positions)):
            if distance[i] > L1[i] + L2[i]:
                # too far away, return theta= spot angle and phi=PI
                flags[i][0] |= self.TOO_FAR_FROM_CENTER
                phi[i][0] = np.pi
                tht[i][0] = (np.angle(relativePositions[i]) - tht0[i]) % (2 * np.pi)
                if tht[i][0] <= (tht1[i] - tht0[i]) % (2 * np.pi):
                    flags[i][0] |= self.IN_OVERLAPPING_REGION
                continue
            if distance[i] < np.abs(L1[i] - L2[i]):
                # too close to center, theta is undetermined, return theta=spot angle and phi=0
                flags[i][0] |= self.TOO_CLOSE_TO_CENTER
                phi[i][0] = 0
                tht[i][0] = (np.angle(relativePositions[i]) - tht0[i]) % (2 * np.pi)
                if tht[i][0] <= (tht1[i] - tht0[i]) % (2 * np.pi):
                    flags[i][0] |= self.IN_OVERLAPPING_REGION
                continue

            ang1 = np.arccos((L1Sq[i] + L2Sq[i] - distanceSq[i]) / (2 * L1[i] * L2[i]))
            ang2 = np.arccos((L1Sq[i] + distanceSq[i] - L2Sq[i]) / (2 * L1[i] * distance[i]))

            # the regular solutions, phi angle is between 0 and pi, no checking for phi hard stops
            flags[i][0] |= self.SOLUTION_OK
            phi[i][0] = ang1 - phiIn[i]
            tht[i][0] = (np.angle(relativePositions[i]) + ang2 - tht0[i]) % (2 * np.pi)
            # check if tht is within two theta hard stops
            if tht[i][0] <= (tht1[i] - tht0[i]) % (2 * np.pi):
                flags[i][0] |= self.IN_OVERLAPPING_REGION

            # check if there are additional solutions
            if ang1 <= np.pi/2 and ang1 > 0:
                if phiIn[i] <= -ang1:
                    flags[i][1] |= self.SOLUTION_OK
                flags[i][1] |= self.PHI_NEGATIVE
                # phiIn < 0
                phi[i][1] = -ang1 - phiIn[i]
                tht[i][1] = (np.angle(relativePositions[i]) - ang2 - tht0[i]) % (2 * np.pi)
                # check if tht is within two theta hard stops
                if tht[i][1] <= (tht1[i] - tht0[i]) % (2 * np.pi):
                    flags[i][1] |= self.IN_OVERLAPPING_REGION
            elif ang1 > np.pi/2 and ang1 < np.pi:
                if phiOut[i] >= 2 * np.pi - ang1:
                    flags[i][1] |= self.SOLUTION_OK
                flags[i][1] |= self.PHI_BEYOND_PI
                # phiOut > np.pi
                phi[i][1] = 2 * np.pi - ang1 - phiIn[i]
                tht[i][1] = (np.angle(relativePositions[i]) - ang2 - tht0[i]) % (2 * np.pi)
                # check if tht is within two theta hard stops
                if tht[i][1] <= (tht1[i] - tht0[i]) % (2 * np.pi):
                    flags[i][1] |= self.IN_OVERLAPPING_REGION
        return (tht, phi, flags)

    def moveXY(self, cobras, startPositions, targetPositions, overlappingCW=False,
               thetaThreshold=1e10, phiThreshold=1e10, delta=10.0):
        """Move the Cobras in XY coordinate.

        Parameters
        ----------
        cobras: a list of cobras
        targetPositions: numpy array
            A complex numpy array with the target fiber positions.
        startPositions: numpy array
            A complex numpy array with the starting fiber positions.
        overlappingCW: a boolean value
            use the theta solution that's close to CW limit if targetPosition is in overlapping region
        thetaThreshold: a double value
            The threshold value for using slow/fast theta motor maps, the default is slow.
        phiThreshold: a double value
            The threshold value for using slow/fast phi motor maps, the default is slow.
        delta: a double value (in degree)
            The tolerance which is used to identify if startPosition is in overlapping region or not.

        If there are more than one possible convertion to theta/phi, this function picks the regular one.
        For better control, the caller should use positionsToAngles to determine which solution is the right one.
        """

        if len(cobras) != len(startPositions):
            raise RuntimeError("number of starting positions must match number of cobras")
        if len(cobras) != len(targetPositions):
            raise RuntimeError("number of target positions must match number of cobras")
        delta = np.deg2rad(delta)

        startTht, startPhi, _ = self.positionsToAngles(cobras, startPositions)
        targetTht, targetPhi, _ = self.positionsToAngles(cobras, targetPositions)

        valids = np.all([np.isnan(startTht[:,0]) == False, np.isnan(targetTht[:,0]) == False], axis=0)
        valid_cobras = cobras[valids]
        if len(valid_cobras) <= 0:
            raise RuntimeError("no valid target positions are found")
        elif not np.all(valids):
            self.logger.info("some target positions are invalid")

        cIdx = [self._mapCobraIndex(c) for c in cobras]
        gapTht = (self.calibModel.tht1[cIdx] - self.calibModel.tht0[cIdx] + np.pi) % (2*np.pi) - np.pi
        for c_i in np.where(valids)[0]:
            if targetTht[c_i,0] < gapTht[c_i] and overlappingCW:
                targetTht[c_i,0] += np.pi*2
            if startTht[c_i,0] < gapTht[c_i] + delta and targetTht[c_i,0] > np.pi:
                startTht[c_i,0] += np.pi*2
            elif startTht[c_i,0] > np.pi*2 - delta and targetTht[c_i,0] < np.pi:
                startTht[c_i,0] -= np.pi*2

        deltaTht = targetTht[valids,0] - startTht[valids,0]
        deltaPhi = targetPhi[valids,0] - startPhi[valids,0]
        thetaFast = np.zeros(len(valid_cobras), 'bool')
        thetaFast[np.abs(deltaTht) > thetaThreshold] = True
        phiFast = np.zeros(len(valid_cobras), 'bool')
        phiFast[np.abs(deltaPhi) > phiThreshold] = True

        # move bobras by angles
        with np.printoptions(precision=2, suppress=True):
            self.logger.info(f"engaged cobras: {[(c.module,c.cobraNum) for c in valid_cobras]}")
            self.logger.info(f"move to: {np.stack((targetTht[valids,0], targetPhi[valids,0]))}")
            self.logger.info(f"move from: {np.stack((startTht[valids,0], startPhi[valids,0]))}")
        self.moveThetaPhi(valid_cobras, deltaTht, deltaPhi, startTht[valids,0], startPhi[valids,0], thetaFast, phiFast)

    def moveXYfromHome(self, cobras, targetPositions, ccwLimit=True, thetaThreshold=-1.0, phiThreshold=-1.0):
        """Move the Cobras in XY coordinate from hard stops.

        Parameters
        ----------
        cobras: a list of cobras
        targetPositions: numpy array
            A complex numpy array with the target fiber positions.
        ccwLimit: 'ccw'(default) or 'cw' hard stop for current theta arms
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
        valid_cobras = cobras[valids]
        if len(valid_cobras) <= 0:
            raise RuntimeError("no valid target position found")
        elif not np.all(valids):
            self.logger.warn("some target positions are invalid")

        # define home positions
        phiHomes = np.zeros(len(valid_cobras))
        if ccwLimit:
            thtHomes = phiHomes
        else:
            cIdx = np.array([self._mapCobraIndex(c) for c in valid_cobras])
            thtHomes = (self.calibModel.tht1[cIdx] - self.calibModel.tht0[cIdx] + np.pi) % (2*np.pi) + np.pi
        self.logger.info(f"engaged cobras: {[(c.module,c.cobraNum) for c in valid_cobras]}")
        self.logger.info(f"move to: {list(zip(targetTht[valids,0], targetPhi[valids,0]))}")
        self.logger.info(f"move from: {list(zip(thtHomes, phiHomes))}")

        # move cobras by angles
        deltaTht = targetTht[valids,0] - thtHomes
        deltaPhi = targetPhi[valids,0] - phiHomes
        thetaFast = np.full(len(valid_cobras), True)
        thetaFast[np.abs(deltaTht) < thetaThreshold] = False
        phiFast = np.full(len(valid_cobras), True)
        phiFast[deltaPhi < phiThreshold] = False
        self.moveThetaPhi(valid_cobras, deltaTht, deltaPhi, thtHomes, phiHomes, thetaFast, phiFast)

    def moveXYfromHomeSafe(self, cobras, targetPositions, ccwLimit=False):
        """Move the Cobras in XY coordinate from hard stops in the safe way.
           Phi motors should be twice as fast as theta motors and in opposite direction so the tips go in
           straight lines. Using phi: fast on-time, theta: slow on-time setting.

        Parameters
        ----------
        cobras: a list of cobras
        targetPositions: numpy array
            A complex numpy array with the target fiber positions.
        ccwLimit: 'cw'(default) or 'ccw' hard stop for current theta arms

        If there are more than one possible convertion to theta/phi, this function picks the regular one.
        For better control, the caller should use positionsToAngles to determine which solution is the right one.
        """
        if ccwLimit:
            raise RuntimeError("should be in theta CW limits, stop here!!!")

        if len(cobras) != len(targetPositions):
            raise RuntimeError("number of target positions must match number of cobras")

        targetTht, targetPhi, _ = self.positionsToAngles(cobras, targetPositions)

        # check if there is a solution
        valids = np.isnan(targetTht[:,0]) == False
        valid_cobras = cobras[valids]
        if len(valid_cobras) <= 0:
            raise RuntimeError("no valid target positions are found")
        elif not np.all(valids):
            self.logger.warn("some target positions are invalid")

        # define home positions
        phiHomes = np.zeros(len(valid_cobras))
        cIdx = np.array([self._mapCobraIndex(c) for c in valid_cobras])
        thtHomes = (self.calibModel.tht1[cIdx] - self.calibModel.tht0[cIdx]) % (2*np.pi) + (2*np.pi)
        with np.printoptions(precision=2, suppress=True):
            self.logger.info(f"engaged cobras: {[(c.module,c.cobraNum) for c in valid_cobras]}")
            self.logger.info(f"move to: {np.stack((targetTht[valids,0], targetPhi[valids,0]))}")
            self.logger.info(f"move from: {np.stack((thtHomes, phiHomes))}")

        # move cobras by angles
        deltaTht = targetTht[valids,0] - thtHomes
        deltaPhi = targetPhi[valids,0] - phiHomes
        self.moveThetaPhi(valid_cobras, deltaTht, deltaPhi, thtHomes, phiHomes, thetaFast=False, phiFast=True)


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
    def allocateCobraModule(cls, moduleId=1):
        moduleId = pfiDesign.PFIDesign.getRealModuleId(moduleId)
        cobraIds = range(1,cls.nCobrasPerModule+1)
        cobras = []
        for c in cobraIds:
            cobras.append(func.Cobra(moduleId, c))

        return cobras

    @classmethod
    def allocateCobraBoard(cls, module, board):
        module = pfiDesign.PFIDesign.getRealModuleId(module)
        if module < 1 or module > cls.nModules:
            raise IndexError(f'module numbers are 1..{cls.nModules}')
        if board not in (1,2):
            raise IndexError('board numbers are 1 or 2.')
        cobras = []
        for c in range(1,cls.nCobrasPerModule+1):
            if (c%2 == 1 and board == 1) or (c%2 == 0 and board == 2):
                cobras.append(func.Cobra(module, c))

        return cobras

    @classmethod
    def allocateBoardCobra(cls, module, board, cobra):
        module = pfiDesign.PFIDesign.getRealModuleId(module)
        if module < 1 or module > cls.nModules:
            raise IndexError(f'module numbers are 1..{cls.nModules}')

        cobras = []
        if board == 1:
            cobras.append(func.Cobra(module, cobra*2-1))
        elif board == 2:
            cobras.append(func.Cobra(module, cobra*2))
        else:
            raise IndexError('board numbers are 1 or 2.')

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

        if boards % 2 != 0:
            raise RuntimeError("number of boards are not even.")

        cobras = self.allocateCobraRange(range(1,boards//2+1))

        return cobras


    @classmethod
    def setModelParameters(cls, thetaParameter=None, phiParameter=None):
        if thetaParameter is not None:
            cls.thetaParameter = thetaParameter
        if phiParameter is not None:
            cls.phiParameter = phiParameter

    @classmethod
    def getModelParameters(cls, thetaParameter=None, phiParameter=None):
        return cls.thetaParameter, cls.phiParameter
