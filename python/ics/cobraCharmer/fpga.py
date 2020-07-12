from importlib import reload
import logging

import numpy as np

from pfs.utils import fiberids

class FPGA(object):
    def __init__(self, fpgaHost='localhost', logDir=None, doConnect=True, debug=False):
        """ Initialize a PFI class
        Args:
           fpgaHost    - fpga device
           logDir      - directory name for logs
           doConnect   - do connection or not
        """

        self.logger = logging.getLogger('fpga')
        self.logger.setLevel(logging.INFO)
        if logDir is not None:
            log.setupLogPaths(logDir)

        self.protoLogger = fpgaLogger.FPGAProtocolLogger(logRoot=logDir)

    def moveSteps(self, cobras, thetaMoves, phiMoves, interval=2.5, force=False):
        """ Move cobras with theta and phi steps and ontimes

        Args
        ----
        thetaSteps: A numpy array with theta steps to go
            phiSteps: A numpy array with phi steps to go
            waitThetaSteps: A numpy array for theta delay
            waitPhiSteps: A numpy array with phi delay
            interval: the FPGA interval parameter for RUN command
            thetaFast: a boolean value for using fast/slow theta movement
            phiFast: a boolean value for using fast/slow phi movement
            force: ignore disabled cobra status

        """

        if len(cobras) != len(thetaMoves):
            raise RuntimeError("number of theta steps must match number of cobras")
        if len(cobras) != len(phiMoves):
            raise RuntimeError("number of phi steps must match number of cobras")

        if len(cobras) == 0:
            self.logger.debug(f'skipping RUN command: no cobras')
            return

        for c_i, c in enumerate(cobras):
            thetaSteps, thetaOntime, thetaWaitSteps = thetaMoves[c_i]
            phiSteps, phiOntime, phiWaitSteps = phiMoves[c_i]

            if np.abs(thetaSteps) > self.maxThetaSteps:
                newSteps = np.sign(thetaSteps) * self.maxThetaSteps
                self.logger.warn(f'clipping #{c_i+1} theta steps from {thetaSteps} to {newSteps}')
                thetaSteps = newSteps
            if np.abs(phiSteps) > self.maxPhiSteps:
                newSteps = np.sign(phiSteps) * self.maxPhiSteps
                self.logger.warn(f'clipping #{c_i+1} phi steps from {phiSteps} to {newSteps}')
                phiSteps = newSteps

            dirs1 = ['cw', 'cw']

            if thetaSteps < 0:
                dirs1[0] = 'ccw'
            if phiSteps < 0:
                dirs1[1] = 'ccw'

            en = [thetaSteps != 0, phiSteps != 0]
            okMotors = c.usableStatus()
            if force:
                okMotors = (True, True)
            en = (en[0] and okMotors[0],
                  en[1] and okMotors[1])

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
            self.logger.debug(f'send RUN command succeeded')
