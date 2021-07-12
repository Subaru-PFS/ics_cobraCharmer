import numpy as np


class FpgaState(object):
    """ Track state of FPGA. Currently just tracks the last move made, per-cobra """

    def __init__(self):
        self.cobraMoves = dict()

    def cobraId(self, cobra):
        """ Return an indexable id for a cobra. """
        return (cobra.module, cobra.cobraNum)

    def clearMoves(self):
        """ Declare a new move, by setting all known cobras to not have move info.

        The expectation is that all _moving_ cobras will be set with .runCobra
        """
        for cm in self.cobraMoves.values():
            cm['thetaSteps'] = np.nan
            cm['phiSteps'] = np.nan

    def _blankMove(self):
        cobraInfo = dict()
        cobraInfo['thetaSteps'] = np.nan
        cobraInfo['phiSteps'] = np.nan
        cobraInfo['thetaOntime'] = np.nan
        cobraInfo['phiOntime'] = np.nan

        return cobraInfo

    def runCobra(self, cobraBlock):
        moveParams = cobraBlock.p
        thetaEnabled, phiEnabled = moveParams.en

        cobraInfo = self._blankMove()
        if thetaEnabled:
            cobraInfo['thetaSteps'] = (moveParams.steps[0] *
                                       (-1 if moveParams.dir[0] == 'ccw' else 1))
        if phiEnabled:
            cobraInfo['phiSteps'] = (moveParams.steps[1] *
                                     (-1 if moveParams.dir[1] == 'ccw' else 1))

        cobraInfo['thetaOntime'] = moveParams.pulses[0] / 1000
        cobraInfo['phiOntime'] = moveParams.pulses[1] / 1000

        self.cobraMoves[self.cobraId(cobraBlock)] = cobraInfo

    def cobraLastMove(self, cobraBlock):
        """ Return the last move we know about for a given cobra """

        cobraId = self.cobraId(cobraBlock)
        try:
            return self.cobraMoves[cobraId]
        except KeyError:
            return self._blankMove()


fpgaState = FpgaState()
