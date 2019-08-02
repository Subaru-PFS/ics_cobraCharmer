import numpy as np

class FpgaState(object):
    """ Track state of FPGA. Currently just tracks the last move made, per-cobra """

    def  __init__(self):
        self.cobraMoves = dict()

    def cobraId(self, cobra):
        """ Return an indexable id for a cobra. """
        return (cobra.module, cobra.cobraNum)

    def runCobra(self, cobraBlock):
        moveParams = cobraBlock.p
        thetaEnabled, phiEnabled = moveParams.en

        cobraInfo = dict()
        if thetaEnabled:
            cobraInfo['thetaSteps'] = (moveParams.steps[0] *
                                       (-1 if moveParams.dir[0] == 'ccw' else 1))
        else:
            cobraInfo['thetaSteps'] = np.nan
        if phiEnabled:
            cobraInfo['phiSteps'] = (moveParams.steps[1] *
                                     (-1 if moveParams.dir[1] == 'ccw' else 1))
        else:
            cobraInfo['phiSteps'] = np.nan

        cobraInfo['thetaOntime'] = moveParams.pulses[0] / 1000
        cobraInfo['phiOntime'] = moveParams.pulses[1] / 1000

        self.cobraMoves[self.cobraId(cobraBlock)] = cobraInfo

    def cobraLastMove(self, cobraBlock):
        """ Return the last move we know about for a given cobra """

        cobraId = self.cobraId(cobraBlock)
        return self.cobraMoves[cobraId]

fpgaState = FpgaState()
