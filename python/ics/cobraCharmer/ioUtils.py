from importlib import reload
import logging

import numpy as np

from . import cobra, motormap
reload(cobra)
reload(motormap)

import pfs.utils.fiberids
reload(pfs.utils.fiberids)
fiberIds = pfs.utils.fiberids.FiberIds()

def convertCobrasFromPfiDesign(butler, pfiDesign):
    """ Convert an XML file to our per-cobra encoding.

    Given a loaded PFIDesign, save all its cobras using the given butler.

    Args
    ----
    butler : `cobraCharmer.Butler`
    pfiDesign : `cobraCharmer.PFIDesign`
      A PFIDesign object loaded with some number of cobras.

    """

    for c_i in range(pfiDesign.nCobras):
        parts = dict()
        parts['cobraId'] = fiberIds.cobraIdForModulePlusCobra(pfiDesign.moduleIds[c_i],
                                                              pfiDesign.positionerIds[c_i])
        parts['serial'] = pfiDesign.serialIds[c_i]
        parts['status'] = pfiDesign.status[c_i]
        parts['center'] = pfiDesign.centers[c_i].imag, pfiDesign.centers[c_i].real
        parts['thetaLimits'] = np.rad2deg(pfiDesign.tht0[c_i]), np.rad2deg(pfiDesign.tht1[c_i])
        parts['phiLimits'] = np.rad2deg(pfiDesign.phiIn[c_i]), np.rad2deg(pfiDesign.phiOut[c_i])
        parts['L1'] = pfiDesign.L1[c_i]
        parts['L2'] = pfiDesign.L2[c_i]
        parts['thetaMotorFrequency'] = pfiDesign.motorFreq1[c_i]
        parts['phiMotorFrequency'] = pfiDesign.motorFreq2[c_i]

        c = cobra.Cobra(parts['cobraId'])
        c.initFromParts(**parts)
        createMotormapsFromPfiDesign(butler, pfiDesign, c, c_i)

        butler.put(c, 'cobraGeometry', dict(moduleName=c.moduleName,
                                            cobraInModule=c.cobraInModule))

def createMotormapsFromPfiDesign(butler, pfiDesign, cobra, cobraIndex):
    """ Convert the motormaps in an XML file-based cobra to our per-cobra encoding.

    Given a loaded PFIDesign, save all six of the motormaps for a
    single cobra using the given butler.

    Args
    ----
    butler : `cobraCharmer.Butler`
    pfiDesign : `cobraCharmer.PFIDesign`
      A PFIDesign object loaded with some number of cobras.
    """

    def createOneMap(cobra, motor, direction, mapName, ontime, angles, steps):
        mm = motormap.MotorMap(mapName, cobra.cobraId,
                               motor=motor, direction=direction,
                               angles=angles, steps=steps,
                               ontimes=ontime)
        butler.put(mm, 'motorMap',
                   idDict=dict(moduleName=cobra.moduleName,
                               cobraInModule=cobra.cobraInModule,
                               motor=motor, direction=direction,
                               mapName=mapName))

    stepSize = pfiDesign.angularSteps[cobraIndex]
    phiCount = int(np.deg2rad(200.0)/stepSize + 0.5)
    thetaCount = int(np.deg2rad(400.0)/stepSize + 0.5)
    angles = np.linspace(0, stepSize*thetaCount, thetaCount)

    createOneMap(cobra, 'theta', 'fwd', 'fast',
                 pfiDesign.motorOntimeFwd1[cobraIndex],
                 angles[:thetaCount], pfiDesign.F1Pm[cobraIndex][:thetaCount])
    createOneMap(cobra, 'theta', 'rev', 'fast',
                 pfiDesign.motorOntimeRev1[cobraIndex],
                 angles[:thetaCount], pfiDesign.F1Nm[cobraIndex][:thetaCount])
    createOneMap(cobra, 'theta', 'fwd', '',
                 pfiDesign.motorOntimeSlowFwd1[cobraIndex],
                 angles[:thetaCount], pfiDesign.S1Pm[cobraIndex][:thetaCount])
    createOneMap(cobra, 'theta', 'rev', '',
                 pfiDesign.motorOntimeSlowRev1[cobraIndex],
                 angles[:thetaCount], pfiDesign.S1Nm[cobraIndex][:thetaCount])

    createOneMap(cobra, 'phi', 'fwd', 'fast',
                 pfiDesign.motorOntimeFwd2[cobraIndex],
                 angles[:phiCount], pfiDesign.F2Pm[cobraIndex][:phiCount])
    createOneMap(cobra, 'phi', 'rev', 'fast',
                 pfiDesign.motorOntimeRev2[cobraIndex],
                 angles[:phiCount], pfiDesign.F2Nm[cobraIndex][:phiCount])
    createOneMap(cobra, 'phi', 'fwd', '',
                 pfiDesign.motorOntimeSlowFwd2[cobraIndex],
                 angles[:phiCount], pfiDesign.S2Pm[cobraIndex][:phiCount])
    createOneMap(cobra, 'phi', 'rev', '',
                 pfiDesign.motorOntimeSlowRev2[cobraIndex],
                 angles[:phiCount], pfiDesign.S2Nm[cobraIndex][:phiCount])
