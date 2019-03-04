import os
from importlib import reload
import numpy as np

import logging
from copy import deepcopy

from ics.cobraCharmer import pfi as pfiControl
from ics.cobraCharmer import imageSet
from ics.cobraCharmer.utils import fileManager
from ics.cobraCharmer.camera import cameraFactory

from . import adjustMotorOntime
from . import utils
reload(utils)

logger = logging.getLogger('proc')
logger.setLevel(logging.INFO)

def runThetaMotormap(pfi, output, modules=None,
                     repeat=1,
                     steps=50,
                     thetaRange=12000,
                     ontime=None,
                     startingModel=None,
                     reprocess=False,
                     updateModel=True):

    # We need to be more flexible....
    if modules is None:
        modules = [1]
    if len(modules) != 1 and modules[0] != 1:
        raise ValueError('sorry, lazy programmer cannot map several modules')

    module = modules[0]

    # Define the cobra range.
    allCobras = []
    for m in modules:
        allCobras.extend(pfiControl.PFI.allocateCobraModule(module))

    # CRAP: Move and flesh out. At least provide a commandline list.
    brokens = utils.getBrokenCobras(pfi, module)

    # define the broken/good cobras
    visibles= [e for e in range(1,58) if e not in brokens]
    goodCobras = [c for c in allCobras if c.cobra not in brokens]
    goodIdx = np.array(visibles) - 1

    # HACKS: use the same weighting as Johannes to calculate motor maps,
    #        plus some suspect definitions.
    regions = 112

    if not reprocess:
        pfi.reset()
        pfi.setFreq()

    np.seterr(divide='raise')
    xmlFiles = []

    if ontime is not None:
        setName = f'thetaOntime_{thetaOntime}ms'
    else:
        setName = None

    if not reprocess:
        # fastOnTime = np.full(57, 0.08)
        # utils.movePhiToSafeOut(pfi, goodCobras, output, bootstrap=True)
        # pfi.calibModel.updateOntimes(phiFwd=fastOnTime, phiRev=fastOnTime)
        pfi.moveAllSteps(goodCobras, -thetaRange, 0)

        if ontime is not None:
            thetaOnTime = np.full(57, ontime/1000.0)
            pfi.calibModel.updateOntimes(thtFwd=thetaOnTime, thtRev=thetaOnTime)

        thetaDataset = utils.takethetaMap(pfi, output, goodCobras, setName=setName
                                          steps=steps, thetaRange=thetaRange)
    else:
        logger.info(f'reloading data')
        thetaDataset = imageSet.ImageSet.makeFromDirectory(os.path.join(output.imageDir, setName))

        logger.info(f'processing data')
        thetaFW, thetaRV = utils.thetaMeasure(pfi.calibModel.centers, thetaDataset, stepSize=steps)
        phiCenter, phiAngFW, phiAngRV, phiRadius = utils.calcPhiGeometry(phiFW, phiRV, goodIdx=goodIdx)
        phiMMFW, phiMMRV = utils.calcPhiMotorMap(phiCenter, phiAngFW, phiAngRV, regions, steps,
                                                 goodIdx=None)

        pfi.calibModel.updateMotorMaps(phiFwd=phiMMFW, phiRev=phiMMRV, useSlowMaps=True)
        pfi.calibModel.updateMotorMaps(phiFwd=phiMMFW, phiRev=phiMMRV, useSlowMaps=False)

        xmlPath = os.path.join(output.xmlDir, f'phiOntime_{t_ms}ms.xml')
        logger.info(f'writing {t_ms}ms map to {xmlPath}')
        pfi.calibModel.createCalibrationFile(xmlPath)
        xmlFiles.append(xmlPath)

        pfi.loadModel(startingModel)

        adjustMotorOntime.doAdjustOnTime(output.xmlDir,
                                         startingModel,
                                         'phiOntimes.xml',
                                         xmlFiles, doTheta=False)

    print("Process Finised")

def main(args=None):
    if isinstance(args, str):
        import shlex
        args = shlex.split(args)

    import argparse

    parser = argparse.ArgumentParser('runPhiOntime', add_help=True)
    parser.add_argument('moduleName', type=str,
                        help='the name of the module (e.g. "SC03", "Spare1", or "PFI")')
    parser.add_argument('--steps', type=int, default=50,
                        help='size of step to take for the motormaps')
    parser.add_argument('--phiRange', type=int, default=5000,
                        help='expected full range of phi motors')
    parser.add_argument('--ontimes', type=int, nargs='+', default=(25,40,55),
                        help='ontimes to test. At least two required')
    parser.add_argument('--fpgaHost', type=str, default='localhost',
                        help='connect to the given FPGA host instead of the simulator.')
    parser.add_argument('--modelName', type=str, default=None,
                        help='load the given PFI model before calibrating.')
    parser.add_argument('--module', type=int, default=0,
                        help='calibrate the given module. Or all.')
    parser.add_argument('--saveModelFile', type=str, default='',
                        help='save the updated model in the given file.')
    parser.add_argument('--reprocess', type=str, default=False,
                        help='do not acquire data, but re-process existing data in this directory.')

    opts = parser.parse_args(args)

    if len(opts.ontimes) < 2:
        parser.exit(1, '--ontimes must have at least two times')

    if opts.reprocess:
        cam = None
        output = fileManager.ProcedureDirectory.loadFromPath(opts.reprocess)
    else:
        import pfiSite

        cam = cameraFactory(pfiSite.location)  # Tell the factory the location.
        output = fileManager.ProcedureDirectory(opts.moduleName, experimentName='map')

    pfi = pfiControl.PFI(fpgaHost=opts.fpgaHost, logDir=output.logDir,
                         doLoadModel=False, doConnect=False)
    pfi.loadModel(opts.modelName)

    runPhiOntime(pfi, output,
                 modules=[opts.module] if opts.module != 0 else None,
                 startingModel=opts.modelName,
                 steps=opts.steps,
                 phiRange=opts.phiRange,
                 ontimes=opts.ontimes,
                 reprocess=opts.reprocess)

    if opts.saveModelFile:
        pfi.calibModel.createCalibrationFile(opts.saveModelFile)

if __name__ == "__main__":
    main()
