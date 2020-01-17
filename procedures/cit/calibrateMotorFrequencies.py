from importlib import reload
import logging
import pathlib
import numpy as np

from ics.cobraCharmer import pfi as pfiModule
from ics.cobraCharmer.utils import butler
reload(pfiModule)

def calibrateMotorFrequencies(pfi, modules=None, updateModel=True,
                              thetaLow=55.0, thetaHigh=75.0,
                              phiLow=90.0, phiHigh=120.0,
                              enabled=(True,True)):
    """ Calibrate cobras.

    Args
    ----
    pfi : PFI instance
       A collection of 1+ cobra modules, with loaded model.
    modules : a list of module numbers.
       If set, limit calibration to those modules.
    updateModel : bool
       Whether to update the pfi's internal model

    We limit calibrations to individual full modules. The _calibration_
    step will run on individual cobras, but readback of measured
    frequencies happens to full boards. Avoid confusion by limiting
    this step.


    """
    logger = logging.getLogger('pfi')

    if modules is not None:
        cobras = []
        for m in modules:
            c1 = pfi.allocateCobraModule(m)
            cobras.extend(c1)
    else:
        cobras = pfi.getAllDefinedCobras()

    boards = {(c.module, c.board) for c in cobras}

    pfi.calibrateFreq(cobras,
                      thetaLow=thetaLow, thetaHigh=thetaHigh,
                      phiLow=phiLow, phiHigh=phiHigh,
                      enabled=enabled)
    for b in boards:
        mod, brd = b
        err, t1, t2, v, freq1, curr1, freq2, curr2 = pfi.hk(mod, brd, updateModel=updateModel)
        for i in range(len(curr1)):
            logger.info(f"cobra {i+1:-2d} theta={curr1[i]:-6.3f} phi={curr2[i]:-6.3f}")

    return pfi

def main(args=None):
    if isinstance(args, str):
        import shlex
        args = shlex.split(args)

    import argparse

    parser = argparse.ArgumentParser('calibrateMotorFrequencies', add_help=True)
    parser.add_argument('moduleName', type=str,
                        help='the name of the module (e.g. "SC03", "Spare1", or "PFI")')

    parser.add_argument('--initOntimes', action='store_true',
                        help='set the model ontimes to some sane initial value.')
    parser.add_argument('--fpgaHost', type=str, default='localhost',
                        help='connect to the given FPGA host instead of the simulator.')
    parser.add_argument('--modelVersion', type=str, default='init',
                        help='specify the version of the input model.')
    parser.add_argument('--saveModelFile', type=pathlib.Path, default=None,
                        help='save the updated model in the given file.')
    opts = parser.parse_args(args)
    print(opts)

    output = butler.RunTree()

    pfi = pfiModule.PFI(fpgaHost=opts.fpgaHost, logDir=output.logDir,
                        doLoadModel=False)
    mapPath = butler.mapPathForModule(opts.moduleName, opts.modelVersion)
    print(mapPath)
    pfi.loadModel(mapPath)

    pfi = calibrateMotorFrequencies(pfi=pfi,
                                    updateModel=True)
    if opts.initOntimes:
        thetaOntimes = np.full(57, 0.065)
        phiOntimes = np.full(57, 0.045)

        pfi.calibModel.updateOntimes(thtFwd=thetaOntimes, thtRev=thetaOntimes,
                                     phiFwd=phiOntimes, phiRev=phiOntimes)

    if opts.saveModelFile is not None:
        if opts.saveModelFile.is_absolute():
            outputPath = opts.saveModelFile
        else:
            outputPath = output.outputDir / opts.saveModelFile
        pfi.calibModel.createCalibrationFile(outputPath)
        print(f"wrote updated map file to {outputPath}")

if __name__ == "__main__":
    main()
