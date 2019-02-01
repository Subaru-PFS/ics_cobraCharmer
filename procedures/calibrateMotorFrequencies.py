from ics.cobraCharmer import pfi as pfiModule
from ics.cobraCharmer import pfiDesign
from ics.cobraCharmer.utils import fileManager

def calibrateMotorFrequencies(pfi, modules=None, updateModel=True):
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

    if modules is not None:
        cobras = []
        for m in modules:
            c1 = pfi.allocateCobraModule(m)
            cobras.extend(c1)
    else:
        cobras = pfi.getAllConnectedCobras()

    boards = {(c.module, c.board) for c in cobras}

    pfi.calibrate(cobras)
    for b in boards:
        mod, brd = b
        pfi.hk(mod, brd, updateModel=updateModel)

    return pfi

def main(args=None):
    if isinstance(args, str):
        import shlex
        args = shlex.split(argv)

    import argparse

    parser = argparse.ArgumentParser('calibrateMotorFrequencies', add_help=True)
    parser.add_argument('moduleName', type=str,
                        help='the name of the module (e.g. "SC03", "Spare1", or "PFI")')

    parser.add_argument('--fpgaHost', type=str, default='localhost',
                        help='connect to the given FPGA host instead of the simulator.')
    parser.add_argument('--modelName', type=str, default=None,
                        help='load the given PFI model before calibrating.')
    parser.add_argument('--module', type=int, default=0,
                        help='calibrate the given module. Or all.')
    parser.add_argument('--saveModelFile', type=str, default='',
                        help='save the updated model in the given file.')

    opts = parser.parse_args(args)

    output = fileManager.ProcedureDirectory(opts.moduleName, experimentName='calibrate')

    pfi = pfiModule.PFI(fpgaHost=opts.fpgaHost, logDir=output.logDir,
                        doLoadModel=False)
    pfi.loadModel(opts.modelName)

    pfi = calibrateMotorFrequencies(pfi=pfi, output=output,
                                    modules=[opts.module] if opts.module != 0 else None0

    if opts.saveModelFile:
        pfi.calibModel.createCalibrationFile(opts.saveModelFile)

if __name__ == "__main__":
    main()
