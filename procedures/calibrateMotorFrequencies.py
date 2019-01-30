import ics.cobraCharmer.pfi as pfiModule
import ics.cobraCharmer.pfiDesign as pfiDesign

def calibrateMotorFrequencies(pfi=None, boards=None, updateModel=True, fpgaHost='localhost'):
    """ Calibrate cobras.

    Args
    ----
    pfi : PFI instance
       A collection of 1+ cobra modules, with loaded model.
    boards : a list of (module, board) ids.
       If set, limit calibration to those boards.
    updateModel : bool
       Update

    We limit calibrations to individual full boards. The _calibration_
    step will run on individual cobras, but readback of measured
    frequencies happens to full boards. Avoid confusion by limiting
    this step.


    """

    if pfi is None:
        pfi = pfiModule.PFI(fpgaHost=fpgaHost, doLoadModel=False)

    if boards is not None:
        cobras = []
        for b in boards:
            mod, brd = b
            c1 = pfi.allocateCobraBoard(mod, brd)
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

    parser = argparse.ArgumentParser('calibrateMotorFrequencies')
    parser.add_argument('--fpgaHost', type=str, default='localhost',
                        help='connect to the given FPGA host instead of the simulator.')
    parser.add_argument('--modelName', type=str, default='',
                        help='load the given PFI model before calibrating.')
    parser.add_argument('--updateModel', action='store_true',
                        help='Update the loaded model. Requires --modelName')
    parser.add_argument('--saveModelFile', type=str, default='',
                        help='save the updated model in the given file. Requires --updateModel')

    opts = parser.parse_args(args)

    if opts.updateModel and not opts.modelName:
        parser.error('--updateModel requires --modelName')

    if opts.saveModelFile and not opts.updateModel:
        parser.error('--saveModelFile requires --updatemodel')

    if opts.modelName:
        pfi = pfiModule.PFI(fpgaHost=opts.fpgaHost, doLoadModel=False)
        pfi.loadModel(opts.modelName)
    else:
        pfi = None

    pfi = calibrateMotorFrequencies(pfi=pfi, fpgaHost=opts.fpgaHost,
                                    updateModel=opts.updateModel)

    if opts.saveModelFile:
        pfi.calibModel.createCalibrationFile(opts.saveModelFile)

if __name__ == "__main__":
    main()
