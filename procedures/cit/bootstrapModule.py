from importlib import reload
import numpy as np

from ics.cobraCharmer import pfiDesign
from ics.cobraCharmer import pfi as pfiControl

from ics.cobraCharmer.utils import butler
from ics.cobraCharmer.utils import coordinates
from procedures.moduleTest.mcs import camera
from procedures.cit import calibrateMotorFrequencies

reload(butler)
reload(camera)
reload(calibrateMotorFrequencies)

def bootstrapModule(moduleName, initialXml=None, outputName=None,
                    fpgaHost='fpga',
                    simulationPath=None,
                    setCenters=True,
                    clearGeometry=True,
                    doCalibrate=True,
                    numberCobrasFromRight=False,
                    setModuleId=True):

    run = butler.RunTree()

    if fpgaHost == 'fpga':
        fpgaHost = '128.149.77.24' # See INSTRM-464
    elif fpgaHost == 'None' or fpgaHost == '':
        fpgaHost = None

    if initialXml is None:
        initialXml = butler.mapPathForModule(moduleName, 'init')
    if outputName is None:
        outputName = f"{moduleName}_bootstrap.xml"

    cam = camera.cameraFactory(runManager=run, simulationPath=simulationPath,
                               doClear=True)
    cam.resetStack(doStack=False)
    _ = cam.expose() # Just to record in case calibrartion moves far.

    if fpgaHost is None:
        pfi = None
        pfiModel = pfiDesign.PFIDesign(initialXml)
    else:
        pfi = pfiControl.PFI(fpgaHost=fpgaHost, logDir=run.logDir,
                             doLoadModel=False)
        pfi.loadModel(initialXml)
        pfiModel = pfi.calibModel
        pfi.reset()

        # if we need to calibrate motor frequencies , assume the worst
        # (as seen in the assembly station init files): the values
        # would leave the motors not safe to run. So calibrate phi now, so
        # that we can reliably move it to home.
        if doCalibrate:
            calibrateMotorFrequencies.calibrateMotorFrequencies(pfi,
                                                                enabled=(False, True))
        else:
            pfi.setFreq()

    cam = camera.cameraFactory(runManager=run, simulationPath=simulationPath)
    cam.resetStack(doStack=False)

    _ = cam.expose() # Leave out of no-PFI block so that simulationPath reads always work.
    if pfi is not None:
            pfi.moveAllSteps(None, 0, -4000)

        # Define the cobra range.
        allCobras = pfiControl.PFI.allocateCobraModule(1)
        pfi.moveAllSteps(allCobras, 0, -5000)

    cs, im, _ = cam.expose()
    imCenters = np.stack((cs['x'], cs['y']), 1)

    nspots = len(cs)
    if nspots != 57:
        raise RuntimeError(f'need 57 spots, got {nspots}')
    oldCenters = pfiModel.centers
    modelCenters = np.stack((np.real(oldCenters), np.imag(oldCenters)), 1)

    # The _only_ point of this step is to match up the fiber
    # numbers with the spots.
    # Forcing the images to have the boards run horizontally is a robust way
    # to do this.
    imIdx, _ = coordinates.laydown(imCenters)
    modelIdx, _ = coordinates.laydown(modelCenters)

    # These are now sorted descending by x, alternating between top
    # and bottom cobras. But the cobras centers in the xml files can
    # have real trash coordinates. For all the existing science
    # modules at CIT, several cobras have the same coordinates.  So do
    # _not_ use the model coordinates by default. Simply assign the
    # cobras by increasing/decreasing X.
    #
    # We want the cobras to be numbered from the left by default.
    #
    if numberCobrasFromRight:
        imIdx = imIdx[::-1]         # The ASRD camera has cobra 1 in the top-right.
    homes = imCenters[imIdx,0] + imCenters[imIdx,1]*(1j)

    # Usually only use scale from this. Only use the reset if we neither assign new centers nor
    # clear the geometry.
    offset1, scale1, tilt1, convert1 = coordinates.makeTransform(oldCenters[modelIdx], homes)

    if setModuleId:
        pfiModel.setModuleId(moduleName)

    if setCenters:
        centers = homes
    else:
        centers = convert1(pfiModel.centers)

    if clearGeometry:
        tht0 = pfiModel.tht0[:] * 0.0
        tht1 = pfiModel.tht1[:] * 0.0 + 20.0 * np.pi/180

        phiIn = pfiModel.phiIn[:] * 0.0
        phiOut = pfiModel.phiOut[:] * 0.0 + 180.0 * np.pi/180

    else:
        tht0 = (pfiModel.tht0+tilt1)%(2*np.pi)
        tht1 = (pfiModel.tht1+tilt1)%(2*np.pi)
        phiIn = pfiModel.phiIn
        phiOut = pfiModel.phiOut
    L1 = pfiModel.L1*scale1
    L2 = pfiModel.L2*scale1

    pfiModel.updateGeometry(centers, L1, L2)
    pfiModel.updateThetaHardStops(tht0, tht1)
    pfiModel.updatePhiHardStops(phiIn, phiOut)

    if doCalibrate:
        # Now can calibrate theta motors.
        calibrateMotorFrequencies.calibrateMotorFrequencies(pfi,
                                                            enabled=(True, False))
    xmlDir = run.outputDir
    outPath = xmlDir / outputName
    pfiModel.createCalibrationFile(outPath, name='bootstrap')

    return outPath

def main(args=None):
    if isinstance(args, str):
        import shlex
        args = shlex.split(args)

    import argparse

    parser = argparse.ArgumentParser('bootstrapModule', add_help=True)
    parser.add_argument('moduleName', type=str,
                        help='the name of the module (e.g. "SC03", "Spare1", or "PFI")')

    parser.add_argument('--fpgaHost', type=str, default='fpga',
                        help='connect to the given FPGA host instead of real one.')
    parser.add_argument('--modelName', type=str, default=None,
                        help='load the given model before calibrating.')
    parser.add_argument('--saveModelFile', type=str, default=None,
                        help='save the updated model in the given file.')
    parser.add_argument('--simulationPath', type=str, default=None,
                        help='use the given path to get camera images from.')
    parser.add_argument('--noSetCenters', action='store_true',
                        help='transform the old centers instead of using the image.')
    parser.add_argument('--noSetModuleId', action='store_true',
                        help='leave the existing module id.')
    parser.add_argument('--noClearGeometry', action='store_true',
                        help='transform the old geometry intsead of clearing it.')
    parser.add_argument('--noCalibrate', action='store_true',
                        help='do not calibrate the motor frequencies.')
    parser.add_argument('--numberCobrasFromRight', action='store_true',
                        help='have the cobras be numbered with decreasing X')

    opts = parser.parse_args(args)
    bootstrapModule(opts.moduleName,
                    opts.modelName, opts.saveModelFile,
                    fpgaHost=opts.fpgaHost,
                    simulationPath=opts.simulationPath,
                    setModuleId=not opts.noSetModuleId,
                    setCenters=not opts.noSetCenters,
                    doCalibrate=not opts.noCalibrate,
                    numberCobrasFromRight=opts.numberCobrasFromRight,
                    clearGeometry=not opts.noClearGeometry)

if __name__ == "__main__":
    main()
