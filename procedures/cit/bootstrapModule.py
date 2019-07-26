from importlib import reload
import numpy as np

from ics.cobraCharmer import pfiDesign
from ics.cobraCharmer import pfi as pfiControl

from ics.cobraCharmer.utils import butler
from ics.cobraCharmer.utils import coordinates
from procedures.moduleTest.mcs import camera
reload(butler)
reload(camera)

def bootstrapModule(moduleName, initialXml=None, outputName=None,
                    fpgaHost=None,
                    simulationPath=None,
                    setCenters=True, clearGeometry=True,
                    setModuleId=True):

    run = butler.RunTree()

    if fpgaHost == 'fpga':
        fpgaHost = '128.149.77.24'
    if initialXml is None:
        initialXml = butler.mapPathForModule(moduleName, 'init')

    if fpgaHost is None:
        pfi = None
        pfiModel = pfiDesign.PFIDesign(initialXml)
    else:
        pfi = pfiControl.PFI(fpgaHost=fpgaHost, logDir=run.logDir,
                             doLoadModel=False)
        pfi.loadModel(initialXml)
        pfiModel = pfi.calibModel
        pfi.reset()
        pfi.setFreq()

    cam = camera.cameraFactory(runManager=run, simulationPath=simulationPath)
    cam.resetStack(doStack=False)

    _ = cam.expose() # Leave out of no-PFI block so that simulationPath reads always work.
    if pfi is not None:

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

    # These are now sorted descending by x, or alternating between top
    # and bottom cobras. But the cobras centers in the xml files can
    # have real trash coordinates. For all the existing science
    # modules at CIT, several cobras have the same coordinates.
    #
    imIdx = imIdx[::-1]         # The CIT and ASRD cameras have cobra 1 in the top-right.
    homes = imCenters[imIdx,0] + imCenters[imIdx,1]*(1j)
    offset1, scale1, tilt1, convert1 = coordinates.makeTransform(oldCenters[modelIdx], homes)

    if setModuleId:
        pfiModel.setModuleId(moduleName)

    if setCenters:
        centers = homes
    else:
        centers = convert1(pfiModel.centers)

    if clearGeometry:
        tht0 = pfiModel.tht0[:] * 0.0
        tht1 = pfiModel.tht1[:] * 0.0 + 380.0 * np.pi/180

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

    xmlDir = run.outputDir
    outPath = xmlDir / outputName
    pfiModel.createCalibrationFile(outPath, name='bootstrap')

def main(args=None):
    if isinstance(args, str):
        import shlex
        args = shlex.split(args)

    import argparse

    parser = argparse.ArgumentParser('bootstrapModule', add_help=True)
    parser.add_argument('moduleName', type=str,
                        help='the name of the module (e.g. "SC03", "Spare1", or "PFI")')

    parser.add_argument('--fpgaHost', type=str, default=None,
                        help='connect to the given FPGA host instead of the simulator.')
    parser.add_argument('--modelName', type=str, default=None,
                        help='load the given model before calibrating.')
    parser.add_argument('--saveModelFile', type=str, default='',
                        help='save the updated model in the given file.')
    parser.add_argument('--simulationPath', type=str, default=None,
                        help='use the given path to get camera images from.')
    parser.add_argument('--noSetCenters', action='store_true',
                        help='transform the old centers instead of using the image.')
    parser.add_argument('--noSetModuleId', action='store_true',
                        help='leave the existing module id.')
    parser.add_argument('--noClearGeometry', action='store_true',
                        help='transform the old geometry intsead of clearing it.')

    opts = parser.parse_args(args)

    bootstrapModule(opts.moduleName,
                    opts.modelName, opts.saveModelFile,
                    fpgaHost=opts.fpgaHost,
                    simulationPath=opts.simulationPath,
                    setModuleId=not opts.noSetModuleId,
                    setCenters=not opts.noSetCenters,
                    clearGeometry=not opts.noClearGeometry)

if __name__ == "__main__":
    main()
