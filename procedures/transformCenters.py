from importlib import reload
import numpy as np
import os

from ics.cobraCharmer import pfi as pfiControl

from ics.cobraCharmer.utils import fileManager
from ics.cobraCharmer.utils import coordinates
from ics.cobraCharmer.camera import cameraFactory
from ics.cobraCharmer import imageSet

def transformCenters(pfi, output, initialXml, outputXml, setCenters=False):
    if outputXml is None:
        outputXml = os.path.join(output.xmlDir, outputXml)

    # Define the cobra range.
    mod1Cobras = pfiControl.PFI.allocateCobraModule(1)
    allCobras = mod1Cobras

    pfi.loadModel(initialXml)
    pfi.reset()
    pfi.setFreq()

    cam = cameraFactory()
    dataSet = imageSet.ImageSet(pfi, cam, output)
    im, name = dataSet.expose(name='preHome')

    # Take two steps since we may be coming from a place we need to
    # be slightly careful about.
    pfi.moveAllSteps(allCobras, 0, -5000)
    # pfi.moveAllSteps(allCobras, -10000, 0)

    im, name = dataSet.expose(name='centers')
    cs, im = dataSet.spots('centers')
    imCenters = np.stack((cs['x'], cs['y']), 1)

    print("nspots = %d" % (len(cs)))
    oldCenters = pfi.calibModel.centers
    modelCenters = np.stack((np.real(oldCenters), np.imag(oldCenters)), 1)

    # The _only_ point of this exercise is to match up the fiber
    # numbers with the spots.
    # Forcing the images to have the boards run horizontally is a robust way
    # to do this.
    imIdx, _ = coordinates.laydown(imCenters)
    modelIdx, _ = coordinates.laydown(modelCenters)

    homes = imCenters[imIdx,0] + imCenters[imIdx,1]*(1j)
    offset1, scale1, tilt1, convert1 = coordinates.makeTransform(oldCenters[modelIdx], homes)

    old = pfi.calibModel

    if setCenters:
        centers = homes
    else:
        centers = convert1(old.centers)
    tht0 = (old.tht0+tilt1)%(2*np.pi)
    tht1 = (old.tht1+tilt1)%(2*np.pi)
    L1 = old.L1*scale1
    L2 = old.L2*scale1

    old.updateGeometry(centers, L1, L2)
    old.updateThetaHardStops(tht0, tht1)
    old.createCalibrationFile(outputXml)

    print("Process Finised")

def main(args=None):
    if isinstance(args, str):
        import shlex
        args = shlex.split(args)

    import argparse

    parser = argparse.ArgumentParser('recenterMap', add_help=True)
    parser.add_argument('moduleName', type=str,
                        help='the name of the module (e.g. "SC03", "Spare1", or "PFI")')

    parser.add_argument('--fpgaHost', type=str, default='localhost',
                        help='connect to the given FPGA host instead of the simulator.')
    parser.add_argument('--modelName', type=str, default=None,
                        help='load the given PFI model before calibrating.')
    parser.add_argument('--saveModelFile', type=str, default='',
                        help='save the updated model in the given file.')
    parser.add_argument('--setCenters', action='store_true',
                        help='Simply _set_ the centers.')

    opts = parser.parse_args(args)

    cam = cameraFactory('cit')
    output = fileManager.ProcedureDirectory(opts.moduleName, experimentName='recenter')

    pfi = pfiControl.PFI(fpgaHost=opts.fpgaHost, logDir=output.logDir,
                         doLoadModel=False)
    pfi.loadModel(opts.modelName)

    transformCenters(pfi, output,
                     opts.modelName, opts.saveModelFile,
                     setCenters=opts.setCenters)

    if opts.saveModelFile:
        if os.path.isabs(opts.saveModelFile):
            savePath = opts.saveModelFile
        else:
            savePath = os.path.join(output.xmlDir, opts.saveModelFile)
        pfi.calibModel.createCalibrationFile(savePath)


if __name__ == "__main__":
    main()
