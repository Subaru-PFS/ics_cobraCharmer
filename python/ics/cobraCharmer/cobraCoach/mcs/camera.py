from importlib import reload
import logging
import pathlib
import platform
import time

import numpy as np
import numpy.lib.recfunctions as recfuncs

import astropy.io.fits as pyfits

from ics.cobraCharmer.utils import butler
from ics.fpsActor import najaVenator
reload(najaVenator)

# Configure the default formatter and logger.
logging.basicConfig(datefmt = "%Y-%m-%d %H:%M:%S", level=logging.DEBUG,
                    format = "%(asctime)s.%(msecs)03dZ %(name)-16s %(levelno)s %(filename)s:%(lineno)d %(message)s")

def whereAmI():
    """ Guess our location/camera.

    For now we only look for the CIT cube and the ASRD bench. Windows v. Unix.

    """
    import platform

    if platform.system() == 'Windows':
        return 'cit'

    # Need to distinguish the ASRD bench, the ASRD MCS, the Subaru MCS.
    # Can use network address for Subaru

    return 'rmod'

def cameraFactory(name=None, doClear=False, simulationPath=None, runManager=None,
                  actor=None, cmd=None):
    if doClear or simulationPath is not None:
        try:
            del cameraFactory.__camera
        except AttributeError:
            pass
    try:
        return cameraFactory.__camera
    except:
        if name is None:
            name = whereAmI()

        if name == 'mcsActor':
            from . import mcsActorCam
            reload(mcsActorCam)
            if simulationPath is not None:
                raise NotImplementedError("for mcsActorCam, you *must* use the mcsActor-side simulation.")
            cameraFactory.__camera = mcsActorCam.McsActorCamera(actor=actor,
                                                                cmd=cmd,runManager=runManager)
        elif name == 'mcs':
            from . import mcsCam
            reload(mcsCam)
            cameraFactory.__camera = mcsCam.McsCamera(simulationPath=simulationPath,actor=actor,
                                                      cmd=cmd,runManager=runManager)
        elif name == 'cit':
            from . import citCam
            reload(citCam)
            cameraFactory.__camera = citCam.CitCamera(simulationPath=simulationPath,
                                                      runManager=runManager)
        elif name == 'asrd':
            from . import asrdCam
            reload(asrdCam)
            cameraFactory.__camera = asrdCam.AsrdCamera(simulationPath=simulationPath,
                                                        runManager=runManager)
        elif name == 'rmod':
            from . import rmodCam
            reload(rmodCam)
            cameraFactory.__camera = rmodCam.RmodCamera(simulationPath=simulationPath,
                                                        runManager=runManager)
        elif name == 'sim':
            cameraFactory.__camera = SimCamera(simulationPath=simulationPath,
                                               runManager=runManager)
        else:
            raise ValueError(f'camera type must be specified and known, not {name}')

        return cameraFactory.__camera

class Camera(object):
    filePrefix = 'PFXC'

    def __init__(self, runManager=None, simulationPath=None, logLevel=logging.INFO, cmd=None):
        self.logger = logging.getLogger('camera')
        self.logger.setLevel(logLevel)
        self.frameId = None

        self._cam = None
        self.dark = None
        self.exptime = 0.8
        self.doWriteFitsSpots = True
        
        if runManager is None:
            runManager = butler.RunTree()
        self.runManager = runManager
        self.dataRoot = runManager.rootDir
        self.imageDir = runManager.dataDir
        self.outputDir = runManager.outputDir
        self.sequenceNumberFilename = "nextSequenceNumber"
        self.resetStack(doStack=False)
        self.cmd = cmd

        if simulationPath is not None:
            simulationPath = pathlib.Path(simulationPath)
            self.simulationPath = (simulationPath, 0)
        else:
            self.simulationPath = None

    def newDir(self, doStack=True):
        """ Change the directory for output files (images and spots). """

        self.runManager.newRun()
        self.imageDir = self.runManager.dataDir
        self.outputDir = self.runManager.outputDir
        self.resetStack(doStack=doStack)

    def resetStack(self, doStack=False):
        self.doStack = "stack.fits" if doStack is True else doStack

    def __repr__(self):
        if self.simulationPath is None:
            return f"{self.__class__.__name__}(dir={self.imageDir})"
        else:
            return f"{self.__class__.__name__}(dir={self.imageDir}, simulationPath={self.simulationPath})"

    @property
    def cam(self):
        if self._cam is None:
            self._camConnect()
        return self._cam

    def _camClose(self):
        """ Default camera deallocation routine. Assumes that __del() works and suffices. """
        self._cam = None

    def _camConnect(self):
        """ Implement in subclass. Must set self._cam """
        raise NotImplementedError()

    def _camExpose(self, exptime, _takeDark=False):
        """ Implement in subclass. Returns image """
        raise NotImplementedError()

    def _readNextSimulatedImage(self):
        path, idx = self.simulationPath
        files = sorted(path.glob('*.fits'))
        self.logger.info(f"simulation path={path}, nfiles={len(files)}")

        nextFile = files[idx]
        idx = idx+1
        if idx >= len(files):
            idx = 0
        self.simulationPath = path, idx

        return pyfits.getdata(nextFile)

    def takeDark(self, exptime=None):
        """ Take and save a dark frame. """

        if exptime is None:
            exptime = self.exptime
        im = self._camExpose(exptime, _takeDark=True)
        self.dark = im

        filename = self.saveImage(im, doStack=False)
        self.darkFile = filename

        return filename

    def getPositionsForFrame(self, frameId):
        self.nv = najaVenator.NajaVenator()
        mcsData = self.nv.readCentroid(frameId)

        df=mcsData.loc[mcsData['fiberId'] > 0]
        self.logger.info(f'mcs data {mcsData.shape[0]}')
        #centroids = {'x':mcsData['centroidx'].values.astype('float'),
        #             'y':mcsData['centroidy'].values.astype('float')}
        
        centroids=np.rec.array([df['centroidx'].values.astype('float'),
                                df['centroidy'].values.astype('float')],
                               formats='float,float',names='x,y')
        return centroids

    def getObjects(self, im, expid, sigma=1.5, threshold=None):
        """ Measure the centroids in the image.

        Args
        ----
        im : `ndarray`
           The image to process. Possibly ISR'ed
        expid : `str`
           Some unique exposure identifier. Expected to be `pathlib.Path.stem`.
        sigma : `float`
           Bullshit.

        Returns
        -------
        objects : `ndarray`
           The measured centroids, along with IDs.
           For now, full `sep.extract` output, along with the `expid` and a spot index.
        data_sub : `ndarray`
           The background-subtracted image.
        background : `sep.Background`
           The background image.

        Notes
        -----
        `background` is probably st
        """
        import sep

        t0 = time.time()
        data = im.astype('f4')
        bkg = sep.Background(data)
        bkg.subfrom(data)
        data_sub = data

        # thresh = np.percentile(data_sub, percentile)
        std = np.std(data_sub)
        thresh = std*sigma
        t1 = time.time()
        objects = sep.extract(data_sub,
                              thresh=thresh,
                              filter_type='conv', clean=False,
                              deblend_cont=1.0)
        self.logger.debug(f'median={np.median(data_sub)} std={np.std(data_sub)} '
                          f'thresh={thresh} {len(objects)} objects')

        # Add exposure and spot IDs
        expids = np.zeros((len(objects)), dtype='U12')
        expids[:] = expid
        spotIds = np.arange(len(objects))
        objects = recfuncs.append_fields(objects,
                                         ['expId','spotId'], [expids,spotIds], dtypes=['U12','i4'],
                                         usemask=False)

        t2 = time.time()
        self.logger.debug(f'spots, bknd: {t1-t0:0.3f} spots: {t2-t1:0.3f} total: {t2-t0:0.3f}')

        return objects, data_sub, bkg

    def expose(self, name=None, exptime=None, doCentroid=True, frameNum=None):
        """Take an exposure, usually measure centroids, and save the outputs.

        Args
        ----
        name : `pathlib.Path` or `str`
          An optional _extra_ name to assign to the image file. By
          default, just uses the automatica PFS-compliant filename.

        exptime : `float`
          Override the camera's default exposure time.

        doCentroid: `bool`
          Whether to measure and save centroids.

        frameNum : `int`
          The visit+frame nmber for the file.

        Returns
        -------
        objects : `ndarray`
           All the measured centroids. Note that this might include fiducials, etc.
        filename : `pathlib.Path`
           The full path of the PFS-compliant image file.
        background : `sep.Background`
           The background which was subracted.

        Notes
        -----

        The image saved in the image file is raw: not dark or
        background subtracted. I think the only time you would _look_
        at the image file is when there is a problem, and so you want
        the raw image.

        Returns the background, only because that is otherwise
        unavailable. I think this is probably stupid.
        """
        t0 = time.time()
        if self.simulationPath is not None:
            im = self._readNextSimulatedImage()
        else:
            if exptime is None:
                self.logger.info(f'camera exposure time = {self.exptime}')
                exptime = self.exptime
            im = self._camExpose(exptime)
            if np.all(im == 0):
                self.logger.warn('image is all 0s; reconnecting')
                self._camClose()
                _ = self.cam
                time.sleep(2)
                im = self._camExpose(exptime, frameNum=frameNum)

            if self.dark is not None:
                im -= self.dark

        filename = self.saveImage(im, extraName=name, frameNum=frameNum)
        self.logger.info(f'Saving image as filename {filename}')

        if doCentroid:
            t0 = time.time()
            #objects, data_sub, bkgd = self.getObjects(im, filename.stem)
            if self.frameId is None:
                self.logger.info(f'Gen2 Frame ID is not given. Run SExtractor.')
                objects, data_sub, bkgd = self.getObjects(im, filename.stem)
            else:
                self.logger.info(f'Gen2 Frame ID {self.frameId} is given. Get position from DB')
                objects = self.getPositionsForFrame(self.frameId)
                bkgd = None
            t1 = time.time()
            self.appendSpots(filename, objects)
            t2=time.time()

            self.logger.info(f'{filename.stem}: {len(objects)} spots, '
                              f'get: {t1-t0:0.3f} save: {t2-t1:0.3f} total: {t2-t0:0.3f}')
        else:
            objects = None
            bkgd = None

        return objects, filename, bkgd

    def trim(self, x=None, y=None):
        """ Returns mask of valid points. """

        return np.arange(len(x))

    def _consumeNextSeqno(self):
        """ Return the next free sequence number.

        We manage this sequence number using a file in our root
        directory.

        This functionality should be pulled out into pfs_utils
        """

        sequenceNumberFile = pathlib.Path(self.dataRoot, self.sequenceNumberFilename)
        if not sequenceNumberFile.exists():
            with open(sequenceNumberFile, 'wt') as sf:
                sf.write("1\n")
                sf.close()
        try:
            sf = open(sequenceNumberFile, "rt")
            seq = sf.readline()
            seq = seq.strip()
            seqno = int(seq)
        except Exception as e:
            raise RuntimeError("could not read sequence integer from %s: %s" %
                               (sequenceNumberFile, e))
        nextSeqno = seqno+1
        try:
            sf = open(sequenceNumberFile, "wt")
            sf.write("%d\n" % (nextSeqno))
            sf.truncate()
            sf.close()
        except Exception as e:
            raise RuntimeError("could not WRITE sequence integer to %s: %s" %
                               (sequenceNumberFile, e))

        return seqno

    def _getNextName(self):
        if self.imageDir is None:
            self.newDir(doStack=False)
        self.imageDir = self.runManager.dataDir

        self.seqno = self._consumeNextSeqno()
        return pathlib.Path(self.imageDir,
                            f'{self.filePrefix}{self.seqno:08d}.fits')

    def _updateStack(self, img):
        """ Add an image to a "stack" image file.
        """
        if self.doStack is False:
            return
        stackPath = pathlib.Path(self.imageDir, self.doStack)

        try:
            stackFits = pyfits.open(stackPath, mode="update")
            stack = stackFits['IMAGE'].data
            stack += img
        except FileNotFoundError:
            stackFits = pyfits.open(stackPath, mode="append")
            stackFits.append(pyfits.CompImageHDU(img.astype('f4'), name='IMAGE', uint=True))

        stackFits.flush()
        stackFits.close()
        del stackFits

    def saveImage(self, img, cleanImg=None, extraName=None, doStack=True, frameNum=None):
        filename = self._getNextName(frameNum=frameNum)

        hdus = pyfits.HDUList()
        hdus.append(pyfits.CompImageHDU(img, name='IMAGE', uint=True))

        hdus.writeto(filename, overwrite=True)
        self.logger.debug('saveImage: %s', filename)

        if extraName is not None:
            linkname = filename.parent / extraName
            if platform.system() == 'Windows':
                hdus.writeto(linkname)  # Creating sylink requires admin!
            else:
                linkname.symlink_to(filename.name)

        if doStack:
            if cleanImg is None:
                cleanImg = img
            self._updateStack(cleanImg)

        return filename

    def appendSpots(self, filename, spots):
        """ Add spots to existing image file and append them to summary spot file.

        Args
        ----
        filename : `pathlib.Path` or `str`
          The existing FITS file to append a new Binary Table to.
        spots : `ndarray`
          The array of spots to write.

        This is not done efficiently, but it turns out that loading a
        numpy save file, appending to the array, and writing it back
        out is not bad.
        """

        t0 = time.time()
        if self.doWriteFitsSpots:
            hdulist = pyfits.open(filename, mode='append')
            hdulist.append(pyfits.BinTableHDU(spots, name='SPOTS'))
            hdulist.close()

        spotfile = self.outputDir / 'spots.npz'
        if spotfile.exists():
            with open(spotfile, 'rb') as f:
                oldData = np.load(f)
                oldSpots = oldData['spots']
            allSpots = np.concatenate([oldSpots, spots])
        else:
            allSpots = spots
        t1 = time.time()

        with open(spotfile, 'wb') as f:
            np.savez_compressed(f, spots=allSpots)
        t2 = time.time()

        self.logger.debug(f'appendSpots: len: {len(spots)},{len(allSpots)} read={t1-t0:0.3f} write={t2-t1:0.3f} total={t2-t0:0.3f}')

class SimCamera(Camera):
    filePrefix = 'PFFC'

    def _camConnect(self):
        pass
