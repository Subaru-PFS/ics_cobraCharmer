from importlib import reload
import logging
import pathlib
import platform
import time

import numpy as np

import astropy.io.fits as pyfits

from ics.cobraCharmer.utils import butler

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

    return 'asrd'

def cameraFactory(name=None, doClear=False, simulationPath=None, runManager=None):
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
        if name == 'cit':
            from . import citCam
            reload(citCam)
            cameraFactory.__camera = citCam.CitCamera(simulationPath=simulationPath,
                                                      runManager=runManager)
        elif name == 'asrd':
            from . import asrdCam
            reload(asrdCam)
            cameraFactory.__camera = asrdCam.AsrdCamera(simulationPath=simulationPath,
                                                        runManager=runManager)
        elif name == 'sim':
            cameraFactory.__camera = SimCamera(simulationPath=simulationPath,
                                               runManager=runManager)
        else:
            raise ValueError(f'camera type must be specified and known, not {name}')

        return cameraFactory.__camera

class Camera(object):
    filePrefix = 'PFXC'

    def __init__(self, runManager=None, simulationPath=None, logLevel=logging.INFO):
        self.logger = logging.getLogger('camera')
        self.logger.setLevel(logLevel)

        self._cam = None
        self.dark = None
        self.exptime = 0.25

        if runManager is None:
            runManager = butler.RunTree()
        self.runManager = runManager
        self.dataRoot = runManager.rootDir
        self.imageDir = None
        self.sequenceNumberFilename = "nextSequenceNumber"
        self.doStack = False

        if simulationPath is not None:
            simulationPath = pathlib.Path(simulationPath)
            self.simulationPath = (simulationPath, 0)
        else:
            self.simulationPath = None

    def newDir(self, doStack=True):
        """ Change the directory for output files (images and spots). """

        self.runManager.newRun()
        self.imageDir = self.runManager.dataDir
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

    def getObjects(self, im, sigma=20.0):
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
        self.logger.warn(f'median={np.median(data_sub)} std={np.std(data_sub)} '
                         f'thresh={thresh} {len(objects)} objects')

        keep_w = self.trim(objects['x'], objects['y'])
        if len(keep_w) != len(objects):
            self.logger.info(f'trimming {len(objects)} objects to {len(keep_w)}')
        objects = objects[keep_w]
        t2 = time.time()
        self.logger.info(f'spots, bknd: {t1-t0:0.3f} spots: {t2-t1:0.3f} total: {t2-t0:0.3f}')

        return objects, data_sub, bkg

    def expose(self, name=None, exptime=None, doCentroid=True, steps=None, guess=None):
        t0 = time.time()
        if self.simulationPath is not None:
            im = self._readNextSimulatedImage()
        else:
            if exptime is None:
                exptime = self.exptime
            im = self._camExpose(exptime)

            if self.dark is not None:
                im -= self.dark

        filename = self.saveImage(im, extraName=name)

        if doCentroid:
            t0 = time.time()
            objects, data_sub, bkgd = self.getObjects(im)
            t1 = time.time()
            self.appendSpots(filename, objects, steps=steps, guess=guess)
            t2=time.time()

            self.logger.info(f'{filename}: {len(objects)} spots, get: {t1-t0:0.3f} save: {t2-t1:0.3f} total: {t2-t0:0.3f}')
        else:
            objects = None

        return objects, filename, bkgd

    def trim(self, x=None, y=None):
        """ Returns mask of valid points. """

        return np.arange(len(x))

    def _consumeNextSeqno(self):
        """ Return the next free sequence number.

        We manage this sequence number using a file in our root
        directory.
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

        self.seqno = self._consumeNextSeqno()
        return pathlib.Path(self.imageDir,
                            f'{self.filePrefix}{self.seqno:08d}.fits')

    def _updateStack(self, img):
        if self.doStack is False:
            return
        stackPath = pathlib.Path(self.imageDir, self.doStack)

        try:
            stackFits = pyfits.open(stackPath, mode="update")
            stack = stackFits['IMAGE'].data
            stack += img
        except FileNotFoundError:
            stackFits = pyfits.open(stackPath, mode="append")
            stackFits.append(pyfits.CompImageHDU(img, name='IMAGE'))

        stackFits.flush()
        stackFits.close()
        del stackFits

    def saveImage(self, img, extraName=None, doStack=True):
        filename = self._getNextName()

        hdus = pyfits.HDUList()
        hdus.append(pyfits.CompImageHDU(img, name='IMAGE', uint=True))

        hdus.writeto(filename, overwrite=True)
        self.logger.info('saveImage: %s', filename)

        if extraName is not None:
            linkname = filename.parent / extraName
            if platform.system() == 'Windows':
                hdus.writeto(linkname)  # Creating sylink requires admin!
            else:
                linkname.symlink_to(filename.name)

        if doStack:
            self._updateStack(img)

        return filename

    def appendSpots(self, filename, spots, guess=None, steps=None):
        """ Add spots to existing image file and append them to summary spot file.

            This is not done efficiently.
        """

        t0 = time.time()
        hdulist = pyfits.open(filename, mode='append')
        hdulist.append(pyfits.BinTableHDU(spots, name='SPOTS'))
        hdulist.close()

        spotfile = self.imageDir / 'spots.npz'
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