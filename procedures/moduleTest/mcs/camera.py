from importlib import reload
import logging
import pathlib
import time

import numpy as np

import astropy.io.fits as pyfits
from . import spots
reload(spots)

# Configure the default formatter and logger.
logging.basicConfig(datefmt = "%Y-%m-%d %H:%M:%S", level=logging.DEBUG,
                    format = "%(asctime)s.%(msecs)03dZ %(name)-16s %(levelno)s %(filename)s:%(lineno)d %(message)s")

def cameraFactory(name=None, doClear=False, simulationPath=None, dataRoot=None):
    if doClear or simulationPath is not None:
        try:
            del cameraFactory.__camera
        except AttributeError:
            pass
    try:
        return cameraFactory.__camera
    except:
        if name == 'cit':
            from . import citCam
            reload(citCam)
            cameraFactory.__camera = citCam.CitCamera(simulationPath=simulationPath, dataRoot=dataRoot)
        elif name == 'asrd':
            from . import asrdCam
            reload(asrdCam)
            cameraFactory.__camera = asrdCam.AsrdCamera(simulationPath=simulationPath, dataRoot=dataRoot)
        elif name == 'sim':
            cameraFactory.__camera = SimCamera(simulationPath=simulationPath, dataRoot=dataRoot)
        else:
            raise ValueError(f'camera type must be specified and known, not {name}')

        return cameraFactory.__camera

class Camera(object):
    filePrefix = 'PFXC'

    def __init__(self, simulationPath=None, dataRoot=None, logLevel=logging.INFO):
        self.logger = logging.getLogger('camera')
        self.logger.setLevel(logLevel)

        self._cam = None
        self.dark = None
        self.exptime = 0.25

        if dataRoot is None:
            dataRoot = '/data/MCS'
        self.dataRoot = dataRoot
        self.dirpath = None
        self.sequenceNumberFilename = "nextSequenceNumber"

        if simulationPath is not None:
            simulationPath = pathlib.Path(simulationPath)
            self.simulationPath = (simulationPath, 0)
        else:
            self.simulationPath = None

    def _now(self):
        return time.strftime('%Y%m%d_%H%M%S')

    def setDirpath(self, dirname=None):
        """ Set the directory for output files (images and spots). Created if does not exist. """

        if dirname is None:
            dirname = pathlib.Path(self.dataRoot, self._now())
        dirpath = pathlib.Path(dirname)
        dirpath = dirpath.expanduser().resolve()
        dirpath.mkdir(parents=True)
        self.dirpath = dirpath

    def __repr__(self):
        if self.simulationPath is None:
            return f"{self.__class__.__name__}(dir={self.dirpath})"
        else:
            return f"{self.__class__.__name__}(dir={self.dirpath}, simulationPath={self.simulationPath})"

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

    def dark(self, exptime=None):
        """ Take and save a dark frame. """

        im, filename = self.expose(exptime=exptime, noCentroids=True, _takeDark=True)
        self.dark = im

    def expose(self, exptime=None, name=None, _takeDark=False, doCentroid=True):
        if self.simulationPath is not None:
            im = self._readNextSimulatedImage()
        else:
            if exptime is None:
                exptime = self.exptime
            im = self._camExpose(exptime, _takeDark=_takeDark)

        if not _takeDark and self.dark is not None:
            im -= self.dark

        filename = self.saveImage(im, name)

        if _takeDark:
            return im, filename

        if doCentroid:
            t0 = time.time()
            objects = spots.getObjects(im)
            t1 = time.time()
            self.appendSpots(filename, objects)
            t2=time.time()

            self.logger.info(f'{filename.stem}: {len(objects)} spots, get: {t1-t0:0.3f} save: {t2-t1:0.3f} total: {t2-t0:0.3f}')

        return im, filename

    def trim(self, x=None, y=None):
        """ Returns mask of valid points. """

        return np.arange(len(x))

    def _consumeNextSeqno(self):
        """ Return the next free sequence number. """

        sequenceNumberFile = pathlib.Path(self.dataRoot, self.sequenceNumberFilename)

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
        if self.dirpath is None:
            self.setDirpath()

        self.seqno = self._consumeNextSeqno()
        return pathlib.Path(self.dirpath,
                            f'{self.filePrefix}{self.seqno:08d}.fits')

    def saveImage(self, img, name=None):
        if name is None:
            name = self._getNextName()

        hdus = pyfits.HDUList()
        hdus.append(pyfits.CompImageHDU(img, name='IMAGE', uint=True))

        hdus.writeto(name, overwrite=True)

        return name

    def appendSpots(self, filename, spots):
        """ Add spots to existing image file and append them to summary spot file.

            This is not done efficiently.
        """

        t0 = time.time()
        hdulist = pyfits.open(filename, mode='append')
        hdulist.append(pyfits.BinTableHDU(spots, name='SPOTS'))
        hdulist.close()

        spotfile = self.dirpath / 'spots.npz'
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
