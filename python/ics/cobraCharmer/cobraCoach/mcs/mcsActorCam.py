from importlib import reload
import numpy as np
import time
import subprocess as sub
import astropy.io.fits as pyfits
import threading
import pathlib
import ics.utils.cmd as cmdUtils
from ics.cobraCharmer.cobraCoach.mcs import camera
reload(camera)

class McsActorCamera(camera.Camera):
    filePrefix = 'PFSC'

    def __init__(self, actor=None, **kw):
        super().__init__(**kw)

        self.actor = actor
        self.doWriteFitsSpots = False # the mcsActor takes care of this.

        self.logger.info('mcsActor camera')
        self._lock = threading.Lock()

    def _camConnect(self):
        self.logger.info('text="Starting camera initialization."')
        cmdString = f'status'
        cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
                                      forUserCmd=cmd)
        if cmdVar.didFail:
            self.logger.warn('text="Camera initialization failed"')
            return None

    def _getCameraName(self):
        
        self.cameraName = self.actor.models['mcs'].keyVarDict['cameraName']

    def _camExpose(self, exptime=None, frameNum=None, _takeDark=False,
                   doCentroid=True, doFibreID=True, cmd=None):
        """Actually arrange for an mcsActor exposure.

        Args:
        =====
        exptime : `float`
          The requested exposure time.
        frameNum : `int`
          The exposure (visit)(frameIdx) number. For fpsActor and mcsActorCam, this must be set.
        _takeDark : `bool`
          If True, this should be labelled a dark.

        Returns:
        ========
        filePath : `pathlib.Path`-equivalent
          NOTE: THIS is different from other cameras, which return raw images.
        """

        if frameNum is not None:
            frameArg = f"frameId={frameNum}"
        else:
            frameArg = ""
        expType = "dark" if _takeDark else "object"
        doCentroidArg = "doCentroid" if doCentroid else ""
        doFiberIDArg = "doFibreID" if doFibreID else ""

        t1=time.time()
        cmdString = f"expose {expType} expTime={exptime:0.2f} {frameArg} {doCentroidArg} {doFiberIDArg}"
        self.logger.info(f'calling mcs {cmdString} with cmd={cmd} from {threading.current_thread()}')
        cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
                                      forUserCmd=cmd, timeLim=exptime+60)
        if cmdVar.didFail:
            cmd.fail(f'text="MCS expose failed: {cmdUtils.interpretFailure(cmdVar)}"')
            raise RuntimeError(f'FAILED to read mcs image!')

        t2=time.time()

        filekey= self.actor.models['mcs'].keyVarDict['filename'][0]
        filename = pathlib.Path(filekey)
        datapath = filename.parents[0]
        frameId = int(filename.stem[4:], base=10)

        self.logger.info(f'MCS frame ID={frameId} filename={filename}')
        self.logger.info('Time for exposure = %f. '% (t2-t1))

        return filename

    def _readNextSimulatedImage(self):
        raise NotImplementedError("for mcsActorCam, you *must* use the mcsActor-side simulation.")

    def _consumeNextSeqno(self):
        raise NotImplementedError("for mcsActorCam, there is no local sequence/frame number.")

    def _getNextName(self):
        raise NotImplementedError("for mcsActorCam, there are no local raw filenames.")

    def trim(self, x, y):
        raise NotImplementedError("trimming would be done on the mcsActor side")

    def expose(self, name=None, exptime=None,
               doCentroid=True, doFibreID=True,
               frameNum=None, cmd=None, doStack=False):
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

        doFibreId: `bool`
          Whether to match spots and cobras.

        frameNum : `int`
          The visit+frame nmber for the file.

        cmd : `actorcore.Command`
          Our controlling MHS Command.

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

        For mcsActorCam, all measurements must happen on the actor
        side, and the primary file is saved by the mcsActor. Here, we
        populate the managed run directory with *links*.
        """

        if cmd is None:
            cmd = self.actor.bcast

        t0 = time.time()
        if exptime is None:
            #raise NotImplementedError("sorry Craig, you need to set a default exptime")
            exptime = 0.8

        filename = self._camExpose(exptime, doCentroid=doCentroid, frameNum=frameNum,
                                   doFibreID=doFibreID, cmd=cmd)
        t1 = time.time()
        self.linkImage(filename, extraName=name, doStack=doStack)
        t2 = time.time()

        self.logger.info(f"Getting positions from DB for frame {frameNum}")
        objects = self.getPositionsForFrame(frameNum)

        t3 = time.time()
        self.appendSpots(filename, objects)
        t4=time.time()

        self.logger.info(f'{filename.stem}: {len(objects)} spots, '
                         f'get: {t1-t0:0.3f} saveAndStack: {t2-t1:0.3f} db: {t3-t2:0.3f} spots: {t4-t3:0.3f}'
                         f'total: {t4-t0:0.3f}')
        return objects, filename, None

    def linkImage(self, filename, extraName=None, doStack=False):
        """Link mcsActor raw image file into our per-run directory.

        Args
        ----
        filename : `pathlib.Path`
           The absolute path of the just-saved raw image.

        extraName : `str`
           Optional name for the image file. If exists, add a link.

        doStack : `bool`:
           Whether we should update some stack file with the raw image.
        """

        # We get the full path of the PFSC file. Link to that first.
        linkname = self.imageDir / filename.name
        self.logger.info(f'Linking to {filename} from {linkname}')
        linkname.symlink_to(filename)

        # Now add any alias
        if extraName is not None:
            linkname = self.imageDir / extraName
            self.logger.info(f'Linking to {filename} from {linkname}')
            linkname.symlink_to(filename.name)

        if doStack:
            self.logger.info(f'Adding to a stacked image.')
            img = pyfits.getdata(filename, extname='IMAGE')
            self._updateStack(img)

        return filename

    def _record(self):
        with self._lock:
            im = self._camExpose(self.exptime)
            if self.dark is not None:
                im -= self.dark
            self._imRecord = np.full(im.shape, im, 'float')

        while self._recording:
            with self._lock:
                im = self._camExpose(self.exptime)
                if self.dark is not None:
                    im -= self.dark
                self._imRecord += im

    def startRecord(self):
        """Start recording
        """

        raise NotImplementedError("mcsActoCam being asked to startRecord-ing -- not sure whether that is sane.")

        self._recording = True
        t = threading.Thread(target=self._record, daemon=True)
        t.start()

    def stopRecord(self, name=None):
        """ Stop recording
        """
        self._recording = False
        with self._lock:
            filename = self.saveImage(self._imRecord, extraName=name)

        return filename
