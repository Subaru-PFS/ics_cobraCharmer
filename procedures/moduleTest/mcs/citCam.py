from importlib import reload
import numpy as np
import time

from . import camera
reload(camera)

class CitCamera(camera.Camera):
    filePrefix = 'PFCC'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.logger.warn('cit...')
        self._exptime = 0.25

    def _camConnect(self):
        if self.simulationPath is not None:
            return None

        from Camera import andor
        self._camClose()
        cam = andor.Andor()
        self._initPfsAndor(cam)
        self._cam = cam
        return self._cam

    def _initPfsAndor(self, cam):
        cam.SetVerbose(False)
        cam.SetSingleScan()
        cam.SetShutter(1,0,50,50)

        return cam

    def _camExpose(self, exptime, _takeDark=False):
        cam = self.cam
        if _takeDark or exptime == 0:
            cam.SetShutter(0,0,0,0)

        cam.SetExposureTime(exptime)
        cam.StartAcquisition()

        time.sleep(exptime+0.1)
        data = []
        cam.GetAcquiredData(data)   # ?!?
        if data == []:
            raise RuntimeError("failed to readout image")

        im = np.array(data).astype('u2').reshape(2048,2048)
        im = np.flipud(im)

        return im

    def trim(self, x, y):
        """ Return indices or mask of all valid points. """

        w = (y < (x + 500)) & (y > (x - 500))
        return w
