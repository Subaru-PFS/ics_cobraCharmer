import time
from importlib import reload

import numpy as np

from ics.cobraCharmer.cobraCoach.mcs import camera

try:
    from Camera import andor
except Exception:
    andor = None

reload(camera)

class CitCamera(camera.Camera):
    filePrefix = 'PFCC'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.logger.info('cit...')
        self._exptime = 0.5

    def _camConnect(self):
        if self.simulationPath is not None:
            return None

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
        # im = np.ascontiguousarray(np.rot90(im, 1))

        # The ASRD reduction code curently requires that phi moves go CCW when moving CW.
        # So make sure we get that.
        im = np.fliplr(im)

        return im

    def trim(self, x, y):
        """ Return indices or mask of all valid points. """

        # x = 2000 - x
        w = (y < (x + 500)) & (y > (x - 500))
        return w
