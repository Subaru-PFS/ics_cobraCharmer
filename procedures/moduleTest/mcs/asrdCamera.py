from subprocess import Popen, PIPE
import pathlib

import numpy as np
import astropy.io.fits as pyfits

from .camera import Camera

class AsrdCamera(Camera):
    filePrefix = 'PFAC'

    def __init__(self, **kw):
        super().__init__(**kw)

        self.devIds = 1,2
        self.data = None
        self.logger.info('asrd...')

    def _camConnect(self):
        if self.simulationPath is not None:
            return None

        from idsCamera import idsCamera
        cams = []
        for devId in devIds:
            cam = idsCamera(devId)
            cam.setExpoureTime(20)
            cams.append(cam)

        self.data = None
        self._cams = cams

    def _camExpose(self, exptime, _takeDark=False):
        cams = self.cam

        data1 = cams[0].getCurrentFrame()
        data2 = cams[1].getCurrentFrame()
        data = np.stack((data1, data2), axis=1)
        self.data = data

        return data

    def reload(self):
        self._camClose()
        self._camConnect()

    @property
    def im1(self):
        if self.data is None:
            return None
        h, w = self.data.shape
        return self.data[:, :w//2]

    @property
    def im2(self):
        if self.data is None:
            return None
        h, w = self.data.shape
        return self.data[:, w//2:]

    def trim(objects):
        """ Return indices or mask of all valid points. """

        return np.arange(len(objects))
