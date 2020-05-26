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

        from procedures.idsCamera import idsCamera
        cams = []
        for devId in self.devIds:
            cam = idsCamera(devId)
            cam.setExpoureTime(20)
            cams.append(cam)

        self.data = None
        self._cam = cams

    def _camExpose(self, exptime, _takeDark=False):
        cams = self.cam

        _data = []
        for cam in cams:
            _data.append(cam.getCurrentFrame())
        _data.reverse()
        data = np.concatenate(_data, axis=1)
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
        return self.data[:, w//2:]

    @property
    def im2(self):
        if self.data is None:
            return None
        h, w = self.data.shape
        return self.data[:, :w//2]

    def trim(self, x, y):
        """ Return indices or mask of all valid points. """

        w = (y > 600) & (y < 1400)
        return w
