import threading

import numpy as np

from ics.cobraCharmer.cobraCoach.mcs.camera import Camera

try:
    from procedures.idsCamera import idsCamera
except Exception:
    idsCamera = None


class AsrdCamera(Camera):
    filePrefix = 'PFAC'

    def __init__(self, **kw):
        super().__init__(**kw)

        self.devIds = 1,2
        self.data = None
        self.logger.info('asrd...')
        self._lock = threading.Lock()

    def _camConnect(self):
        if self.simulationPath is not None:
            return None

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
