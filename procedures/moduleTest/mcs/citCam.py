from importlib import reload

from . import camera
reload(camera)
#from .camera import Camera as BaseCamera

class CitCamera(camera.Camera):
    filePrefix = 'PFCC'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.logger.warn('cit...')
        self._exptime = 0.25

    def _camConnect(self):
        if self.simulationPath is None:
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

    def trim(self, x, y):
        """ Return indices or mask of all valid points. """

        w = (y < (x + 500)) & (y > (x - 500))
        return w
