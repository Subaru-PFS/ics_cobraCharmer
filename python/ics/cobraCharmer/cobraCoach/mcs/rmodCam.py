from importlib import reload
import numpy as np
import time
import subprocess as sub
import astropy.io.fits as pyfits
import threading

from . import camera
reload(camera)

class RmodCamera(camera.Camera):
    filePrefix = 'PFSC'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.logger.info('RMOD 71M Camera')
        self.imageSize = (7096, 10000)
        self._exptime = 0.5
        self._lock = threading.Lock()
        

    def _camConnect(self):
        if self.simulationPath is not None:
            return None

        self.logger.info('text="Starting camera initialization."')
        p = sub.Popen(['/opt/EDTpdv/initcam', '-f', '/home/pfs/mhs/devel/ics_cobraCharmer/etc/illusnis-71mp.cfg'],
                      stdout=sub.PIPE, stderr=sub.PIPE)
        output, errors = p.communicate()
        string=errors[23:-1]
        if (string == 'done'):
            self.logger.info('text="Camera initialization message: %s"' % (string))

    def _camExpose(self, exptime, _takeDark=False):
        
        t1=time.time()
    
        # Command camera to do exposure sequence
        slicename='/tmp/rmodexpose.fits'

        self.logger.info('slice name: %s' % (slicename))
        p = sub.Popen(['rmodexposure', '-f', slicename, '-l', '1'],stdout=sub.PIPE, stderr=sub.PIPE)

        output, errors = p.communicate()
        t2=time.time()
    
        self.logger.info('exposureState="reading"')                
        f = pyfits.open(slicename)

        image = f[0].data
        t3=time.time()
        self.logger.info('Time for exposure = %f. '% ((t2-t1)/1.))
        self.logger.info('text="Time for image loading= %f. '% ((t3-t2)/1.))

        return image

    def trim(self, x, y):
        """ Return indices or mask of all valid points. """

        # x = 2000 - x
        w = (y < (x + 500)) & (y > (x - 500))
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
