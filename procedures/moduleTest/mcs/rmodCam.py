from importlib import reload
import numpy as np
import time
import subprocess as sub
import astropy.io.fits as pyfits

from . import camera
reload(camera)

class RmodCamera(camera.Camera):
    filePrefix = 'PFSC'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.logger.info('RMOD 71M Camera')
        self.imageSize = (8960, 5778)
        self._exptime = 0.5

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
