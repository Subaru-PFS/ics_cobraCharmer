from subprocess import Popen, PIPE
import pathlib
import subprocess as sub

import numpy as np
import astropy.io.fits as pyfits

from .camera import Camera

class RmodCamera(Camera):
    filePrefix = 'PFAC'

    def __init__(self, **kw):
        super().__init__(**kw)

        self.exptime = 4
        self.data = None
        self.logger.info('RMOD 71M...')

    def _camConnect(self):
        if self.simulationPath is not None:
            return None

        p = sub.Popen(['/opt/EDTpdv/initcam', '-f', 
                    '/home/pfs/mhs/devel/ics_mcsActor/etc/illunis-71mp.cfg'],
                    stdout=sub.PIPE, stderr=sub.PIPE)

        output, errors = p.communicate()
        string=errors[23:-1]
        if (string == 'done'):
            self.logger.info(f'Camera initialization message: {string}')


        p = sub.Popen(['rmodcontrol', '-e', self.exptime],
                    stdout=sub.PIPE, stderr=sub.PIPE)

        output, errors = p.communicate()
        self.data = None

    def _camExpose(self, exptime, _takeDark=False):
        
        slicename = '/tmp/rmodexposure.fits'
        p = sub.Popen(['rmodexposure', '-f', slicename, '-l', '1'],stdout=sub.PIPE, stderr=sub.PIPE)
        output, errors = p.communicate()

        f = pyfits.open(slicename)
        data = f[0].data
        
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
