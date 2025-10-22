from importlib import reload
import numpy as np
import time
import subprocess as sub
import astropy.io.fits as pyfits
import threading
import pathlib
from ics.cobraCharmer.cobraCoach.mcs import camera
reload(camera)

class McsCamera(camera.Camera):
    filePrefix = 'PFSC'

    def __init__(self,actor=None, **kw):
        super().__init__(**kw)
        if actor is not None:
            self.actor = actor

        self.logger.info('MCS camera')
        self.imageSize = (7096, 10000)
        self._exptime = 0.5
        self._lock = threading.Lock()
        self.frameId = None

    def _camConnect(self):
        if self.simulationPath is not None:
            return None

        self.logger.info('text="Starting camera initialization."')
        
        if self.actor is None:
            p = sub.Popen(['/opt/EDTpdv/initcam', '-f', '/home/pfs/mhs/devel/ics_cobraCharmer/etc/illusnis-71mp.cfg'],
                      stdout=sub.PIPE, stderr=sub.PIPE)
            output, errors = p.communicate()
            string=errors[23:-1]
            if (string == 'done'):
                self.logger.info('text="Camera initialization message: %s"' % (string))
        else:
            cmdString = f'status'
            cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
                                          forUserCmd=cmd)
            if cmdVar.didFail:
                self.logger.warn('text="Camera initialization failed: %s"' % (string))
                return None
        
    def _camExpose(self, exptime, frameNum=None, _takeDark=False):
        t1=time.time()

        if self.actor is None:
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
        
        else:
            if frameNum is not None:
                frameArg = "frameId={frameNum} "
            else:
                frameArg = ""
            cmdString = f"expose object expTime={exptime:0.2f} {frameArg} doCentroid"
            self.logger.info('exposureState="reading"')
            cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
                                          forUserCmd=self.cmd, timeLim=exptime+10)
            if cmdVar.didFail:
                self.cmd.warn('text=%s' % ('Failed to expose with %s' % (cmdString)))
                return None

            t2=time.time()
        
            filekey= self.actor.models['mcs'].keyVarDict['filename'][0]
            filename = pathlib.Path(filekey)
            datapath = filename.parents[0]
            frameId = int(filename.stem[4:], base=10)
            
            self.frameId = frameId

            self.logger.info(f'MCS frame ID={frameId}')
            self.logger.info(f'MCS image datapath={datapath}, filename={filename}')

            f = pyfits.open(filename)
            image = f[1].data

        # t1=time.time()
    
        # # Command camera to do exposure sequence
        # slicename='/tmp/rmodexpose.fits'

        # self.logger.info('slice name: %s' % (slicename))
        # p = sub.Popen(['rmodexposure', '-f', slicename, '-l', '1'],stdout=sub.PIPE, stderr=sub.PIPE)

        # output, errors = p.communicate()
        # t2=time.time()
    
        # self.logger.info('exposureState="reading"')                
        # f = pyfits.open(slicename)

        # image = f[0].data
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
