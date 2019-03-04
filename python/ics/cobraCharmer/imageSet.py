import logging
import os
import numpy as np

import astropy.io.fits as pyfits
import sep

class ImageSet(object):
    def __init__(self, camera, imageDir, setName=None, makeStack=False, saveSpots=False):
        self.logger = logging.getLogger('images')
        self.camera = camera
        self.imageDir = imageDir
        self.namelist = dict()
        self.setName = setName
        self.makeStack = makeStack
        self.stack = None
        self.allSpots = dict()
        self.saveSpots = saveSpots

    @classmethod
    def makeFromDirectory(cls, imageDir):
        self = cls(None, None, imageDir)
        import pathlib

        p = pathlib.Path(imageDir)
        paths = p.glob('*.fits')
        self.namelist = dict()
        for p in paths:
            self.namelist[p.stem] = str(p)

        return self

    def makePathname(self, name, dir=None, nameArgs=None):
        if nameArgs is None:
            nameArgs = dict()
        filename=name.format(nameArgs)

        if dir is None:
            dir = self.imageDir

        if self.setName:
            dir = os.path.join(dir, self.setName)
        if not os.path.isdir(dir):
            os.mkdir(dir, 0o2775)
        return os.path.normpath(os.path.join(dir, filename))

    def expose(self, name, cameraArgs=None, nameArgs=None, saveSpots=False):
        """Acquire an image set.

        Args
        ----
        name : str
          The non-path part of the filename. We prepend the path
        cameraArgs : dict
          Passed down the camera expose() routine
        nameArgs : dict
          Passed in to .format(nameArgs) the filename
        saveSpots : bool
          Measure the spots and save them.

        Returns
        -------
        im : the image itself
        pathname : the entire final pathname for the saved FITS file.
        """

        if cameraArgs is None:
            cameraArgs = dict()
        if nameArgs is None:
            nameArgs = dict()

        if saveSpots is False:
            saveSpots = self.saveSpots

        filename=self.makePathname(name, nameArgs=nameArgs)
        self.logger.info(f'taking {filename}')
        im, filename = self.camera.expose(name=filename, **cameraArgs)
        self.namelist[name] = filename

        if self.makeStack:
            if self.stack is None:
                self.stack = np.zeros_like(im, dtype='f4')
            bkg = sep.Background(im.astype('f4'))
            self.stack += im - bkg

        if saveSpots:
            spots = self.spots(name)
            self.allSpots[name] = spots
        return im, filename

    def saveImage(self, name, img):
        hdus = pyfits.HDUList()
        hdus.append(pyfits.CompImageHDU(img, name='IMAGE', uint=True))

        if not name.endswith('.fits'):
            filename = name+'.fits'
        else:
            filename = name

        filename = os.path.normpath(filename)
        hdus.writeto(filename, overwrite=True)

        return filename

    def saveStack(self, filename):
        if self.stack is None:
            raise RuntimeError('no stack to save')
        self.saveImage(self.makePathname(filename), self.stack)

    def stream(self, name, nFrames=1, cameraArgs=None, nameArgs=None):
        """Acquire a video image set.
        """

        if cameraArgs is None:
            cameraArgs = dict()
        if nameArgs is None:
            nameArgs = dict()

        filename=self.makePathname(name, nameArgs=nameArgs),
        names = self.camera.stream(filename, nFrames, **cameraArgs)
        self.namelist.extend(names)

        return names

    def spots(self, name, sigma=10.0, doTrim=True, disp=None):
        if name in self.namelist:
            name = self.namelist[name]
        if not os.path.isabs(name):
            name = self.makePathname(name)
        if not name.endswith('.fits'):
            name += '.fits'

        name = os.path.normpath(name)
        im = pyfits.getdata(name)
        objects, imSub, _ = self.getObjects(im, sigma=sigma)

        if disp is not None:
            disp.set('frame clear')
            disp.set_np2arr(imSub)
            disp.set('regions color green')
            for o in objects:
                disp.set(f"regions command {{point {o['x']} {o['y']}}}")

        if doTrim:
            # CIT Only -- wrap this, CPL.
            w = (objects['y'] < (objects['x'] + 500)) & (objects['y'] > (objects['x'] - 500))
            objects = objects[w]

            if disp is not None:
                disp.set('regions color red')
                for o in objects:
                    disp.set(f"regions command {{circle {o['x']} {o['y']} 10}}")

        return objects, imSub

    def getObjects(self, im, sigma=10.0):
        data = im.astype('f4')
        bkg = sep.Background(data)
        data_sub = data - bkg

        mn = np.mean(data_sub)
        std = np.std(data_sub)
        thresh = sigma * std
        objects = sep.extract(data_sub, thresh=thresh)

        return objects, data_sub, bkg
