import os
import numpy as np

import astropy.io.fits as pyfits
import sep

class ImageSet(object):
    def __init__(self, pfi, camera, output, makeStack=False):
        self.camera = camera
        self.output = output
        self.namelist = dict()
        self.makeStack = makeStack
        self.stack = None

    def makePathname(self, name, nameArgs=None):
        if nameArgs is None:
            nameArgs = dict()
        filename=name.format(nameArgs)

        return os.path.join(self.output.imageDir,
                            filename)

    def expose(self, name, cameraArgs=None, nameArgs=None):
        """Acquire an image set.
        """

        if cameraArgs is None:
            cameraArgs = dict()
        if nameArgs is None:
            nameArgs = dict()

        filename=self.makePathname(name, nameArgs)
        im, filename = self.camera.expose(name=filename, **cameraArgs)
        self.namelist[name] = filename

        if self.makeStack:
            if self.stack is None:
                self.stack = np.zeros_like(im, dtype='f4')
            self.stack += im

        return im, filename

    def saveImage(self, name, img):
        hdus = pyfits.HDUList()
        hdus.append(pyfits.CompImageHDU(img, name='IMAGE', uint=True))

        if not name.endswith('.fits'):
            filename = name+'.fits'
        else:
            filename = name

        hdus.writeto(filename, overwrite=True)

        return filename

    def saveStack(self, filename):
        if self.stack is None:
            raise RuntimeError('no stack to save')
        self.saveImage(self.makePathname(filename).self.stack)

    def stream(self, name, nFrames=1, cameraArgs=None, nameArgs=None):
        """Acquire a video image set.
        """

        if cameraArgs is None:
            cameraArgs = dict()
        if nameArgs is None:
            nameArgs = dict()

        filename=self.makePathname(name, nameArgs),
        names = self.camera.stream(filename, nFrames, **cameraArgs)
        self.namelist.extend(names)

        return names

    def spots(self, name, sigma=5.0, doTrim=True):
        if name in self.namelist:
            name = self.namelist[name]

        im = pyfits.getdata(name)
        objects, _, _ = self.getObjects(im, sigma=sigma)

        if doTrim:
            # CIT Only -- wrap this, CPL.
            w = objects['y'] < (objects['x'] + 500)
            objects = objects[w]

        return objects, im

    def getObjects(self, im, sigma=5.0):
        data = im.astype('f4')
        bkg = sep.Background(data)
        data_sub = data - bkg

        mn = np.mean(data_sub)
        std = np.std(data_sub)
        thresh = sigma * std
        objects = sep.extract(data_sub, thresh=thresh)

        return objects, data_sub, bkg
