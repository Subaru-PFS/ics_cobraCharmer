from subprocess import Popen, PIPE
import pathlib

import numpy as np
import astropy.io.fits as pyfits

from .camera import Camera

class AsrdCamera(Camera):
    filePrefix = 'PFBC'

def expose(exptime=None, dataPath=None):
    if dataPath is None:
        raise ValueError('need a filepath')

    p1 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "1", "-e", "18", "-f", pathlib.Path(dataPath, "cam1_")], stdout=PIPE)
    p2 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "2", "-e", "18", "-f", pathlib.Path(dataPath, "cam2_")], stdout=PIPE)
    p1.communicate()
    p2.communicate()

    data1 = pyfits.getdata(pathlib.Path(dataPath, 'cam1_0001.fits'))
    data2 = pyfits.getdata(pathlib.Path(dataPath, 'cam2_0001.fits'))
    data = np.stack((data1, data2), axis=1)

    filename = saveImage(dataPath, data)
    return data, filename

def trim(objects):
    """ Return indices or mask of all valid points. """

    return np.arange(len(objects))

def saveImage(name, img):
    hdus = pyfits.HDUList()
    hdus.append(pyfits.CompImageHDU(img, name='IMAGE', uint=True))

    filename = name+'.fits'
    hdus.writeto(filename, overwrite=True)

    return filename
