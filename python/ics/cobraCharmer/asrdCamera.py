from subprocess import Popen
import numpy as np

import astropy.io.fits as fits

def expose(exptime=None, pathname=None):
    if pathname is None:
        raise ValueError('need a filepath')

    p1 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "1", "-e", "18", "-f", os.path.join(pathname, "/cam1_")], stdout=PIPE)
    p2 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "2", "-e", "18", "-f", os.path.join(pathname, "/cam2_")], stdout=PIPE)
    p1.communicate()
    p2.communicate()

    data1 = fits.getdata(dataPath+'/cam1_0001.fits')
    data2 = fits.getdata(dataPath+'/cam2_0001.fits')
    data = np.stack((data1, data2), axis=1)

    filename = saveImage(pathname, data)
    return data, filename

def saveImage(name, img):
    hdus = pyfits.HDUList()
    hdus.append(pyfits.CompImageHDU(img, name='IMAGE', uint=True))

    filename = name+'.fits'
    hdus.writeto(filename, overwrite=True)

    return filename
