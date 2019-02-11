# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:32:53 2018

@author: cobra
"""
import time

import astropy.io.fits as pyfits
import numpy as np

import andor

def getCam():
    try:
        return getCam.cam
    except:
        getCam.cam = andor.Andor()
        initPfsAndor(getCam.cam)
        return getCam.cam

def initPfsAndor(cam=None):
    if cam is None:
        cam = getCam()

    cam.SetVerbose(False)
    cam.SetSingleScan()
    cam.SetShutter(1,0,50,50)

    return cam

def expose(exptime=0.25, dark=False, name=None):
    cam = initPfsAndor()

    if dark or exptime == 0:
        cam.SetShutter(0,0,0,0)

    cam.SetExposureTime(exptime)
    cam.StartAcquisition()

    time.sleep(exptime+0.1)
    data = []
    cam.GetAcquiredData(data)   # ?!?
    if data == []:
        raise RuntimeError("failed to readout image")

    im = np.array(data).astype('u2').reshape(2048,2048)
    im = np.flipud(im)

    if name is not None:
        filename = saveImage(name, im)
    else:
        filename = None

    return im, filename

def saveImage(name, img):
    hdus = pyfits.HDUList()
    hdus.append(pyfits.CompImageHDU(img, name='IMAGE', uint=True))

    fullname = name+'.fits'
    hdus.writeto(fullname, overwrite=True)

    return fullname
