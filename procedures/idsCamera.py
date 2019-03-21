
from pyueye import ueye
import numpy as np
import cv2
import sys
import math
from astropy.io import fits

class idsCamera():
    def setExpoureTime(self, expTime):
        #Pixel-Clock Setting, the range of this camera is 7-35 MHz
        nPixelClockDefault=ueye.INT(200)
        nRet = ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_SET,nPixelClockDefault, 
            ueye.sizeof(nPixelClockDefault))
        
        if nRet != ueye.IS_SUCCESS:
            print("is_PixelClock ERROR")
       
        # Working on exposure time range. Set exposure time to be 20 ms.
        ms = ueye.DOUBLE(expTime)
        
        nRet = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ms,ueye.sizeof(ms))
        if nRet != ueye.IS_SUCCESS:
            print("is_Exposure ERROR")
        
            
    def getExposureTime(self):
        pass

    def getCurrentFrame(self):
        nRet = ueye.is_FreezeVideo(self.hCam, ueye.IS_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")
        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID, self.width, 
            self.height, self.nBitsPerPixel, self.pitch)

        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")

        print("getting image")
        array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
        frame = np.reshape(array,(self.height.value, self.width.value, self.bytes_per_pixel))
        print(frame.shape)
        
        coadd = np.zeros(frame.shape[0:2]).astype('float')
        #coadd[:,:] = (frame[:,:,1]*255).astype('float')+frame[:,:,0].astype('float')
        coadd[:,:] = frame[:,:,1].astype('float')*255+frame[:,:,0].astype('float')
        return coadd

    def __init__(self, deviceID):
        self.deviceID = deviceID
        
        DeviceID =  self.deviceID | ueye.IS_USE_DEVICE_ID
        self.hCam = ueye.HIDS(DeviceID)             
        self.sensorInfo = ueye.SENSORINFO()
        self.camInfo = ueye.CAMINFO()
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        rectAOI = ueye.IS_RECT()
        self.pitch = ueye.INT()
        self.nBitsPerPixel = 10    #24: bits per pixel for color mode; take 8 bits per pixel for monochrome
        channels = 1                    #3: channels for color mode(RGB); take 1 channel for monochrome
        m_nColorMode = ueye.IS_CM_MONO10		# Y8/RGB16/RGB24/REG32
        bytes_per_pixel = int(self.nBitsPerPixel / 8)
        #nColorMode = IS_CM_MONO10;
        #self.nBitsPerPixel = 10;
        # Starts the driver and establishes the connection to the camera
        nRet = ueye.is_InitCamera(self.hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")

        # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that self.camInfo points to
        nRet = ueye.is_GetCameraInfo(self.hCam, self.camInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")

        # You can query additional information about the sensor type used in the camera
        nRet = ueye.is_GetSensorInfo(self.hCam, self.sensorInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")

        nRet = ueye.is_ResetToDefault(self.hCam)
        if nRet != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")
       
        # Set display mode to DIB
        nRet = ueye.is_SetDisplayMode(self.hCam, ueye.IS_SET_DM_DIB)

        if int.from_bytes(self.sensorInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            m_nColorMode = ueye.IS_CM_MONO10
            self.nBitsPerPixel = ueye.INT(10)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 7) + 1
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tm_nColorMode: \t\t", m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        else:
            # for monochrome camera models use Y8 mode
            m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8) + 1
            print("else")

        # Can be used to set the size and position of an "area of interest"(AOI) within an image
        nRet = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

        self.width = rectAOI.s32Width
        self.height = rectAOI.s32Height

        # Prints out some information about the camera and the sensor
        print("Camera model:\t\t", self.sensorInfo.strSensorName.decode('utf-8'))
        print("Camera serial no.:\t", self.camInfo.SerNo.decode('utf-8'))
        print("Maximum image self.width:\t", self.width)
        print("Maximum image self.height:\t", self.height)
        print()

        #---------------------------------------------------------------------------------------------------------------------------------------

        # Allocates an image memory for an image having its dimensions defined by self.width 
        # and self.height and its color depth defined by self.nBitsPerPixel
        nRet = ueye.is_AllocImageMem(self.hCam, self.width, self.height, self.nBitsPerPixel, self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, m_nColorMode)


    def __del__(self):
        # Releases an image memory that was allocated using is_AllocImageMem() and 
        # removes it from the driver management
        ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)

        # Disables the self.hCam camera handle and releases the data structures and 
        # memory areas taken up by the uEye camera
        ueye.is_ExitCamera(self.hCam)

        print("Camera close!")


def main():

    camera = idsCamera(1)
    camera.setExpoureTime(20)
    image = camera.getCurrentFrame()
    hdu = fits.PrimaryHDU(image)
    hdu.writeto('new1.fits',overwrite=True)

if __name__ == '__main__':
    main()