import sys
import glob
import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
import sep
from ics.cobraCharmer import pfi as pfiControl

class AnalyzeMotorMapTask ( object ):
    def __init__ ( self, dirname, steps,
                   ncobras=57,
                   repeat=3,                   
                   cam_split=26,
                   thetaSteps=10000,
                   phiSteps=5000 ):
        self.dataPath = dirname + '/image/'
        self.prodctPath = dirname + '/product/'

        self.ncobras = ncobras
        self.steps = steps
        self.repeat = repeat
        self.cam_split = cam_split
        self.thetaSteps = thetaSteps
        self.phiSteps = phiSteps

        self.define_irregulars () # \\ to-do : clean this


    def load_xml ( self, xml_filename, overwrite_centers=False ):
        '''
        Load module calibration file
        '''
        pfi = pfiControl.PFI(fpgaHost='', doConnect=False,
                             doLoadModel=False, )
        pfi.loadModel( xml_filename )
        self.pfi = pfi
        if overwrite_centers:
            self.get_centers_atasiaa (overwrite=True)

    def define_irregulars ( self ):
        '''
        Indicate broken/bent cobras
        '''
        brokens = [1, 39, 43, 54]
        visibles= [e for e in range(1,1+self.ncobras) if e not in brokens]
        badIdx = np.array(brokens) - 1
        goodIdx = np.array(visibles) - 1
        self.goodIdx = goodIdx

        goodGroupIdx = {}
        for group in range(3):
            goodGroupIdx[group] = goodIdx[goodIdx%3==group]
        self.goodGroupIdx = goodGroupIdx

    def circle_fitting(self, p):
        x = np.real(p)
        y = np.imag(p)
        m = np.vstack([x, y, np.ones(len(p))]).T
        n = np.array(x*x + y*y)
        a, b, c = np.linalg.lstsq(m, n, rcond=None)[0]
        return a/2, b/2, np.sqrt(c+(a*a+b*b)/4)


    def lazyIdentification(self, centers, spots, radii=None):
        '''
        Associate fibers with PID
        '''
        n = len(centers)
        if radii is not None and len(radii) != n:
            raise RuntimeError("number of centers must match number of radii")
        ans = np.empty(n, dtype=int)
        for i in range(n):
            dist = np.absolute(spots - centers[i])
            j = np.argmin(dist)
            if radii is not None and np.absolute(centers[i] - spots[j]) > radii[i]:
                ans[i] = -1
            else:
                ans[i] = j
        return ans

    def extract_sep ( self, data, thresh=50 ):
        '''
        Do fiber image segmentation with sep
        '''
        cs = sep.extract(data.astype(float), thresh)
        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        return spots
    
    def get_centers_atasiaa ( self, overwrite=False ):
        '''
        A hack to get the centers from the ASIAA 2 camera set-up. The IDs from this
        approach may not match the Caltech system.
        '''
        centers = np.zeros([self.repeat, self.ncobras], dtype=complex)
        for camera_idx in [1,2]:
            for n in range(self.repeat):
                home_image = fits.getdata(self.dataPath+f'/phi{camera_idx}Begin{n}_0001.fits.fz')
                cpos = self.extract_sep ( home_image )
                if camera_idx == 1:
                    gidx = np.argsort(np.real(cpos))[::-1][:self.cam_split+1]
                    centers[n,self.goodIdx[:self.cam_split+1]] = cpos[gidx]
                else:
                    gidx = np.argsort(np.real(cpos))[:self.cam_split]
                    centers[n,self.goodIdx[self.cam_split+1:]] = cpos[gidx]

        if overwrite:
            self.pfi.calibModel.centers = centers.mean(axis=0)
        return centers

    def catalog_positions ( self ):
        '''
        From ASIAA images of the cobras, centroid and output xy fiber coordinates
        '''
        
        # \\ to-do : is there any difference between the FW/REV theta/phi code
        #    besides the filenames and Nsteps

        # variable declaration for position measurement
        thetaFW = np.zeros((57, self.repeat, self.thetaSteps//self.steps+1), dtype=complex)
        thetaRV = np.zeros((57, self.repeat, self.thetaSteps//self.steps+1), dtype=complex)
        phiFW = np.zeros((57, self.repeat, self.phiSteps//self.steps+1), dtype=complex)
        phiRV = np.zeros((57, self.repeat, self.phiSteps//self.steps+1), dtype=complex)
        

        # phi stages
        for nCam in [1,2]:
            if (nCam == 1): myIdx = self.goodIdx[self.goodIdx <= self.cam_split]
            if (nCam == 2): myIdx = self.goodIdx[self.goodIdx > self.cam_split]
            centers = self.pfi.calibModel.centers[myIdx]            

            # forward phi
            cnt = self.phiSteps//self.steps
            for n in range(self.repeat):
                data = fits.getdata(self.dataPath+f'/phi{nCam}Begin{n}_0001.fits.fz')
                spots = self.extract_sep ( data )

                idx = self.lazyIdentification(centers, spots)

                phiFW[myIdx,n,0] = spots[idx]
                stack_image = data
                for k in range(cnt):
                    data = fits.getdata(self.dataPath+f'/phi{nCam}Forward{n}N{k}_0001.fits.fz')
                    #print(f'/phi{nCam}Forward{n}N{k}_0001.fits.fz')
                    cs = sep.extract(data.astype(float), 50)
                    spots = np.array([c['x']+c['y']*(1j) for c in cs])
                    idx = self.lazyIdentification(centers, spots)
                    phiFW[myIdx,n,k+1] = spots[idx]
                    stack_image = stack_image + data
                fits.writeto(self.prodctPath+f'/Cam{nCam}phiForwardStack.fits.fz',stack_image,overwrite=True)

            # reverse phi
            for n in range(self.repeat):
                data = fits.getdata(self.dataPath+f'/phi{nCam}End{n}_0001.fits.fz')
                cs = sep.extract(data.astype(float), 50)
                spots = np.array([c['x']+c['y']*(1j) for c in cs])
                idx = self.lazyIdentification(centers, spots)
                phiRV[myIdx,n,0] = spots[idx]
                stack_image = data   
                for k in range(cnt):
                    data = fits.getdata(self.dataPath+f'/phi{nCam}Reverse{n}N{k}_0001.fits.fz')
                    cs = sep.extract(data.astype(float), 50)
                    spots = np.array([c['x']+c['y']*(1j) for c in cs])
                    idx = self.lazyIdentification(centers, spots)
                    phiRV[myIdx,n,k+1] = spots[idx]
                    stack_image = stack_image + data
                fits.writeto(self.prodctPath+f'/Cam{nCam}phiReverseStack.fits.fz',stack_image,overwrite=True)

            # forward theta
            cnt = self.thetaSteps//self.steps
            for n in range(self.repeat):
                data = fits.getdata(self.dataPath+f'/theta{nCam}Begin{n}_0001.fits.fz')
                cs = sep.extract(data.astype(float), 50)
                spots = np.array([c['x']+c['y']*(1j) for c in cs])
                idx = self.lazyIdentification(centers, spots)
                thetaFW[myIdx,n,0] = spots[idx]
                stack_image = data   
                for k in range(cnt):
                    data = fits.getdata(self.dataPath+f'/theta{nCam}Forward{n}N{k}_0001.fits.fz')
                    cs = sep.extract(data.astype(float), 50)
                    spots = np.array([c['x']+c['y']*(1j) for c in cs])
                    idx = self.lazyIdentification(centers, spots)
                    thetaFW[myIdx,n,k+1] = spots[idx]
                    stack_image = stack_image + data
                fits.writeto(self.prodctPath+f'/Cam{nCam}thetaForwardStack.fits.fz',stack_image,overwrite=True)


            # reverse theta
            for n in range(self.repeat):
                data = fits.getdata(self.dataPath+f'/theta{nCam}End{n}_0001.fits.fz')
                cs = sep.extract(data.astype(float), 50)
                spots = np.array([c['x']+c['y']*(1j) for c in cs])
                idx = self.lazyIdentification(centers, spots)
                thetaRV[myIdx,n,0] = spots[idx]
                stack_image = data    
                for k in range(cnt):
                    data = fits.getdata(self.dataPath+f'/theta{nCam}Reverse{n}N{k}_0001.fits.fz')
                    cs = sep.extract(data.astype(float), 50)
                    spots = np.array([c['x']+c['y']*(1j) for c in cs])
                    idx = self.lazyIdentification(centers, spots)
                    thetaRV[myIdx,n,k+1] = spots[idx]
                    stack_image = stack_image + data
                fits.writeto(self.prodctPath+f'/Cam{nCam}thetaReverseStack.fits.fz',stack_image,overwrite=True)
        return thetaFW, thetaRV, phiFW, phiRV

    def fix_geometry ( self, thetaFW, thetaRV, phiFW, phiRV ):
        '''
        Convert (x,y) positions to (theta,phi)
        '''
        # variable declaration for theta, phi angles
        thetaCenter = np.zeros(57, dtype=complex)
        phiCenter = np.zeros(57, dtype=complex)
        thetaAngFW = np.zeros((57, self.repeat, self.thetaSteps//self.steps+1), dtype=float)
        thetaAngRV = np.zeros((57, self.repeat, self.thetaSteps//self.steps+1), dtype=float)
        phiAngFW = np.zeros((57, self.repeat, self.phiSteps//self.steps+1), dtype=float)
        phiAngRV = np.zeros((57, self.repeat, self.phiSteps//self.steps+1), dtype=float)

        # measure centers
        for c in self.goodIdx:
            data = np.concatenate((thetaFW[c].flatten(), thetaRV[c].flatten()))
            x, y, r = self.circle_fitting(data)
            thetaCenter[c] = x + y*(1j)
            data = np.concatenate((phiFW[c].flatten(), phiRV[c].flatten()))
            x, y, r = self.circle_fitting(data)
            phiCenter[c] = x + y*(1j)

        # measure theta angles
        cnt = self.thetaSteps//self.steps
        for c in self.goodIdx:
            for n in range(self.repeat):
                for k in range(cnt+1):
                    thetaAngFW[c,n,k] = np.angle(thetaFW[c,n,k] - thetaCenter[c])
                    thetaAngRV[c,n,k] = np.angle(thetaRV[c,n,k] - thetaCenter[c])
                home = thetaAngFW[c,n,0]
                thetaAngFW[c,n] = (thetaAngFW[c,n] - home) % (np.pi*2)
                thetaAngRV[c,n] = (thetaAngRV[c,n] - home) % (np.pi*2)

        # fix over 2*pi angle issue
        for c in self.goodIdx:
            for n in range(self.repeat):
                for k in range(cnt):
                    if thetaAngFW[c,n,k+1] < thetaAngFW[c,n,k]:
                        thetaAngFW[c,n,k+1] += np.pi*2
                for k in range(cnt):
                    if thetaAngRV[c,n,k+1] > thetaAngRV[c,n,k]:
                        thetaAngRV[c,n,k] += np.pi*2
                    else:
                        break
                for k in range(cnt):
                    if thetaAngRV[c,n,k+1] > thetaAngRV[c,n,k]:
                        thetaAngRV[c,n,k+1] -= np.pi*2

        # measure phi angles
        cnt = self.phiSteps//self.steps + 1
        for c in self.goodIdx:
            for n in range(self.repeat):
                for k in range(cnt):
                    phiAngFW[c,n,k] = np.angle(phiFW[c,n,k] - phiCenter[c])
                    phiAngRV[c,n,k] = np.angle(phiRV[c,n,k] - phiCenter[c])
                home = phiAngFW[c,n,0]
                phiAngFW[c,n] = (phiAngFW[c,n] - home + np.pi/2) % (np.pi*2) - np.pi/2
                phiAngRV[c,n] = (phiAngRV[c,n] - home + np.pi/2) % (np.pi*2) - np.pi/2
        return thetaAngFW,thetaAngRV,phiAngFW,phiAngRV


    def generate_motormap ( self, thetaAngFW,thetaAngRV,phiAngFW,phiAngRV ):
        '''
        Generate a minimal motormap (average speed per step in deg/step).
        '''
        mmap_list = []
        for ang_data in [thetaAngFW,thetaAngRV,phiAngFW,phiAngRV]:
            deg_moves = np.rad2deg(ang_data)
            speed = np.diff(deg_moves)/ self.steps
            mdeg = (deg_moves[:,:,:-1]+deg_moves[:,:,1:])/2.
            mmap_list.append(mdeg + speed*1.j)
        return mmap_list
            
        

