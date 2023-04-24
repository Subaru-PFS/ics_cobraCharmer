import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import glob
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import sys
import math
import pathlib
from scipy import optimize
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable




from bokeh.io import output_notebook, show, export_png,export_svgs
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


from bokeh.plotting import figure, show, output_file
import bokeh.palettes
from bokeh.layouts import column,gridplot
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper
from bokeh.models.glyphs import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable


from bokeh.transform import linear_cmap
from bokeh.palettes import Category20

from ics.cobraCharmer import pfiDesign
from ics.cobraCharmer import func
import fnmatch
from ics.fpsActor import fpsFunction as fpstool
import pandas as pd
from opdb import opdb
import logging

from pfs.utils.butler import Butler
import pfs.utils.coordinates.transform as transformUtils


def findVisit(runDir):
    return int(pathlib.Path(sorted(glob.glob(f'/data/MCS/{runDir}/data/PFSC*.fits'))[0]).name[4:-7])

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def gaussianFit(n, bins, initGuess = None):
    shift = np.abs((bins[0]-bins[1])/2)
    histy = n
    histx = bins[:-1]+shift

    if initGuess is None:
        popt,pcov = curve_fit(gaus,histx,histy,p0=[1,np.sum(histx*histy)/np.sum(histy),0.01])
    else:
        mean, sigma = initGuess[0], initGuess[1]
        
        popt,pcov = curve_fit(gaus,histx,histy,p0=[1,mean, sigma])
    
    sigma=np.abs(popt[2])

    return popt

class VisDianosticPlot(object):

    def __init__(self, runDir=None, xml=None, arm=None, datatype=None):
        
        
        # Initializing the logger
        logging.basicConfig(format="%(asctime)s.%(msecs)03d %(levelno)s %(name)-10s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")
        self.logger = logging.getLogger('visDianosticPlot')
        self.logger.setLevel(logging.INFO)


        if arm != None:
            self.arm = arm


        if runDir != None:
            if os.path.exists(f'/data/MCS/{runDir}/') is False:
                self.path = f'/data/MCS_Subaru/{runDir}/'
            else:
                self.path = f'/data/MCS/{runDir}/'

        if xml != None:
            #xml = pathlib.Path(f'{self._findXML(self.path)[0]}')
            des = pfiDesign.PFIDesign(xml)
            self.calibModel = des
            cobras = []
            for i in des.findAllCobras():
                c = func.Cobra(des.moduleIds[i],
                            des.positionerIds[i])
                cobras.append(c)
            allCobras = np.array(cobras)
            nCobras = len(allCobras)

            goodNums = [i+1 for i,c in enumerate(allCobras) if
                    des.cobraIsGood(c.cobraNum, c.module)]
            badNums = [e for e in range(1, nCobras+1) if e not in goodNums]


            self.goodIdx = np.array(goodNums, dtype='i4') - 1
            self.badIdx = np.array(badNums, dtype='i4') - 1

        if datatype == 'MM':
            self._loadCobraMMData(arm=arm)

    def __del__(self):
        if hasattr(self,'fw'):
            del(self.fw)
            del(self.rv)
            del(self.af)
            del(self.ar)
            del(self.sf)
            del(self.sr)

    def _findXML(self,path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, '*.xml'):
                    result.append(os.path.join(root, name))
        return result

    def _findFITS(self):
        path = self.path
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, 'PFSC*.fits'):
                    result.append(os.path.join(root, name))
    
    
        return result

    def _loadCobraMMData(self, arm=None) :
        
        try:
            self.alt=pyfits.open(self._findFITS()[0])[0].header['ALTITUDE']
        except:
            self.alt=90

        path = self.path+'data/'
        
        if arm is None:
            raise Exception('Define the arm')

        fwdFitsFile= f'{path}{arm}ForwardStack0.fits'
        revFitsFile= f'{path}{arm}ReverseStack0.fits'

        #fwdImage = pyfits.open(fwdFitsFile)
        #self.fwdStack=fwdImage[1].data

        #revImage = pyfits.open(revFitsFile)
        #self.revStack=revImage[1].data


        self.centers = np.load(path + f'{arm}Center.npy')
        self.radius = np.load(path + f'{arm}Radius.npy')
        self.fw = np.load(path + f'{arm}FW.npy')
        self.rv = np.load(path + f'{arm}RV.npy')
        self.af = np.load(path + f'{arm}AngFW.npy')
        self.ar = np.load(path + f'{arm}AngRV.npy')
        self.sf = np.load(path + f'{arm}SpeedFW.npy')
        self.sr = np.load(path + f'{arm}SpeedRV.npy')
        self.mf = np.load(path + f'{arm}MMFW.npy')
        self.mr = np.load(path + f'{arm}MMRV.npy')
        try:
            self.badMM = np.load(path + 'badMotorMap.npy')
        except:
            self.badMM = np.load(path + 'bad.npy')
        self.badrange = np.load(path + 'badRange.npy')
        #self.mf2 = np.load(path + 'phiMMFW2.npy')
        #self.mr2 = np.load(path + 'phiMMRV2.npy')
        #self.bad2 = np.load(path + 'bad2.npy')        
    
    def _addLine(self, centers, length, angle, **kwargs):
        ax = plt.gca()
        x = length*np.cos(angle)
        y = length*np.sin(angle)
        for idx in self.goodIdx:
            ax.plot([centers.real[idx],  centers.real[idx]+x[idx]],
                        [centers.imag[idx],centers.imag[idx]+y[idx]],**kwargs)
        pass


    def visCreateNewPlot(self,title, xLabel, yLabel, size=(8, 8), nRows = 1, nCols = 1, 
        aspectRatio="equal", patchAlpha=0, supTitle=True, **kwargs):
        

        #plt.figure(figsize=size, facecolor="white", tight_layout=True, **kwargs)
        fig, ax = plt.subplots(nRows, nCols,figsize=size, facecolor="white", **kwargs)
        fig.patch.set_alpha(patchAlpha)
        
        if nRows*nCols == 1:
        
            plt.clf()
            if supTitle is True:
                plt.suptitle(title)
            else:
                plt.title(title)
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
            plt.show(block=False)

            # Set the axes aspect ratio
            ax = plt.gca()
            ax.set_aspect(aspectRatio)
        else:
            plt.suptitle(title)
            plt.subplots_adjust(wspace=0.25,hspace=0.3)

        

        self.fig = fig

    def visSetAxesLimits(self, xLim, yLim):
        """Sets the axes limits of an already initialized figure.
        
        Parameters
        ----------
        xLim: object
            A numpy array with the x axis limits.
        yLim: object
            A numpy array with the y axis limits.
        """
        ax = plt.gca()
        ax.set_xlim(xLim)
        ax.set_ylim(yLim)
    


    def visGeometryFromXML(self, newXml=None, thetaAngle=None, phiAngle=None,
        markCobra=False, patrol=False, visHardStops = True, allCobra = False):
        
        if newXml is None:
            des = self.calibModel
        else:
            des = pfiDesign.PFIDesign(newXml)
        
        ax = plt.gca()

        if allCobra is False:
            ax.scatter(des.centers.real[self.goodIdx], des.centers.imag[self.goodIdx],marker='o', color='red', s=20)
        else:
            ax.scatter(des.centers.real, des.centers.imag,marker='o', color='red', s=20)

        # Adding theta hard-stops
        if visHardStops is True:
            length=des.L1+des.L1 

            self._addLine(des.centers,length,des.tht0,color='orange',
                            linewidth=0.5,linestyle='--')

            self._addLine(des.centers,length,des.tht1,color='black',
                            linewidth=0.5,linestyle='-.')

        if thetaAngle is not None:
            self._addLine(des.centers,des.L1,thetaAngle,color='blue',
                    linewidth=2,linestyle='-')

        if phiAngle is not None:
            # Calculate the end point first
            x = des.L1*np.cos(thetaAngle)
            y = des.L1*np.sin(thetaAngle)
            newx = des.centers.real + x
            newy = des.centers.imag + y
            newPos = newx+newy*1j
            phiOpenAngle = phiAngle+thetaAngle - np.pi
            self._addLine(newPos,des.L2,phiOpenAngle,color='blue',
                    linewidth=10,linestyle='-', solid_capstyle='round',alpha=0.5)


        if markCobra is True:
            for idx in range(10):
                ax.text(des.centers[idx].real, des.centers[idx].imag,idx)

            for idx in range(798,808):
                ax.text(des.centers[idx].real, des.centers[idx].imag,idx)

            for idx in range(1596,1606):
                ax.text(des.centers[idx].real, des.centers[idx].imag,idx)

        if patrol is True:
            for i in self.goodIdx:
                d = plt.Circle((des.centers[i].real, des.centers[i].imag), 
                   des.L1[i]+des.L2[i], facecolor=('#D9DAFC'),edgecolor=None, 
                   fill=True,alpha=0.7)
                ax.add_artist(d)
    
    def visVisitAllSpots(self, pfsVisitID = None, subVisit = None, camera = None, dataRange=None):
        '''
            This function plots data points of all spots of a given visitID, especailly for convergence and MM run. 
        '''


        if camera is not None:
            cameraName = camera

        ax = plt.gca()

        if pfsVisitID is None:
            visitID = int(self._findFITS()[0][-12:-7])
        else:
            visitID = pfsVisitID

        path=f'{self.path}/data/'
        tarfile = path+'targets.npy'
        
        if os.path.exists(tarfile):
            targets=np.load(tarfile)

        butler = Butler(configRoot=os.path.join(os.environ["PFS_INSTDATA_DIR"], "data"))

        # Read fiducial and spot geometry
        fids = butler.get('fiducials')
        
        try:
            db=opdb.OpDB(hostname='db-ics', port=5432,dbname='opdb',
                        username='pfs')
        except:     
            db=opdb.OpDB(hostname='pfsa-db01', port=5432,dbname='opdb',
                                username='pfs')
        
        dataRange = None
        subVisit = 1

        if subVisit is None:
            if dataRange is None:
                dataRange  = [0,12]
        else:
            dataRange = [subVisit, subVisit+1]


        for count, sub in enumerate(range(*dataRange)):
            subid=sub

            frameid = visitID*100+subid
            try:
                db=opdb.OpDB(hostname='db-ics', port=5432,dbname='opdb',
                        username='pfs')

                match = db.bulkSelect('cobra_match','select * from cobra_match where '
                      f'mcs_frame_id = {frameid}').sort_values(by=['cobra_id']).reset_index()

                mcsData = db.bulkSelect('mcs_data','select * from mcs_data where '
                        f'mcs_frame_id = {frameid}').sort_values(by=['spot_id']).reset_index()
                teleInfo = db.bulkSelect('mcs_exposure','select altitude, insrot from mcs_exposure where '
                        f'mcs_frame_id = {frameid}')
            except:

                db=opdb.OpDB(hostname='pfsa-db01', port=5432,dbname='opdb',
                                    username='pfs')

                match = db.bulkSelect('cobra_match','select * from cobra_match where '
                        f'mcs_frame_id = {frameid}').sort_values(by=['cobra_id']).reset_index()

                mcsData = db.bulkSelect('mcs_data','select * from mcs_data where '
                        f'mcs_frame_id = {frameid}').sort_values(by=['spot_id']).reset_index()
                teleInfo = db.bulkSelect('mcs_exposure','select altitude, insrot from mcs_exposure where '
                        f'mcs_frame_id = {frameid}')

            #if subid == 0:
            #self.logger.info(f'Using first frame for transformation')
            pt = transformUtils.fromCameraName(cameraName,altitude=teleInfo['altitude'].values[0], 
                        insrot=teleInfo['insrot'].values[0])
        
        
            outerRing = np.zeros(len(fids), dtype=bool)
            for i in [29, 30, 31, 61, 62, 64, 93, 94, 95, 96]:
                    outerRing[fids.fiducialId == i] = True
            pt.updateTransform(mcsData, fids[outerRing], matchRadius=8.0, nMatchMin=0.1)
            
            for i in range(2):
                    rfid, rdmin = pt.updateTransform(mcsData, fids, matchRadius=4.2,nMatchMin=0.1)


            xx , yy = pt.mcsToPfi(mcsData['mcs_center_x_pix'],mcsData['mcs_center_y_pix'])
            
            if count == 0:
                ax.plot(xx,yy,'.',label='Spots from transformaion')
                ax.plot(match['pfi_center_x_mm'],match['pfi_center_y_mm'],'x',label='Spots from match table')
            else:
                ax.plot(xx,yy,'.')
                ax.plot(match['pfi_center_x_mm'],match['pfi_center_y_mm'],'x')

        if os.path.exists(tarfile):
            ax.plot(targets.real,targets.imag,'+', label='Final Target')

        ax.legend()

    
    def visArmlengthComp(self, diff, arm='theta', crange=[-0.025, 0.025], hrange=[-0.05,0.05],
        gauFit = True, extraLable=None):
        
        if arm == 'phi':
            arm = 'L2'
        else: arm = 'L1'
        
        fig, ax = plt.subplots(1,2, figsize=(14,6), facecolor="white")

        ax[0].set_aspect("equal")
        sc=ax[0].scatter(self.calibModel.centers.real[self.goodIdx], self.calibModel.centers.imag[self.goodIdx],
            c=diff[self.goodIdx], vmin=crange[0], vmax=crange[1])
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar = fig.colorbar(sc,cax=cax)


        ax[0].set_xlabel('X (mm)')
        ax[0].set_ylabel('y (mm)')

        n, bins, patches = ax[1].hist(diff[self.goodIdx],range=(hrange[0],hrange[1]),bins=30)

        popt = gaussianFit(n, bins)
        sigma=np.abs(popt[2])

        ax[1].plot(bins[:-1],gaus(bins[:-1],*popt),'ro:',label='fit')

        ax[1].text(0.7, 0.8, f'Median = {np.median(diff[self.goodIdx]):.4f}, $\sigma$={sigma:.4f}', 
                        horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
        ax[1].set_xlabel('Difference (mm)')
        ax[1].set_ylabel('Counts')

        if extraLable is None:
            plt.suptitle(f'{arm} Arm Length Comparison')
        else:
            plt.suptitle(f'{arm} Arm Length Comparison {extraLable}')

    def visStoppedCobra(self, pfsVisitID, tolerance = 0.01, getStoppedNum=False):
        '''
            Visulization of stopped cobra in each iteration for a certain visitID
        '''
        ax = plt.gca()

        path=f'{self.path}/data/'
        tarfile = path+'targets.npy'
        movfile = path+'moves.npy'

        targets=np.load(tarfile)
        mov = np.load(movfile)

        mcs_finised = []
        
        maxIteration = mov.shape[2]


        for subID in range(maxIteration+1):
            frameid = pfsVisitID*100+subID
            db=opdb.OpDB(hostname='db-ics', port=5432,
                        dbname='opdb',username='pfs')
                
            match = db.bulkSelect('cobra_match','select * from cobra_match where '
                f'mcs_frame_id = {frameid}').sort_values(by=['cobra_id']).reset_index()
            
            if len(match['pfi_center_x_mm']) != 0:
                dist=np.sqrt((match['pfi_center_x_mm'].values[self.goodIdx]-targets.real)**2+
                    (match['pfi_center_y_mm'].values[self.goodIdx]-targets.imag)**2)
            
                inx = np.where(dist < tolerance)
                mcs_finised.append(len(inx[0]))
        if len(mcs_finised) > maxIteration:
            mcs_finised = np.array(mcs_finised[1:])
        else:
             mcs_finised = np.array(mcs_finised)

        

        fpga_notDone = []
        for iteration in range(maxIteration):
            ind = np.where(np.abs(mov[0,:,iteration]['position']) > 0)
            notDone = len(np.where(np.abs(mov[0,ind[0],iteration]['position']-targets[ind[0]]) > tolerance)[0])
            fpga_notDone.append(notDone)


        fpga_notDone = np.array(fpga_notDone)   
        fpga_finished = len(self.goodIdx) - fpga_notDone

        ax.set_aspect('auto')

        ax.plot(fpga_finished, linestyle ='-', marker='x', label='FPS')
        ax.plot(mcs_finised, linestyle ='-',marker='.',label='MCS')
        ax.plot(fpga_finished - mcs_finised, label = 'FPS - MCS')
        ax.plot(np.zeros(12)+len(self.goodIdx)*0.95,linestyle ='dotted', label = '95% Threshold')
        ax.legend()

        if getStoppedNum:
            return fpga_finished[-1], mcs_finised[-1]

    def visTargetConvergence(self, pfsVisitID, maxIteration = 11, tolerance = 0.01):
        vmax = 4*tolerance
        ax = plt.gcf().get_axes()[0]
        self.visSubaruConvergence(Axes=ax, pfsVisitID = pfsVisitID,subVisit=maxIteration-1,vmax=vmax)
        ax = plt.gcf().get_axes()[1]
        self.visSubaruConvergence(Axes=ax,pfsVisitID = pfsVisitID,subVisit=3,tolerance=tolerance, histo=True, bins=20,range=(0,vmax))
       


    def visSubaruConvergence(self, Axes = None, pfsVisitID=None, subVisit=11, 
        histo=False, heatmap=True, vectormap=False, range=(0,0.08), bins=20, tolerance=0.01, **kwargs):
        '''
            Visulization of cobra convergence result at certain iteration.  
            This fuction gets the locations of all fibers from database
            and then calculate the distance to the final targets. 


        '''
        if Axes is None:
            ax = plt.gca()
        else:
            ax = Axes

        if pfsVisitID is None:
            visitID = int(self._findFITS()[0][-12:-7])
        else:
            visitID = pfsVisitID
        
        if subVisit is not None:    
            subID = subVisit

        frameid = visitID*100+subID
        try:
            db=opdb.OpDB(hostname='pfsa-db01', port=5432,
                   dbname='opdb',username='pfs')
            match = db.bulkSelect('cobra_match','select * from cobra_match where '
                      f'mcs_frame_id = {frameid}').sort_values(by=['cobra_id']).reset_index()
        except:
            db=opdb.OpDB(hostname='db-ics', port=5432,
                   dbname='opdb',username='pfs')
        
            match = db.bulkSelect('cobra_match','select * from cobra_match where '
                      f'mcs_frame_id = {frameid}').sort_values(by=['cobra_id']).reset_index()

        path=f'{self.path}/data/'

        tarfile = path+'targets.npy'
        movfile = path+'moves.npy'

        targets=np.load(tarfile)
        mov = np.load(movfile)

        try:
            maxIteration = mov.shape[2]
        except:
            mov = np.array([mov])
            maxIteration = mov.shape[2]

        dist=np.sqrt((match['pfi_center_x_mm'].values[self.goodIdx]-targets.real)**2+
            (match['pfi_center_y_mm'].values[self.goodIdx]-targets.imag)**2)
        
        self.logger.info(f'Tolerance: {tolerance}')

        ind = np.where(np.abs(mov[0,:,maxIteration-1]['position']) > 0)
        notDone = len(np.where(np.abs(mov[0,ind[0],maxIteration-1]['position']-targets[ind[0]]) > tolerance)[0])
        
        if histo is True:
            heatmap, vectormap = False, False

            ax.set_aspect('auto')
            for subID in np.arange(subVisit,maxIteration):
                
                frameid = visitID*100+subID

                try:
                    db=opdb.OpDB(hostname='db-ics', port=5432,
                        dbname='opdb',username='pfs')
                except:
                    db=opdb.OpDB(hostname='pfsa-db01', port=5432,
                        dbname='opdb',username='pfs')

                match = db.bulkSelect('cobra_match','select * from cobra_match where '
                            f'mcs_frame_id = {frameid}').sort_values(by=['cobra_id']).reset_index()
        

                dist=np.sqrt((match['pfi_center_x_mm'].values[self.goodIdx]-targets.real)**2+
                            (match['pfi_center_y_mm'].values[self.goodIdx]-targets.imag)**2)
                n, bins, patches = ax.hist(dist,range=range, bins=bins, alpha=0.7,
                    histtype='step',linewidth=3,
                    label=f'{subID+1}-th Iteration')
            
            outIdx = np.where(bins > tolerance)
            #print(np.sum(n[outIdx[0][0]:]))
            #print(np.sum(n[np.where(bins > 0.01)[0]]))
            outRegion =  np.sum(n[outIdx[0][0]-1:])
            ax.text(0.7, 0.45, f'N. of non-converged (MCS) = {outRegion}', 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.text(0.7, 0.4, f'N. of non-converged (FPS) = {notDone}', 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.legend(loc='upper right')
            ax.set_xlabel('Distance (mm)')
            ax.set_ylabel('Counts')
            self.logger.info(f'Number of not done cobra (MCS): {outRegion}')
        
        if vectormap:
            heatmap, histo = False, False
            ind = np.where(dist < 0.02)

            ax.set_aspect('auto')
            x = self.calibModel.centers.real[self.goodIdx][ind]
            y = self.calibModel.centers.imag[self.goodIdx][ind]
            dx = (match['pfi_center_x_mm'].values[self.goodIdx]-targets.real)[ind]
            dy = (match['pfi_center_y_mm'].values[self.goodIdx]-targets.imag)[ind]
            vectorLength = 0.01
            q=ax.quiver(x,y, 
                    dx, dy, color='red',units='xy',**kwargs)

         
            ax.quiverkey(q, X=0.2, Y=0.95, U=vectorLength,
                    label=f'length = {vectorLength} mm', labelpos='E')

        
        if heatmap:
            ax.set_aspect('equal')
            sc=ax.scatter(self.calibModel.centers.real[self.goodIdx],self.calibModel.centers.imag[self.goodIdx],
                c=dist,marker='s', **kwargs)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            colorbar = self.fig.colorbar(sc,cax=cax)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            #plt.colorbar(sc)

        #ind = np.where(np.abs(mov[0,:,maxIteration-1]['position']) > 0)
        #notDone = len(np.where(np.abs(mov[0,ind[0],maxIteration-1]['position']-targets[ind[0]]) > tolerance)[0])
        
        self.logger.info(f'Number of still moving cobra: {(len(ind[0]))}')
        self.logger.info(f'Number of not done cobra (FPS): {notDone}')

    def visCobraCenter(self, baseData, targetData, histo=False, gauFit = True, vectorLength=0.05, **kwargs):
        
        '''
            This function is used to compare the center locations between two datasets. 
            
            Input:
                baseData: The referenced center locations 
                targetData: The target to be compared with.
                compareObj:  The target we compare with.  Typically, it is the data-compareObj
        '''
        
        ax = plt.gca()
        title = ax.get_title()

        x = baseData.real
        y = baseData.imag

        dx = targetData.real - x
        dy = targetData.imag - y


        diff = np.sqrt(dx**2+dy**2)

        if histo is True:
            
            ax1 = plt.subplot(212)
            n, bins, patches = ax1.hist(diff,range=(0,np.mean(diff)+2*np.std(diff)), bins=15, color='#0504aa',
                alpha=0.7)

            popt = gaussianFit(n, bins)
            sigma = popt[2]
            if gauFit is True:
                ax1.plot(bins[:-1],gaus(bins[:-1],*popt),'ro:',label='fit')


            ax1.text(0.8, 0.8, f'Median = {np.median(diff):.4f}, $\sigma$={sigma:.4f}', 
                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
            ax1.set_title('2D')
            ax1.set_xlabel('distance (mm)')
            ax1.set_ylabel('Counts')
            ax1.set_ylim(0,1.2*np.max(n))


            ax2 = plt.subplot(221)
            n, bins, patches = ax2.hist(dx,range=(np.mean(dx)-3*np.std(dx),np.mean(dx)+3*np.std(dx)), 
                bins=30, color='#0504aa',alpha=0.7)
            
            popt = gaussianFit(n, bins)
            sigma = popt[2]
            if gauFit is True:
                ax2.plot(bins[:-1],gaus(bins[:-1],*popt),'ro:',label='fit')
            
            ax2.text(0.7, 0.85, f'Median = {np.median(dx):.4f}, $\sigma$={sigma:.4f}', 
                horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title('X direction')
            ax2.set_xlabel('distance (mm)')
            ax2.set_ylabel('Counts')
            ax2.set_ylim(0,1.2*np.max(n))


            ax3 = plt.subplot(222, sharey = ax2)
            ax3.tick_params(axis='both',labelleft=False)


            n, bins, patches = ax3.hist(dy,range=(np.mean(dy)-3*np.std(dx),np.mean(dy)+3*np.std(dx)), 
                bins=30, color='#0504aa', alpha=0.7)
            popt = gaussianFit(n, bins)
            sigma = np.abs(popt[2])
            
            if gauFit is True:
                ax3.plot(bins[:-1],gaus(bins[:-1],*popt),'ro:',label='fit')
            
            ax3.text(0.7, 0.85, f'Median = {np.median(dy):.4f}, $\sigma$={sigma:.4f}', 
                horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

            ax3.set_title('Y direction')
            ax3.set_xlabel('distance (mm)')
            plt.subplots_adjust(wspace=0,hspace=0.3)

            plt.suptitle(title, fontsize=16)


        else:
            
            sigma = np.std(diff)
            self.logger.info(f'Mean = {np.mean(diff):.3f}, Median = {np.median(diff):.3f} Std={np.std(diff):.3f}')
            self.logger.info(f'CobraIdx for large center variant :{np.where(diff >= np.median(diff)+2.0*sigma)[0]}')

            #indx = np.where(diff < np.median(diff)+2.0*sigma)[0]
            
            q=ax.quiver(x,y, 
                    dx, dy, color='red',units='xy',**kwargs)

         
            ax.quiverkey(q, X=0.2, Y=0.95, U=vectorLength,
                    label=f'length = {vectorLength} mm', labelpos='E')
    
    def visFiducialFiber(self):
        '''
            This function plots the location of FFs used in data model.
        '''


        ax=plt.gca()
        butler = Butler(configRoot=os.path.join(os.environ["PFS_INSTDATA_DIR"], "data"))
        fids = butler.get('fiducials')

        ax.plot(fids['x_mm'].values,fids['y_mm'].values,'b+')


    def visFiducialXYStage(self, temp=[0,0], compEL=[90,60]):
        '''
            This function plots the residuals of FF measuremet from XY stage.
        
        '''
        
        try:
            db=opdb.OpDB(hostname='db-ics', port=5432,dbname='opdb',
                        username='pfs')
            fidDataOne = db.bulkSelect('fiducial_fiber_geometry','select * from fiducial_fiber_geometry where '
                        f' ambient_temp = {temp[0]} and elevation = {compEL[0]} and fiducial_fiber_calib_id > 6').set_index('fiducial_fiber_id')
            fidDataTwo = db.bulkSelect('fiducial_fiber_geometry','select * from fiducial_fiber_geometry where '
                        f' ambient_temp = {temp[1]} and elevation = {compEL[1]} and fiducial_fiber_calib_id > 6').set_index('fiducial_fiber_id')
        except:
            db=opdb.OpDB(hostname='pfsa-db01', port=5432,dbname='opdb',
                        username='pfs')
            fidDataOne = db.bulkSelect('fiducial_fiber_geometry','select * from fiducial_fiber_geometry where '
                        f' ambient_temp = {temp[0]} and elevation = {compEL[0]} and fiducial_fiber_calib_id > 6').set_index('fiducial_fiber_id')
            fidDataTwo = db.bulkSelect('fiducial_fiber_geometry','select * from fiducial_fiber_geometry where '
                        f' ambient_temp = {temp[1]} and elevation = {compEL[1]} and fiducial_fiber_calib_id > 6').set_index('fiducial_fiber_id')

        ax=plt.gca()

        dx = -fidDataOne['ff_center_on_pfi_x_mm']+fidDataTwo['ff_center_on_pfi_x_mm']
        dy = fidDataOne['ff_center_on_pfi_y_mm']-fidDataTwo['ff_center_on_pfi_y_mm']
        
        ax.plot(-fidDataOne['ff_center_on_pfi_x_mm'],fidDataOne['ff_center_on_pfi_y_mm'],'r.', label='EL90')
        ax.plot(-fidDataTwo['ff_center_on_pfi_x_mm'],fidDataTwo['ff_center_on_pfi_y_mm'],'b+',label='EL60')
        q=ax.quiver(-fidDataOne['ff_center_on_pfi_x_mm'],fidDataOne['ff_center_on_pfi_y_mm'],
                dx,dy,color='red',units='xy')
        ax.quiverkey(q, X=0.15, Y=0.95, U=0.05,
                                label='length = 0.05 mm', labelpos='E')
        ax.legend()

    def visAllFFSpots(self, pfsVisitID=None, vector=True, vectorLength=0.05, camera = None, 
        dataRange=None, histo=False, getAllFFPos = False, badFF=None, binNum=7, dataOnly=False):

        '''
            
            Args
            ----
            pfsVisitID:  The pfsVisitID 
            getAllFFPos : If this flag is set to be True, returing the averaged position.
            refXYstage: Compare with insdata or averaged positions
        '''
        
        if dataOnly is False:
            ax=plt.gca()

        butler = Butler(configRoot=os.path.join(os.environ["PFS_INSTDATA_DIR"], "data"))

        # Read fiducial and spot geometry
        fids = butler.get('fiducials')

        try:
            db=opdb.OpDB(hostname='db-ics', port=5432,dbname='opdb',
                        username='pfs')
        except:     
            db=opdb.OpDB(hostname='pfsa-db01', port=5432,dbname='opdb',
                                username='pfs')

        ffpos_array=[]
        
        if camera is None:
            camera = 'canon'

        # Building stable FF list
        stableFF = np.zeros(len(fids), dtype=bool)
        stableFF[:]=True 

        if badFF is not None:
            for idx in fids['fiducialId']:
                if idx in badFF:
                    stableFF[fids.fiducialId == idx] = False
        self.logger.info(f'Stable FF = {stableFF}')
        

        for count, sub in enumerate(range(*dataRange)):
            subid=sub

            frameid = pfsVisitID*100+subid

            

            mcsData = db.bulkSelect('mcs_data','select * from mcs_data where '
                    f'mcs_frame_id = {frameid}').sort_values(by=['spot_id']).reset_index()
            teleInfo = db.bulkSelect('mcs_exposure','select altitude, insrot from mcs_exposure where '
                    f'mcs_frame_id = {frameid}')

            # Getting instrument information from DB
            if camera == 'rmod':
                altitude = 90
            else:
                altitude = teleInfo['altitude'].values[0]
            
            rotation = teleInfo['insrot'].values[0]
            
            pt = transformUtils.fromCameraName(camera, altitude=altitude, 
                    insrot=rotation)
    
            outerRing = np.zeros(len(fids), dtype=bool)
            for i in [29, 30, 31, 61, 62, 64, 93, 94, 95, 96]:
                outerRing[fids.fiducialId == i] = True
            pt.updateTransform(mcsData, fids[outerRing], matchRadius=8.0, nMatchMin=0.1)
            
            for i in range(2):
                rfid, rdmin = pt.updateTransform(mcsData, fids, matchRadius=4.2,nMatchMin=0.1)
            nMatch = sum(rfid > 0)
            self.logger.info(f'Total matched = {nMatch}')   


            xx , yy = pt.mcsToPfi(mcsData['mcs_center_x_pix'],mcsData['mcs_center_y_pix'])
            oriPt = fids['x_mm'].values+fids['y_mm'].values*1j
            
            traPt = xx+yy*1j
            traPt = traPt[~np.isnan(traPt)]
            ranPt = []
            for i in oriPt:
                d = np.abs(i-traPt)
                if np.min(d) < 1:
                    ix = np.where(d == np.min(d))
                    ranPt.append(traPt[ix[0]][0])
                else:
                    ranPt.append(np.nan)
            ranPt = np.array(ranPt)

            ffpos_array.append(ranPt)
            if dataOnly is False:
                if count == 0:
                    ax.plot(ranPt[stableFF].real,ranPt[stableFF].imag,'g.',label='FF observed')
                else:
                    ax.plot(ranPt[stableFF].real,ranPt[stableFF].imag,'g.')    

        ffpos_array=np.array(ffpos_array)

        ffpos = np.mean(ffpos_array,axis=0)
        ffstd = np.abs(np.std(ffpos_array,axis=0))
        
        
        if dataOnly is False:
        
            ax.plot(fids['x_mm'].values, fids['y_mm'].values,'b+',label='FF')
            ax.plot(ffpos.real[stableFF], ffpos.imag[stableFF],'r+',label='Avg')
            
            # Mark the FF IDs
            for i in range(len(fids)):
                ax.text(fids.x_mm.values[i], fids.y_mm.values[i], 
                        fids.fiducialId.values[i].astype('str'), fontsize=8)

            ax.legend()
            
            if vector is True:
                q=ax.quiver(oriPt[stableFF].real, oriPt[stableFF].imag,
                        ffpos.real[stableFF]-oriPt[stableFF].real, ffpos[stableFF].imag-oriPt[stableFF].imag,
                        color='red',units='xy')
            
            
                ax.quiverkey(q, X=0.2, Y=0.95, U=vectorLength,
                            label=f'length = {vectorLength} mm', labelpos='E')

            if histo is True:
            

                dx = ffpos.real - fids['x_mm'].values
                dy = ffpos.imag - fids['y_mm'].values
                
                diff = np.sqrt(dx**2+dy**2)
                #import pdb; pdb.set_trace()

                ax1 = plt.subplot(212)
                n, bins, patches = ax1.hist(diff[stableFF],range=(0,0.15),
                    bins=binNum, color='#0504aa',alpha=0.7)

                popt = gaussianFit(n, bins)
                sigma = popt[2]
                xPeak = popt[1]

                ax1.plot(bins[:-1],gaus(bins[:-1],*popt),'r:',label='fit')
                
                #ax1.text(0.8, 0.8, f'Median = {np.nanmedian(diff[stableFF]):.2f}, $\sigma$={sigma:.2f}', 
                ax1.text(0.8, 0.8, f'Mean = {xPeak:.4f}, $\sigma$={sigma:.2f}', 
                    horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
                ax1.set_title('2D')
                ax1.set_xlabel('distance (mm)')
                ax1.set_ylabel('Counts')
                ax1.set_ylim(0,1.2*np.max(n))


                ax2 = plt.subplot(221)
                n, bins, patches = ax2.hist(dx,range=(np.nanmean(dx[stableFF])-3*np.nanstd(dx[stableFF]),np.nanmean(dx[stableFF])+3*np.nanstd(dx[stableFF])), 
                    bins=(2*binNum)+1, color='#0504aa',alpha=0.7)
                popt = gaussianFit(n, bins)
                sigma = popt[2]
                xPeak = popt[1]

                ax2.plot(bins[:-1],gaus(bins[:-1],*popt),'ro:',label='fit')
                ax2.text(0.5, 0.9, f'Mean = {xPeak:.4f}, $\sigma$={sigma:.2f}', 
                    horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
                ax2.set_title('X direction')
                ax2.set_xlabel('distance (mm)')
                ax2.set_ylabel('Counts')
                ax2.set_ylim(0,1.2*np.max(n))


                ax3 = plt.subplot(222, sharey = ax2)
                ax3.tick_params(axis='both',labelleft=False)


                n, bins, patches = ax3.hist(dy,range=(np.nanmean(dy[stableFF])-3*np.nanstd(dy[stableFF]),np.nanmean(dy[stableFF])+3*np.nanstd(dy[stableFF])), 
                    bins=(2*binNum)+1, color='#0504aa', alpha=0.7)
                
                popt = gaussianFit(n, bins)
                sigma = np.abs(popt[2])
                xPeak = popt[1]
                ax3.plot(bins[:-1],gaus(bins[:-1],*popt),'ro:',label='fit')
                
                #ax3.text(0.7, 0.8, f'Median = {np.nanmedian(dy):.2f}, $\sigma$={sigma:.2f}', 
                ax3.text(0.5, 0.9, f'Mean = {xPeak:.4f}, $\sigma$={sigma:.2f}', 
                    horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

                ax3.set_title('Y direction')
                ax3.set_xlabel('distance (mm)')
                plt.subplots_adjust(wspace=0,hspace=0.3)

        if getAllFFPos is True:
            self.logger.info(f'Returing all fiducial fiber positions.')
            return ffpos_array

    def visIterationFFOffset(self, posData, Axes=None, iteration = 0, offsetThres = 0.006, offsetBox = 0.2, 
        heatMap = False, badFF = None, binNum=40):

        '''
            This function plots the relative offsets for all FF.  It is very useful for 
            identifing the unstable FF.

            posData: FF positions transformed from MCS exposures. 

        '''
        butler = Butler(configRoot=os.path.join(os.environ["PFS_INSTDATA_DIR"], "data"))
        fids = butler.get('fiducials')

        
        if Axes is None:
            ax = plt.gca()
        else:
            ax = Axes

        divider = make_axes_locatable(ax)
        # below height and pad are in inches
        ax_histx = divider.append_axes("top", 1.2, pad=0.3, sharex=ax)
        ax_histy = divider.append_axes("right", 1.2, pad=0.3, sharey=ax)

        # make some labels invisible
        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax_histy.yaxis.set_tick_params(labelleft=False)

        stableFF = np.zeros(len(fids), dtype=bool)
        stableFF[:]=True 

        if badFF is not None:
            for idx in fids['fiducialId']:
                if idx in badFF:
                    stableFF[fids.fiducialId == idx] = False

        avgPos = np.nanmean(posData,axis=0)
        

        ffOffset = posData[iteration,:] - avgPos

        if heatMap is True:
            cax = divider.append_axes('right', size='5%', pad=0.5)

            h, xedge, yedge, im = ax.hist2d(ffOffset.flatten().real,ffOffset.flatten().imag,
              cmap='Blues',
              bins=[40,40],range=[[-offsetBox,offsetBox],[-offsetBox,offsetBox]],cmin=0)

            plt.colorbar(im, cax=cax)
        else:

            #for i in range(ffOffset.shape[1]):
                # Check if this one is in unstable FF id
            #if (stableFF[i]):

                #off = np.mean(np.abs(ffOffset[:,i]))
                #if off > offsetThres:
                #    ax.plot(ffOffset[:,i].real,ffOffset[:,i].imag,'+', label=f'FF ID {fids.fiducialId[i]}')
                #else:
            ax.plot(ffOffset.real,ffOffset.imag,'+')
            ax.legend()

        n, bins, patches = ax_histx.hist(ffOffset[stableFF].flatten().real,bins=binNum,range=(-offsetBox,offsetBox))
        popt = gaussianFit(n, bins)
        sigma = np.abs(popt[2])
        xPeak = popt[1]
        ax_histx.plot(bins[:-1],gaus(bins[:-1],*popt),'r:',label='fit')
        ax_histx.text(0.5, 0.9, f'Mean = {xPeak:.4f}, $\sigma$={sigma:.4f}', 
                    horizontalalignment='center', verticalalignment='center', transform=ax_histx.transAxes)

        n, bins, patches = ax_histy.hist(ffOffset[stableFF].flatten().imag,bins=binNum,range=(-offsetBox,offsetBox),orientation='horizontal')        
        popt = gaussianFit(n, bins)
        sigma = np.abs(popt[2])
        xPeak = popt[1]
        ax_histy.plot(gaus(bins[:-1],*popt),bins[:-1],'r:',label='fit')
        ax_histy.text(0.5, 0.9, f'Mean = {xPeak:.4f}, $\sigma$={sigma:.4f}', 
                    horizontalalignment='center', verticalalignment='center', transform=ax_histy.transAxes)

        ax.set_xlim(-offsetBox,offsetBox)
        ax.set_ylim(-offsetBox,offsetBox)

    def visAllFFOffsetHisto(self, posData, Iteration = None, Axes = None, binNum = 80, offsetBox = 0.05):
        
        if Axes is None:
            ax = plt.gca()
        else:
            ax = Axes

        avgPos = np.nanmean(posData,axis=0)
        if Iteration is not None:
            ffOffset = np.abs(posData[Iteration, :] - avgPos)
            n, bins, patches = ax.hist(ffOffset,bins=binNum,range=(0,offsetBox))
        else:
            ffOffset = np.abs(posData - avgPos)
            n, bins, patches = ax.hist(ffOffset.flatten(),bins=binNum,range=(0,offsetBox))
        
        popt = gaussianFit(n, bins)
        sigma = np.abs(popt[2])
        xPeak = popt[1]
        ax.plot(bins[:-1],gaus(bins[:-1],*popt),'r:',label='fit')
        ax.text(0.5, 0.9, f'Mean = {xPeak:.4f}, $\sigma$={sigma:.4f}', 
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        
        pass


    def visAllFFOffset(self, posData, Axes=None, offsetThres = 0.006, offsetBox = 0.2, 
        heatMap = False, badFF = None, binNum=40):
        
        '''
            This function plots the relative offsets for all FF.  It is very useful for 
            identifing the unstable FF.

            posData: FF positions transformed from MCS exposures. 

        '''
        butler = Butler(configRoot=os.path.join(os.environ["PFS_INSTDATA_DIR"], "data"))
        fids = butler.get('fiducials')

        
        if Axes is None:
            ax = plt.gca()
        else:
            ax = Axes

        divider = make_axes_locatable(ax)
        # below height and pad are in inches
        ax_histx = divider.append_axes("top", 1.2, pad=0.3, sharex=ax)
        ax_histy = divider.append_axes("right", 1.2, pad=0.3, sharey=ax)

        # make some labels invisible
        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax_histy.yaxis.set_tick_params(labelleft=False)

        stableFF = np.zeros(len(fids), dtype=bool)
        stableFF[:]=True 

        if badFF is not None:
            for idx in fids['fiducialId']:
                if idx in badFF:
                    stableFF[fids.fiducialId == idx] = False

        avgPos = np.nanmean(posData,axis=0)
        

        ffOffset = posData - avgPos

        if heatMap is True:
            cax = divider.append_axes('right', size='5%', pad=0.5)

            h, xedge, yedge, im = ax.hist2d(ffOffset.flatten().real,ffOffset.flatten().imag,
              cmap='Blues',
              bins=[40,40],range=[[-offsetBox,offsetBox],[-offsetBox,offsetBox]],cmin=0)

            plt.colorbar(im, cax=cax)
        else:

            for i in range(ffOffset.shape[1]):
                # Check if this one is in unstable FF id
                if (stableFF[i]):

                    off = np.mean(np.abs(ffOffset[:,i]))
                    if off > offsetThres:
                        ax.plot(ffOffset[:,i].real,ffOffset[:,i].imag,'+', label=f'FF ID {fids.fiducialId[i]}')
                    else:
                        ax.plot(ffOffset[:,i].real,ffOffset[:,i].imag,'+')
            ax.legend()

        n, bins, patches = ax_histx.hist(ffOffset[:,stableFF].flatten().real,bins=binNum,range=(-offsetBox,offsetBox))
        popt = gaussianFit(n, bins)
        sigma = np.abs(popt[2])
        xPeak = popt[1]
        ax_histx.plot(bins[:-1],gaus(bins[:-1],*popt),'r:',label='fit')
        ax_histx.text(0.5, 0.9, f'Mean = {xPeak:.4f}, $\sigma$={sigma:.4f}', 
                    horizontalalignment='center', verticalalignment='center', transform=ax_histx.transAxes)

        n, bins, patches = ax_histy.hist(ffOffset[:,stableFF].flatten().imag,bins=binNum,range=(-offsetBox,offsetBox),orientation='horizontal')        
        popt = gaussianFit(n, bins)
        sigma = np.abs(popt[2])
        xPeak = popt[1]
        ax_histy.plot(gaus(bins[:-1],*popt),bins[:-1],'r:',label='fit')
        ax_histy.text(0.5, 0.9, f'Mean = {xPeak:.4f}, $\sigma$={sigma:.4f}', 
                    horizontalalignment='center', verticalalignment='center', transform=ax_histy.transAxes)

        ax.set_xlim(-offsetBox,offsetBox)
        ax.set_ylim(-offsetBox,offsetBox)
            

    def visFiducialResidual(self, visitID, subID, temp=0, elevation=90, ffdata='opdb',
        vectorOnly=False, vectorLength=0.05):

        butler = Butler(configRoot=os.path.join(os.environ["PFS_INSTDATA_DIR"], "data"))
        fids = butler.get('fiducials')

        frameid=visitID*100+subID
        firstFrame = visitID*100
        self.logger.info(f'frameID = {frameid}, firstFrame={firstFrame}')
        try:
            db=opdb.OpDB(hostname='db-ics', port=5432,dbname='opdb',
                        username='pfs')

            mcsData = db.bulkSelect('mcs_data','select * from mcs_data where '
                        f'mcs_frame_id = {frameid}').sort_values(by=['spot_id']).reset_index()
            mcsDataFirstFrame = db.bulkSelect('mcs_data','select * from mcs_data where '
                        f'mcs_frame_id = {firstFrame}').sort_values(by=['spot_id']).reset_index()
            teleInfo = db.bulkSelect('mcs_exposure','select altitude, insrot from mcs_exposure where '
                    f'mcs_frame_id = {frameid}')
            fidData = db.bulkSelect('fiducial_fiber_geometry','select * from fiducial_fiber_geometry where '
                    f' ambient_temp = {temp} and elevation = {elevation} '
                    f'and fiducial_fiber_calib_id > 6').set_index('fiducial_fiber_id')

        except:
            db=opdb.OpDB(hostname='pfsa-db01', port=5432,dbname='opdb',
                        username='pfs')

            mcsData = db.bulkSelect('mcs_data','select * from mcs_data where '
                            f'mcs_frame_id = {frameid}').sort_values(by=['spot_id']).reset_index()
            mcsDataFirstFrame = db.bulkSelect('mcs_data','select * from mcs_data where '
                            f'mcs_frame_id = {firstFrame}').sort_values(by=['spot_id']).reset_index()
            teleInfo = db.bulkSelect('mcs_exposure','select altitude, insrot from mcs_exposure where '
                        f'mcs_frame_id = {frameid}')
            fidData = db.bulkSelect('fiducial_fiber_geometry','select * from fiducial_fiber_geometry where '
                        f' ambient_temp = {temp} and elevation = {elevation} '
                        f'and fiducial_fiber_calib_id > 6').set_index('fiducial_fiber_id')
            
        pfiTransform = transformUtils.fromCameraName('canon50M',
            altitude=teleInfo['altitude'].values[0],
            insrot=teleInfo['insrot'].values[0])


        if ffdata == 'insdata':
            outerRing = np.zeros(len(fids), dtype=bool)
            for i in [29, 30, 31, 61, 62, 64, 93, 94, 95, 96]:
                outerRing[fids.fiducialId == i] = True
        
            pfiTransform.updateTransform(mcsDataFirstFrame, fids[outerRing], matchRadius=8.0, nMatchMin=0.1)
            
            for i in range(2):
                pfiTransform.updateTransform(mcsDataFirstFrame, fids, matchRadius=4.2,nMatchMin=0.1)
        else:
            # massage the data from opdb first.
            ffData =  { 'fiducialId': fidData.index, 
                    'x_mm' :-fidData['ff_center_on_pfi_x_mm'].values,
                     'y_mm' :fidData['ff_center_on_pfi_y_mm'].values
            }
            outerRing = np.zeros(len(ffData['fiducialId']), dtype=bool)
            for i in [29, 30, 31, 61, 62, 64, 93, 94, 95, 96]:
                outerRing[ffData['fiducialId'] == i] = True
        
            pfiTransform.updateTransform(mcsDataFirstFrame, pd.DataFrame(ffData)[outerRing], matchRadius=8.0, nMatchMin=0.1)
            
            for i in range(2):
                pfiTransform.updateTransform(mcsDataFirstFrame, pd.DataFrame(ffData), matchRadius=4.2,nMatchMin=0.1)
            
            #pfiTransform.updateTransform(mcsData, pd.DataFrame(ffData),matchRadius=3.2)

        #matchid= np.where(pfiTransform.match_fid != -1)[0]
        ff_mcs_x=mcsData['mcs_center_x_pix'].values
        ff_mcs_y=mcsData['mcs_center_y_pix'].values

        x_mm, y_mm = pfiTransform.mcsToPfi(ff_mcs_x,ff_mcs_y)
        traPt = x_mm+y_mm*1j
        traPt = traPt[~np.isnan(traPt)]
        if ffdata == 'insdata':
            oriPt = fids['x_mm'].values+fids['y_mm'].values*1j
        else:
            oriPt = -fidData['ff_center_on_pfi_x_mm'].values+fidData['ff_center_on_pfi_y_mm'].values*1j

        ranPt = []
        for i in oriPt:
            d = np.abs(i-traPt)
            if np.min(d) < 8:
                ix = np.where(d == np.min(d))
                ranPt.append(traPt[ix[0]][0])
            else:
                ranPt.append(i)
        ranPt = np.array(ranPt)

        dx=ranPt.real-oriPt.real
        dy=ranPt.imag-oriPt.imag
        self.logger.info(f'Number of total matched = {len(dx)}')
        diff = np.sqrt(dx**2+dy**2)
        self.logger.info(f'Mean = {np.mean(diff):.5f} Std = {np.std(diff):.5f}')

        if vectorOnly is True:
            ax=plt.gca()
            ax.plot(ranPt.real,ranPt.imag,'r.', label='MCS projection')
            if ffdata == 'insdata':
                ax.plot(fids['x_mm'].values, fids['y_mm'].values,'b+',label='XY stage')

            else:
                ax.plot(oriPt.real, oriPt.imag,'b+',label='XY stage')
            q=ax.quiver(oriPt.real, oriPt.imag,dx,dy,color='red',units='xy')
            ax.quiverkey(q, X=0.2, Y=0.95, U=vectorLength,
                        label=f'length = {vectorLength} mm', labelpos='E')
            ax.legend()
        else:

            ax0 = plt.subplot(224)
            ax0.plot(ranPt.real,ranPt.imag,'r.', label='MCS projection')
            if ffdata == 'insdata':
                ax0.plot(fids['x_mm'].values, fids['y_mm'].values,'b+',label='XY stage')
                q=ax0.quiver(fids['x_mm'].values, fids['y_mm'].values,dx,dy,color='red',units='xy')
            else:
                ax0.plot(oriPt.real, oriPt.imag,'b+',label='XY stage')
                q=ax0.quiver(oriPt.real, oriPt.imag, dx,dy,color='red',units='xy')
            ax0.quiverkey(q, X=0.2, Y=0.95, U=vectorLength,
                        label=f'length = {vectorLength} mm', labelpos='E')
            ax0.legend()

            ax1 = plt.subplot(223)
            n, bins, patches = ax1.hist(diff,range=(0,np.mean(diff)+1*np.std(diff)), bins=10, color='#0504aa',
                alpha=0.7)
            ax1.text(0.8, 0.8, f'Mean = {np.mean(diff):.2f}, $\sigma$={np.std(diff):.2f}', 
                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
            ax1.set_title('2D')
            ax1.set_xlabel('distance (mm)')
            ax1.set_ylabel('Counts')
            ax1.set_ylim(0,1.5*np.max(n))


            ax2 = plt.subplot(221)
            ax2.hist(dx,range=(np.mean(dx)-2*np.std(dx),np.mean(dx)+2*np.std(dx)), 
                bins=10, color='#0504aa',alpha=0.7)
            ax2.text(0.7, 0.8, f'Mean = {np.mean(dx):.2f}, $\sigma$={np.std(dx):.2f}', 
                horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title('X direction')
            ax2.set_xlabel('distance (mm)')
            ax2.set_ylabel('Counts')
            ax2.set_ylim(0,2.0*np.max(n))


            ax3 = plt.subplot(222, sharey = ax2)
            ax3.tick_params(axis='both',labelleft=False)


            ax3.hist(dy,range=(np.mean(dy)-2*np.std(dx),np.mean(dy)+2*np.std(dx)), 
                bins=10, color='#0504aa', alpha=0.7)
            ax3.text(0.7, 0.8, f'Mean = {np.mean(dy):.2f}, $\sigma$={np.std(dy):.2f}', 
                horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

            ax3.set_title('Y direction')
            ax3.set_xlabel('distance (mm)')
            plt.subplots_adjust(wspace=0,hspace=0.3)

        return ranPt
    
    def visRemeasurePhiCenter(self, data, radiusTolerance = 0.5):

        """
        Remeasuring the phi center and radius of from fiber spots.
        
        Parameters
        ----------
        data: compelex array
            fiber spots of forward movement

        rv: compelex array
            fiber spots of reverse movement
        
        """
        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x-xc)**2 + (y-yc)**2)

        def f_2(c):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        
        estCenter = np.mean([data[0],data[-1]])
        medianR = np.median(np.abs((data-estCenter)))
        indx = np.where(np.abs((data-estCenter)) < medianR+radiusTolerance)
        data=data[indx]

        x = data.real
        y = data.imag

        center_estimate = np.mean(x), np.mean(y)
        center_2, ier = optimize.leastsq(f_2, center_estimate)

        xc_2, yc_2 = center_2
        Ri_2       = calc_R(*center_2)
        R_2        = Ri_2.mean()
        residu_2   = sum((Ri_2 - R_2)**2)

        ax=plt.gca()
        ax.plot(data.real,data.imag,'.',label='data spot')
        
        ax.plot(xc_2,yc_2,'+',label='New center')

        d = plt.Circle((xc_2, yc_2), R_2, color='blue', fill=False)
        ax.add_artist(d)

        ax.text(xc_2,yc_2,f'{xc_2:.3f}+{yc_2:.3f}j')

        ax.legend()
                
        self.logger.info(f'The center is at (x ,y) = {xc_2+yc_2*1j} R = {R_2:.4f}')

        return xc_2+yc_2*1j, R_2
        


    def visRemeasureThetaCenter(self, fw, rv, estCenter, radiusTolerance = 0.5, badAngle = None,
        doPlots = False) :

        """
        Remeasuring the theta center and radius of from fiber spots.
        
        Parameters
        ----------
        fw: compelex array
            fiber spots of forward movement

        rv: compelex array
            fiber spots of reverse movement
        
        """
        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x-xc)**2 + (y-yc)**2)

        def f_2(c):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        data = np.append(fw,rv)
        medianR = np.median(np.abs((data-estCenter)))
        indx = np.where(np.abs((data-estCenter)) < medianR+radiusTolerance)
        data=data[indx]
        
        if badAngle is not None:
            pointAngle = np.rad2deg(np.angle(data-estCenter))% 360
            angleIndx = np.where((pointAngle < badAngle[0]) | (pointAngle > badAngle[1]))[0]
            data=data[angleIndx]
       

        x = data.real
        y = data.imag

        center_estimate = estCenter.real, estCenter.imag
        center_2, ier = optimize.leastsq(f_2, center_estimate)

        xc_2, yc_2 = center_2
        Ri_2       = calc_R(*center_2)
        R_2        = Ri_2.mean()
        residu_2   = sum((Ri_2 - R_2)**2)

        if doPlots == True:
            ax=plt.gca()
            ax.plot(fw.real,fw.imag,'.',label='FW')
            ax.plot(rv.real,rv.imag,'.',label='RV')
            
            d = plt.Circle((xc_2, yc_2), R_2, color='blue', fill=False)
            ax.add_artist(d)

            ax.text(xc_2,yc_2,f'{xc_2:.3f}+{yc_2:.3f}j')

            ax.legend()
                    
        self.logger.info(f'The center is at (x ,y) = {xc_2+yc_2*1j} R = {R_2:.4f}')
        return xc_2+yc_2*1j, R_2


    def visSaveFigure(self, fileName, **kwargs):
        """
        Saves an image of the current figure.
        
        Parameters
        ----------
        fileName: object
            The image file name path.
        kwargs: figure.savefig properties
            Any additional property that should be passed to the savefig method.
        
        """

        plt.gcf().savefig(fileName, **kwargs)

        plt.close()

    def visPauseExecution(self):
        """Pauses the general program execution to allow figure inspection.
        
        """
        plt.show()


    def visCobraLocation(self, moveRunDir, phiGeoRunDir=None, thetaGeoRunDir = None):

        if phiGeoRunDir is not None:
            
            angleList=np.load(f'/data/MCS/{phiGeoRunDir}/output/phiOpenAngle.npy')
                


    def visPlotGeometry(self, arm=None, pngfile=None):
        try:
            self.centers
        except AttributeError:
            self._loadCobraData(arm=arm)
        
        if arm is None:
            raise Exception('Define the arm')

        plt.figure(figsize=(10, 20))
        plt.clf()

        plt.subplot(211)
        ax = plt.gca()

        ax.plot(self.centers[self.group1].real, self.centers[self.group1].imag, 'ro')
        ax.axis('equal')
        for idx in self.group1:
            c = plt.Circle((self.centers[idx].real, self.centers[idx].imag), self.radius[idx], color='g', fill=False)
            ax.add_artist(c)
        ax.set_title(f'1st camera')

        plt.subplot(212)
        ax = plt.gca()

        ax.plot(self.centers[self.group2].real, self.centers[self.group2].imag, 'ro')
        ax.axis('equal')
        for idx in self.group2:
            c = plt.Circle((self.centers[idx].real, self.centers[idx].imag), self.radius[idx], color='g', fill=False)
            ax.add_artist(c)
        ax.set_title(f'2nd camera')

        if pngfile is not None:
            plt.savefig(pngfile)
        else:
            plt.show()

    def visDotLocation(self, dotDF=None, dotColor = None):
        
        ax = plt.gca()

        if dotDF is None:
            dotFile = '/software/devel/pfs/pfs_instdata/data/pfi/dot/black_dots_mm.csv'
            newDot = pd.read_csv(dotFile)
        else:
            neeDot = dotDF

        if dotColor is None:
            dotColor = 'grey'
        # Plot DOT location
        for dotidx in range(len(newDot)):
            e = plt.Circle((newDot['x'].values[dotidx], newDot['y'].values[dotidx]), newDot['r'].values[dotidx], 
                        color=dotColor, fill=True, alpha=0.5)
            ax.add_artist(e)



    def visPlotFiberSpots(self, cobraIdx=None, moveData=None, color=None, 
        markCobra=False, markGeometry=True):
        
        '''
            This function is mainly used to plot round trip data, for example, motor map or geometry

            Args
            -------
            cobraIdx: cobra index to show
            moveData: input of FW and RV data 
        
        '''

        ax = plt.gca()
        
        
        if cobraIdx is None:
            cobra = self.goodIdx
        else:
            cobra = cobraIdx

        # By default, we do not want to see cobra marked as bad
        

        for idx in cobra:
            c = plt.Circle((self.calibModel.centers[idx].real, 
                self.calibModel.centers[idx].imag), 
                self.calibModel.L1[idx]+self.calibModel.L2[idx], facecolor='g', edgecolor=None,alpha=0.5)
            ax.add_artist(c)

            if markCobra is True:
                ax.text(self.calibModel.centers[idx].real, self.calibModel.centers[idx].imag,idx, fontsize=8)


        if markGeometry is True:
            if moveData is None:
                for idx in cobra:
                    d = plt.Circle((self.centers[idx].real, self.centers[idx].imag), self.radius[idx], color='red', fill=False)
                    ax.add_artist(d)
        
                ax.scatter(self.centers[cobra].real,self.centers[cobra].imag,color='red')    

        if moveData is None:
            for n in range(1):
                for k in cobra:
                    if k % 3 == 0:
                        c = 'r'
                        d = 'c'
                    elif k % 3 == 1:
                        c = 'g'
                        d = 'm'
                    else:
                        c = 'b'
                        d = 'y'
                    if k == 0:
                        ax.plot(self.fw[k][n,0].real, self.fw[k][n,0].imag, c + 'o',label='FW0')
                        ax.plot(self.rv[k][n,0].real, self.rv[k][n,0].imag, d + 's',label='RV0')
                    ax.plot(self.fw[k][n,0].real, self.fw[k][n,0].imag, c + 'o')
                    ax.plot(self.rv[k][n,0].real, self.rv[k][n,0].imag, d + 's')
                    ax.plot(self.fw[k][n,1:].real, self.fw[k][n,1:].imag, c + '.')
                    ax.plot(self.rv[k][n,1:].real, self.rv[k][n,1:].imag, d + '.')
        else:
            fw = moveData[0]
            rv = moveData[1]
            for n in range(1):
                for k in cobra:
                    if k % 3 == 0:
                        c = 'r'
                        d = 'c'
                    elif k % 3 == 1:
                        c = 'g'
                        d = 'm'
                    else:
                        c = 'b'
                        d = 'y'
                    if color is not None:
                        c = color
                        d = color
                    if k == 0:
                        ax.plot(self.fw[k][n,0].real, self.fw[k][n,0].imag, c + 'o',label='FW0')
                        ax.plot(self.rv[k][n,0].real, self.rv[k][n,0].imag, d + 's',label='RV0')
                    ax.plot(fw[k][n,0].real, fw[k][n,0].imag, c + 'o')
                    ax.plot(rv[k][n,0].real, rv[k][n,0].imag, d + 's')
                    ax.plot(fw[k][n,1:].real, fw[k][n,1:].imag, c + '.')
                    ax.plot(rv[k][n,1:].real, rv[k][n,1:].imag, d + '.')

        ax.legend()

        #plt.show()
    
    def visStackedImage(self, direction='fwd', flip=False):
        if direction == 'fwd':
            data = self.fwdStack
        else:
            data = self.revStack

        if flip is True:
            image=(np.flip(data).T).copy(order='C')
        else:
            image=data
        m, s = np.mean(image), np.std(data)
        ax = plt.gca()
        im = ax.imshow(image, interpolation='nearest', 
                    cmap='gray', vmin=m-s, vmax=m+3*s, origin='lower')
    
    def visCobraMotorMap(self, stepsize=50, figPath=None, arm=None, pdffile=None, debug=False):
        try:
            self.mf
        except AttributeError:
            self._loadCobraData(arm=arm)
        
        if arm is None:
            raise Exception('Define the arm')
        

        #ymax = 1.1*np.rad2deg(np.max(self.mf))
        #ymin = -1.1*np.rad2deg(np.max(self.mr))



        for fiber in self.goodIdx:
            if arm == 'phi':
                ymax = 0.15
                ymin = -0.15
            else:
                ymax = 0.25
                ymin = -0.25

            #width = (fiber + 1) / 2
            
            #bar = "\r[" + "#" * int(math.ceil(width)) + " " * int(29 - width) + "]"
            #sys.stdout.write(f'{bar}')
            #sys.stdout.flush()

            x=np.arange(112)*3.6
            c = fiber

            plt.figure(figsize=(8,6))
            plt.clf()
            ax = plt.gca()
            ax.set_title(f'Fiber {arm} #{c+1}')


            daf = np.zeros(len(self.af[c][0])-1)
            dar = np.zeros(len(self.ar[c][0])-1)

            for data in self.af[c]: 
                for i,item in enumerate(data):
                    if i < len(daf):
                        daf[i] = np.rad2deg(data[i+1] - data[i])/stepsize
                        ax.plot([np.rad2deg(data[i+1]),np.rad2deg(data[i])],[daf[i],daf[i]],color='grey')
            ax.plot(x,np.rad2deg(self.mf[c]), 'r')        
           


            for data in self.ar[c]: 
                for i,item in enumerate(data):
                    if i < len(daf):
                        dar[i] = np.rad2deg(data[i+1] - data[i])/stepsize
                        ax.plot([np.rad2deg(data[i+1]),np.rad2deg(data[i])],[dar[i],dar[i]],color='grey')
            ax.plot(x,-np.rad2deg(self.mr[c]), 'r')
            
            
            
            if np.max(np.rad2deg(self.mf[c])) > ymax: 
                ymax = 1.1*np.max(np.rad2deg(self.mf[c]))
            if np.min(-np.rad2deg(self.mr[c])) < ymin: 
                ymin = 1.1*np.min(-np.rad2deg(self.mr[c]))
            
            #print(np.max(np.rad2deg(self.mf[c])),
            #np.max(np.rad2deg(self.mr[c])),ymax,ymin)
            ax.set_ylim([ymin,ymax])
            if arm == 'phi':
                ax.set_xlim([0,200])
            else:
                ax.set_xlim([0,400])
            
            if figPath is not None:
                if not (os.path.exists(figPath)):
                    os.mkdir(figPath)
                plt.ioff()
                plt.savefig(f'{figPath}/motormap_{arm}_{c+1}.png')
            else:
                plt.show()
            plt.close()

        self.visSpeedHisto(arm=f'{arm}', figPath=f'{figPath}')

        if pdffile is not None:
            cmd=f"""convert {figPath}motormap*_[0-9].png {figPath}motormap*_[0-9]?.png \
            {figPath}motormap*_[0-9]??.png {figPath}motormap*_[0-9]???.png {pdffile}"""
            retcode = subprocess.call(cmd,shell=True)
            if debug is True:
                print(cmd)

    def _visSpeedHistogram(self, avg1, Title, Legend1):
        
        hist1, edges1 = np.histogram(avg1, bins=np.arange(0.0, 0.3, 0.01))
        #hist2, edges2 = np.histogram(avg2, bins=np.arange(0.0, 0.3, 0.01))

        TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']
        p = figure(title=Title, tools=TOOLS, background_fill_color="#fafafa")
        #p.quad(top=hist1, bottom=0, left=edges1[:-1], right=edges1[1:],
        #    fill_color="navy", line_color="white", alpha=0.3,legend=Legend1)

        p.step(x=edges1[0:-2],y=hist1[0:-1], color='black',legend=Legend1,line_width=2,mode="after")

        return p


    def _visSpeedStdHistogram(self, std,Title, Legend1):
            
        hist1, edges1 = np.histogram(std, bins=np.arange(0.0, 0.1, 0.005))
        #hist2, edges2 = np.histogram(avg2, bins=np.arange(0.0, 0.1, 0.005))

        TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']
        p = figure(title=Title, tools=TOOLS, background_fill_color="#fafafa")
        #p.quad(top=hist1, bottom=0, left=edges1[:-1], right=edges1[1:],
        #    fill_color="navy", line_color="white", alpha=0.3,legend=Legend1)

        p.step(x=edges1[0:-2],y=hist1[0:-1], color='black',legend=Legend1,line_width=2,mode="after")

        return p

    def visSpeedHisto(self, arm=None, figPath=None):
        try:
            self.sf
        except AttributeError:
            self._loadCobraData(arm=arm)

        p1=self._visSpeedHistogram(np.rad2deg(self.sf), f'{arm} Fwd', 'ASIAA')
        p2=self._visSpeedHistogram(np.rad2deg(self.sr), f'{arm} Rev', 'ASIAA')
        
        fwdstd = []
        revstd = []

        for i in self.mf:
            fwdstd.append(np.std(np.rad2deg(i)))
        for i in self.mr:
            revstd.append(np.std(np.rad2deg(i)))
        
        fwdstd = np.array(fwdstd)
        revstd = np.array(revstd)

        q1=self._visSpeedStdHistogram(fwdstd, f'{arm} Fwd Std', 'ASIAA')
        q2=self._visSpeedStdHistogram(revstd, f'{arm} Rev Std', 'ASIAA')
        #q3=makeStdHistoPlot(j2fwd_std1, j2fwd_std2, 'Phi Fwd Std', 'Caltech', 'ASIAA')
        #q4=makeStdHistoPlot(j2rev_std1, j2rev_std2, 'Phi Rev Std', 'Caltech', 'ASIAA')
        #show(column(p1,p2,p3,p4))
        grid = gridplot([[p1, p2]])
        qgrid = gridplot([[q1, q2]])
        #show(grid)
        

        if figPath is not None:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            driver = webdriver.Chrome(ChromeDriverManager().install(),options=options)

            export_png(grid,filename=figPath+f"{arm}_motor_speed_histogram.png",webdriver=driver)
            export_png(qgrid,filename=figPath+f"{arm}_motor_speed_std.png",webdriver=driver)


    def visAngleMovement(self, figPath=None, arm = 'phi', pdffile=None):
        try:
            self.fw
        except AttributeError:
            self._loadCobraData(arm=arm)
        
        if arm is None:
            raise Exception('Define the arm')


        plt.figure(figsize=(10, 8))
        plt.clf()
        ax = plt.gca()
        ax.set_title(f'Fiber {arm} Speed')

        ax.plot(self.goodIdx+1,np.rad2deg(self.sf[self.goodIdx]),'o',label='Fwd')
        ax.plot(self.goodIdx+1,np.rad2deg(self.sr[self.goodIdx]),'o',label='Rev')
        ax.legend()
        if figPath is not None:
            if not (os.path.exists(figPath)):
                    os.mkdir(figPath)
            plt.savefig(figPath+f'FiberSpeed.png')
        else:
            plt.show()
        plt.close()
        
        for fiberIdx in self.goodIdx:
            c = fiberIdx
            plt.figure(figsize=(10, 8))
            plt.clf()
            ax = plt.gca()
            ax.set_title(f'Fiber {arm} #{c+1}')
        
            for n in range(self.af.shape[1]):
                ax.plot(np.rad2deg(self.af[c, n]), '.')
                ax.plot(np.rad2deg(self.ar[c, n]), '.')
            ax.set_xlabel("Steps",fontsize=10)
            ax.set_ylabel("Angle from hard-stop (Degree)",fontsize=10)
            if figPath is not None:
                if not (os.path.exists(figPath)):
                    os.mkdir(figPath)
                plt.savefig(figPath+f'AngleMove_{arm}_{c+1}.png')
            else:
                plt.show()
            plt.close()

        if pdffile is not None:
            cmd=f"""convert {figPath}FiberSpeed.png {figPath}AngleMove*_[0-9].png \
                {figPath}AngleMove*_[0-9]?.png {figPath}AngleMove*_[0-9]??.png {figPath}AngleMove*_[0-9]???.png {pdffile}"""
            retcode = subprocess.call(cmd,shell=True)
            print(cmd)

    def visConverge(self, figPath = None, arm = 'phi', runs = 50, margin = 15, montage=None, pdffile=None):
        
        if arm == 'phi':
            phiPath = self.path
            moveData = np.load(phiPath+'phiData.npy')
            angleLimit = 180
        else:
            thetaPath =  self.path
            moveData = np.load(thetaPath+'thetaData.npy')
            angleLimit = 360
        
        
        snr_list = np.array([])
        fiber_list = np.array([])
        repeat_list = np.array([])

        for fiberIdx in self.goodIdx:
            xdata=[]
            ydata=[]
            z=[]
            
            x =np.arange(10)
            #fig, (vax, hax) = plt.subplots(1, 2, figsize=(12, 6),sharey='all')
            fig = plt.figure(figsize=(12, 12),constrained_layout=True)
            gs = gridspec.GridSpec(2, 2)
            vax = fig.add_subplot(gs[0, 0])
            hax = fig.add_subplot(gs[0, 1])
            sax = fig.add_subplot(gs[1, :])

            for i in range(runs):
                angle = margin + (angleLimit - 2 * margin) * i / (runs - 1)
                if i == 0:
                    delAngle = (angleLimit - 2 * margin) / (runs - 1)
                y=np.rad2deg(np.append([0], moveData[fiberIdx,i,:,0]))
                xdata.append(np.arange(10))
                ydata.append(np.full(len(x), angle))
                z.append(np.log10(np.abs(angle - np.rad2deg(np.append([0], moveData[fiberIdx,i,:,0])))))

                hax.plot(x[:9],y,marker='o',fillstyle='none',markersize=3)
                #hax.scatter(x,y,marker='o',fillstyle='none')
            
            """Adding one extra data for pcolor function requirement"""
            xdata.append(np.arange(10))
            ydata.append(np.full(len(x), angle+delAngle))
            xdata=np.array(xdata)
            ydata=np.array(ydata)
            #if arm is 'theta':
            ydata=ydata[:,:]-delAngle 
            #else:
            #    ydata=ydata[:,:]-0.5*delAngle            
            z=np.array(z)

            sc=vax.pcolor(xdata,ydata,z,cmap='inferno_r',vmin=-1.0,vmax=1.0)
            
            #plt.xticks(np.arange(8)+0.5,np.arange(8)+1)
            cbaxes = fig.add_axes([0.05,0.13,0.01,0.75]) 
            cb = plt.colorbar(sc, cax = cbaxes,orientation='vertical')
            #cbar=vax.colorbar(sc)
            
            cbaxes.yaxis.set_ticks_position('left')
            cbaxes.yaxis.set_label_position('left')
            cb.set_label('Angle Different (log)', labelpad=-1)
            vax.set_xlabel("Iteration",fontsize=10)
            vax.set_ylabel("Cabra Location (Degree)",fontsize=10)
            
            #Plot SNR
            snr_array=[]
            k_offset=1/(.074)**2
            tmax=76
            tobs=900
            tstep=x*8+12
            
            linklength=2.35
            

            for i in range(runs):
                angle = margin + (angleLimit - 2 * margin) * i / (runs - 1)

                dist=2*np.sin(0.5*(np.abs(np.deg2rad(angle)-(np.append([0], moveData[fiberIdx,i,:,0])))))*linklength
                snr=(1-k_offset*dist**2)*(np.sqrt((tmax+tobs-tstep[0:9])/(tobs)))
                snr[snr < 0]=0
                
                if np.max(snr) < 0.95:
                    if fiberIdx+1 not in [139,172,194,226,322,399,400,470]:
                        print(f'Fiber {fiberIdx+1}, angle = {angle}, SNR= {np.max(snr)}')
                sax.scatter(tstep[0:9],snr,s=50)
                snr_array.append(snr)

            snr_array=np.array(snr_array)
            snr_avg=np.mean(snr_array,axis=0)

            sax.plot(tstep[0:9],snr_avg,color='green',linewidth=4)
            sax.scatter(tstep[0:9],snr_avg,color='green',s=100)
            sax.set_title('SNR = %.3f'%(np.max(snr_avg)),fontsize=20)
            #if np.max(snr_avg) < 0.95:
                #print(f'Fiber {fiberIdx+1} SNR {np.max(snr_avg)}')

            if arm == 'phi':
                vax.set_ylim([-10,200])
                hax.set_ylim([-10,200])
                sax.set_ylim([0.5,1.20])
            else:
                vax.set_ylim([-10,400])
                vax.set_ylim([-10,400])
                sax.set_ylim([0.5,1.20])

            fig.suptitle(f'Fiber No. {fiberIdx+1}',fontsize=15)
            #plt.subplots_adjust(bottom=0.15, wspace=0.05)
            if figPath is not None:
                if not (os.path.exists(figPath)):
                    os.mkdir(figPath)
                plt.savefig(figPath+f'Converge_{arm}_{fiberIdx+1}.png')
            else:
                plt.show()
            plt.close()

        if montage is not None:
            cmd=f"""montage {figPath}Con*_[0-9].png {figPath}Con*_[0-9]?.png \
                -tile 4x -geometry +4+4 {montage}"""
            retcode = subprocess.call(cmd,shell=True)
            if retcode != 0:
                raise Exception
            #if figPath is not None:
            #    plt.savefig(figPath+f'Converge_{arm}_{fiberIdx+1}.png')
        
        if pdffile is not None:
            cmd=f"""convert {figPath}Con*_{arm}_[0-9].png {figPath}Con*_{arm}_[0-9]?.png \
            {figPath}Con*_{arm}_[0-9]??.png {figPath}Con*_{arm}_[0-9]???.png  {pdffile}"""
            retcode = subprocess.call(cmd,shell=True)
            print(cmd)

        del(moveData)
    
    def visModuleSNR(self, figPath = None, arm = 'phi', runs = 50, 
        margin = 15, pdffile=None):

        if arm == 'phi':
            phiPath = self.path
            moveData = np.load(phiPath+'phiData.npy')
            angleLimit = 180
        else:
            thetaPath =  self.path
            moveData = np.load(thetaPath+'thetaData.npy')
            angleLimit = 360

        all_snr=[]
        max_snr=[]
        it_max =[]

        for fiberIdx in self.goodIdx:
            max_array=[]
            snr_array=[]
            it_array = []
            
            k_offset=1/(.075)**2
            tmax=76
            tobs=900
            tstep=np.arange(10)*8+12

            linklength=2.35


            for i in range(runs):
                angle = margin + (angleLimit - 2 * margin) * i / (runs - 1)

                dist=2*np.sin(0.5*(np.abs(np.deg2rad(angle)-(np.append([0], moveData[fiberIdx,i,:,0])))))*linklength            
                
                snr=(1-k_offset*dist**2)*(np.sqrt((tmax+tobs-tstep[0:9])/(tobs)))
                snr[snr < 0]=0
        
                snr_array.append(snr)
                max_array.append(np.max(snr))
                it_array.append(np.argmax(snr))
                
            all_snr.append(snr_array)
            max_snr.append(max_array)
            it_max.append(it_array)
            
            
        max_snr = np.array(max_snr)    
        all_snr = np.array(all_snr)    
        it_max = np.array(it_max)    

        all_angle = margin + (angleLimit - 2 * margin) * np.arange(runs+1) / (runs)
        all_fiber = np.arange(1197)


        fig = plt.figure(figsize=(10, 8),constrained_layout=True)
        inx=8
        cmin = np.min(all_snr[:,:,inx])
        cmax = np.max(all_snr[:,:,inx])
        sc=plt.pcolor(all_angle, all_fiber, all_snr[:,:,inx],cmap='inferno',vmin=0.95,vmax=cmax)
        plt.title(f"SNR at {inx}-th iteration", fontsize = 20)
        plt.xlabel("Angle (Degree)", fontsize = 15)
        plt.ylabel("Fiber", fontsize = 15)
        cbaxes = fig.add_axes([1.02,0.1,0.02,0.8]) 
        cb = fig.colorbar(sc, cax = cbaxes,orientation='vertical')
        
        if figPath is not None:
            if not (os.path.exists(figPath)):
                os.mkdir(figPath)
            plt.savefig(figPath+f'snr_iteration_{arm}.png', bbox_inches='tight')
        else:
            plt.show()
        plt.close()



        fig = plt.figure(figsize=(10, 8),constrained_layout=True)
        cmin = np.min(max_snr)
        cmax = np.max(max_snr)
        sc=plt.pcolor(all_angle, all_fiber,max_snr,cmap='inferno',vmin=0.95,vmax=cmax)
        plt.title(f"Maximum SNR", fontsize = 20)
        plt.xlabel("Angle (Degree)", fontsize = 15)
        plt.ylabel("Fiber", fontsize = 15)
        cbaxes = fig.add_axes([1.02,0.1,0.02,0.8]) 
        cb = fig.colorbar(sc, cax = cbaxes,orientation='vertical')

        if figPath is not None:
            if not (os.path.exists(figPath)):
                os.mkdir(figPath)
            plt.savefig(figPath+f'snr_max_{arm}.png', bbox_inches='tight')
        else:
            plt.show()
        plt.close()


        fig = plt.figure(figsize=(10, 8),constrained_layout=True)
        cmin = np.min(it_max)
        cmax = np.max(it_max)
        sc=plt.pcolor(all_angle, all_fiber,it_max,cmap='inferno',vmin=0,vmax=cmax)
        plt.title(f"Iteration of maximum SNR", fontsize = 20)
        plt.xlabel("Angle (Degree)", fontsize = 15)
        plt.ylabel("Fiber", fontsize = 15)

        cbaxes = fig.add_axes([1.02,0.1,0.02,0.8]) 
        cb = fig.colorbar(sc, cax = cbaxes,orientation='vertical')       

        if figPath is not None:
            if not (os.path.exists(figPath)):
                os.mkdir(figPath)
            plt.savefig(figPath+f'snrmax_it_{arm}.png', bbox_inches='tight')
        else:
            plt.show()
        plt.close()

        if pdffile is not None:
            cmd=f"""convert {figPath}snr_iteration_{arm}.png {figPath}snr_max_{arm}.png \
                {figPath}snrmax_it_{arm}.png {pdffile}"""
            retcode = subprocess.call(cmd,shell=True)
        print(cmd)
        del(moveData)

    def visConvergeHisto(self, figPath = None, filename = None, arm ='phi', 
        brokens=None, runs = 16, margin = 15, title=None):
        #brokens = []
        #badIdx = np.array(brokens) - 1
        #goodIdx = np.array([e for e in range(57) if e not in badIdx])
        
        if arm == 'phi':
            phiPath = self.path
            moveData = np.load(phiPath+'phiData.npy')
            angleLimit = 180
        else:
            thetaPath =  self.path
            moveData = np.load(thetaPath+'thetaData.npy')
            angleLimit = 360
                
        fig, fax = plt.subplots(figsize=(15, 6),ncols=4, nrows=2, constrained_layout=True)
        fig.suptitle(f'{title}', fontsize=16)
        for ite in range(8):
            diff = []
            #runs = 16
            #margin = 15
            for i in range(runs):
                if runs > 1:
                    angle = np.deg2rad(margin + (angleLimit - 2 * margin) * i / (runs - 1))
                else:
                    angle = np.deg2rad((angleLimit - 2 * margin) / (runs - 1))
                for cob in self.goodIdx:
                    diff.append(np.rad2deg(angle - moveData[cob,i,ite,0]))
            x= (np.arange(100)-50)/100
            diff = np.array(diff)
            
            
            if ite < 4:
                fax[int(ite/4), ite%4].hist(diff, bins=100,range=(-10,10))
                fax[int(ite/4), ite%4].set_xlim([-10, 10])
                fax[int(ite/4), ite%4].set_ylim([0, 800])
            else:
                fax[int(ite/4), ite%4].hist(diff, bins=50,range=(-1,1))
                fax[int(ite/4), ite%4].set_xlim([-1, 1])
                fax[int(ite/4), ite%4].set_ylim([0, 400])

            
            fax[int(ite/4), ite%4].title.set_text(f'{ite} Iteration')   

        if filename is None:
            plt.savefig(figPath+f'{arm}_convergeHisto.png')
        else:
            plt.savefig(figPath+f'{filename}')
        plt.close()
        del(moveData)


    def visMultiMotorMapfromXML(self, xmlList, cobraIdx = None, figPath=None, arm='phi', pdffile=None, fast=False):
        binSize = 3.6
        x=np.arange(112)*3.6
        
        if cobraIdx is None:
            cobraIdx = self.goodIdx
        
        for i in cobraIdx:
            plt.figure(figsize=(10, 8))
            #plt.clf()

            ax = plt.gca()
            
            for xml in xmlList:
                model = pfiDesign.PFIDesign(pathlib.Path(xml))

                if arm == 'phi':
                    slowFWmm = np.rad2deg(model.angularSteps[i]/model.S2Pm[i])
                    slowRVmm = np.rad2deg(model.angularSteps[i]/model.S2Nm[i])
                    fastFWmm = np.rad2deg(model.angularSteps[i]/model.F2Pm[i])
                    fastRVmm = np.rad2deg(model.angularSteps[i]/model.F2Nm[i])
                else:
                    slowFWmm = np.rad2deg(model.angularSteps[i]/model.S1Pm[i])
                    slowRVmm = np.rad2deg(model.angularSteps[i]/model.S1Nm[i])
                    fastFWmm = np.rad2deg(model.angularSteps[i]/model.F1Pm[i])
                    fastRVmm = np.rad2deg(model.angularSteps[i]/model.F1Nm[i])

                labeltext=os.path.splitext(os.path.basename(xml))[0]
                
                if fast is True:
                    ln1 = ax.plot(x,fastFWmm,label=f'{labeltext} Fwd')
                    ln2 = ax.plot(x,-fastRVmm,label=f'{labeltext} Rev')
                else:
                    ln1 = ax.plot(x,slowFWmm,label=f'{labeltext} Fwd')
                    ln2 = ax.plot(x,-slowRVmm,label=f'{labeltext} Rev')

            ax.set_title(f'Fiber {arm} #{i+1}')
            ax.legend()
            if arm == 'phi':
                ax.set_xlim([0,200])
                if fast is False:
                    ax.set_ylim([-0.15,0.15])
                else:
                    ax.set_ylim([-0.25,0.25])
            else:
                ax.set_xlim([0,400])
                ax.set_ylim([-0.3,0.3])
            
            if figPath is not None:
                if not (os.path.exists(figPath)):
                    os.mkdir(figPath)

                plt.ioff()
                plt.tight_layout()
                plt.savefig(f'{figPath}/motormap_{arm}_{i+1}.png')
            #else:
            #    plt.show()
            #plt.close()


        if pdffile is not None:
            cmd=f"""convert {figPath}motormap*{arm}*_[0-9].png {figPath}motormap*{arm}*_[0-9]?.png \
            {figPath}motormap*{arm}*_[0-9]?.png {pdffile}"""
            retcode = subprocess.call(cmd,shell=True)
            print(cmd)

    def visMMVariant(self, baseXML, xmlList, figPath=None, arm='phi', pdffile=None):
        x=np.arange(112)*3.6


        basemodel = pfiDesign.PFIDesign(baseXML)
        var_fwd=[]
        var_rev=[]        
        for i in self.goodIdx:
            plt.figure(figsize=(10, 12))
            plt.clf()

            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2)
          
            m = 0
            if arm == 'phi':
                baseSlowFWmm = np.rad2deg(basemodel.angularSteps[i]/basemodel.S2Pm[i])
                baseSlowRVmm = np.rad2deg(basemodel.angularSteps[i]/basemodel.S2Nm[i])
            else:
                baseSlowFWmm = np.rad2deg(basemodel.angularSteps[i]/basemodel.S1Pm[i])
                baseSlowRVmm = np.rad2deg(basemodel.angularSteps[i]/basemodel.S1Nm[i])
            

            var_fwd_array=[]
            var_rev_array=[]
            for f in xmlList:

                model = pfiDesign.PFIDesign(f)

                if arm == 'phi':
                    slowFWmm = np.rad2deg(model.angularSteps[i]/model.S2Pm[i])
                    slowRVmm = np.rad2deg(model.angularSteps[i]/model.S2Nm[i])
                else:
                    slowFWmm = np.rad2deg(model.angularSteps[i]/model.S1Pm[i])
                    slowRVmm = np.rad2deg(model.angularSteps[i]/model.S1Nm[i])
                
                labeltext=os.path.splitext(os.path.basename(f))[0]
                ln1=ax1.plot(x,np.abs(slowFWmm-baseSlowFWmm)/baseSlowFWmm,
                    marker=f'${m}$', markersize=13, label=f'{labeltext} Fwd' )
                ln2=ax2.plot(x,np.abs(slowRVmm-baseSlowRVmm)/baseSlowRVmm,
                    marker=f'${m}$', markersize=13, label=f'{labeltext} Rev')
                
                var_fwd_array.append(np.abs(slowFWmm-baseSlowFWmm)/baseSlowFWmm)
                var_rev_array.append(np.abs(slowFWmm-baseSlowFWmm)/baseSlowFWmm)
                m=m+1

            ax1.set_title(f'Fiber {arm} #{i+1} FWD')
            ax1.legend()
            ax2.set_title(f'Fiber {arm} #{i+1} REV')
            ax2.legend()
            if arm == 'phi':
                ax1.set_xlim([0,200])
                ax2.set_xlim([0,200])
            else:
                ax1.set_xlim([0,400])
                ax2.set_xlim([0,400])


            if figPath is not None:
                if not (os.path.exists(figPath)):
                    os.mkdir(figPath)

                plt.ioff()
                plt.tight_layout()
                plt.savefig(f'{figPath}/motormap_{arm}_{i+1}.png')
            else:
                plt.show()
            plt.close()

            var_fwd.append(var_fwd_array)
            var_rev.append(var_rev_array)
        
        
        var_fwd=np.array(var_fwd)
        var_rev=np.array(var_rev)
        
        
        plt.figure(figsize=(10, 12))
        plt.clf()
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
            
        for i in self.goodIdx:
            x=range(len(xmlList))
            divfwd_array=[]
            divrev_array=[]
            for f in range(len(xmlList)):
                if arm == 'phi':
                    # applying 5-sigma filter
                    data = var_fwd[i,f,0:42]
                    divfwd_avg=np.mean(data[abs(data - np.mean(data)) < 5 * np.std(data)])
                    
                    data = var_rev[i,f,0:42]
                    divrev_avg=np.mean(data[abs(data - np.mean(data)) < 5 * np.std(data)])
                else:
                    data = var_fwd[i,f,0:84]
                    divfwd_avg=np.mean(data[abs(data - np.mean(data)) < 5 * np.std(data)])
                    
                    data = var_rev[i,f,0:84]
                    divrev_avg=np.mean(data[abs(data - np.mean(data)) < 5 * np.std(data)])

                divfwd_array.append(divfwd_avg)
                divrev_array.append(divrev_avg)
            
            divfwd_array=np.array(divfwd_array)
            divrev_array=np.array(divfwd_array)

            if np.max(divfwd_array) > 0.4:
                ax1.scatter(x,divfwd_array,marker='.',s=25,label=f'{i+1}')
            else:
                ax1.scatter(x,divfwd_array,marker='.',s=25)
            
            if np.max(divrev_array) > 0.4:
                ax2.scatter(x,divrev_array,marker='.',s=25,label=f'{i+1}')
            else:
                ax2.scatter(x,divrev_array,marker='.',s=25)
        ax1.legend()
        ax2.legend()
        ax1.set_title(f'Fiber FWD Variation')
        ax2.set_title(f'Fiber REV Variation')
        ax2.set_xlabel("Motor Map Run")
        ax1.set_ylabel("Variation")
        ax2.set_ylabel("Variation")
        plt.savefig(f'{figPath}/allvariant{arm}.png')
        plt.close()

        if pdffile is not None:
            cmd=f"""convert {figPath}motormap*_[0-9].png {figPath}motormap*_[0-9]?.png \
            {figPath}motormap*_[0-9]?.png {pdffile}"""
            retcode = subprocess.call(cmd,shell=True)
            print(cmd)
        return var_fwd,var_rev
