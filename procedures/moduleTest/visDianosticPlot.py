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

from bokeh.io import output_notebook, show, export_png,export_svgs
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


from bokeh.plotting import figure, show, output_file
import bokeh.palettes
from bokeh.layouts import column,gridplot
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper
from bokeh.models.glyphs import Text

from bokeh.transform import linear_cmap
from bokeh.palettes import Category20

from ics.cobraCharmer import pfiDesign
from ics.cobraCharmer import func
import fnmatch
from ics.fpsActor import fpsFunction as fpstool
import pandas as pd
from opdb import opdb


class VisDianosticPlot(object):

    def __init__(self, runDir=None, xml=None, arm=None, datatype=None):
        
        
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
        path = self.path+'data/'
        
        if arm is None:
            raise Exception('Define the arm')

        fwdFitsFile= f'{path}{arm}ForwardStack0.fits'
        revFitsFile= f'{path}{arm}ReverseStack0.fits'

        fwdImage = pyfits.open(fwdFitsFile)
        self.fwdStack=fwdImage[1].data

        revImage = pyfits.open(revFitsFile)
        self.revStack=revImage[1].data


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
        self.badMM = np.load(path + 'badMotorMap.npy')
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


    def visCreateNewPlot(self,title, xLabel, yLabel, size=(8, 8), 
        aspectRatio="equal", **kwargs):


        #plt.figure(figsize=size, facecolor="white", tight_layout=True, **kwargs)
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.show(block=False)

        # Set the axes aspect ratio
        ax = plt.gca()
        ax.set_aspect(aspectRatio)
    
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
    


    def visGeometryFromXML(self, newXml, thetaAngle=None, markCobra=False, patrol=False):
        des = pfiDesign.PFIDesign(newXml)
        
        ax = plt.gca()

        ax.scatter(des.centers.real, des.centers.imag,marker='o', color='white', s=20)


        # Adding theta hard-stops
        length=des.L1+des.L1 

        self._addLine(des.centers,length,des.tht0,color='orange',
                        linewidth=0.5,linestyle='--')

        self._addLine(des.centers,length,des.tht1,color='black',
                        linewidth=0.5,linestyle='-.')

        if thetaAngle is not None:
            self._addLine(des.centers,des.L1,thetaAngle,color='blue',
                    linewidth=2,linestyle='-')


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
        

    def visSubaruConvergence(self):
        ax = plt.gca()

        visitID = int(self._findFITS()[0][-12:-7])
        subID = 11
        frameid = visitID*100+subID

        db=opdb.OpDB(hostname='pfsa-db01', port=5432,
                   dbname='opdb',username='pfs')
        
        match = db.bulkSelect('cobra_match','select * from cobra_match where '
                      f'mcs_frame_id = {frameid}').sort_values(by=['cobra_id']).reset_index()

        path=f'{self.path}/data/'

        tarfile = path+'targets.npy'
        targets=np.load(tarfile)

        dist=np.sqrt((match['pfi_center_x_mm'].values[self.goodIdx]-targets.real)**2+
            (match['pfi_center_y_mm'].values[self.goodIdx]-targets.imag)**2)

        sc=ax.scatter(self.calibModel.centers.real[self.goodIdx],self.calibModel.centers.imag[self.goodIdx],
            c=dist,marker='s', vmax=0.8)
        
        plt.colorbar(sc)



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

    def visDotLocation(self):
        
        ax = plt.gca()

        dotFile = '/software/devel/pfs/pfs_instdata/data/pfi/dot/black_dots_mm.csv'
        newDot=pd.read_csv(dotFile)

        # Plot DOT location
        for dotidx in range(len(newDot)):
            e = plt.Circle((newDot['x'].values[dotidx], newDot['y'].values[dotidx]), newDot['r'].values[dotidx], 
                        color='grey', fill=True, alpha=0.5)
            ax.add_artist(e)



    def visPlotFiberSpots(self, cobraIdx=None):

        data = self.imgdata
        m, s = np.mean(data), np.std(data)
        #fig, ax = plt.subplots(figsize=(10,10))
        ax = plt.gca()
        im = ax.imshow(data, interpolation='nearest', 
                    cmap='gray', vmin=m-s, vmax=m+3*s, origin='lower')
        
        if cobraIdx is None:
            cobra = self.goodIdx
        else:
            cobra = cobraIdx

        for idx in cobra:
            c = plt.Circle((self.calibModel.centers[idx].real, self.calibModel.centers[idx].imag), 5, color='g', fill=False)
            ax.add_artist(c)


        for idx in cobra:
            d = plt.Circle((self.centers[idx].real, self.centers[idx].imag), self.radius[idx], color='red', fill=False)
            ax.add_artist(d)
  
        ax.scatter(self.centers[cobra].real,self.centers[cobra].imag,color='red')    

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
                ax.plot(self.fw[k][n,0].real, self.fw[k][n,0].imag, c + 'o')
                ax.plot(self.rv[k][n,0].real, self.rv[k][n,0].imag, d + 's')
                ax.plot(self.fw[k][n,1:].real, self.fw[k][n,1:].imag, c + '.')
                ax.plot(self.rv[k][n,1:].real, self.rv[k][n,1:].imag, d + '.')

        plt.show()
    
    def visStackedImage(self, direction='fwd', flip=False):
        if direction == 'fwd':
            data = self.fwdStack
        else:
            data = self.fwdStack

        if flip is True:
            image=(np.flip(data).T).copy(order='C')
        
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

        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(ChromeDriverManager().install(),options=options)


        if figPath is not None:
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


    def visMultiMotorMapfromXML(self, xmlList, figPath=None, arm='phi', pdffile=None, fast=False):
        binSize = 3.6
        x=np.arange(112)*3.6
        
        
        for i in self.goodIdx:
            plt.figure(figsize=(10, 8))
            plt.clf()

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
            else:
                plt.show()
            plt.close()


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
