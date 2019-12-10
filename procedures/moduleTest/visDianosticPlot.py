import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import glob
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import sys
import math

from bokeh.io import output_notebook, show, export_png,export_svgs
from bokeh.plotting import figure, show, output_file
import bokeh.palettes
from bokeh.layouts import column,gridplot
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper
from bokeh.models.glyphs import Text

from bokeh.transform import linear_cmap
from bokeh.palettes import Category20

from ics.cobraCharmer import pfiDesign


class VisDianosticPlot(object):

    def __init__(self, datapath, brokens=None, camSplit=26):
        self.path = datapath
        
        self.camSplit = camSplit
        self.brokens = []

        if brokens is None:
            self.brokens = []
        else:
            self.brokens = brokens
     
        self.visibles= [e for e in range(1,58) if e not in brokens]
        self.badIdx = np.array(brokens) - 1
        self.goodIdx = np.array(self.visibles) - 1

        # two groups for two cameras
        cam_split = self.camSplit
        self.group1 = self.goodIdx[self.goodIdx <= cam_split]
        self.group2 = self.goodIdx[self.goodIdx > cam_split]


    def _loadCobraData(self, arm=None) :
        path = self.path
        
        if arm == 'phi':
            self.centers = np.load(path + 'phiCenter.npy')
            self.radius = np.load(path + 'phiRadius.npy')
            self.fw = np.load(path + 'phiFW.npy')
            self.rv = np.load(path + 'phiRV.npy')
            self.af = np.load(path + 'phiAngFW.npy')
            self.ar = np.load(path + 'phiAngRV.npy')
            self.sf = np.load(path + 'phiSpeedFW.npy')
            self.sr = np.load(path + 'phiSpeedRV.npy')
            self.mf = np.load(path + 'phiMMFW.npy')
            self.mr = np.load(path + 'phiMMRV.npy')
            self.bad = np.load(path + 'bad.npy')
            self.mf2 = np.load(path + 'phiMMFW2.npy')
            self.mr2 = np.load(path + 'phiMMRV2.npy')
            self.bad2 = np.load(path + 'bad2.npy')

        if arm == 'theta':
            self.centers = np.load(path + 'thetaCenter.npy')
            self.radius = np.load(path + 'thetaRadius.npy')
            self.fw = np.load(path + 'thetaFW.npy')
            self.rv = np.load(path + 'thetaRV.npy')
            self.af = np.load(path + 'thetaAngFW.npy')
            self.ar = np.load(path + 'thetaAngRV.npy')
            self.sf = np.load(path + 'thetaSpeedFW.npy')
            self.sr = np.load(path + 'thetaSpeedRV.npy')
            self.mf = np.load(path + 'thetaMMFW.npy')
            self.mr = np.load(path + 'thetaMMRV.npy')
            self.bad = np.load(path + 'bad.npy')
            self.mf2 = np.load(path + 'thetaMMFW2.npy')
            self.mr2 = np.load(path + 'thetaMMRV2.npy')
            self.bad2 = np.load(path + 'bad2.npy')

        if arm is None:
            raise Exception('Define the arm')

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

    def visPlotFiberDot(self, arm = None):
        try:
            self.fw
        except AttributeError:
            self._loadCobraData(arm=arm)
        
        if arm is None:
            raise Exception('Define the arm')

        plt.figure(figsize=(10, 20))
        plt.clf()

        plt.subplot(211)
        ax = plt.gca()
        ax.axis('equal')

        for n in range(1):
            for k in self.group1:
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

        plt.subplot(212)
        ax = plt.gca()
        ax.axis('equal')

        for n in range(1):
            for k in self.group2:
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
    
    def visStackedImage(self, arm = None, repeat=1):
        try:
            self.centers
        except AttributeError:
            self._loadCobraData(arm=arm)
       
        if arm is None:
            raise Exception('Define the arm')

        for n in range(repeat):
            cam1_list = glob.glob(self.path+f'/{arm}1*Stack{n}.fits.gz')
            cam1_list.sort() 
            cam2_list = glob.glob(self.path+f'/{arm}2*Stack{n}.fits.gz')
            cam2_list.sort()

            cam1_list.extend(cam2_list)
            a = 1000.0
            fig=plt.figure(figsize=(9, 7))
            columns = 1
            rows = len(cam1_list)
            i=0
            ax = []
            
            for f in cam1_list:
                hdu = fits.open(f)
                #xmax = np.max(self.centers[self.group1].real).astype('int')+200
                #xmin = np.min(self.centers[self.group1].real).astype('int')-200
                ymin = np.min(self.centers[self.group1].imag).astype('int')-150
                ymax = np.max(self.centers[self.group1].imag).astype('int')+150

                #print(xmax,xmin)
                #plt.subplots_adjust(hspace = .3)
                image = np.log10(a*hdu[0].data+1)/np.log10(a)
                basename = os.path.basename(f)
                ax.append( fig.add_subplot(rows, columns, i+1) )
                
                if i < len(cam1_list)-1:
                    ax[-1].get_xaxis().set_ticks([])
                
                ax[-1].set_title(basename, fontsize = 10)
                #ax[-1].set_ylim(0., 1.)
                plt.imshow(image[ymin:ymax:,:],cmap='gray')
                i=i+1
            plt.tight_layout()    
            plt.show()


    
    def visCobraMotorMap(self, stepsize=50, figPath=None, arm=None, pdffile=None, debug=False):
        try:
            self.mf
        except AttributeError:
            self._loadCobraData(arm=arm)
        
        if arm is None:
            raise Exception('Define the arm')
        
        ymax = 1.1*np.rad2deg(np.max(self.mf))
        ymin = -1.1*np.rad2deg(np.max(self.mr))



        for fiber in self.goodIdx:
            width = (fiber + 1) / 2
            
            bar = "\r[" + "#" * int(math.ceil(width)) + " " * int(29 - width) + "]"
            sys.stdout.write(f'{bar}')
            sys.stdout.flush()

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
            ax.plot(x,np.rad2deg(self.mf2[c]), 'pink')


            for data in self.ar[c]: 
                for i,item in enumerate(data):
                    if i < len(daf):
                        dar[i] = np.rad2deg(data[i+1] - data[i])/stepsize
                        ax.plot([np.rad2deg(data[i+1]),np.rad2deg(data[i])],[dar[i],dar[i]],color='grey')
            ax.plot(x,-np.rad2deg(self.mr[c]), 'r')
            ax.plot(x,-np.rad2deg(self.mr2[c]), color='pink')
            ax.set_ylim([ymin,ymax])
            if arm is 'phi':
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
            cmd=f"""convert {figPath}motormap*_[0-9].png {figPath}motormap*_[0-9]?.png {pdffile}"""
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

        if figPath is not None:
            export_png(grid,filename=figPath+"motor_speed_histogram.png")
            export_png(qgrid,filename=figPath+"motor_speed_std.png")

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
                {figPath}AngleMove*_[0-9]?.png {pdffile}"""
            retcode = subprocess.call(cmd,shell=True)
            print(cmd)

    def visConverge(self, figPath = None, arm = 'phi', runs = 50, margin = 15, montage=None, pdffile=None):
        
        if arm is 'phi':
            phiPath = self.path
            moveData = np.load(phiPath+'phiData.npy')
            angleLimit = 180
        else:
            thetaPath =  self.path
            moveData = np.load(thetaPath+'thetaData.npy')
            angleLimit = 360
        
        
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
            
            # Plot SNR
            snr_array=[]
            k_offset=1/(.075)**2
            tmax=105
            tobs=900
            tstep=x*8+12
            if arm is 'phi':
                linklength=0.026
            else:
                l1=0.026
                l2=0.028
                linklength = np.sqrt((l2-(l1*np.cos(np.deg2rad(60))))**2+(l1*np.sin(np.deg2rad(60)))**2)

            for i in range(runs):
                angle = margin + (360 - 2 * margin) * i / (runs - 1)
                if i == 0:
                    delAngle = (360 - 2 * margin) / (runs - 1)


                dist=np.abs(np.deg2rad(angle)-(np.append([0], moveData[fiberIdx,i,:,0])))*linklength
                snr=(1-k_offset*dist**2)*(np.sqrt((tmax+tobs-tstep[0:9])/(tobs)))
                snr_array.append(snr)
                sax.scatter(tstep[0:9],snr)

            snr_array=np.array(snr_array)
            snr_avg=np.mean(snr_array,axis=0)

            sax.plot(tstep[0:9],snr_avg,color='green',linewidth=4)
            sax.scatter(tstep[0:9],snr_avg,color='green',s=100)
            sax.set_title('SNR = %.3f'%(np.max(snr_avg)),fontsize=20)


            if arm is 'phi':
                vax.set_ylim([-10,200])
                hax.set_ylim([-10,200])
                sax.set_ylim([0.5,1.05])
            else:
                vax.set_ylim([-10,400])
                vax.set_ylim([-10,400])
                sax.set_ylim([0.5,1.05])
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
            if retcode is not 0:
                raise Exception
            #if figPath is not None:
            #    plt.savefig(figPath+f'Converge_{arm}_{fiberIdx+1}.png')
        
        if pdffile is not None:
            cmd=f"""convert {figPath}Con*_[0-9].png {figPath}Con*_[0-9]?.png {pdffile}"""
            retcode = subprocess.call(cmd,shell=True)
            print(cmd)
    
    def visMultiMotorMapfromXML(self, xmlList, figPath=None, arm='phi', pdffile=None):
        binSize = 3.6
        x=np.arange(112)*3.6
        
        
        for i in self.goodIdx:
            plt.figure(figsize=(10, 8))
            plt.clf()

            ax = plt.gca()
            
            for xml in xmlList:
                model = pfiDesign.PFIDesign(xml)

                if arm is 'phi':
                    slowFWmm = np.rad2deg(model.angularSteps[i]/model.S2Pm[i])
                    slowRVmm = np.rad2deg(model.angularSteps[i]/model.S2Nm[i])
                else:
                    slowFWmm = np.rad2deg(model.angularSteps[i]/model.S1Pm[i])
                    slowRVmm = np.rad2deg(model.angularSteps[i]/model.S1Nm[i])
                
                labeltext=os.path.basename(os.path.dirname(xml))
                ln1 = ax.plot(x,slowFWmm,label=f'{labeltext} Fwd')
                ln2 = ax.plot(x,-slowRVmm,label=f'{labeltext} Rev')

            ax.set_title(f'Fiber {arm} #{i+1}')
            ax.legend()
            if arm is 'phi':
                ax.set_xlim([0,200])
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
            cmd=f"""convert {figPath}motormap*_[0-9].png {figPath}motormap*_[0-9]?.png {pdffile}"""
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
            if arm is 'phi':
                baseSlowFWmm = np.rad2deg(basemodel.angularSteps[i]/basemodel.S2Pm[i])
                baseSlowRVmm = np.rad2deg(basemodel.angularSteps[i]/basemodel.S2Nm[i])
            else:
                baseSlowFWmm = np.rad2deg(basemodel.angularSteps[i]/basemodel.S1Pm[i])
                baseSlowRVmm = np.rad2deg(basemodel.angularSteps[i]/basemodel.S1Nm[i])
            

            var_fwd_array=[]
            var_rev_array=[]
            for f in xmlList:

                model = pfiDesign.PFIDesign(f)

                if arm is 'phi':
                    slowFWmm = np.rad2deg(model.angularSteps[i]/model.S2Pm[i])
                    slowRVmm = np.rad2deg(model.angularSteps[i]/model.S2Nm[i])
                else:
                    slowFWmm = np.rad2deg(model.angularSteps[i]/model.S1Pm[i])
                    slowRVmm = np.rad2deg(model.angularSteps[i]/model.S1Nm[i])
                
                labeltext=os.path.basename(os.path.dirname(f))[-6:]
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
            if arm is 'phi':
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
                if arm is 'phi':
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
            cmd=f"""convert {figPath}motormap*_[0-9].png {figPath}motormap*_[0-9]?.png {pdffile}"""
            retcode = subprocess.call(cmd,shell=True)
            print(cmd)
        return var_fwd,var_rev