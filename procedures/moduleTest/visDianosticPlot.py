import matplotlib.pyplot as plt
import numpy as np
import glob
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os

from bokeh.io import output_notebook, show, export_png,export_svgs
from bokeh.plotting import figure, show, output_file
import bokeh.palettes
from bokeh.layouts import column,gridplot
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper
from bokeh.models.glyphs import Text

from bokeh.transform import linear_cmap
from bokeh.palettes import Category20


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


    
    def visCobraMotorMap(self, stepsize=50, figpath=None, arm=None):
        try:
            self.mf
        except AttributeError:
            self._loadCobraData(arm=arm)
        
        if arm is None:
            raise Exception('Define the arm')
        
        ymax = 1.1*np.rad2deg(np.max(self.mf))
        ymin = -1.1*np.rad2deg(np.max(self.mr))

        for fiber in self.goodIdx:
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
            
            if figpath is not None:
                plt.ioff()
                plt.savefig(f'{figpath}/motormap_{arm}_{c+1}.png')
            else:
                plt.show()
            plt.close()

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

    def visSpeedHisto(self, arm=None, figpath=None):
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

        if figpath is not None:
            export_png(grid,filename=figpath+"motor_speed_histogram.png")
            export_png(qgrid,filename=figpath+"motor_speed_std.png")