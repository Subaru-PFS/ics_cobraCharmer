import sys, os
from importlib import reload
import numpy as np
import time
import datetime
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import glob
from copy import deepcopy
from ics.cobraCharmer import pfi as pfiControl
import pandas as pd

from bokeh.models.annotations import Title
from bokeh.io import output_notebook, show, export_png,export_svgs
from bokeh.plotting import figure, show, output_file
import bokeh.palettes
from bokeh.layouts import column,gridplot
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper
from bokeh.models.glyphs import Text
from bokeh.palettes import inferno
from bokeh.models import BasicTicker, PrintfTickFormatter, ColorBar

from bokeh.transform import linear_cmap
from copy import deepcopy

class targetConvergence():

    def setFiberUDPos(self):
         # Home phi
        self.pfi.moveAllSteps(self.allCobras, 0, -5000)

        # Home theta
        self.pfi.moveAllSteps(self.allCobras, -10000, 0)
        self.pfi.moveAllSteps(self.allCobras, -2000, 0)
        
        # Move the bad cobras to up/down positions
        self.pfi.moveSteps(getCobras([0,38,42,53]), [3200,800,4200,5000], np.zeros(4))

        # move visible positioners to outwards positions, phi arms are moved out for 60 degrees
        # (outTargets) otherwise we can't measure the theta angles
        thetas = np.empty(57, dtype=float)
        thetas[::2] = self.pfi.thetaToLocal(self.oddCobras, np.full(len(self.oddCobras), np.deg2rad(270)))
        thetas[1::2] = self.pfi.thetaToLocal(self.evenCobras, np.full(len(self.evenCobras), np.deg2rad(90)))
        phis = np.full(57, np.deg2rad(60.0))
        outTargets = self.pfi.anglesToPositions(self.allCobras, thetas, phis)

        # Home the good cobras
        self.pfi.moveAllSteps(getCobras(self.goodIdx), -10000, -5000)
        self.pfi.moveAllSteps(getCobras(self.goodIdx), -5000, -5000)
        
        # move to outTargets
        _ = moveToXYfromHome(self.pfi, self.goodIdx, outTargets[self.goodIdx], self.datapath)

        # move phi arms in
        self.pfi.moveAllSteps(getCobras(self.goodIdx), 0, -5000)

    def oneDimensionTest(self, repeat, Theta=False, Phi=False):
        
        print(f"Repeating numbers = {repeat}")
        for i in range(repeat):
            print(f"runngin {i} iteration")
            # Given random angels
            target = np.random.rand(len(self.allCobras))    
            
            if Theta == True:
                print("running theta convergence test.")
                # Allowing theta angel from 10 to 350
                target = target*340 + 10

                # move visible positioners to outwards positions, phi arms are moved out for 60 degrees
                # (outTargets) otherwise we can't measure the theta angles
                thetas = np.full(len(self.allCobras), np.deg2rad(target))
                #thetas[::2] = self.pfi.thetaToLocal(self.oddCobras, np.full(len(self.oddCobras), np.deg2rad(270)))
                #thetas[1::2] = self.pfi.thetaToLocal(self.evenCobras, np.full(len(self.evenCobras), np.deg2rad(90)))
                phis = np.full(57, np.deg2rad(60.0))
                outTargets = self.pfi.anglesToPositions(self.allCobras, thetas, phis)

                # Home the good cobras
                self.pfi.moveAllSteps(getCobras(self.goodIdx), -10000, -5000)
                self.pfi.moveAllSteps(getCobras(self.goodIdx), -2000, -5000)

                data = moveToXYfromHome(self.pfi, self.goodIdx, outTargets[self.goodIdx], self.datapath, 
                    stackImage=str(i), maxTries=13)

                np.array(data)
                np.save(f'{self.datapath}/outTargetTheta_{i}',outTargets)
                np.save(f'{self.datapath}/curPositionTheta_{i}',data)

            if Phi == True:
                print("running phi convergence test.")
                # Allowing phi angel from 10 to 170
                #phis = np.deg2rad(target*130 + 20)
                #print(phis)
                thetas = np.empty(len(self.allCobras), dtype=float)
                phis = np.full(len(self.allCobras), np.deg2rad(110))
                
                thetas[::2] = self.pfi.thetaToLocal(self.oddCobras, np.full(len(self.oddCobras), np.deg2rad(270)))
                thetas[1::2] = self.pfi.thetaToLocal(self.evenCobras, np.full(len(self.evenCobras), np.deg2rad(90)))
                outTargets = self.pfi.anglesToPositions(self.allCobras, thetas, phis)

                # Home the good cobras
                self.pfi.moveAllSteps(getCobras(self.goodIdx), -10000, -5000)
                self.pfi.moveAllSteps(getCobras(self.goodIdx), -2000, -5000)

                data = moveToXYfromHome(self.pfi, self.goodIdx, outTargets[self.goodIdx], self.datapath, 
                    stackImage=str(i), maxTries=13)

                np.array(data)
                np.save(f'{self.datapath}/outTargetPhi_{i}',outTargets)
                np.save(f'{self.datapath}/curPositionPhi_{i}',data)

        print("End of convergence test.")

        pass

    def plotFiberSNR(self, idx, distArray, outputPath):
        dist_array = distArray

        N=4000
        x = np.random.random(size=N) * 100
        y = np.random.random(size=N) * 100

        colors = [
            "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)
        ]


        TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']
        p = figure(tools=TOOLS, x_range=[0, 10], y_range=[0,1.1],plot_height=500,plot_width=800)

        p.xaxis.axis_label = "Iteration No."
        p.yaxis.axis_label = "SNR"

        snr_arr=[]
        msnr_arr=[]
        for dist in dist_array[:,:,idx]:
            snr=(1-self.k_offset*dist**2)*self.factor 
            snr[snr<0]=np.nan
            snr_arr.append(snr)

            inx = np.isnan(snr)
            new_snr = snr
            new_snr[inx] = 0

            msnr_arr.append(np.max(new_snr))
            iteration = np.arange(0,14,1)
            #print(dist)
            #print(snr)
            p.scatter(x=iteration, y=snr, radius=0.05,
                    fill_color=colors[0:14], fill_alpha=0.8,
                    line_color=None)

        snr_arr=np.array(snr_arr)
        snr_avg=np.nanmean(snr_arr, axis=0)

        where_are_NaNs = np.isnan(snr_avg)
        snr_avg[where_are_NaNs] = 0
        
        p.circle(x=np.arange(0,9,1),y=snr_avg[:9], radius=0.08,color='green',)
        p.line(x=np.arange(0,9,1),y=snr_avg[:9],color='green', line_width=3) 

        t = Title()
        t.text = f'Fiber No {self.visibles[idx]}, Max SNR = {np.max(snr_avg):.3f}' #at {ind[0][0]}th iteration'
        p.title = t

        #show(p)
        export_png(p,filename=outputPath+"ConvergeTest_"+str(int(self.visibles[idx]))+".png")

        return msnr_arr

    def visualizeFiberSNR(self, reps, dataPath, outputPath):
        #dataPath='/Volumes/Disk/Data/Converge_20190213/'


        tobs=900
        tmax=105
        tstep=np.array(range(0,14))*8+12

        pixelscale=83
        self.k_offset=1/(.075)**2
        self.factor=(np.sqrt((tmax+tobs-tstep)/(tobs)))


        target_array = []
        pos_array = []
        dist_array = []
        for i in range(reps):
            tar = np.load(dataPath+f'outTargetPhi_{i}.npy')
            pos = np.load(dataPath+f'curPositionPhi_{i}.npy')
            target_array.append(tar)
            pos_array.append(pos)
            
            dist=pixelscale*(np.abs(pos-tar[self.goodIdx]))/1000
            dist_array.append(dist)
            
            
        pos_array = np.array(pos_array)
        target_array = np.array(target_array)
        dist_array = np.array(dist_array)

        #outputPath='/Volumes/Disk/Data/Convergence/'


        snr_list = np.array([])
        fiber_list = np.array([])
        repeat_list = np.array([])
        for idx in range(len(self.goodIdx)):
            #print(idx)
            maxsnr = self.plotFiberSNR(idx, dist_array, outputPath)
            fids = np.full(len(maxsnr), self.visibles[idx])    
            repeats = np.arange(1,reps+1)

            snr_list = np.append(snr_list,maxsnr)
            fiber_list = np.append(fiber_list,fids)
            repeat_list = np.append(repeat_list,repeats)   
        
        #snr_list=np.array(snr_list)
        #fiber_list=np.array(fiber_list)
        #repeat_list = np.array(repeat_list)
        
        df = pd.DataFrame(data={"x":fiber_list, "y":repeat_list, "value":snr_list})
        TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']
        colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
        mapper = LinearColorMapper(palette="Viridis256", low=df.value.min(), high=df.value.max())
        
        p = figure(plot_width=1000, plot_height=800, x_range=[0,58], y_range=[0,51],tools=TOOLS)
        p.xaxis.axis_label = "Fiber No."
        p.yaxis.axis_label = "Max SNR for Each Convergence Test"
        
        p.rect(x="x", y="y", width=1, height=1,
           source=df,
           fill_color={'field': 'value', 'transform': mapper},
           line_color=None)
        
        color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%4.2f"))

        p.add_layout(color_bar, 'right')

        export_png(p,filename=outputPath+"ConvergeTest_MaxSNR.png")
    
    
    def __init__(self, IPstring, XML, dataPath, fiberlist=False, Connect=True):
        
        if fiberlist is not False:
            self.visibles = fiberlist
        else:
            self.visibles =  range(1,58)
            
        self.goodIdx = np.array(self.visibles) - 1

        
        self.datapath = dataPath
         # Define the cobra range.
        mod1Cobras = pfiControl.PFI.allocateCobraRange(range(1,2))
        self.allCobras = mod1Cobras
        # partition module 1 cobras into odd and even sets
        moduleCobras2 = {}
        for group in 1,2:
            cm = range(group,58,2)
            mod = [1]*len(cm)
            moduleCobras2[group] = pfiControl.PFI.allocateCobraList(zip(mod,cm))
        self.oddCobras = moduleCobras2[1]
        self.evenCobras = moduleCobras2[2]
        
        '''
            Make connection to module 
        '''   
        # Initializing COBRA module
        if Connect is True:
                self.pfi = pfiControl.PFI(fpgaHost=IPstring) #'fpga' for real device.
                if not os.path.exists(XML):
                    print(f"Error: {XML} not presented!")
                    sys.exit()
        
                self.pfi.loadModel(XML)
                self.pfi.setFreq(self.allCobras)
        else:
                self.pfi = pfiControl.PFI(fpgaHost=IPstring, doConnect=False) #'fpga' for real device.
        #preciseXML=cobraCharmerPath+'/xml/updateOntime_'+datetoday+'.xml'

        # if not os.path.exists(XML):
        #     print(f"Error: {XML} not presented!")
        #     sys.exit()
        
        # self.pfi.loadModel(XML)
        # self.pfi.setFreq(self.allCobras)


        # Preparing data path for storage.
        storagePath = dataPath+'/data'
        prodctPath = dataPath+'/product'

        # Prepare the data path for the work
        if not (os.path.exists(dataPath)):
            os.makedirs(dataPath)
        if not (os.path.exists(storagePath)):
            os.makedirs(storagePath)
        if not (os.path.exists(prodctPath)):
            os.makedirs(prodctPath)
        
        #pass
    
    def __del__(self):
        pass

def getMaxSNR(pid):
    brokens = [1, 39, 43, 54]
    visibles= [e for e in range(1,58) if e not in brokens]


    
    pid = pid
    tobs=900
    tmax=105
    tstep=np.array(range(0,11))*8+12

    pixelscale=83
    k_offset=1/(.075)**2
    factor=(np.sqrt((tmax+tobs-tstep)/(tobs)))

    N=4000
    x = np.random.random(size=N) * 100
    y = np.random.random(size=N) * 100

    colors = [
        "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)
    ]

    TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']

    p = figure(tools=TOOLS, x_range=[-1,10], y_range=[0.8,1.05],plot_height=400,plot_width=500, 
               title="Fiber No. "+str(int(visibles[pid])))

    p.xaxis.axis_label = "Iteration"
    p.yaxis.axis_label = "SNR"


    snr_arr=np.array([])
    for i in range(1,4):
        pos=np.load('Data/pos'+str(i)+'.npy')
        target=np.load('Data/targets'+str(i)+'.npy')

        dist=pixelscale*(np.abs(pos[pid,:]-target[pid]))/1000
        
        snr=(1-k_offset*dist**2)*factor    
        #snr[snr<0]=0
        #print(np.abs(pos[pid-1,:]-target[pid-1]))
        #print(snr)
        if i == 1:
            snr_arr=snr
        else:
            snr_arr=np.vstack((snr_arr,snr))

        iteration = np.arange(0,11,1)
        p.scatter(x=iteration, y=snr, radius=0.1,
              fill_color=colors[0:11], fill_alpha=0.8,
              line_color=None)

    #print(snr_arr)    
    snr_avg=np.nanmean(snr_arr, axis=0)
    p.line(x=np.arange(0,9,1),y=snr_avg[:9],color='green', line_width=3)    

    #print(snr_avg)
    #show(p)
    ind=np.where(snr_avg == np.nanmax(snr_avg))
    
    if np.nanmax(snr_avg) < 0.45:
        return 10, 0, p
    else:
        return ind[0][0], np.nanmax(snr_avg), p


def lazyIdentification(centers, spots, radii=None):
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

def getCobras(cobs):
    # cobs is 0-indexed list
    return pfiControl.PFI.allocateCobraList(zip(np.full(len(cobs), 1), np.array(cobs) + 1))


# function to move cobras to target positions
def moveToXYfromHome(pfi, idx, targets, dataPath, stackImage=None, threshold=3.0, maxTries=10, cam_split=26):
    cobras = getCobras(idx)
    pfi.moveXYfromHome(cobras, targets)

    ntries = 1

    posArray = []
    while True:
        # check current positions, first exposing
        p1 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "1", "-e", "18", "-f", dataPath+"/cam1_"], stdout=PIPE)
        p1.communicate()
        p2 = Popen(["/home/pfs/IDSControl/idsexposure", "-d", "2", "-e", "18", "-f", dataPath+"/cam2_"], stdout=PIPE)
        p2.communicate()

        # extract sources and fiber identification
        data1 = fits.getdata(dataPath+f'/cam1_0001.fits').astype(float)
        ext1 = sep.extract(data1, 100)
        idx1 = lazyIdentification(pfi.calibModel.centers[idx[idx <= cam_split]], ext1['x'] + ext1['y']*(1j))
        data2 = fits.getdata(dataPath+f'/cam2_0001.fits').astype(float)
        ext2 = sep.extract(data2, 100)
        idx2 = lazyIdentification(pfi.calibModel.centers[idx[idx > cam_split]], ext2['x'] + ext2['y']*(1j))
        curPos = np.concatenate((ext1[idx1]['x'] + ext1[idx1]['y']*(1j), ext2[idx2]['x'] + ext2[idx2]['y']*(1j)))
        
        if stackImage is not None:
            if ntries == 1:
                stackImg1 = data1
                stackImg2 = data2
            else:
                stackImg1 =stackImg1 + data1
                stackImg2 =stackImg2 + data2
            
            fits.writeto(dataPath+f'/product/Cam1_stacked_'+stackImage+'.fits',stackImg1,overwrite=True)
            fits.writeto(dataPath+f'/product/Cam2_stacked_'+stackImage+'.fits',stackImg2,overwrite=True)


        
        print(curPos)
        print(np.abs(curPos - targets))
        posArray.append(curPos)
        
        # check position errors
        done = np.abs(curPos - targets) <= threshold
        if np.all(done):
            print('Convergence sequence done')
            break
        if ntries > maxTries:
            print(f'Reach max {maxTries} tries, gave up')
            break
        ntries += 1

        # move again
        pfi.moveXY(cobras, curPos, targets)
        
    return posArray

def setFiberUDPOS(XML, DataPath):
    # Define the cobra range.
    mod1Cobras = pfiControl.PFI.allocateCobraRange(range(1,2))
    allCobras = mod1Cobras
    oneCobra = pfiControl.PFI.allocateCobraList([(1,2)])
    twoCobras = pfiControl.PFI.allocateCobraList([(1,2), (1,5)])

    # partition module 1 cobras into non-interfering sets
    moduleCobras = {}
    for group in 1,2,3:
        cm = range(group,58,3)
        mod = [1]*len(cm)
        moduleCobras[group] = pfiControl.PFI.allocateCobraList(zip(mod,cm))
    group1Cobras = moduleCobras[1]
    group2Cobras = moduleCobras[2]
    group3Cobras = moduleCobras[3]

    # partition module 1 cobras into odd and even sets
    moduleCobras2 = {}
    for group in 1,2:
        cm = range(group,58,2)
        mod = [1]*len(cm)
        moduleCobras2[group] = pfiControl.PFI.allocateCobraList(zip(mod,cm))
    oddCobras = moduleCobras2[1]
    evenCobras = moduleCobras2[2]

    # Initializing COBRA module
    pfi = pfiControl.PFI(fpgaHost='128.149.77.24') #'fpga' for real device.
    #preciseXML=cobraCharmerPath+'/xml/updateOntime_'+datetoday+'.xml'

    if not os.path.exists(XML):
        print(f"Error: {XML} not presented!")
        sys.exit()
        
    pfi.loadModel(XML)
    pfi.setFreq(allCobras)

    # Calculate up/down(outward) angles
    oddMoves = pfi.thetaToLocal(oddCobras, [np.deg2rad(270)]*len(oddCobras))
    oddMoves[oddMoves>1.85*np.pi] = 0

    evenMoves = pfi.thetaToLocal(evenCobras, [np.deg2rad(90)]*len(evenCobras))
    evenMoves[evenMoves>1.85*np.pi] = 0

    allMoves = np.zeros(57)
    allMoves[::2] = oddMoves
    allMoves[1::2] = evenMoves

    allSteps, _ = pfi.calculateSteps(np.zeros(57), allMoves, np.zeros(57), np.zeros(57))

    # define the broken/good cobras
    brokens = [1, 39, 43, 54]
    visibles= [e for e in range(1,58) if e not in brokens]
    badIdx = np.array(brokens) - 1
    goodIdx = np.array(visibles) - 1

    # two groups for two cameras
    cam_split = 26
    group1 = goodIdx[goodIdx <= cam_split]
    group2 = goodIdx[goodIdx > cam_split]

    # three non-interfering groups for good cobras
    goodGroupIdx = {}
    for group in range(3):
        goodGroupIdx[group] = goodIdx[goodIdx%3==group]

    def getCobras(cobs):
        # cobs is 0-indexed list
        return pfiControl.PFI.allocateCobraList(zip(np.full(len(cobs), 1), np.array(cobs) + 1))

    # Home phi
    pfi.moveAllSteps(allCobras, 0, -5000)

    # Home theta
    pfi.moveAllSteps(allCobras, -10000, 0)
    pfi.moveAllSteps(allCobras, -10000, 0)
    
    # Move the bad cobras to up/down positions
    pfi.moveSteps(getCobras(badIdx), allSteps[badIdx], np.zeros(len(brokens)))

    # move visible positioners to outwards positions, phi arms are moved out for 60 degrees
    # (outTargets) otherwise we can't measure the theta angles
    thetas = np.empty(57, dtype=float)
    thetas[::2] = pfi.thetaToLocal(oddCobras, np.full(len(oddCobras), np.deg2rad(270)))
    thetas[1::2] = pfi.thetaToLocal(evenCobras, np.full(len(evenCobras), np.deg2rad(90)))
    phis = np.full(57, np.deg2rad(60.0))
    outTargets = pfi.anglesToPositions(allCobras, thetas, phis)

    # Home the good cobras
    pfi.moveAllSteps(getCobras(goodIdx), -10000, -5000)
    pfi.moveAllSteps(getCobras(goodIdx), -10000, -5000)
    
    # move to outTargets
    moveToXYfromHome(pfi, goodIdx, outTargets[goodIdx], DataPath)

    # move phi arms in
    pfi.moveAllSteps(getCobras(goodIdx), 0, -5000)


def main():

    cobraCharmerPath='/home/pfs/mhs/devel/ics_cobraCharmer/'
    xml=cobraCharmerPath+'xml/precise_20190212.xml'
    #xml=cobraCharmerPath+'/xml/precise6.xml'

    datetoday=datetime.datetime.now().strftime("%Y%m%d")
    dataPath = '/data/pfs/Converge_'+datetoday+'/'
    
    outputPath = '/data/pfs/Converge_'+datetoday+'/'

    IP = '128.149.77.24'
    #IP = 'localhost'
    
    brokens = [1, 39, 43, 54]
    visibles= [e for e in range(1,58) if e not in brokens]

    targetCon = targetConvergence(IP, xml, dataPath, fiberlist=visibles, Connect=True)
    #targetCon.setFiberUDPos()
    targetCon.oneDimensionTest(2, Phi=True)
    #targetCon.oneDimensionTest(50, Theta=True)


    targetCon.visualizeFiberSNR(2,dataPath, outputPath)


if __name__ == '__main__':
    main()
