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
            
            if Phi == True:
                print("running phi convergence test.")
                # Allowing phi angel from 10 to 170
                target = target*160 + 10
               
                thetas = np.empty(len(self.allCobras), dtype=float)
                phis = np.full(len(self.allCobras), np.deg2rad(target))
                
                thetas[::2] = self.pfi.thetaToLocal(self.oddCobras, np.full(len(self.oddCobras), np.deg2rad(270)))
                thetas[1::2] = self.pfi.thetaToLocal(self.evenCobras, np.full(len(self.evenCobras), np.deg2rad(90)))
                outTargets = self.pfi.anglesToPositions(self.allCobras, thetas, phis)

                # Home the good cobras
                self.pfi.moveAllSteps(getCobras(self.goodIdx), -10000, -5000)
                self.pfi.moveAllSteps(getCobras(self.goodIdx), -2000, -5000)

                data = moveToXYfromHome(self.pfi, self.goodIdx, outTargets[self.goodIdx], self.datapath, maxTries=13)

                np.array(data)
                np.save(f'{self.datapath}/outTarget_{i}',outTargets)
                np.save(f'{self.datapath}/curPosition_{i}',data)

        print("End of convergence test.")

        pass
    
    def __init__(self, IPstring, XML, dataPath, fiberlist=False):
        
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
        self.pfi = pfiControl.PFI(fpgaHost=IPstring) #'fpga' for real device.
        #preciseXML=cobraCharmerPath+'/xml/updateOntime_'+datetoday+'.xml'

        if not os.path.exists(XML):
            print(f"Error: {XML} not presented!")
            sys.exit()
        
        self.pfi.loadModel(XML)
        self.pfi.setFreq(self.allCobras)


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
def moveToXYfromHome(pfi, idx, targets, dataPath, threshold=3.0, maxTries=10, cam_split=26):
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
    dataPath = '/data/pfs/Converge_'+datetoday
    
    IP = '128.149.77.24'
    
    brokens = [1, 39, 43, 54]
    visibles= [e for e in range(1,58) if e not in brokens]

    targerCon = targetConvergence(IP, xml, dataPath, fiberlist=visibles)
    targerCon.setFiberUDPos()
    targerCon.oneDimensionTest(50, Phi=True)


if __name__ == '__main__':
    main()
