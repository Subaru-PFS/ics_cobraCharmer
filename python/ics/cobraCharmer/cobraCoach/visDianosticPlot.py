import fnmatch
import glob
import logging
import os
import pathlib
import re
import subprocess

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pfs.utils.coordinates.transform as transformUtils
import psycopg2

#from webdriver_manager.chrome import ChromeDriverManager
from ics.cobraCharmer import func, pfiDesign
from mcsActor.mcsRoutines import fiducials
from mpl_toolkits.axes_grid1 import make_axes_locatable
from opdb import opdb
from pfs.utils.butler import Butler
from pfs.utils.fiberids import FiberIds
from scipy.optimize import curve_fit, least_squares
from sqlalchemy import create_engine


def findDesignFromVisit(visit):
    command = f"grep moveToPfsDesign /data/logs/actors/fps/202?-*-*.log | grep {visit}"

    try:
        output = subprocess.check_output(command, shell=True, text=True)
        slist_str = str(output)
        pattern = r"designI[dD]\)=\[Long\((\d+)\)\]"
        match = re.search(pattern, str(slist_str))


        if match:
            value = int(match.group(1))  # Extract the captured numerical value and convert it to float
            #print(value)
        else:
            value = None

    except subprocess.CalledProcessError:
        conn = psycopg2.connect("dbname='opdb' host='db-ics' port=5432 user='pfs'")
        engine = create_engine('postgresql+psycopg2://', creator=lambda: conn)


        pfsConfig = pd.read_sql('''
                    SELECT 
                        visit0,pfs_design_id,converg_tolerance   FROM public.pfs_config
                    WHERE 
                        pfs_config.visit0 = %(pfs_config)s
                ''', engine, params={'pfs_config': visit})

        conn.close()

        value = pfsConfig['pfs_design_id'].values[0]

    return value


def findToleranceFromVisit(visit):
    #command = f"grep moveToPfsDesign /data/logs/actors/fps/2024-*-*.log | grep {visit}"

    #try:
    #    output = subprocess.check_output(command, shell=True, text=True)
    #    slist_str = str(output)
    #    pattern = r"\{KEY\(tolerance\)=\[Float\((\d+(\.\d+)?)\)\]\}"

    #    match = re.search(pattern, str(slist_str))

    #    if match:
    #        value = float(match.group(1))  # Extract the captured numerical value and convert it to float
    #        #print(value)
    #    else:
    #        print("No match found. Set to default 0.01")
    #        value = 0.01

    #except subprocess.CalledProcessError as e:
    conn = psycopg2.connect("dbname='opdb' host='db-ics' port=5432 user='pfs'")
    engine = create_engine('postgresql+psycopg2://', creator=lambda: conn)


    pfsConfig = pd.read_sql('''
                SELECT 
                    visit0,converg_tolerance  FROM public.pfs_config
                WHERE 
                    pfs_config.visit0 = %(pfs_config)s
            ''', engine, params={'pfs_config': visit})

    conn.close()

    if pfsConfig['converg_tolerance'].values[0] is None:
        value = 0.01
    else:
        value = round(pfsConfig['converg_tolerance'].values[0],3)

    return value


def findRunDir(pfsVisitId):
    import fnmatch
    import os
    base_dir = "/data/MCS/"
    pattern = f"PFSC{pfsVisitId:06d}??.fits"
    # Only look in directories matching 202[2345]*
    for entry in os.scandir(base_dir):
        if entry.is_dir() and fnmatch.fnmatch(entry.name, "202[2345]*"):
            data_dir = os.path.join(base_dir, entry.name, "data")
            if os.path.isdir(data_dir):
                for fname in os.listdir(data_dir):
                    if fnmatch.fnmatch(fname, pattern):
                        return entry.name  # run_dir
    return None

def findVisit(runDir):
    try:
        filename = pathlib.Path(sorted(glob.glob(f'/data/MCS/{runDir}/data/PFSC*.fits'))[0]).name
        digits = ''.join(filter(str.isdigit, filename))
        return int(digits[:-2])
    except:
        return None

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


def circle_residuals(params, x, y):
    """
    Calculate the residuals (errors) between the data points and the circle.
    params: (cx, cy, r) - circle center (cx, cy) and radius r
    x, y: arrays of data points coordinates
    """
    cx, cy, r = params
    return np.sqrt((x - cx)**2 + (y - cy)**2)-r

def fit_circle(x, y, initial_guess=None):
    """
    Fit a circle to the given x, y points using the Least Squares Circle Fitting method.
    x, y: arrays of data points coordinates
    initial_guess: initial estimation of the circle (optional)
    return: (cx, cy, r) - circle center (cx, cy) and radius r
    """
    if initial_guess is None:
        # Choose the first three points as the initial estimation
        initial_guess = (x[0], y[0], np.sqrt((x[1]-x[0])**2 + (y[1]-y[0])**2))

    # Use least_squares to perform the iterative nonlinear least squares optimization
    result = least_squares(circle_residuals, initial_guess, args=(x, y))

    cx, cy, r = result.x

    return cx, cy, r


def fit_circle_ransac(x, y, num_iterations=100, threshold=1.0):
    best_inliers = None
    best_params = None

    for _ in range(num_iterations):
        # Randomly sample 3 data points to form an initial estimation of the circle
        indices = np.random.choice(len(x), 9, replace=False)
        initial_guess = (x[indices[0]], y[indices[0]], np.sqrt((x[indices[1]]-x[indices[0]])**2 + (y[indices[1]]-y[indices[0]])**2))

        # Fit the circle using the initial estimation
        cx, cy, r = fit_circle(x, y, initial_guess)

        # Calculate residuals and identify inliers
        residuals = circle_residuals((cx, cy, r), x, y)
        inliers = np.abs(residuals) < threshold

        # Update best parameters if we found more inliers
        if best_inliers is None or np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_params = (cx, cy, r)


    # Check if we have at least three inliers before refitting the circle
    if np.sum(best_inliers) >= 3:
        cx, cy, r = fit_circle(x[best_inliers], y[best_inliers])
    else:
        raise ValueError("Not enough inliers to fit the circle.")

    return cx, cy, r, best_inliers


class VisDianosticPlot:

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

            brokenNums = [i+1 for i,c in enumerate(allCobras) if
                    des.fiberIsBroken(c.cobraNum, c.module)]
            goodNums = [i+1 for i,c in enumerate(allCobras) if
                    des.cobraIsGood(c.cobraNum, c.module)]
            badNums = [e for e in range(1, nCobras+1) if e not in goodNums]


            self.invisibleIdx = np.array(brokenNums, dtype='i4') - 1
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
            #plt.show(block=False)

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

    def visSetCobra(self, cobraIdx, scale=1.1):
        """
            Set plot region to a given cobra index
        """
        center = self.calibModel.centers[cobraIdx]
        dist = scale*(self.calibModel.L1[cobraIdx]+self.calibModel.L2[cobraIdx])

        self.visSetAxesLimits([center.real-dist,center.real+dist],[center.imag-dist,center.imag+dist])

    def visUnassignedFibers(self, pfsVisitID):

        conn = psycopg2.connect("dbname='opdb' host='db-ics' port=5432 user='pfs'")
        engine = create_engine('postgresql+psycopg2://', creator=lambda: conn)


        with conn:
                fiberData = pd.read_sql('''
                    SELECT DISTINCT 
                        fiber_id, pfi_center_final_x_mm, pfi_center_final_y_mm, 
                        pfi_nominal_x_mm, pfi_nominal_y_mm
                    FROM 
                        pfs_config_fiber
                    WHERE
                        pfs_config_fiber.visit0 = %(visit0)s
                    -- limit 10
                ''', engine, params={'visit0': pfsVisitID})


        fid=FiberIds()
        fiberData['cobra_id']=fid.fiberIdToCobraId(fiberData['fiber_id'].values)
        fiberData=fiberData.sort_values('cobra_id')
        df = fiberData.loc[fiberData['cobra_id'] != 65535]
        unassigned_rows = df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].isna().all(axis=1)]
        unassigned_cobraIdx =  unassigned_rows['cobra_id'].values - 1

        ax = plt.gca()

        ax.scatter(self.calibModel.centers.real[unassigned_cobraIdx],
                   self.calibModel.centers.imag[unassigned_cobraIdx], marker='X', color='red', s=20)

    def visCobraArms(self, center, thetaLength, phiLength, thetaAngle, phiAngle):

        ax = plt.gca()
        self._addLine(center,thetaLength, thetaAngle,color='blue',
                    linewidth=2,linestyle='-')

        x = thetaLength*np.cos(thetaAngle)
        y = thetaLength*np.sin(thetaAngle)
        newx = center.real + x
        newy = center.imag + y
        newPos = newx+newy*1j
        phiOpenAngle = phiAngle+thetaAngle - np.pi
        self._addLine(newPos,phiLength,phiOpenAngle,color='blue',
                linewidth=10,linestyle='-', solid_capstyle='round',alpha=0.5)

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

        ax[1].text(0.7, 0.8, rf'Median = {np.median(diff[self.goodIdx]):.4f}, $\sigma$={sigma:.4f}',
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

        try:
            maxIteration = mov.shape[2]
        except:
            maxIteration = mov.shape[-1]

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

        print(f'MCS report: {mcs_finised}')



        fpga_notDone = []
        for iteration in range(maxIteration):
            try:
                ind = np.where(np.abs(mov[0,:,iteration]['position']) > 0)
                notDone = len(np.where(np.abs(mov[0,ind[0],iteration]['position']-targets[ind[0]]) > tolerance)[0])
            except:
                ind = np.where(np.abs(mov[:,iteration]['position']) > 0)
                notDone = len(np.where(np.abs(mov[ind[0],iteration]['position']-targets[ind[0]]) > tolerance)[0])
            fpga_notDone.append(notDone)
        conn = psycopg2.connect("dbname='opdb' host='db-ics' port=5432 user='pfs'")
        engine = create_engine('postgresql+psycopg2://', creator=lambda: conn)


        with conn:
            fiberData = pd.read_sql('''
                SELECT DISTINCT 
                    fiber_id, pfi_center_final_x_mm, pfi_center_final_y_mm, 
                    pfi_nominal_x_mm, pfi_nominal_y_mm
                FROM 
                    pfs_config_fiber
                WHERE
                    pfs_config_fiber.visit0 = %(visit0)s
                -- limit 10
            ''', engine, params={'visit0': pfsVisitID})


        fid=FiberIds()
        fiberData['cobra_id']=fid.fiberIdToCobraId(fiberData['fiber_id'].values)
        fiberData=fiberData.sort_values('cobra_id')
        df = fiberData.loc[fiberData['cobra_id'] != 65535]
        #unassigned_rows = df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].isna().all(axis=1)]
        #unassigned_cobraIdx =  unassigned_rows['cobra_id'].values - 1


        assigned_row= df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].notna().all(axis=1)]
        assigned_cobraIdx =  assigned_row['cobra_id'].values - 1

        print(f'FPGA report: {fpga_notDone}')
        fpga_notDone = np.array(fpga_notDone)
        fpga_finished = len(self.goodIdx) - fpga_notDone

        ax.set_aspect('auto')
        ax.set_xlim(0,maxIteration+1)
        ax.plot(fpga_finished, linestyle ='-', marker='x', label='FPS')
        ax.plot(mcs_finised, linestyle ='-',marker='.',label='MCS')
        ax.plot(fpga_finished - mcs_finised, label = 'FPS - MCS')

        # Plot the 95% of all assigned cobra
        ax.plot(np.zeros(12)+len(assigned_cobraIdx)*0.95,linestyle ='dotted', label = '95% Threshold')
        ax.legend()

        if getStoppedNum:
            return fpga_finished[-1], mcs_finised[-1]

    def visNotDoneCobra(self, pfsVisitID, tolerance = 0.1, doPlots=False):
        '''
            This function is used to plot the location of not done cobra
        '''
        if doPlots:
            ax = plt.gca()

        if pfsVisitID is None:
            visitID = int(self._findFITS()[0][-12:-7])
        else:
            visitID = pfsVisitID

        conn = psycopg2.connect("dbname='opdb' host='db-ics' port=5432 user='pfs'")
        engine = create_engine('postgresql+psycopg2://', creator=lambda: conn)


        with conn:
            fiberData = pd.read_sql('''
                SELECT DISTINCT 
                    fiber_id, pfi_center_final_x_mm, pfi_center_final_y_mm, 
                    pfi_nominal_x_mm, pfi_nominal_y_mm
                FROM 
                    pfs_config_fiber
                WHERE
                    pfs_config_fiber.visit0 = %(visit0)s
                -- limit 10
            ''', engine, params={'visit0': pfsVisitID})


        fid=FiberIds()
        fiberData['cobra_id']=fid.fiberIdToCobraId(fiberData['fiber_id'].values)
        fiberData=fiberData.sort_values('cobra_id')
        df = fiberData.loc[fiberData['cobra_id'] != 65535]
        unassigned_rows = df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].isna().all(axis=1)]
        unassigned_cobraIdx =  unassigned_rows['cobra_id'].values - 1


        assigned_row= df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].notna().all(axis=1)]
        assigned_cobraIdx =  assigned_row['cobra_id'].values - 1

        targetFromDB = df['pfi_nominal_x_mm'].values+df['pfi_nominal_y_mm'].values*1j
        isNan = np.isnan(targetFromDB)
        targetFromDB[isNan] = self.calibModel.centers[isNan]

        path=f'{self.path}/data/'
        tarfile = path+'targets.npy'
        movfile = path+'moves.npy'

        mov = np.load(movfile)

        frameid = pfsVisitID*100+(mov.shape[2]-1)

        db=opdb.OpDB(hostname='db-ics', port=5432,
                        dbname='opdb',username='pfs')
        match = db.bulkSelect('cobra_match','select * from cobra_match where '
                f'mcs_frame_id = {frameid}').sort_values(by=['cobra_id']).reset_index()

        # path=f'{self.path}/data/'

        # tarfile = path+'targets.npy'
        # movfile = path+'moves.npy'

        # mov = np.load(movfile)

        targets = targetFromDB
        dist=np.sqrt((match['pfi_center_x_mm'].values[assigned_cobraIdx]-targets[assigned_cobraIdx].real)**2+
                        (match['pfi_center_y_mm'].values[assigned_cobraIdx]-targets[assigned_cobraIdx].imag)**2)


        try:
            maxIteration = mov.shape[2]
        except:
            mov = np.array([mov])
            maxIteration = mov.shape[2]


        print(f'Tolerance: {tolerance}')

        #targets=np.load(tarfile)
        tar = targetFromDB[self.goodIdx]
        ind = np.where(np.abs(mov[0,:,maxIteration-1]['position']) > 0)
        notDone = len(np.where(np.abs(mov[0,ind[0],maxIteration-1]['position']-tar[ind[0]]) > tolerance)[0])

        #notDone = fpga_notDone[-1]
        print(f'Number of not done cobra (FPGA): {notDone}')

        if doPlots:
            sc=ax.scatter(self.calibModel.centers.real,self.calibModel.centers.imag,
                    c='grey',marker='s')
            ax.scatter(self.calibModel.centers.real[self.goodIdx[ind]],self.calibModel.centers.imag[self.goodIdx[ind]],
                c='red',marker='s',label='Not Converged')

        return ind[0]

    def visCobraMovement(self, pfsVisitID, cobraIdx=0, iteration = 8, newPlot=True):
        """
        Visualize the movement of a specific cobra during a PFS visit.
        Args:
            pfsVisitID (int): The PFS visit ID.
            cobraIdx (int, optional): The index of the cobra. Defaults to 0.
            iteration (int, optional): The iteration number. Defaults to 8.
        """

        #visit = 107682
        visit = pfsVisitID

        runDir = findRunDir(pfsVisitID)
        iteration=8

        mov=np.load(f'/data/MCS/{runDir}/data/moves.npy')
        tar=np.load(f'/data/MCS/{runDir}/data/targets.npy')

        movIdx = np.where(self.goodIdx == cobraIdx)[0][0]

        if newPlot is True:
            self.visCreateNewPlot(f'Visit = {visit} Cobra Index = {self.goodIdx[movIdx]}','X','Y')

        self.visGeometryFromXML(thetaAngle=mov[0,movIdx,iteration-1]['thetaAngle']+self.calibModel.tht0[self.goodIdx[movIdx]],
                            phiAngle=mov[0,movIdx,iteration-1]['phiAngle'],patrol=True)


        #self.visUnassignedFibers(pfsVisitID)

        ax=plt.gca()
        x = mov[0,movIdx,:]['position'].real
        y = mov[0,movIdx,:]['position'].imag

        for i in range(len(x)):
            if x[i]==0 and y[i]==0:
                x[i] = x[i-1]
                y[i] = y[i-1]
            ax.scatter(x[i], y[i], marker='.', alpha=0.5,label=f'{i+1}')
            ax.text(x[i], y[i],f'{i}')
        ax.scatter(tar.real[movIdx],tar.imag[movIdx],c='red',marker='x',label='target',s=80)
        #ax.scatter(self.calibModel.centers.real[idx], vis.calibModel.centers.imag[idx])

        #ax.legend()
        self.visSetCobra(self.goodIdx[movIdx], scale=1.0)



    def visTargetConvergence(self, pfsVisitID, maxIteration = 11, excludeUnassign=True, tolerance = None):

        if tolerance is None:
            tolerance=findToleranceFromVisit(pfsVisitID)

        vmax = 4*tolerance

        ax = plt.gcf().get_axes()[0]
        self.visSubaruConvergence(Axes=ax, pfsVisitID = pfsVisitID,subVisit=maxIteration-1,excludeUnassign=excludeUnassign,
                        tolerance=tolerance,vmax=vmax)
        ax = plt.gcf().get_axes()[1]
        self.visSubaruConvergence(Axes=ax,pfsVisitID = pfsVisitID,subVisit=3,tolerance=tolerance, excludeUnassign=excludeUnassign,
                        histo=True, bins=20,range=(0,vmax))



    def visSubaruConvergence(self, Axes = None, pfsVisitID=None, subVisit=11,
        histo=False, histoThres = True, heatmap=True, vectormap=False, excludeUnassign = True,
        plot_range=(0,0.08), bins=20, tolerance=0.01, **kwargs):
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



        import psycopg2
        from sqlalchemy import create_engine

        if excludeUnassign is True:
            # Getting unassined fiber
            #pfsDesignID = int(findDesignFromVisit(pfsVisitID))
            conn = psycopg2.connect("dbname='opdb' host='db-ics' port=5432 user='pfs'")
            engine = create_engine('postgresql+psycopg2://', creator=lambda: conn)


            with conn:
                fiberData = pd.read_sql('''
                    SELECT DISTINCT 
                        fiber_id, pfi_center_final_x_mm, pfi_center_final_y_mm, 
                        pfi_nominal_x_mm, pfi_nominal_y_mm
                    FROM 
                        pfs_config_fiber
                    WHERE
                        pfs_config_fiber.visit0 = %(visit0)s
                    -- limit 10
                ''', engine, params={'visit0': pfsVisitID})


            fid=FiberIds()
            fiberData['cobra_id']=fid.fiberIdToCobraId(fiberData['fiber_id'].values)
            fiberData=fiberData.sort_values('cobra_id')
            df = fiberData.loc[fiberData['cobra_id'] != 65535]
            unassigned_rows = df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].isna().all(axis=1)]
            unassigned_cobraIdx =  unassigned_rows['cobra_id'].values - 1


            assigned_row= df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].notna().all(axis=1)]
            assigned_cobraIdx =  assigned_row['cobra_id'].values - 1

            targetFromDB = df['pfi_nominal_x_mm'].values+df['pfi_nominal_y_mm'].values*1j
            isNan = np.isnan(targetFromDB)
            targetFromDB[isNan] = self.calibModel.centers[isNan]

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

        if excludeUnassign is True:
            targets = targetFromDB
            dist=np.sqrt((match['pfi_center_x_mm'].values[assigned_cobraIdx]-targets[assigned_cobraIdx].real)**2+
                            (match['pfi_center_y_mm'].values[assigned_cobraIdx]-targets[assigned_cobraIdx].imag)**2)
        else:
            targets=np.load(tarfile)
            dist=np.sqrt((match['pfi_center_x_mm'].values[self.goodIdx]-targets.real)**2+
                (match['pfi_center_y_mm'].values[self.goodIdx]-targets.imag)**2)

        mov = np.load(movfile)

        try:
            maxIteration = mov.shape[2]
        except:
            mov = np.array([mov])
            maxIteration = mov.shape[2]

        #dist=np.sqrt((match['pfi_center_x_mm'].values[self.goodIdx]-targets[self.goodIdx].real)**2+
        #    (match['pfi_center_y_mm'].values[self.goodIdx]-targets[self.goodIdx].imag)**2)


        self.logger.info(f'Tolerance: {tolerance}')

        #targets=np.load(tarfile)
        tar = targetFromDB[self.goodIdx]
        ind = np.where(np.abs(mov[0,:,maxIteration-1]['position']) > 0)
        notDone = len(np.where(np.abs(mov[0,ind[0],maxIteration-1]['position']-tar[ind[0]]) > tolerance)[0])

        #notDone = fpga_notDone[-1]
        print(f'Number of not done cobra (FPGA): {notDone}')

        if doPlots:
            sc=ax.scatter(self.calibModel.centers.real,self.calibModel.centers.imag,
                    c='grey',marker='s')
            ax.scatter(self.calibModel.centers.real[self.goodIdx[ind]],self.calibModel.centers.imag[self.goodIdx[ind]],
                c='red',marker='s',label='Not Converged')

        return ind[0]

    def visCobraMovement(self, pfsVisitID, cobraIdx=0, iteration = 8, newPlot=True):
        """
        Visualize the movement of a specific cobra during a PFS visit.
        Args:
            pfsVisitID (int): The PFS visit ID.
            cobraIdx (int, optional): The index of the cobra. Defaults to 0.
            iteration (int, optional): The iteration number. Defaults to 8.
        """

        #visit = 107682
        visit = pfsVisitID

        runDir = findRunDir(pfsVisitID)
        iteration=8

        mov=np.load(f'/data/MCS/{runDir}/data/moves.npy')
        tar=np.load(f'/data/MCS/{runDir}/data/targets.npy')

        movIdx = np.where(self.goodIdx == cobraIdx)[0][0]

        if newPlot is True:
            self.visCreateNewPlot(f'Visit = {visit} Cobra Index = {self.goodIdx[movIdx]}','X','Y')

        self.visGeometryFromXML(thetaAngle=mov[0,movIdx,iteration-1]['thetaAngle']+self.calibModel.tht0[self.goodIdx[movIdx]],
                            phiAngle=mov[0,movIdx,iteration-1]['phiAngle'],patrol=True)


        #self.visUnassignedFibers(pfsVisitID)

        ax=plt.gca()
        x = mov[0,movIdx,:]['position'].real
        y = mov[0,movIdx,:]['position'].imag

        for i in range(len(x)):
            if x[i]==0 and y[i]==0:
                x[i] = x[i-1]
                y[i] = y[i-1]
            ax.scatter(x[i], y[i], marker='.', alpha=0.5,label=f'{i+1}')
            ax.text(x[i], y[i],f'{i}')
        ax.scatter(tar.real[movIdx],tar.imag[movIdx],c='red',marker='x',label='target',s=80)
        #ax.scatter(self.calibModel.centers.real[idx], vis.calibModel.centers.imag[idx])

        #ax.legend()
        self.visSetCobra(self.goodIdx[movIdx], scale=1.0)



    def visTargetConvergence(self, pfsVisitID, maxIteration = 11, excludeUnassign=True, tolerance = None):

        if tolerance is None:
            tolerance=findToleranceFromVisit(pfsVisitID)

        vmax = 4*tolerance

        ax = plt.gcf().get_axes()[0]
        self.visSubaruConvergence(Axes=ax, pfsVisitID = pfsVisitID,subVisit=maxIteration-1,excludeUnassign=excludeUnassign,
                        tolerance=tolerance,vmax=vmax)
        ax = plt.gcf().get_axes()[1]
        self.visSubaruConvergence(Axes=ax,pfsVisitID = pfsVisitID,subVisit=3,tolerance=tolerance, excludeUnassign=excludeUnassign,
                        histo=True, bins=20,range=(0,vmax))



    def visSubaruConvergence(self, Axes = None, pfsVisitID=None, subVisit=11,
        histo=False, histoThres = True, heatmap=True, vectormap=False, excludeUnassign = True,
        plot_range=(0,0.08), bins=20, tolerance=0.01, **kwargs):
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



        import psycopg2
        from sqlalchemy import create_engine

        if excludeUnassign is True:
            # Getting unassined fiber
            #pfsDesignID = int(findDesignFromVisit(pfsVisitID))
            conn = psycopg2.connect("dbname='opdb' host='db-ics' port=5432 user='pfs'")
            engine = create_engine('postgresql+psycopg2://', creator=lambda: conn)


            with conn:
                fiberData = pd.read_sql('''
                    SELECT DISTINCT 
                        fiber_id, pfi_center_final_x_mm, pfi_center_final_y_mm, 
                        pfi_nominal_x_mm, pfi_nominal_y_mm
                    FROM 
                        pfs_config_fiber
                    WHERE
                        pfs_config_fiber.visit0 = %(visit0)s
                    -- limit 10
                ''', engine, params={'visit0': pfsVisitID})


            fid=FiberIds()
            fiberData['cobra_id']=fid.fiberIdToCobraId(fiberData['fiber_id'].values)
            fiberData=fiberData.sort_values('cobra_id')
            df = fiberData.loc[fiberData['cobra_id'] != 65535]
            unassigned_rows = df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].isna().all(axis=1)]
            unassigned_cobraIdx =  unassigned_rows['cobra_id'].values - 1


            assigned_row= df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].notna().all(axis=1)]
            assigned_cobraIdx =  assigned_row['cobra_id'].values - 1

            targetFromDB = df['pfi_nominal_x_mm'].values+df['pfi_nominal_y_mm'].values*1j
            isNan = np.isnan(targetFromDB)
            targetFromDB[isNan] = self.calibModel.centers[isNan]

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

        if excludeUnassign is True:
            targets = targetFromDB
            dist=np.sqrt((match['pfi_center_x_mm'].values[assigned_cobraIdx]-targets[assigned_cobraIdx].real)**2+
                            (match['pfi_center_y_mm'].values[assigned_cobraIdx]-targets[assigned_cobraIdx].imag)**2)
        else:
            targets=np.load(tarfile)
            dist=np.sqrt((match['pfi_center_x_mm'].values[self.goodIdx]-targets.real)**2+
                (match['pfi_center_y_mm'].values[self.goodIdx]-targets.imag)**2)

        mov = np.load(movfile)

        try:
            maxIteration = mov.shape[2]
        except:
            mov = np.array([mov])
            maxIteration = mov.shape[2]

        #dist=np.sqrt((match['pfi_center_x_mm'].values[self.goodIdx]-targets[self.goodIdx].real)**2+
        #    (match['pfi_center_y_mm'].values[self.goodIdx]-targets[self.goodIdx].imag)**2)


        self.logger.info(f'Tolerance: {tolerance}')

        #targets=np.load(tarfile)
        tar = targetFromDB[self.goodIdx]
        ind = np.where(np.abs(mov[0,:,maxIteration-1]['position']) > 0)
        notDone = len(np.where(np.abs(mov[0,ind[0],maxIteration-1]['position']-tar[ind[0]]) > tolerance)[0])

        #notDone = fpga_notDone[-1]
        print(f'Number of not done cobra (FPGA): {notDone}')

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

                if excludeUnassign is True:
                    dist=np.sqrt((match['pfi_center_x_mm'].values[assigned_cobraIdx]-targets[assigned_cobraIdx].real)**2+
                            (match['pfi_center_y_mm'].values[assigned_cobraIdx]-targets[assigned_cobraIdx].imag)**2)
                else:
                    dist=np.sqrt((match['pfi_center_x_mm'].values[self.goodIdx]-targets.real)**2+
                            (match['pfi_center_y_mm'].values[self.goodIdx]-targets.imag)**2)

                # This setting will set all value larger than the threshold to be exact the
                #   thresold.  So that there will be a big bar in histogram.
                if histoThres is True:
                    dist[dist > plot_range[1]]=plot_range[1]

                n, bins, patches = ax.hist(dist,range=plot_range, bins=bins, alpha=0.7,
                    histtype='step',linewidth=3,
                    label=f'{subID+1}-th Iteration')


            x_vertical = tolerance  # x-coordinate for the vertical line
            #ax.axvline(x=x_vertical, color='r', linestyle='--')
            #ax.axhline(np.percentile(dist, 50), color='green', linewidth=2,label='50 percentile')
            #ax.axhline(np.percentile(dist, 75), color='blue', linewidth=2,label='75 percentile')
            #ax.axhline(np.percentile(dist, 95), color='brown', linewidth=2,label='95 percentile')

            #print(bins)
            outIdx = np.where(bins > tolerance)
            #print(len(outIdx), outIdx)
            if np.size(outIdx) != 0:
            #print(np.sum(n[outIdx[0][0]:]))
            #print(np.sum(n[np.where(bins > 0.01)[0]]))
                outRegion =  np.sum(n[outIdx[0][0]-1:])
            else:
                outRegion = 0
            #ax.text(0.7, 0.45, f'N. of non-converged (MCS) = {outRegion}',
            #            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            #ax.text(0.7, 0.4, f'N. of non-converged (FPS) = {notDone}',
            #            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
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
            if excludeUnassign is True:
                sc=ax.scatter(self.calibModel.centers.real[assigned_cobraIdx],self.calibModel.centers.imag[assigned_cobraIdx],
                    c=dist,marker='s', **kwargs)
            else:
                sc=ax.scatter(self.calibModel.centers.real[self.goodIdx],self.calibModel.centers.imag[self.goodIdx],
                    c=dist,marker='s', **kwargs)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            colorbar = self.fig.colorbar(sc,cax=cax)
            if excludeUnassign is True:
                #print('here')
                ax.scatter(self.calibModel.centers.real[unassigned_cobraIdx],self.calibModel.centers.imag[unassigned_cobraIdx],
                c='red',marker='s',label='UNASSIGNED', **kwargs)

            # Plot the location of broken fibers
            ax.scatter(self.calibModel.centers.real[self.badIdx],self.calibModel.centers.imag[self.badIdx],
                c='lightpink',marker='s',label='BROKEN', **kwargs)

            # Add a red line on the colorbar
            line_color = 'red'
            line_position = tolerance  # Position of the line on the colorbar (between 0 and 1)
            cax.axhline(line_position, color=line_color, linewidth=2,label='tolerance')


            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.legend()
            #plt.colorbar(sc)

        #ind = np.where(np.abs(mov[0,:,maxIteration-1]['position']) > 0)
        #notDone = len(np.where(np.abs(mov[0,ind[0],maxIteration-1]['position']-targets[ind[0]]) > tolerance)[0])
        self.logger.info(f'75% perceitile: {np.percentile(dist, 75)}')
        self.logger.info(f'95% perceitile: {np.percentile(dist, 95)}')
        self.logger.info(f'Number of still moving cobra: {(len(ind[0]))}')
        self.logger.info(f'Number of not done cobra (FPS): {notDone}')

    def visCobraCenter(self, baseData, targetData, histo=False, gauFit = True, vectorLength=0.05, **kwargs):

        '''
            This function is used to compare the center locations between two datasets (target - base).  
            
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


            ax1.text(0.8, 0.8, rf'Median = {np.median(diff):.4f}, $\sigma$={sigma:.4f}',
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

            ax2.text(0.7, 0.85, rf'Median = {np.median(dx):.4f}, $\sigma$={sigma:.4f}',
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

            ax3.text(0.7, 0.85, rf'Median = {np.median(dy):.4f}, $\sigma$={sigma:.4f}',
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
            This function is used to plot the averaged vector map of all fiducial fiber related to
            PRE-MEASURED (on XY stage) position.  

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
        #fids = butler.get('fiducials')
        fids = fiducials.Fiducials.read(butler)

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

            fidsGood = fids[fids.goodMask]
            fidsOuterRing = fids[fids.goodMask & fids.outerRingMask]

            pt.updateTransform(mcsData, fidsOuterRing, matchRadius=8.0, nMatchMin=0.1)

            for i in range(2):
                rfid, rdmin = pt.updateTransform(mcsData, fidsGood, matchRadius=4.2,nMatchMin=0.1)
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
                ax1.text(0.8, 0.8, rf'Mean = {xPeak:.4f}, $\sigma$={sigma:.2f}',
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
                ax2.text(0.5, 0.9, rf'Mean = {xPeak:.4f}, $\sigma$={sigma:.2f}',
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
                ax3.text(0.5, 0.9, rf'Mean = {xPeak:.4f}, $\sigma$={sigma:.2f}',
                    horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

                ax3.set_title('Y direction')
                ax3.set_xlabel('distance (mm)')
                plt.subplots_adjust(wspace=0,hspace=0.3)

        if getAllFFPos is True:
            self.logger.info('Returing all fiducial fiber positions.')
            return ffpos_array

    def visIterationFFOffset(self, posData, Axes=None, iteration = 0, offsetThres = 0.006, offsetBox = 0.2,
        heatMap = False, badFF = None, binNum=40):

        '''
            This function plots the relative offsets for all FF of A SINGLE ITERATION
            against to the averaged position. It is very useful for identifing the unstable FF.

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
        ax_histx.text(0.5, 0.9, rf'Mean = {xPeak:.4f}, $\sigma$={sigma:.4f}',
                    horizontalalignment='center', verticalalignment='center', transform=ax_histx.transAxes)

        n, bins, patches = ax_histy.hist(ffOffset[stableFF].flatten().imag,bins=binNum,range=(-offsetBox,offsetBox),orientation='horizontal')
        popt = gaussianFit(n, bins)
        sigma = np.abs(popt[2])
        xPeak = popt[1]
        ax_histy.plot(gaus(bins[:-1],*popt),bins[:-1],'r:',label='fit')
        ax_histy.text(0.5, 0.9, rf'Mean = {xPeak:.4f}, $\sigma$={sigma:.4f}',
                    horizontalalignment='center', verticalalignment='center', transform=ax_histy.transAxes)

        ax.set_xlim(-offsetBox,offsetBox)
        ax.set_ylim(-offsetBox,offsetBox)

    def visAllFFOffsetHisto(self, posData, Iteration = None, Axes = None, binNum = 80, offsetBox = 0.05, dataOnly=False):

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
        ax.text(0.5, 0.9, rf'Mean = {xPeak:.4f}, $\sigma$={sigma:.4f}',
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.text(0.5, 0.8, rf'Median = {np.nanmedian(ffOffset.flatten()):.4f}, $\sigma$={sigma:.4f}',
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        if dataOnly:
            data = ffOffset.flatten()
            clean_data = np.array(data)[~np.isnan(data)]
            return xPeak, np.nanmedian(clean_data), np.percentile(clean_data, 75)


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
        ax_histx.text(0.5, 0.9, rf'Mean = {xPeak:.4f}, $\sigma$={sigma:.4f}',
                    horizontalalignment='center', verticalalignment='center', transform=ax_histx.transAxes)

        n, bins, patches = ax_histy.hist(ffOffset[:,stableFF].flatten().imag,bins=binNum,range=(-offsetBox,offsetBox),orientation='horizontal')
        popt = gaussianFit(n, bins)
        sigma = np.abs(popt[2])
        xPeak = popt[1]
        ax_histy.plot(gaus(bins[:-1],*popt),bins[:-1],'r:',label='fit')
        ax_histy.text(0.5, 0.9, rf'Mean = {xPeak:.4f}, $\sigma$={sigma:.4f}',
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

            popt = gaussianFit(n, bins)
            sigma = popt[2]
            xPeak = popt[1]

            ax1.plot(bins[:-1],gaus(bins[:-1],*popt),'r:',label='fit')

            #ax1.text(0.8, 0.8, f'Median = {np.nanmedian(diff[stableFF]):.2f}, $\sigma$={sigma:.2f}',
            ax1.text(0.8, 0.8, rf'Mean = {xPeak:.4f}, $\sigma$={sigma:.2f}',
                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
            ax1.set_title('2D')
            ax1.set_xlabel('distance (mm)')
            ax1.set_ylabel('Counts')
            ax1.set_ylim(0,1.5*np.max(n))


            ax2 = plt.subplot(221)
            ax2.hist(dx,range=(np.nanmean(dx[stableFF])-3*np.nanstd(dx[stableFF]),np.nanmean(dx[stableFF])+3*np.nanstd(dx[stableFF])),
                bins=10, color='#0504aa',alpha=0.7)
            ax2.text(0.7, 0.8, rf'Mean = {xPeak:.4f}, $\sigma$={sigma:.2f}',
                horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title('X direction')
            ax2.set_xlabel('distance (mm)')
            ax2.set_ylabel('Counts')
            ax2.set_ylim(0,2.0*np.max(n))


            ax3 = plt.subplot(222, sharey = ax2)
            ax3.tick_params(axis='both',labelleft=False)


            ax3.hist(dy,range=(np.nanmean(dy[stableFF])-3*np.nanstd(dy[stableFF]),np.nanmean(dy[stableFF])+3*np.nanstd(dy[stableFF])),
                bins=10, color='#0504aa', alpha=0.7)

            popt = gaussianFit(n, bins)
            sigma = np.abs(popt[2])
            xPeak = popt[1]
            ax3.plot(bins[:-1],gaus(bins[:-1],*popt),'ro:',label='fit')

            #ax3.text(0.7, 0.8, f'Median = {np.nanmedian(dy):.2f}, $\sigma$={sigma:.2f}',
            ax3.text(0.5, 0.9, rf'Mean = {xPeak:.4f}, $\sigma$={sigma:.2f}',
                horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

            ax3.set_title('Y direction')
            ax3.set_xlabel('distance (mm)')
            plt.subplots_adjust(wspace=0,hspace=0.3)

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
        plt.title("Maximum SNR", fontsize = 20)
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
        plt.title("Iteration of maximum SNR", fontsize = 20)
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
            {figPath}motormap*{arm}*_[0-9]??.png {figPath}motormap*{arm}*_[0-9]???.png {pdffile}"""
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
        ax1.set_title('Fiber FWD Variation')
        ax2.set_title('Fiber REV Variation')
        ax2.set_xlabel("Motor Map Run")
        ax1.set_ylabel("Variation")
        ax2.set_ylabel("Variation")
        plt.savefig(f'{figPath}/allvariant{arm}.png')
        plt.close()

        if pdffile is not None:
            cmd=f"""convert {figPath}motormap*_[0-9].png {figPath}motormap*_[0-9]?.png \
            {figPath}motormap*_[0-9]??.png {figPath}motormap*_[0-9]???.png {pdffile}"""
            retcode = subprocess.call(cmd,shell=True)
            print(cmd)
        return var_fwd,var_rev
