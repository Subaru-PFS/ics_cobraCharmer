import fnmatch
import logging
import os
import pathlib
import re
import warnings
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pfs.utils.coordinates.transform as transformUtils
import psycopg2
from astropy.io import fits
from ics.cobraCharmer.cobraCoach import visDianosticPlot
from mcsActor.mcsRoutines import fiducials
from pfs.utils import butler
from pfs.utils.fiberids import FiberIds
from sqlalchemy import create_engine


class FitsImageProcessor:
    def __init__(self, pfsVisit, cameraName):

        self.pfsVisit = pfsVisit
        self.directory = f'/data/MCS/{visDianosticPlot.findRunDir(pfsVisit)}/data'
        self.cameraName = cameraName


        logging.basicConfig(format="%(asctime)s.%(msecs)03d %(levelno)s %(name)-10s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")
        self.logger = logging.getLogger('FitsImageProcessor')
        self.logger.setLevel(logging.INFO)

        self.butler = butler.Butler()
        self.fids = fiducials.Fiducials.read(self.butler)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            centX, centY = self.loadXmlfile()
            self.centX = centX
            self.centY = centY

    def loadXmlfile(self):
        newXml = pathlib.Path('/software/mhs/products/Linux64/pfs_instdata/1.8.9/data/pfi/modules/ALL/ALL.xml')
        vis=visDianosticPlot.VisDianosticPlot(xml=newXml)
        self.vis = vis
        return vis.calibModel.centers.real, vis.calibModel.centers.imag

    def fetchTarget(self):
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
            ''', engine, params={'visit0': self.pfsVisit})

        fid=FiberIds()
        fiberData['cobra_id']=fid.fiberIdToCobraId(fiberData['fiber_id'].values)
        fiberData=fiberData.sort_values('cobra_id')
        df = fiberData.loc[fiberData['cobra_id'] != 65535]
        unassigned_rows = df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].isna().all(axis=1)]
        unassigned_cobraIdx =  unassigned_rows['cobra_id'].values - 1

        self.unassigned_cobraIdx = unassigned_cobraIdx
        targetFromDB = df['pfi_nominal_x_mm'].values+df['pfi_nominal_y_mm'].values*1j

        return targetFromDB.real, targetFromDB.imag

    def findFitsFiles(self):
        pattern = f'PFSC{self.pfsVisit:06d}??.fits'
        fitsFiles = [os.path.join(self.directory, file) for file in os.listdir(self.directory) if fnmatch.fnmatch(file, pattern)]
        fitsFiles.sort()
        self.fitsFiles = fitsFiles
        return fitsFiles

    def extractFrameNum(self, fitsFile):
        filename = os.path.basename(fitsFile)
        match = re.search(r'PFSC(\d{8})\.fits', filename)
        return int(match.group(1)) if match else None

    def extractImageBox(self, fitsFile, x, y, boxSize=50):
        with fits.open(fitsFile, memmap=True) as hdul:
            imageData = hdul[1].data
            x = min(max(x, boxSize // 2), imageData.shape[1] - boxSize // 2)
            y = min(max(y, boxSize // 2), imageData.shape[0] - boxSize // 2)
            box = imageData[y - boxSize // 2:y + boxSize // 2, x - boxSize // 2:x + boxSize // 2]
        return box, x, y

    def fetchMcsData(self, frameNum):
        connStr = "postgresql://pfs@db-ics/opdb"
        engine = create_engine(connStr)
        with engine.connect() as conn:
            mcsData = pd.read_sql(f'''
                SELECT DISTINCT 
                    spot_id, mcs_center_x_pix, mcs_center_y_pix
                FROM mcs_data
                WHERE
                  mcs_frame_id = {frameNum}
            ''', conn)
        return mcsData

    def fetchTelescopeInfo(self, frameNum):
        connStr = "postgresql://pfs@db-ics/opdb"
        engine = create_engine(connStr)
        with engine.connect() as conn:
            telescopeInfo = pd.read_sql(f'''
                SELECT DISTINCT 
                    insrot, altitude
                FROM mcs_exposure
                WHERE
                  mcs_frame_id = {frameNum}
            ''', conn)

            return telescopeInfo

    def updateTransformation(self, telescopeInfo, mcsData):
        #self.logger.info(f'Initiating the transformation function')
        if 'rmod' in self.cameraName.lower():
            altitude = 90.0
            insrot = 0
            pfiTransform = transformUtils.fromCameraName('usmcs',
                altitude=altitude, insrot=insrot, nsigma=0, alphaRot=0)
        else:
            altitude = telescopeInfo['altitude'].values[0]  # Default value; adjust if needed
            insrot = telescopeInfo['insrot'].values[0]  # Default value; adjust if needed
            pfiTransform = transformUtils.fromCameraName(self.cameraName,
                altitude=altitude, insrot=insrot, nsigma=0, alphaRot=1)

        #self.logger.info(f'Camera name: {self.cameraName}')
        #self.logger.info(f'Calculating transformation using FF at outer region')

        self.fidsGood = self.fids[self.fids.goodMask]
        self.fidsOuterRing = self.fids[self.fids.goodMask & self.fids.outerRingMask]
        num_matches = []
        ffid, dist = pfiTransform.updateTransform(mcsData, self.fidsOuterRing, matchRadius=8.0, nMatchMin=0.1)
        num_matches.append((ffid != -1).sum())

        nsigma = 0
        pfiTransform.nsigma = nsigma
        pfiTransform.alphaRot = 0

        #self.logger.info(f'Re-calcuating transofmtaion using ALL FFs.')
        for i in range(2):
            #ifig = 1
            #fig = plt.figure(ifig); plt.clf()
            ffid, dist = pfiTransform.updateTransform(mcsData, self.fidsGood, matchRadius=4.2,nMatchMin=0.1)
            #num_matches = (ffid != -1).sum()
            num_matches.append((ffid != -1).sum())

            # Log or print the number of matches

        self.logger.info(f'Number of matched elements: {num_matches}')
        return pfiTransform



    def processFitsFiles(self, fitsFiles, x=None, y=None, cobraIdx=None, boxSize=50):
        results = {}

        with ThreadPoolExecutor() as executor:
            futures = {}
            for fitsFile in fitsFiles:
                future = executor.submit(self._processSingleFile, fitsFile, x, y, cobraIdx, boxSize)
                futures[future] = fitsFile

            for future in futures:
                fitsFile = futures[future]
                results[fitsFile] = future.result()

        return results

    def _processSingleFile(self, fitsFile, x, y, cobraIdx, boxSize):
        frameNum = self.extractFrameNum(fitsFile)
        mcsData = self.fetchMcsData(frameNum)
        telescopeInfo = self.fetchTelescopeInfo(frameNum)
        pfiTransform = self.updateTransformation(telescopeInfo, mcsData)
        self.logger.info(f'{fitsFile} {frameNum}')

        if cobraIdx is not None:
            x, y = pfiTransform.pfiToMcs([self.centX[cobraIdx]], [self.centY[cobraIdx]])
            x = int(x)
            y = int(y)

        box, x_adjusted, y_adjusted = self.extractImageBox(fitsFile, x, y, boxSize)
        return (box, x_adjusted, y_adjusted, mcsData, pfiTransform)


    def plotImageBoxesWithData(self, pixelCoord=None, cobraIdx=None, boxSize=150):
        fitsFiles = self.findFitsFiles()

        if cobraIdx is not None:
            # Convert cobraIdx to pixel coordinates (x, y) for each frame
            results = self.processFitsFiles(fitsFiles, x=None, y=None, boxSize=boxSize, cobraIdx=cobraIdx)
        elif pixelCoord is not None:
            x, y = pixelCoord
            results = self.processFitsFiles(fitsFiles, x, y, boxSize=boxSize)
        else:
            raise ValueError("Either pixelCoord or cobraIdx must be provided.")

        targetX, targetY = self.fetchTarget()
        self.target = targetX+targetY*1j

        numFiles = len(fitsFiles)
        ncols = 3
        nrows = (numFiles + ncols - 1) // ncols+3 # Add 1 row for the additional plot

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes = axes.flatten()

        for i, fitsFile in enumerate(fitsFiles):
            box, x_adjusted, y_adjusted, mcsData, pfiTransform = results[fitsFile]

            c_xx, c_yy = pfiTransform.pfiToMcs(self.centX, self.centY)
            t_xx, t_yy = pfiTransform.pfiToMcs(targetX, targetY)
            f_xx, f_yy = pfiTransform.pfiToMcs(self.fids['x_mm'].values, self.fids['y_mm'].values)

            m, s = np.mean(box), np.std(box)
            axes[i].imshow(box, vmin=m-2.0*s, vmax=m+2.0*s, cmap='gray', origin='lower')
            axes[i].set_title(os.path.basename(fitsFile))

            mcs_x = mcsData['mcs_center_x_pix'].to_numpy() - (x_adjusted - boxSize // 2)
            mcs_y = mcsData['mcs_center_y_pix'].to_numpy() - (y_adjusted - boxSize // 2)

            center_ff_x = f_xx - (x_adjusted - boxSize // 2)
            center_ff_y = f_yy - (y_adjusted - boxSize // 2)

            center_x = c_xx - (x_adjusted - boxSize // 2)
            center_y = c_yy - (y_adjusted - boxSize // 2)
            tar_x = t_xx - (x_adjusted - boxSize // 2)
            tar_y = t_yy - (y_adjusted - boxSize // 2)

            valid_points = (center_ff_x >= 0) & (center_ff_x < boxSize) & (center_ff_y >= 0) & (center_ff_y < boxSize)
            axes[i].scatter(center_ff_x[valid_points], center_ff_y[valid_points], c='blue', s=30, label='FF')

            valid_points = (center_x >= 0) & (center_x < boxSize) & (center_y >= 0) & (center_y < boxSize)
            axes[i].scatter(center_x[valid_points], center_y[valid_points], marker='.', c='red', s=30, label='Cobra Center')

            valid_points = (tar_x >= 0) & (tar_x < boxSize) & (tar_y >= 0) & (tar_y < boxSize)
            axes[i].scatter(tar_x[valid_points], tar_y[valid_points], marker='x', c='green', s=80, label='Target')
            axes[i].axis('off')

        #for j in range(i + 1, len(axes)):
            #axes[j].axis('off')

        # Create the bottom plot
        ax_bottom = plt.subplot2grid((nrows, ncols), (ncols, 0), colspan=3,rowspan=3)

        self.vis.visCobraMovement(self.pfsVisit, cobraIdx=cobraIdx, newPlot=False)
        #ax_bottom.set_aspect('equal')  # Set equal aspect ratio to make height = width

        plt.tight_layout()
        plt.show()



def main():
    # Example usage
    # directory = 'path/to/fits/files'
    pfsVisit = 113029
    #pfsVisit = 113364
    pfsVisit = 113641
    #runDir = f'/data/MCS/{visDianosticPlot.findRunDir(pfsVisit)}/data'
    #cameraName = 'canon'  # Or 'canon', depending on the camera being used
    #processor = FitsImageProcessor(pfsVisit, cameraName)
    x, y = 4682, 3754  # Example coordinates
    #processor.plotImageBoxesWithData(pixelCoord = (x, y))
    #processor.plotImageBoxesWithData(cobraIdx = 544)

if __name__:
    main()
