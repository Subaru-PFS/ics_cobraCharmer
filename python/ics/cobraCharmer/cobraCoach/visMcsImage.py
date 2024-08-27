%matplotlib inline
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor
import re
import pandas as pd
from sqlalchemy import create_engine
import pfs.utils.coordinates.transform as transformUtils
import mcsActor.mcsRoutines.fiducials as fiducials
from pfs.utils import butler
import matplotlib.pyplot as plt
import numpy as np
import os
from ics.cobraCharmer.cobraCoach import visDianosticPlot
from pfs.utils.fiberids import FiberIds


class FitsImageProcessor:
    def __init__(self, pfsVisit, cameraName, logger):
        self.pfsVisit = pfsVisit
        self.directory = f'/data/MCS/{visDianosticPlot.findRunDir(pfsVisit)}/data'
        self.cameraName = cameraName
        self.logger = logger
        self.butler = butler.Butler()
        self.fids = fiducials.Fiducials.read(self.butler)

    def loadXmlfile(self):
        newXml = pathlib.Path('/software/mhs/products/Linux64/pfs_instdata/1.7.57/data/pfi/modules/ALL/ALL.xml')
        vis=visDianosticPlot.VisDianosticPlot(xml=newXml)
        return vis.calibModel.centers.real, vis.calibModel.centers.imag
    
    def fetchTarget(self):
        conn = psycopg2.connect("dbname='opdb' host='db-ics' port=5432 user='pfs'") 
        engine = create_engine('postgresql+psycopg2://', creator=lambda: conn)

        
        with conn:
            fiberData = pd.read_sql(f'''
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
        targetFromDB = df['pfi_nominal_x_mm'].values+df['pfi_nominal_y_mm'].values*1j
        
        return targetFromDB.real, targetFromDB.imag
        
    def findFitsFiles(self):
        fitsFiles = [os.path.join(self.directory, file) for file in os.listdir(self.directory) if file.endswith('.fits')]
        fitsFiles.sort()
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

    def updateTransformation(self, mcsData):
        #self.logger.info(f'Initiating the transformation function')
        if 'rmod' in self.cameraName.lower():
            altitude = 90.0
            insrot = 0
            pfiTransform = transformUtils.fromCameraName('usmcs', 
                altitude=altitude, insrot=insrot, nsigma=0, alphaRot=0)
        else:
            altitude = 90.0  # Default value; adjust if needed
            insrot = 0  # Default value; adjust if needed
            pfiTransform = transformUtils.fromCameraName(self.cameraName, 
                altitude=altitude, insrot=insrot, nsigma=0, alphaRot=1)

        #self.logger.info(f'Camera name: {self.cameraName}')
        #self.logger.info(f'Calculating transformation using FF at outer region')

        self.fidsGood = self.fids[self.fids.goodMask]
        self.fidsOuterRing = self.fids[self.fids.goodMask & self.fids.outerRingMask]
        
        pfiTransform.updateTransform(mcsData, self.fidsOuterRing, matchRadius=8.0, nMatchMin=0.1)

        return pfiTransform

    def processFitsFiles(self, fitsFiles, x, y, boxSize=50):
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.extractImageBox, fitsFile, x, y, boxSize): fitsFile for fitsFile in fitsFiles}
            for future in futures:
                fitsFile = futures[future]
                results[fitsFile] = future.result()

            futures = {executor.submit(self.fetchMcsData, self.extractFrameNum(fitsFile)): fitsFile for fitsFile in fitsFiles}
            for future in futures:
                fitsFile = futures[future]
                mcsData = future.result()
                pfiTransform = self.updateTransformation(mcsData)
                results[fitsFile] += (mcsData, pfiTransform)

        return results

    def plotImageBoxesWithData(self, x, y, boxSize=150):
        fitsFiles = self.findFitsFiles()
        results = self.processFitsFiles(fitsFiles, x, y, boxSize)
        centX, centY = self.loadXmlfile()
        targetX, targetY = self.fetchTarget()
        
        numFiles = len(fitsFiles)
        ncols = 3
        nrows = (numFiles + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes = axes.flatten()
        
        for i, fitsFile in enumerate(fitsFiles):
            box, x_adjusted, y_adjusted, mcsData, pfiTransform = results[fitsFile]
            
            c_xx, c_yy = pfiTransform.pfiToMcs(centX, centY)
            t_xx, t_yy = pfiTransform.pfiToMcs(targetX, targetY)
            m, s = np.mean(box), np.std(box)
            axes[i].imshow(box, vmin=m-2.0*s, vmax=m+2.0*s, cmap='gray', origin='lower')
            axes[i].set_title(os.path.basename(fitsFile))

            mcs_x = mcsData['mcs_center_x_pix'].to_numpy() - (x - boxSize // 2)
            mcs_y = mcsData['mcs_center_y_pix'].to_numpy() - (y - boxSize // 2)
            center_x = c_xx - (x - boxSize // 2)
            center_y = c_yy - (y - boxSize // 2)
            tar_x = t_xx - (x - boxSize // 2)
            tar_y = t_yy - (y - boxSize // 2)
            
            valid_points = (mcs_x >= 0) & (mcs_x < boxSize) & (mcs_y >= 0) & (mcs_y < boxSize)
            axes[i].scatter(mcs_x[valid_points], mcs_y[valid_points], c='red', s=30, label='MCS Data')
            
            valid_points = (center_x >= 0) & (center_x < boxSize) & (center_y >= 0) & (center_y < boxSize)
            axes[i].scatter(center_x[valid_points], center_y[valid_points], c='blue', s=30, label='Cobra Center')
            
            valid_points = (tar_x >= 0) & (tar_x < boxSize) & (tar_y >= 0) & (tar_y < boxSize)
            axes[i].scatter(tar_x[valid_points], tar_y[valid_points], marker='x', c='green', label='Target')
            
            axes[i].axis('off')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

# Example usage
# directory = 'path/to/fits/files'
pfsVisit = 113029
runDir = f'/data/MCS/{visDianosticPlot.findRunDir(pfsVisit)}/data'
logger = None  # Replace with an actual logger instance if available
cameraName = 'rmod'  # Or 'canon', depending on the camera being used
processor = FitsImageProcessor(pfsVisit, cameraName, logger)
x, y = 4682, 3754  # Example coordinates
processor.plotImageBoxesWithData(x, y)
