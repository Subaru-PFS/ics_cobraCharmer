
import numpy as np
import pandas as pd
import psycopg2
from ics.cobraCharmer.cobraCoach import visDianosticPlot
from opdb import opdb
from sqlalchemy import create_engine


def getMaxIterationFromRunDir(runDir):
    try:
        path=f'/data/MCS/{runDir}/data/'
        movfile = path+'moves.npy'
        mov=np.load(movfile)
        iteration = mov.shape[-1]
    except:
        iteration = None

    return iteration


def getMeanTargetDist(runDir, iteration):


    pfsVisitID = visDianosticPlot.findVisit(runDir)
    pfsDesignID = int(np.int64(visDianosticPlot.findDesignFromVisit(98501)))
    conn = psycopg2.connect("dbname='opdb' host='db-ics' port=5432 user='pfs'")
    engine = create_engine('postgresql+psycopg2://', creator=lambda: conn)

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

    if len(fiberData) != 0:
        from pfs.utils.fiberids import FiberIds
        fid=FiberIds()
        fiberData['cobra_id']=fid.fiberIdToCobraId(fiberData['fiber_id'].values)
        fiberData=fiberData.sort_values('cobra_id')
        df = fiberData.loc[fiberData['cobra_id'] != 65535]
        unassigned_rows = df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].isna().all(axis=1)]
        unassigned_cobraIdx =  unassigned_rows['cobra_id'].values - 1

        assigned_row= df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].notna().all(axis=1)]
        assigned_cobraIdx =  assigned_row['cobra_id'].values - 1

        targetFromDB = df['pfi_nominal_x_mm'].values+df['pfi_nominal_y_mm'].values*1j

        targets = targetFromDB
    else:
        # conraId of the disabled Cobras
        disabled = np.array([46,   49,  172,  192,  343,  346,  360,  442,  492,  647,  737,
                753,  798,  820,  852,  948, 1149, 1207, 1209, 1302, 1459, 1493,
               1519, 1538, 1579, 1636, 1652, 1723, 1789, 1790, 1791, 1824, 1835,
               1881, 1902, 2052, 2351, 2379])
        assigned_cobraIdx= np.array([i for i in range(2394) if i not in disabled])

        tarfile=f'/data/MCS/{runDir}/data/targets.npy'
        targets=np.load(tarfile)

    moveFile = f'/data/MCS/{runDir}/data/moves.npy'
    mov = np.load(moveFile)
    try:
        maxMove = mov.shape[2]
    except:
        maxMove = mov.shape[-1]

    frameid = pfsVisitID*100+12

    db=opdb.OpDB(hostname='db-ics', port=5432,
           dbname='opdb',username='pfs')

    match = db.bulkSelect('cobra_match','select * from cobra_match where '
              f'pfs_visit_id = {pfsVisitID} and iteration = {iteration}').sort_values(by=['cobra_id']).reset_index()

    if len(fiberData) != 0:
        dist=np.sqrt((match['pfi_center_x_mm'].values[assigned_cobraIdx]-targets[assigned_cobraIdx].real)**2+
                                (match['pfi_center_y_mm'].values[assigned_cobraIdx]-targets[assigned_cobraIdx].imag)**2)
    else:
        dist=np.sqrt((match['pfi_center_x_mm'].values[assigned_cobraIdx]-targets.real)**2+
                                (match['pfi_center_y_mm'].values[assigned_cobraIdx]-targets.imag)**2)
    conn.close()
    db.close()
    return np.mean(dist), np.median(dist), np.percentile(dist, 75), np.percentile(dist, 95)
