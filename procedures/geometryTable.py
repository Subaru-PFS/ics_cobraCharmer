import sys, os
from importlib import reload
import numpy as np
import time
import datetime
from ics.cobraCharmer import pfi as pfiControl
from astropy.table import Table

def geometryStat(XMLarray, tablename, skipfiber=False):

    Xarray = []
    Yarray = []

    L1array = []
    L2array = []
    tht0array = []
    tht1array = []
    
    phiInarray = []
    phiOutarray = []


    for count, file in enumerate(XMLarray):
        print(file)
        pfi = pfiControl.PFI(fpgaHost='localhost', doConnect=False) #'fpga' for real device.
        pfi.loadModel(file)
        model = pfi.calibModel
        if count == 0:
            L1array = model.L1
            L2array = model.L2
            tht0array = np.rad2deg(model.tht0)
            tht1array = np.rad2deg(model.tht1)

            phiInarray = np.rad2deg(model.phiIn+np.pi)
            phiOutarray = np.rad2deg(model.phiOut+np.pi)

            Xarray = pfi.calibModel.centers.real
            Yarray = pfi.calibModel.centers.imag

        
        else:
            L1array = np.vstack((L1array,model.L1))
            L2array = np.vstack((L2array,model.L2))
            tht0array = np.vstack((tht0array,np.rad2deg(model.tht0)))
            tht1array = np.vstack((tht1array,np.rad2deg(model.tht1)))

            phiInarray = np.vstack((phiInarray, np.rad2deg(model.phiIn+np.pi)))
            phiOutarray = np.vstack((phiOutarray, np.rad2deg(model.phiOut+np.pi)))

            Xarray = np.vstack((Xarray, pfi.calibModel.centers.real))
            Yarray = np.vstack((Yarray, pfi.calibModel.centers.imag))

    L1mean= np.mean(L1array, axis=0)
    L1std=np.std(L1array, axis=0)

    L2mean= np.mean(L2array, axis=0)
    L2std=np.std(L2array, axis=0)

    tht0mean = np.mean(tht0array, axis=0)
    tht0std = np.std(tht0array, axis=0)

    tht1mean = np.mean(tht1array, axis=0)
    tht1std = np.std(tht1array, axis=0)

    phiInmean = np.mean(phiInarray, axis=0)
    PhiInstd = np.std(phiInarray, axis=0)

    phiOutmean = np.mean(phiOutarray, axis=0)
    PhiOutstd = np.std(phiOutarray, axis=0)

    Xmean = np.mean(Xarray, axis = 0)
    Xstd = np.std(Xarray, axis = 0)

    Ymean = np.mean(Yarray, axis = 0)
    Ystd = np.std(Yarray, axis = 0)

    print(tht0std)
    print(tht1std)

    pid = np.arange(1,58,1)
    
    t = Table([pid, Xmean, Xstd, Ymean, Ystd, L1mean, L1std, L2mean, L2std, 
               tht0mean, tht0std, tht1mean, tht1std, phiInmean, PhiInstd, phiOutmean, PhiOutstd],
    names=('Fiber No.', 'X', 'Xstd', 'Y', 'Ystd', 'L1mean', 'L1std', 'L2mean', 'L2std', 
           'ThetaCCWLimitMean','ThetaCCWLimitStd','ThetaCWLimitMean','ThetaCWLimitStd',
           'PhiCCWLimitMean','PhiCCWLimitStd','PhiCWLimitMean','PhiCWLimitStd'))
    
    if skipfiber is not False:
        t.remove_rows(skipfiber)


    t.write(tablename,format='ascii.ecsv',overwrite=True, formats={'Fiber No.': '%i', 'X': '%f', 'Xstd': '%f', 'Y':'%f', 'Ystd':'%f', 
            'L1mean': '%f', 'L1std': '%f', 'L2mean': '%f', 'L2std': '%f', 
           'ThetaCCWLimitMean': '%f','ThetaCCWLimitStd': '%f','ThetaCWLimitMean': '%f','ThetaCWLimitStd': '%f',
           'PhiCCWLimitMean': '%f','PhiCCWLimitStd': '%f','PhiCWLimitMean': '%f','PhiCWLimitStd': '%f'})

def makeTablefromXML(XML, filename):

    XMLfile = XML
    tablename = filename

    pfi = pfiControl.PFI(fpgaHost='localhost', doConnect=False) #'fpga' for real device.
    pfi.loadModel(XMLfile)

    model = pfi.calibModel
    pid = np.arange(1,58,1)
    
    # Formaing XY from complex number to string
    centers = []
    for num in pfi.calibModel.centers:
        cen = f'({num.real:.2f} {num.imag:.2f})'
        centers.append(cen)

    #centerX = pfi.calibModel.centers.real
    #centerY = pfi.calibModel.centers.imag


    t = Table([pid, centers, model.L1, np.rad2deg(model.tht0), np.rad2deg(model.tht1), model.motorOntimeFwd1, model.motorOntimeRev1,
                             model.L2, np.rad2deg(model.phiIn+np.pi), np.rad2deg(model.phiOut+np.pi), model.motorOntimeFwd2, model.motorOntimeRev2],
        names=('Fiber No.', 'Center', 'L1', 'ThetaCCWLimit', 'ThetaCWLimit', 'ThetaOntimeFwd', 'ThetaOntimeRev',
                                      'L2', 'PhiCCWLimit', 'PhiCWLimit', 'PhiOntimeFwd', 'PhiOntimeRev'))
    
 
    t.write(tablename,format='ascii.ecsv',overwrite=True,
        formats={'Fiber No.':'%i','Center': '%s', 'L1':'%f', 'ThetaCCWLimit':'%f', 'ThetaCWLimit':'%f', 'ThetaOntimeFwd':'%f', 'ThetaOntimeRev':'%f',
                                                  'L2':'%f', 'PhiCCWLimit':'%f', 'PhiCWLimit':'%f', 'PhiOntimeFwd':'%f', 'PhiOntimeRev':'%f'})
   

def main():
    file=['/Volumes/Disk/Data/xml/coarse.xml',
          '/Volumes/Disk/Data/xml/precise.xml',
          '/Volumes/Disk/Data/xml/precise2.xml',
          '/Volumes/Disk/Data/xml/precise3.xml',
          '/Volumes/Disk/Data/xml/precise5.xml',
          '/Volumes/Disk/Data/xml/precise6.xml',
          '/Volumes/Disk/Data/xml/precise_20190212.xml']

    file=[
        '/Volumes/Disk/Data/xml/precise_20190312_1.xml',
        '/Volumes/Disk/Data/xml/precise_20190312_3.xml']

    #brokens = [1, 39, 43, 54]
    #brokens = [0, 38, 42, 53]

    tablename = 'measurement.csv'
    geometryStat(file,tablename)


if __name__ == '__main__':
    main()