from procedures.moduleTest import calculus as cal
import numpy as np
import pandas as pd
import logging
import pandas as pd

logging.basicConfig(format="%(asctime)s.%(msecs)03d %(levelno)s %(name)-10s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")
logger = logging.getLogger('motormaps')
logger.setLevel(logging.INFO)

moveDtype = np.dtype(dict(names=['position', 'thetaAngle', 'thetaSteps', 'thetaOntime',
                                 'thetaFast', 'phiAngle', 'phiSteps', 'phiOntime', 'phiFast'],
                          formats=['c16', 'f4', 'i4', 'f4', '?', 'f4', 'i4', 'f4', '?']))

cc = None

def setCobraCoach(cobraCoach):
    global cc
    cc = cobraCoach

#def buildThetaMotorMaps(xml, steps=500, repeat=1, fast=False, tries=10, homed=True):
#    if homed:
#        logger.info(f'Move theta arms CW and phi arms CCW to the hard stops')
#        cc.moveToHome(cc.goodCobras, thetaEnable=True, phiEnable=True, thetaCCW=False)
#    for group in range(3):
#        prepareThetaMotorMaps(group=group, tries=tries, homed=False)
#        homePhiArms(group=group)
#        runThetaMotorMaps(xml, group=group, steps=steps, repeat=repeat, fast=fast)

#def buildPhiMotorMaps(xml, steps=250, repeat=1, fast=False, tries=10, homed=True):
#    if homed:
#        logger.info(f'Move theta arms CW and phi arms CCW to the hard stops')
#        cc.moveToHome(cc.goodCobras, thetaEnable=True, phiEnable=True, thetaCCW=False)
#    preparePhiMotorMaps(tries=tries, homed=False)
#    runPhiMotorMaps(xml, steps=steps, repeat=repeat, fast=fast)

def moveThetaPhi(cIds, thetas, phis, relative=False, local=True,
                 tolerance=0.1, tries=6, homed=False,
                 newDir=True, thetaFast=False, phiFast=False,
                 threshold=10.0, thetaMargin=np.deg2rad(15.0)):
    """
    move cobras to the target angles

    Parameters
    ----
    cIds : index for the active cobras
    thetas : angles to move for theta arms
    phis : angles to move for phi arms
    relative : angles are offsets from current positions or not
    local : the angles are from the CCW hard stops or not
    tolerance : tolerance for target positions in pixels
    tries : number of iterations
    homed : go home first or not, if true, move in the safe way
    newDir : create a new directory for data or not
    thetaFast: using fast if true else slow theta motor maps
    phiFast: using fast if true else slow phi motor maps
    threshold: using slow motor maps if the distance to the target is below this value
    thetaMargin : the minimum theta angles to the theta hard stops

    Returns
    ----
    A tuple with three elements:
    - dataPath
    - errors for theta angles
    - errors for phi angles
    - a numpy array for the moving history
    """
    if cc.getMode() != 'normal':
        raise RuntimeError('Switch to normal mode first!!!')
    if np.isscalar(thetas):
        thetas = np.full(len(cIds), thetas)
    elif len(thetas) != len(cIds):
        raise RuntimeError('number of theta angles must match the number of cobras')
    if np.isscalar(phis):
        phis = np.full(len(cIds), phis)
    elif len(phis) != len(cIds):
        raise RuntimeError('number of phi angles must match the number of cobras')
    if not isinstance(thetaFast, bool) and len(cIds) != len(thetaFast):
        raise RuntimeError("number of thetaFast must match number of cobras")
    else:
        _thetaFast = np.zeros(cc.nCobras, 'bool')
        _thetaFast[cIds] = thetaFast
        thetaFast = _thetaFast
    if not isinstance(phiFast, bool) and len(cobras) != len(phiFast):
        raise RuntimeError("number of phiFast must match number of cobras")
    else:
        _phiFast = np.zeros(cc.nCobras, 'bool')
        _phiFast[cIds] = phiFast
        phiFast = _phiFast

    if newDir:
        cc.connect(False)
    dataPath = cc.runManager.dataDir
    moves = np.zeros((len(cIds), tries), dtype=moveDtype)

    nowDone = np.zeros(cc.nCobras, 'bool')
    notDoneMask = np.zeros(cc.nCobras, 'bool')
    farAwayMask = np.zeros(cc.nCobras, 'bool')
    targets = np.zeros(cc.nCobras, 'complex')
    targetThetas = np.zeros(cc.nCobras)
    targetPhis = np.zeros(cc.nCobras)
    atThetas = np.zeros(cc.nCobras)
    atPhis = np.zeros(cc.nCobras)
    notDoneMask[cIds] = True
    farAwayMask[cIds] = True

    if relative:
        targetThetas[cIds] = thetas + cc.thetaInfo[cIds]['angle']
        targetPhis[cIds] = phis + cc.phiInfo[cIds]['angle']
    elif not local:
        targetThetas[cIds] = (thetas - cc.calibModel.tht0[cIds]) % (np.pi*2)
        targetThetas[targetThetas < thetaMargin] += np.pi*2
        thetaRange = (cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (np.pi*2) + np.pi
        tooBig = targetThetas > thetaRange - thetaMargin
        targetThetas[tooBig] = thetaRange[tooBig] - thetaMargin
        targetPhis[cIds] = phis - cc.calibModel.phiIn[cIds] - np.pi
    else:
        targetThetas[cIds] = thetas
        targetPhis[cIds] = phis
    targets = cc.pfi.anglesToPositions(cc.allCobras, targetThetas, targetPhis)

    cc.camResetStack(f'Stack.fits')
    logger.info(f'Move theta arms to angle={np.round(np.rad2deg(targetThetas[cIds]),2)} degree')
    logger.info(f'Move phi arms to angle={np.round(np.rad2deg(targetPhis[cIds]),2)} degree')

    if homed:
        # go home for safe movement
        cobras = cc.allCobras[cIds]
        logger.info(f'Move theta arms CW and phi arms CCW to the hard stops')
        cc.moveToHome(cobras, thetaEnable=True, phiEnable=True, thetaCCW=False)

    for j in range(tries):
        cobras = cc.allCobras[notDoneMask]
        _thetaFast = farAwayMask[notDoneMask] & thetaFast[notDoneMask]
        _phiFast = farAwayMask[notDoneMask] & phiFast[notDoneMask]
        cc.thetaScaling = np.logical_not(farAwayMask)
        cc.phiScaling = np.logical_not(farAwayMask)
        atThetas[notDoneMask], atPhis[notDoneMask] = \
            cc.moveToAngles(cobras, targetThetas[notDoneMask], targetPhis[notDoneMask], _thetaFast, _phiFast, True)

        atPositions = cc.cobraInfo['position']
        distances = np.abs(atPositions - targets)
        nowDone[distances < tolerance] = True
        newlyDone = nowDone & notDoneMask
        farAwayMask = (distances > threshold) & farAwayMask

        ndIdx = notDoneMask[cIds]
        moves['position'][ndIdx,j] = atPositions[notDoneMask]
        moves['thetaAngle'][ndIdx,j] = atThetas[notDoneMask]
        moves['thetaSteps'][ndIdx,j] = cc.moveInfo['thetaSteps'][notDoneMask]
        moves['thetaOntime'][ndIdx,j] = cc.moveInfo['thetaOntime'][notDoneMask]
        moves['thetaFast'][ndIdx,j] = cc.moveInfo['thetaFast'][notDoneMask]
        moves['phiAngle'][ndIdx,j] = atPhis[notDoneMask]
        moves['phiSteps'][ndIdx,j] = cc.moveInfo['phiSteps'][notDoneMask]
        moves['phiOntime'][ndIdx,j] = cc.moveInfo['phiOntime'][notDoneMask]
        moves['phiFast'][ndIdx,j] = cc.moveInfo['phiFast'][notDoneMask]

        if np.any(newlyDone):
            notDoneMask &= ~newlyDone
            logger.info(f'done: {np.where(newlyDone)[0]}, {(notDoneMask == True).sum()} left')
        if not np.any(notDoneMask):
            logger.info(f'all cobras are in positions')
            break

    cc.camResetStack()
    cc.thetaScaling = np.full(cc.nCobras, True)
    cc.phiScaling = np.full(cc.nCobras, True)
    if np.any(notDoneMask):
        logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                         f'{np.where(notDoneMask)[0]}, '
                         f'{np.round(distances[notDoneMask],2)}')
    
    badMoveIdx = np.where(notDoneMask)[0]

    return dataPath, atThetas[cIds], atPhis[cIds], moves


def moveThetaPhi2Steps(cIds, thetas, phis, relative=False, local=True,
                       tolerance=0.1, tries=8, homed=True, newDir=True,
                       threshold=10.0, thetaMargin=np.deg2rad(15.0),
                       phiMargin=np.deg2rad(5.0)):
    """
    move cobras to the target angles in the radial direction

    Parameters
    ----
    cIds : index for the active cobras
    thetas : angles to move for theta arms
    phis : angles to move for phi arms
    relative : angles are offsets from current positions or not
    local : the angles are from the CCW hard stops or not
    tolerance : tolerance for target positions in pixels
    tries : number of iterations
    homed : go home first or not, if true, move in the safe way
    newDir : create a new directory for data or not
    threshold: using slow motor maps if the distance to the target is below this value
    thetaMargin : the minimum theta angles to the theta hard stops
    phiMargin : the minimum theta angles to the phi hard stops

    Returns
    ----
    A tuple with three elements:
    - dataPath
    - errors for theta angles
    - errors for phi angles
    - a numpy array for the moving history
    """
    if cc.getMode() != 'normal':
        raise RuntimeError('Switch to normal mode first!!!')
    if np.isscalar(thetas):
        thetas = np.full(len(cIds), thetas)
    elif len(thetas) != len(cIds):
        raise RuntimeError('number of theta angles must match the number of cobras')
    if np.isscalar(phis):
        phis = np.full(len(cIds), phis)
    elif len(phis) != len(cIds):
        raise RuntimeError('number of phi angles must match the number of cobras')

    # convert to local coordinate
    if relative:
        thetas += cc.thetaInfo[cIds]['angle']
        phis += cc.phiInfo[cIds]['angle']
    elif not local:
        thetas = (thetas - cc.calibModel.tht0[cIds]) % (np.pi*2)
        thetas[thetas < thetaMargin] += np.pi*2
        thetaRange = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (np.pi*2) + np.pi)[cIds]
        tooBig = thetas > thetaRange - thetaMargin
        thetas[tooBig] = thetaRange[tooBig] - thetaMargin
        phis -= cc.calibModel.phiIn[cIds] + np.pi
        phis[phis < 0] = 0
        phiRange = ((cc.calibModel.phiOut - cc.calibModel.phiIn) % (np.pi*2))[cIds]
        tooBig = phis > phiRange - phiMargin
        phis[tooBig] = phiRange[tooBig] - phiMargin

    # limit phi angle for first two tries
    limitPhi = np.pi/3 - cc.calibModel.phiIn[cIds] - np.pi
    thetasVia = np.copy(thetas)
    phisVia = np.copy(phis)
    for c in range(len(cIds)):
        if phis[c] > limitPhi[c]:
            phisVia[c] = limitPhi[c]
            thetasVia[c] = thetas[c] + (phis[c] - limitPhi[c])/2
            if thetasVia[c] > thetaRange[c]:
                thetasVia[c] = thetaRange[c]

    moves = np.zeros((len(cIds), tries), dtype=moveDtype)
    _useScaling, _maxSegments, _maxTotalSteps = cc.useScaling, cc.maxSegments, cc.maxTotalSteps
    cc.useScaling, cc.maxSegments, cc.maxTotalSteps = False, _maxSegments * 2, _maxTotalSteps * 2
    dataPath, atThetas, atPhis, moves[:,:2] = \
        moveThetaPhi(cIds, thetasVia, phisVia, False, True, tolerance, 2, homed, newDir, True, True, threshold)

    cc.useScaling, cc.maxSegments, cc.maxTotalSteps = _useScaling, _maxSegments, _maxTotalSteps
    dataPath, atThetas, atPhis, moves[:,2:] = \
        moveThetaPhi(cIds, thetas, phis, False, True, tolerance, tries-2, False, False, False, True, threshold)

    return dataPath, atThetas, atPhis, moves

def prepareThetaMotorMaps(group=0, phi_limit=np.pi/3*2, tolerance=0.1, tries=10, homed=True, threshold=1.0, elbow_radius=1.0, margin=0.1):

    """ move cobras to safe positions for generating theta motor maps
        cobras are divided into three non-interfering groups
        group = (0,1,2)
        phi_limit: the maximum phi angle
        tolerace: target error tolerance(pixel)
        tries: maximum number of movements
        homed: run go home if True
        threshold: threshold(pixel) to switch to slow motor maps
        elbow_radius: the radius of elbows and tips
        margin: margin(pixel) to avoid collision
    """
    if cc.getMode() != 'normal':
        raise RuntimeError('Switch to normal mode first!!!')

    # check cobras configuration
    ydir = np.angle(cc.calibModel.centers[1] - cc.calibModel.centers[55])
    if np.abs(ydir - np.pi/2) > 0.1:
        raise RuntimeError("Check the cobra configuration file")

    # only for good cobras
    goodIdx = cc.goodIdx
    centers = cc.calibModel.centers

    # Loading dot locations
    dotFile = '/software/devel/pfs/pfs_instdata/data/pfi/dot/black_dots_mm.csv'
    dotDF=pd.read_csv(dotFile)
    
    dots = (dotDF['x'].values)+(dotDF['y'].values)*1j
    dots_radii = dotDF['r'].values
    
    #dots = cc.calibModel.dotpos
    #dots_radii = cc.calibModel.dotradii
    elbows = np.zeros(len(centers), 'complex')
    thetaAngles = np.zeros(len(centers))
    phiAngles = np.zeros(len(centers))

    # calculate theta angles and elbow positions
    for n in range(len(centers)):
        gidx = (n + (n//57) + (n//(14*57))) % 3
        if not n in goodIdx or gidx == group:
            thetaAngles[n] = np.pi/3
            elbows[n] = np.nan
        else:
            if (gidx - group) % 3 == 1:
                thetaAngles[n] = np.pi/2
                elbows[n] = centers[n] + cc.calibModel.L1[n] * np.exp(1j * np.pi/2)
            else:
                thetaAngles[n] = np.pi/6
                elbows[n] = centers[n] + cc.calibModel.L1[n] * np.exp(1j * np.pi/6)

    # calculate phi angles
    for n in range(len(centers)):
        gidx = (n + (n//57) + (n//(14*57))) % 3
        if not n in goodIdx or gidx != group:
            phiAngles[n] = np.pi/3
            continue

        elbows_dist = np.abs(elbows - centers[n]) - elbow_radius * 2
        dots_dist = np.abs(dots - centers[n]) - dots_radii[n]
        dots_dist[n] = np.nan
        radius_max = np.nanmin((elbows_dist, dots_dist)) - margin
        radius_do = np.abs(dots[n] - centers[n]) + dots_radii[n] + margin
        radius_di = np.abs(dots[n] - centers[n]) - dots_radii[n] - margin
        radius_limit = np.abs(cc.calibModel.L1[n] - cc.calibModel.L2[n] * np.exp(1j * phi_limit))

        if radius_max >= radius_limit and radius_limit >= radius_do:
            radius_fit = radius_limit
        elif radius_limit >= radius_max and radius_max >= radius_do:
            radius_fit = radius_max
            print(f'Running cobra#{n} outside with radius={radius_fit}')
        else:
            radius_fit = radius_di
            print(f'Running cobra#{n} inside with radius={radius_fit}')

        _, phi_angles, _ = cc.pfi.positionsToAngles([cc.allCobras[n]], [centers[n] + radius_fit])
        phiAngles[n] = phi_angles[0,0] + cc.calibModel.phiIn[n] + np.pi

    cc.pfi.resetMotorScaling()
    dataPath, thetas, phis, moves = moveThetaPhi2Steps(goodIdx, thetaAngles[goodIdx], phiAngles[goodIdx],
        False, False, tolerance, tries, homed, True, threshold)
    np.save(dataPath / 'moves', moves)

def homePhiArms(group=0):
    """ home phi arms for the specified group """
    groupIdx = np.zeros(cc.nCobras, 'bool')
    for n in range(cc.nCobras):
        gidx = (n + (n//57) + (n//(14*57))) % 3
        if gidx != group and n in cc.goodIdx:
            groupIdx[n] = True

    cc.moveToHome(cc.allCobras[groupIdx], phiEnable=True)

def runThetaMotorMaps(newXml, group=0, steps=500, totalSteps=10000, repeat=1, fast=False, thetaOnTime=None,
                       limitOnTime=0.08, limitSteps=10000, updateGeometry=False, phiRunDir=None,
                       delta=np.deg2rad(5.0), force=False):
    """
    generate theta motor maps, it accepts custom thetaOnTIme parameter.
    all cobras should have been placed in the proper position,
    cobras are divided into three non-intefere groups to avoid collision,
    if thetaOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime
    Example:
        makethetaMotorMap(xml, fast=True)               // update fast motor maps
        makethetaMotorMap(xml, fast=False)              // update slow motor maps
        makethetaMotorMap(xml, thetaOnTime=0.06)        // motor maps for on-time=0.06
    """
    lastMode = cc.getMode()
    if lastMode != 'theta':
        cc.setMode('theta')

    # select only the specified group
    groupIdx = np.zeros(cc.nCobras, 'bool')
    for n in range(cc.nCobras):
        gidx = (n + (n//57) + (n//(14*57))) % 3
        if gidx == group and n in cc.goodIdx:
            groupIdx[n] = True

    defaultGoodIdx = cc.goodIdx
    cc.goodIdx = np.where(groupIdx)[0]
    cc.goodCobras = cc.allCobras[cc.goodIdx]

    # aquire data for motor maps
    dataPath, posF, posR = cc.roundTripForTheta(steps, totalSteps, repeat, fast,
                                                thetaOnTime, limitOnTime, limitSteps, force)

    # calculate centers and theta angles
    center = np.zeros(cc.nCobras, dtype=complex)
    radius = np.zeros(cc.nCobras, dtype=float)
    angF = np.zeros(posF.shape, dtype=float)
    angR = np.zeros(posR.shape, dtype=float)
    bad = np.zeros(cc.nCobras, dtype=bool)
    for ci in cc.goodIdx:
        center[ci], radius[ci], angF[ci], angR[ci], bad[ci] = cal.thetaCenterAngles(posF[ci], posR[ci])

    for short in np.where(bad)[0]:
        logger.warn(f'theta range for {short+1:-2d} is short: '
                         f'back={np.rad2deg(angF[short,0,0]):-6.2f} '
                         f'out={np.rad2deg(angR[short,0,-1]):-6.2f}')
    np.save(dataPath / 'thetaCenter', center)
    np.save(dataPath / 'thetaRadius', radius)
    np.save(dataPath / 'thetaAngFW', angF)
    np.save(dataPath / 'thetaAngRV', angR)
    np.save(dataPath / 'badRange', np.where(bad)[0])

    # update theta geometry
    cwHome = np.zeros(cc.nCobras)
    ccwHome = np.zeros(cc.nCobras)
    for m in range(cc.nCobras):
        maxR = 0
        maxIdx = 0
        for n in range(posF.shape[1]):
            ccwH = np.angle(posF[m,n,0] - center[m])
            cwH = np.angle(posR[m,n,0] - center[m])
            curR = (cwH - ccwH + np.pi) % (np.pi*2) + np.pi
            if curR > maxR:
                maxR = curR
                maxIdx = n
        if cc.thetaInfoIsValid:
            lastR = (cc.thetaInfo['cwHome'][m] - cc.thetaInfo['ccwHome'][m] + np.pi) % (np.pi*2) + np.pi
        else:
            lastR = 0
        if curR >= lastR:
            ccwHome[m] = np.angle(posF[m,maxIdx,0] - center[m]) % (np.pi*2)
            cwHome[m] = np.angle(posR[m,maxIdx,0] - center[m]) % (np.pi*2)
        else:
            ccwHome[m] = cc.thetaInfo['ccwHome'][m]
            cwHome[m] = cc.thetaInfo['cwHome'][m]
            center[m] = cc.thetaInfo['center'][m]
    cc.setThetaGeometry(center, ccwHome, cwHome, angle=0, onlyIfClear=False)

    # calculate average speeds
    spdF = np.zeros(cc.nCobras, dtype=float)
    spdR = np.zeros(cc.nCobras, dtype=float)
    for ci in cc.goodIdx:
        spdF[ci], spdR[ci] = cal.speed(angF[ci], angR[ci], steps, delta)
    np.save(dataPath / 'thetaSpeedFW', spdF)
    np.save(dataPath / 'thetaSpeedRV', spdR)

    # calculate motor maps in Johannes weighting
    mmF = np.zeros((cc.nCobras, cal.regions), dtype=float)
    mmR = np.zeros((cc.nCobras, cal.regions), dtype=float)
    mmBad = np.zeros(cc.nCobras, dtype=bool)
    for ci in cc.goodIdx:
        mmF[ci], mmR[ci], mmBad[ci] = cal.motorMaps(angF[ci], angR[ci], steps, delta)
    for bad_i in np.where(mmBad)[0]:
        logger.warn(f'theta map for {bad_i+1} is bad')
    np.save(dataPath / 'thetaMMFW', mmF)
    np.save(dataPath / 'thetaMMRV', mmR)
    np.save(dataPath / 'badMotorMap', np.where(mmBad)[0])

    # update XML file, using Johannes weighting
    slow = not fast
    mmBad[cc.badIdx] = True
    cal.updateThetaMotorMaps(cc.calibModel, mmF, mmR, mmBad, slow)

    if thetaOnTime is not None:
        if np.isscalar(thetaOnTime):
            onTime = np.full(cc.nCobras, thetaOnTime)
            cc.calibModel.updateOntimes(thetaFwd=onTime, thetaRev=onTime, fast=fast)
        else:
            cc.calibModel.updateOntimes(thetaFwd=thetaOnTime[0], thetaRev=thetaOnTime[1], fast=fast)

    if updateGeometry:
        phiCenter = np.load(phiRunDir / 'data' / 'phiCenter.npy')
        phiRadius = np.load(phiRunDir / 'data' / 'phiRadius.npy')
        phiPosF = np.load(phiRunDir / 'data' / 'phiFW.npy')
        phiPosR = np.load(phiRunDir / 'data' / 'phiRV.npy')

        thetaL, phiL, thetaCCW, thetaCW, phiCCW, phiCW = cal.geometry(cc.goodIdx,
            center, radius, posF, posR, phiCenter, phiRadius, phiPosF, phiPosR)

        cc.calibModel.updateGeometry(center, thetaL, phiL)
        cc.calibModel.updateThetaHardStops(thetaCCW, thetaCW)
        cc.calibModel.updatePhiHardStops(phiCCW, phiCW)

    cc.calibModel.createCalibrationFile(cc.runManager.outputDir / newXml)

    if lastMode != 'theta':
        cc.setMode(lastMode)
    cc.goodIdx = defaultGoodIdx
    cc.goodCobras = cc.allCobras[cc.goodIdx]

    return cc.runManager.runDir, np.where(np.logical_or(bad, mmBad))[0]

def preparePhiMotorMaps(thetaAngle=np.pi/3, tolerance=0.01, tries=12, homed=True, threshold=2.0):
    """ move cobras to safe positions for generating phi motor maps
        that is, theta arms = 60 degree on PFI coordonate
        thetaAngle: global theta arm direction
        tolerace: target error tolerance(pixel)
        tries: maximum number of movements
        homed: run go home if True
        threshold: threshold(pixel) to switch to slow motor maps
    """
    if cc.getMode() != 'normal':
        raise RuntimeError('Switch to normal mode first!!!')

    # check cobras configuration
    ydir = np.angle(cc.calibModel.centers[1] - cc.calibModel.centers[55])
    if np.abs(ydir - np.pi/2) > 0.1:
        raise RuntimeError("Check the cobra configuration file")

    # set phi arms to 60 degree
    phiAngle = np.pi/3

    # only for good cobras
    goodIdx = np.where(cc.calibModel.tht0 != 0.0)[0]

    cc.pfi.resetMotorScaling()
    dataPath, thetas, phis, moves = moveThetaPhi2Steps(goodIdx, thetaAngle, phiAngle,
        False, False, tolerance, tries, homed, True, threshold)
    np.save(dataPath / 'moves', moves)

def runPhiMotorMaps(newXml, steps=250, totalSteps=5000, repeat=1, fast=False, phiOnTime=None,
                     limitOnTime=0.05, limitSteps=5000, delta=np.deg2rad(5.0)):
    """ generate phi motor maps, it accepts custom phiOnTIme parameter.

        if phiOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime

        Example:
            makePhiMotorMap(xml, fast=True)             // update fast motor maps
            makePhiMotorMap(xml, fast=False)            // update slow motor maps
            makePhiMotorMap(xml, phiOnTime=0.06)        // motor maps for on-time=0.06
    """
    lastMode = cc.getMode()
    if lastMode != 'phi':
        cc.setMode('phi')

    # aquire data for motor maps
    dataPath, posF, posR = cc.roundTripForPhi(steps, totalSteps, repeat, fast, phiOnTime, limitOnTime, limitSteps)

    # calculate centers and phi angles
    center = np.zeros(cc.nCobras, dtype=complex)
    radius = np.zeros(cc.nCobras, dtype=float)
    angF = np.zeros(posF.shape, dtype=float)
    angR = np.zeros(posR.shape, dtype=float)
    bad = np.zeros(cc.nCobras, dtype=bool)
    for ci in cc.goodIdx:
        center[ci], radius[ci], angF[ci], angR[ci], bad[ci] = cal.phiCenterAngles(posF[ci], posR[ci])

    for short in np.where(bad)[0]:
        logger.warn(f'phi range for {short+1:-2d} is short: '
                         f'back={np.rad2deg(angF[short,0,0]):-6.2f} '
                         f'out={np.rad2deg(angR[short,0,-1]):-6.2f}')
    np.save(dataPath / 'phiCenter', center)
    np.save(dataPath / 'phiRadius', radius)
    np.save(dataPath / 'phiAngFW', angF)
    np.save(dataPath / 'phiAngRV', angR)
    np.save(dataPath / 'badRange', np.where(bad)[0])
    #np.save(dataPath / 'dotlocation',cc.calibModel.dotpos)
    #np.save(dataPath / 'dotradii',cc.calibModel.dotradii)

    # update phi geometry
    cwHome = np.zeros(cc.nCobras)
    ccwHome = np.zeros(cc.nCobras)
    for m in range(cc.nCobras):
        maxR = 0
        maxIdx = 0
        for n in range(posF.shape[1]):
            ccwH = np.angle(posF[m,n,0] - center[m])
            cwH = np.angle(posR[m,n,0] - center[m])
            curR = (cwH - ccwH) % (np.pi*2)
            if curR > maxR:
                maxR = curR
                maxIdx = n
        if cc.phiInfoIsValid:
            lastR = (cc.phiInfo['cwHome'][m] - cc.phiInfo['ccwHome'][m]) % (np.pi*2)
        else:
            lastR = 0
        if curR >= lastR:
            ccwHome[m] = np.angle(posF[m,maxIdx,0] - center[m]) % (np.pi*2)
            cwHome[m] = np.angle(posR[m,maxIdx,0] - center[m]) % (np.pi*2)
        else:
            ccwHome[m] = cc.phiInfo['ccwHome'][m]
            cwHome[m] = cc.phiInfo['cwHome'][m]
            center[m] = cc.phiInfo['center'][m]
    cc.setPhiGeometry(center, ccwHome, cwHome, angle=0, onlyIfClear=False)

    # calculate average speeds
    spdF = np.zeros(cc.nCobras, dtype=float)
    spdR = np.zeros(cc.nCobras, dtype=float)
    for ci in cc.goodIdx:
        spdF[ci], spdR[ci] = cal.speed(angF[ci], angR[ci], steps, delta)
    np.save(dataPath / 'phiSpeedFW', spdF)
    np.save(dataPath / 'phiSpeedRV', spdR)

    # calculate motor maps in Johannes weighting
    mmF = np.zeros((cc.nCobras, cal.regions), dtype=float)
    mmR = np.zeros((cc.nCobras, cal.regions), dtype=float)
    mmBad = np.zeros(cc.nCobras, dtype=bool)
    for ci in cc.goodIdx:
        mmF[ci], mmR[ci], mmBad[ci] = cal.motorMaps(angF[ci], angR[ci], steps, delta)
    for bad_i in np.where(mmBad)[0]:
        logger.warn(f'phi map for {bad_i+1} is bad')
    np.save(dataPath / 'phiMMFW', mmF)
    np.save(dataPath / 'phiMMRV', mmR)
    np.save(dataPath / 'badMotorMap', np.where(mmBad)[0])

    # update XML file, using Johannes weighting
    slow = not fast
    mmBad[cc.badIdx] = True
    cal.updatePhiMotorMaps(cc.calibModel, mmF, mmR, mmBad, slow)

    if phiOnTime is not None:
        if np.isscalar(phiOnTime):
            onTime = np.full(cc.nCobras, phiOnTime)
            cc.calibModel.updateOntimes(phiFwd=onTime, phiRev=onTime, fast=fast)
        else:
            cc.calibModel.updateOntimes(phiFwd=phiOnTime[0], phiRev=phiOnTime[1], fast=fast)

    cc.calibModel.createCalibrationFile(cc.runManager.outputDir / newXml)

    if lastMode != 'phi':
        cc.setMode(lastMode)

    return cc.runManager.runDir, np.where(np.logical_or(bad, mmBad))[0]
