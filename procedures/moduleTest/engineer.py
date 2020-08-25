#from procedures.moduleTest.cobraCoach import cobraCoach as cc
from procedures.moduleTest import calculus as cal
import numpy as np
import logging
from copy import deepcopy
import pathlib
from procedures.moduleTest.speedModel import SpeedModel

logging.basicConfig(format="%(asctime)s.%(msecs)03d %(levelno)s %(name)-10s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")
logger = logging.getLogger('engineer')
logger.setLevel(logging.INFO)

thetaModel = SpeedModel(p1=0.09)
phiModel = SpeedModel(p1=0.07)

move1Dtype = np.dtype(dict(names=['position', 'angle', 'steps', 'ontime', 'fast'],
                           formats=['c16', 'f4', 'i4', 'f4', '?']))
moveDtype = np.dtype(dict(names=['position', 'thetaAngle', 'thetaSteps', 'thetaOntime',
                                 'thetaFast', 'phiAngle', 'phiSteps', 'phiOntime', 'phiFast'],
                          formats=['c16', 'f4', 'i4', 'f4', '?', 'f4', 'i4', 'f4', '?']))

cc = None
mmTheta = None
mmPhi = None

def setCobraCoach(cobraCoach):
    global cc
    cc = cobraCoach

def setThetaMode():
    """ switch to theta mode """
    if cc.getMode() == 'theta':
        logger.info('Already in THETA mode')
        return
    cc.setMode('theta')

def setPhiMode():
    """ switch to theta mode """
    if cc.getMode() == 'phi':
        logger.info('Already in PHI mode')
        return
    cc.setMode('phi')

def setNormalMode():
    """ switch to theta mode """
    if cc.getMode() == 'normal':
        logger.info('Already in NORMAL mode')
        return
    cc.setMode('normal')

def moveThetaAngles(cIds, thetas, relative=False, local=True, tolerance=0.002, tries=6, fast=False, newDir=True):
    """
    move theta arms to the target angles

    Parameters
    ----
    cIds : index for the active cobras
    thetas : angles to move
    relative : angles are offsets from current positions or not
    local : the angles are from the CCW hard stops or in global coordinate
    newDir : create a new directory for data or not

    Returns
    ----
    A tuple with two elements:
    - dataPath
    - errors for the theta angles
    - a numpy array for the moving history
    """
    if cc.getMode() != 'theta':
        raise RuntimeError('Switch to theta mode first!!!')
    if not cc.thetaInfoIsValid:
        raise RuntimeError('Please set theta Geometry or build a Motor-maps first!!!')
    if np.isscalar(thetas):
        thetas = np.full(len(cIds), thetas)
    elif len(thetas) != len(cIds):
        raise RuntimeError("number of theta angles must the match number of cobras")

    if newDir:
        cc.connect(False)
    dataPath = cc.runManager.dataDir
    moves = np.zeros((len(cIds), tries), dtype=move1Dtype)

    nowDone = np.zeros(cc.nCobras, 'bool')
    notDoneMask = np.zeros(cc.nCobras, 'bool')
    targets = np.zeros(cc.nCobras)
    atAngles = np.zeros(cc.nCobras)
    notDoneMask[cIds] = True

    if relative:
        targets[cIds] = thetas + cc.thetaInfo[cIds]['angle']
    elif not local:
        targets[cIds] = (thetas - cc.calibModel.tht0[cIds]) % (np.pi*2)
    else:
        targets[cIds] = thetas

    cc.cam.resetStack(f'thetaStack.fits')
    logger.info(f'Move theta arms to angle={np.round(np.rad2deg(targets[cIds]),2)} degree')

    for j in range(tries):
        cobras = cc.allCobras[notDoneMask]
        atAngles[notDoneMask], _ = cc.moveToAngles(cobras, targets[notDoneMask], None, fast, False, True)

        for ci in range(len(cIds)):
            if notDoneMask[cIds[ci]]:
                moves['position'][ci, j] = cc.cobraInfo['position'][cIds[ci]]
                moves['steps'][ci, j] = cc.moveInfo['thetaSteps'][cIds[ci]]
                moves['angle'][ci, j] = cc.thetaInfo['angle'][cIds[ci]]
                moves['ontime'][ci, j] = cc.moveInfo['thetaOntime'][cIds[ci]]
                moves['fast'][ci, j] = cc.moveInfo['thetaFast'][cIds[ci]]

        diffAngles = cal.absDiffAngle(atAngles, targets)
        nowDone[diffAngles < tolerance] = True
        newlyDone = nowDone & notDoneMask

        if np.any(newlyDone):
            notDoneMask &= ~newlyDone
            logger.info(f'done: {np.where(newlyDone)[0]}, {(notDoneMask == True).sum()} left')
        if not np.any(notDoneMask):
            logger.info(f'all cobras are in positions')
            break

    cc.cam.resetStack()
    if np.any(notDoneMask):
        logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                         f'{np.where(notDoneMask)[0]}, '
                         f'{np.round(np.rad2deg(diffAngles)[notDoneMask], 2)}')

    return dataPath, cal.diffAngle(atAngles, targets)[cIds], moves

def movePhiAngles(cIds, phis, relative=False, local=True, tolerance=0.002, tries=6, fast=False, newDir=True):
    """
    move phi arms to the target angles

    Parameters
    ----
    cIds : index for the active cobras
    phis : angles to move
    relative : angles are offsets from current positions or not
    local : the angles are from the CCW hard stops or the theta arms

    Returns
    ----
    A tuple with two elements:
    - dataPath
    - errors for the phi angles
    - a numpy array for the moving history
    """
    if cc.getMode() != 'phi':
        raise RuntimeError('Switch to phi mode first!!!')
    if not cc.phiInfoIsValid:
        raise RuntimeError('Please set phi Geometry or build a Motor-maps first!!!')
    if np.isscalar(phis):
        phis = np.full(len(cIds), phis)
    elif len(phis) != len(cIds):
        raise RuntimeError("number of phi angles must the match number of cobras")

    if newDir:
        cc.connect(False)
    dataPath = cc.runManager.dataDir
    moves = np.zeros((len(cIds), tries), dtype=move1Dtype)

    nowDone = np.zeros(cc.nCobras, 'bool')
    notDoneMask = np.zeros(cc.nCobras, 'bool')
    targets = np.zeros(cc.nCobras)
    atAngles = np.zeros(cc.nCobras)
    notDoneMask[cIds] = True

    if relative:
        targets[cIds] = phis + cc.phiInfo[cIds]['angle']
    elif not local:
        targets[cIds] = phis - cc.calibModel.phiIn[cIds] - np.pi
    else:
        targets[cIds] = phis

    cc.cam.resetStack(f'phiStack.fits')
    logger.info(f'Move phi arms to angle={np.round(np.rad2deg(targets[cIds]),2)} degree')

    for j in range(tries):
        cobras = cc.allCobras[notDoneMask]
        atThetas, atPhis = cc.moveToAngles(cobras, None, targets[notDoneMask], False, fast, True)
        atAngles[notDoneMask] = atPhis

        for ci in range(len(cIds)):
            if notDoneMask[cIds[ci]]:
                moves['position'][ci, j] = cc.cobraInfo['position'][cIds[ci]]
                moves['steps'][ci, j] = cc.moveInfo['phiSteps'][cIds[ci]]
                moves['angle'][ci, j] = cc.phiInfo['angle'][cIds[ci]]
                moves['ontime'][ci, j] = cc.moveInfo['phiOntime'][cIds[ci]]
                moves['fast'][ci, j] = cc.moveInfo['phiFast'][cIds[ci]]

        diffAngles = cal.absDiffAngle(atAngles, targets)
        nowDone[diffAngles < tolerance] = True
        newlyDone = nowDone & notDoneMask

        if np.any(newlyDone):
            notDoneMask &= ~newlyDone
            logger.info(f'done: {np.where(newlyDone)[0]}, {(notDoneMask == True).sum()} left')
        if not np.any(notDoneMask):
            logger.info(f'all cobras are in positions')
            break

    cc.cam.resetStack()
    if np.any(notDoneMask):
        logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                         f'{np.where(notDoneMask)[0]}, '
                         f'{np.round(np.rad2deg(diffAngles)[notDoneMask], 2)}')

    return dataPath, cal.diffAngle(atAngles, targets)[cIds], moves

def thetaConvergenceTest(cIds, margin=15.0, runs=50, tries=8, fast=False, tolerance=0.1):
    """ theta convergence test """
    moves = np.zeros((runs, len(cIds), tries), dtype=move1Dtype)
    cc.connect(False)
    tolerance = np.deg2rad(tolerance)

    for i in range(runs):
        if runs > 1:
            angle = np.deg2rad(margin + (360 - 2*margin) * i / (runs - 1))
        else:
            angle = np.deg2rad(180)
        logger.info(f'Run {i+1}: angle={np.rad2deg(angle):.2f} degree')
        cc.pfi.resetMotorScaling(cc.allCobras[cIds], 'theta')

        cc.moveToHome(cc.allCobras[cIds], thetaEnable=True)
        dataPath, diffAngles, moves[i,:,:] = \
            moveThetaAngles(cIds, angle, False, True, tolerance, tries, fast, newDir=False)

    np.save(dataPath / 'moves', moves)
    return moves

def phiConvergenceTest(cIds, margin=15.0, runs=50, tries=8, fast=False, tolerance=0.1):
    """ phi convergence test """
    moves = np.zeros((runs, len(cIds), tries), dtype=move1Dtype)
    cc.connect(False)
    tolerance = np.deg2rad(tolerance)

    for i in range(runs):
        if runs > 1:
            angle = np.deg2rad(margin + (180 - 2*margin) * i / (runs - 1))
        else:
            angle = np.deg2rad(90)
        logger.info(f'Run {i+1}: angle={np.rad2deg(angle):.2f} degree')
        cc.pfi.resetMotorScaling(cc.allCobras[cIds], 'phi')

        cc.moveToHome(cc.allCobras[cIds], phiEnable=True)
        dataPath, diffAngles, moves[i,:,:] = \
            movePhiAngles(cIds, angle, False, True, tolerance, tries, fast, newDir=False)

    np.save(dataPath / 'moves', moves)
    return moves

def moveThetaPhi(cIds, thetas, phis, relative=False, local=True, tolerance=0.1, tries=6, homed=False, newDir=True):
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
    homed : go home first or not, if true, move in the safe way
    newDir : create a new directory for data or not

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

    if newDir:
        cc.connect(False)
    dataPath = cc.runManager.dataDir
    moves = np.zeros((len(cIds), tries), dtype=moveDtype)

    nowDone = np.zeros(cc.nCobras, 'bool')
    notDoneMask = np.zeros(cc.nCobras, 'bool')
    targets = np.zeros(cc.nCobras, 'complex')
    targetThetas = np.zeros(cc.nCobras)
    targetPhis = np.zeros(cc.nCobras)
    atThetas = np.zeros(cc.nCobras)
    atPhis = np.zeros(cc.nCobras)
    notDoneMask[cIds] = True

    if relative:
        targetThetas[cIds] = thetas + cc.thetaInfo[cIds]['angle']
        targetPhis[cIds] = phis + cc.phiInfo[cIds]['angle']
    elif not local:
        targetThetas[cIds] = (thetas - cc.calibModel.tht0[cIds]) % (np.pi*2)
        targetPhis[cIds] = phis - cc.calibModel.phiIn[cIds] - np.pi
    else:
        targetThetas[cIds] = thetas
        targetPhis[cIds] = phis
    targets = cc.pfi.anglesToPositions(cc.allCobras, targetThetas, targetPhis)

    cc.cam.resetStack(f'Stack.fits')
    logger.info(f'Move theta arms to angle={np.round(np.rad2deg(targetThetas[cIds]),2)} degree')
    logger.info(f'Move phi arms to angle={np.round(np.rad2deg(targetPhis[cIds]),2)} degree')

    if homed:
        # go home for safe movement
        cobras = cc.allCobras[cIds]
        logger.info(f'Move theta arms CW and phi arms CCW to the hard stops')
        cc.moveToHome(cobras, thetaEnable=True, phiEnable=True, thetaCCW=False)

    for j in range(tries):
        cobras = cc.allCobras[notDoneMask]
        if homed and j == 0:
            # move in the safe way
            atThetas[notDoneMask], atPhis[notDoneMask] = \
                cc.moveToAngles(cobras, targetThetas[notDoneMask], targetPhis[notDoneMask], False, True, True)
        else:
            atThetas[notDoneMask], atPhis[notDoneMask] = \
                cc.moveToAngles(cobras, targetThetas[notDoneMask], targetPhis[notDoneMask], False, False, True)
        atPositions = cc.pfi.anglesToPositions(cc.allCobras, atThetas, atPhis)
        distances = np.abs(atPositions - targets)
        nowDone[distances < tolerance] = True
        newlyDone = nowDone & notDoneMask

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

    cc.cam.resetStack()
    if np.any(notDoneMask):
        logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                         f'{np.where(notDoneMask)[0]}, '
                         f'{np.round(distances[notDoneMask],2)}')

    return dataPath, atThetas[cIds], atPhis[cIds], moves

def movePositions(cIds, targets, tolerance=0.1, tries=6, thetaMarginCCW=0.1, homed=False, newDir=True):
    """
    move cobras to the target positions

    Parameters
    ----
    cIds : index for the active cobras
    targets : target positions
    tolerance : tolerance for target positions in pixels
    thetaMarginCCW : the minimum theta angles to the CCW hard stops
    homed : go home first or not, if true, move in the safe way
    newDir : create a new directory for data or not

    Returns
    ----
    A tuple with two elements:
    - dataPath
    - the errors to the target angles
    - a numpy array for the moving history
    """
    if cc.getMode() != 'normal':
        raise RuntimeError('Switch to normal mode first!!!')
    if np.isscalar(targets):
        targets = np.full(len(cIds), targets)
    elif len(targets) != len(cIds):
        raise RuntimeError("number of targets must match the number of cobras")

    cobras = cc.allCobras[cIds]
    thetas, phis, flags = cc.pfi.positionsToAngles(cobras, targets)
    valid = (flags[:,0] & cc.pfi.SOLUTION_OK) != 0
    if not np.all(valid):
        raise RuntimeError(f"Given positions are invalid: {np.where(valid)[0]}")

    # adjust theta angles that is too closed to the CCW hard stops
    thetas[thetas < thetaMarginCCW] += np.pi*2

    dataPath, atThetas, atPhis, moves = \
        moveThetaPhi(cIds, thetas[:,0], phis[:,0], False, True, tolerance, tries, homed, newDir)

    return dataPath, cc.pfi.anglesToPositions(cobras, atThetas, atPhis), moves

def convergenceTest(cIds, runs=3, thetaMargin=np.deg2rad(10.0), phiMargin=np.deg2rad(20.0), tries=8, tolerance=0.2):
    """ convergence test """
    cc.connect(False)
    targets = np.zeros((runs, len(cIds), 2))
    moves = np.zeros((runs, len(cIds), tries), dtype=moveDtype)
    positions = np.zeros((runs, len(cIds)), dtype=complex)

    g = 0
    thetaRange = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (np.pi*2) + np.pi)[cIds]
    for i in range(runs):
        thetas = np.random.rand(len(cIds)) * (thetaRange - 2*thetaMargin) + thetaMargin
        phis = np.random.rand(len(cIds))
        phis[(cIds+g)%3==0] = phis[(cIds+g)%3==0] * (np.pi*2/3 - phiMargin) + np.pi/3
        phis[(cIds+g)%3!=0] = phis[(cIds+g)%3!=0] * (np.pi/3 - phiMargin) + phiMargin
        targets[i,:,0] = thetas
        targets[i,:,1] = phis
        positions[i] = cc.pfi.anglesToPositions(cc.allCobras[cIds], thetas, phis)
        g += 1

        logger.info(f'=== Run {i+1}: Convergence test ===')
        cc.pfi.resetMotorScaling(cc.allCobras[cIds])
        dataPath, atThetas, atPhis, moves[i,:,:] = \
            moveThetaPhi(cIds, thetas, phis, False, True, tolerance, tries, True, False)


    np.save(dataPath / 'positions', positions)
    np.save(dataPath / 'targets', targets)
    np.save(dataPath / 'moves', moves)
    return targets, moves

def makeThetaMotorMaps(newXml, steps=500, totalSteps=10000, repeat=1, fast=False, thetaOnTime=None,
                       limitOnTime=0.08, limitSteps=10000, updateGeometry=False, phiRunDir=None,
                       delta=np.deg2rad(5.0), force=False):
    """
    generate theta motor maps, it accepts custom thetaOnTIme parameter.
    it assumes that theta arms already point to the outward direction and phi arms inwards,
    also there is good geometry measurement and motor maps in the XML file.
    cobras are divided into three non-intefere groups so phi arms can be moved all way out
    if thetaOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime
    Example:
        makethetaMotorMap(xml, path, fast=True)               // update fast motor maps
        makethetaMotorMap(xml, path, fast=False)              // update slow motor maps
        makethetaMotorMap(xml, path, thetaOnTime=0.06)        // motor maps for on-time=0.06
    """
    if cc.getMode() != 'theta':
        raise RuntimeError('Not in theta mode!!!')

    # aquire data for motor maps
    dataPath, posF, posR = cc.roundTripForTheta(steps, totalSteps, repeat, fast, thetaOnTime, limitOnTime, limitSteps, force)

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
    if not cc.thetaInfoIsValid:
        maxIdx = np.argmax(angR[:,:,0], axis=1)
        cwHome = np.zeros(cc.nCobras)
        ccwHome = np.zeros(cc.nCobras)
        for m in range(cc.nCobras):
            n = maxIdx[m]
            ccwHome[m] = np.angle(posF[m,n,0] - center[m])
            cwHome[m] = np.angle(posR[m,n,0] - center[m])
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

    return cc.runManager.runDir, np.where(np.logical_or(bad, mmBad))[0]

def makePhiMotorMaps(newXml, steps=250, totalSteps=5000, repeat=1, fast=False, phiOnTime=None,
                     limitOnTime=0.05, limitSteps=5000, delta=np.deg2rad(5.0)):
    """ generate phi motor maps, it accepts custom phiOnTIme parameter.
        it assumes that theta arms have been move to up/down positions to avoid collision
        if phiOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime

        Example:
            makePhiMotorMap(xml, path, fast=True)             // update fast motor maps
            makePhiMotorMap(xml, path, fast=False)            // update slow motor maps
            makePhiMotorMap(xml, path, phiOnTime=0.06)        // motor maps for on-time=0.06
    """
    if cc.getMode() != 'phi':
        raise RuntimeError('Not in phi mode!!!')

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

    # update phi geometry
    if not cc.phiInfoIsValid:
        maxIdx = np.argmax(angR[:,:,0], axis=1)
        cwHome = np.zeros(cc.nCobras)
        ccwHome = np.zeros(cc.nCobras)
        for m in range(cc.nCobras):
            n = maxIdx[m]
            ccwHome[m] = np.angle(posF[m,n,0] - center[m])
            cwHome[m] = np.angle(posR[m,n,0] - center[m])
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

    return cc.runManager.runDir, np.where(np.logical_or(bad, mmBad))[0]

def updateXML(newXml, thetaRunDir, phiRunDir):
    """ update Cobras geometry """
    if isinstance(thetaRunDir, str):
        thetaRunDir = pathlib.Path(thetaRunDir)
    if isinstance(phiRunDir, str):
        phiRunDir = pathlib.Path(phiRunDir)

    thetaCenter = np.load(thetaRunDir / 'data' / 'thetaCenter.npy')
    thetaRadius = np.load(thetaRunDir / 'data' / 'thetaRadius.npy')
    thetaPosF = np.load(thetaRunDir / 'data' / 'thetaFW.npy')
    thetaPosR = np.load(thetaRunDir / 'data' / 'thetaRV.npy')

    phiCenter = np.load(phiRunDir / 'data' / 'phiCenter.npy')
    phiRadius = np.load(phiRunDir / 'data' / 'phiRadius.npy')
    phiPosF = np.load(phiRunDir / 'data' / 'phiFW.npy')
    phiPosR = np.load(phiRunDir / 'data' / 'phiRV.npy')

    thetaL, phiL, thetaCCW, thetaCW, phiCCW, phiCW = cal.geometry(cc.goodIdx,
        thetaCenter, thetaRadius, thetaPosF, thetaPosR,
        phiCenter, phiRadius, phiPosF, phiPosR)

    cc.calibModel.updateGeometry(thetaCenter, thetaL, phiL)
    cc.calibModel.updateThetaHardStops(thetaCCW, thetaCW)
    cc.calibModel.updatePhiHardStops(phiCCW, phiCW)

    cc.calibModel.createCalibrationFile(cc.runManager.outputDir / newXml)

    return cc.runManager.outputDir / newXml

def searchOnTime(speed, sData, tData):
    """ There should be some better ways to do!!! """
    onTime = np.zeros(cc.nCobras)

    for c in cc.goodIdx:
        s = sData[:,c]
        t = tData[:,c]
        model = SpeedModel()
        err = model.buildModel(s, t)

        if err:
            logger.warn(f'Building model failed #{c+1}, set to max value')
            onTime[c] = np.max(t)
        else:
            onTime[c] = model.toOntime(speed)
            if not np.isfinite(onTime[c]):
                logger.warn(f'Curve fitting failed #{c+1}, set to median value')
                onTime[c] = np.median(t)

    return onTime

def thetaOnTimeSearch(newXml, speeds=(0.06,0.12), steps=(1000,500), iteration=3, repeat=1):
    """ search the on-time parameters for specified motor speeds """
    onTimeHigh = 0.08
    onTimeLow = 0.015
    onTimeHighSteps = 200
    speedLow = np.deg2rad(0.02)

    if iteration < 3:
        logger.warn(f'Change iteration parameter from {iteration} to 3!')
        iteration = 3
    if np.isscalar(speeds) or len(speeds) != 2:
        raise ValueError(f'speeds parameter must be a tuple with two values: {speeds}')
    if speeds[0] > speeds[1]:
        speeds = speeds[1], speeds[0]
    speeds = np.deg2rad(speeds)
    if np.isscalar(steps) or len(steps) != 2:
        raise ValueError(f'steps parameter must be a tuple with two values:s: {steps}')
    if steps[0] < steps[1]:
        steps = steps[1], steps[0]

    slopeF = np.zeros(cc.nCobras)
    slopeR = np.zeros(cc.nCobras)
    ontF = np.zeros(cc.nCobras)
    ontR = np.zeros(cc.nCobras)
    _ontF = []
    _ontR = []
    _spdF = []
    _spdR = []

    # get the average speeds for onTimeHigh, small step size since it's fast
    logger.info(f'Starting theta on-time search for speed = {np.rad2deg(speeds)}')
    logger.info(f'Initial run, onTime = {onTimeHigh}')
    runDir, duds = makeThetaMotorMaps(newXml, repeat=repeat, steps=onTimeHighSteps, thetaOnTime=onTimeHigh)
    spdF = np.load(runDir / 'data' / 'thetaSpeedFW.npy')
    spdR = np.load(runDir / 'data' / 'thetaSpeedRV.npy')

    # assume a typical value for bad cobras, sticky??
    spdF[spdF<speedLow] = speedLow
    spdR[spdR<speedLow] = speedLow

    _ontF.append(np.full(cc.nCobras, onTimeHigh))
    _ontR.append(np.full(cc.nCobras, onTimeHigh))
    _spdF.append(spdF.copy())
    _spdR.append(spdR.copy())

    for (speed, step) in zip(speeds, steps):
        logger.info(f'Search on-time for speed={np.rad2deg(speed)}.')

        # calculate on time
        for c_i in cc.goodIdx:
            ontF[c_i] = thetaModel.getOntimeFromData(speed, _spdF[0][c_i], onTimeHigh)
            ontR[c_i] = thetaModel.getOntimeFromData(speed, _spdR[0][c_i], onTimeHigh)

        for n in range(iteration):
            ontF[ontF>onTimeHigh] = onTimeHigh
            ontR[ontR>onTimeHigh] = onTimeHigh
            ontF[ontF<onTimeLow] = onTimeLow
            ontR[ontR<onTimeLow] = onTimeLow
            logger.info(f'Run {n+1}/{iteration}, onTime = {np.round([ontF, ontR],4)}')
            runDir, duds = makeThetaMotorMaps(newXml, repeat=repeat, steps=step, thetaOnTime=[ontF, ontR])
            spdF = np.load(runDir / 'data' / 'thetaSpeedFW.npy')
            spdR = np.load(runDir / 'data' / 'thetaSpeedRV.npy')
            _ontF.append(ontF.copy())
            _ontR.append(ontR.copy())
            _spdF.append(spdF.copy())
            _spdR.append(spdR.copy())

            # try the same on-time again for bad measuement
            spdF[spdF<=0.0] = speed
            spdR[spdR<=0.0] = speed

            # calculate on time
            for c_i in cc.goodIdx:
                ontF[c_i] = thetaModel.getOntimeFromData(speed, spdF[c_i], ontF[c_i])
                ontR[c_i] = thetaModel.getOntimeFromData(speed, spdR[c_i], ontR[c_i])

    # try to find best on time, maybe.....
    ontF = searchOnTime(speeds[0], np.array(_spdF), np.array(_ontF))
    ontR = searchOnTime(speeds[0], np.array(_spdR), np.array(_ontR))
    ontF[ontF>onTimeHigh] = onTimeHigh
    ontR[ontR>onTimeHigh] = onTimeHigh

    # build motor maps
    logger.info(f'Build slow motor maps, best onTime = {np.round([ontF, ontR],4)}')
    runDir, duds = makeThetaMotorMaps(newXml, repeat=repeat, steps=steps[0], thetaOnTime=[ontF, ontR], fast=False)
    cc.pfi.loadModel(runDir / 'output' / newXml)

    # for fast on time
    ontF = searchOnTime(speeds[1], np.array(_spdF), np.array(_ontF))
    ontR = searchOnTime(speeds[1], np.array(_spdR), np.array(_ontR))
    ontF[ontF>onTimeHigh] = onTimeHigh
    ontR[ontR>onTimeHigh] = onTimeHigh

    # build motor maps
    logger.info(f'Build fast motor maps, best onTime = {np.round([ontF, ontR],4)}')
    runDir, duds = makeThetaMotorMaps(newXml, repeat=repeat, steps=steps[1], thetaOnTime=[ontF, ontR], fast=True)
    cc.pfi.loadModel(runDir / 'output' / newXml)

    return runDir / 'output' / newXml

def phiOnTimeSearch(newXml, speeds=(0.06,0.12), steps=(500,250), iteration=3, repeat=1):
    """ search the on time parameters for a specified motor speed """
    onTimeHigh = 0.05
    onTimeLow = 0.01
    onTimeHighSteps = 100
    speedLow = np.deg2rad(0.02)

    if iteration < 3:
        logger.warn(f'Change iteration parameter from {iteration} to 3!')
        iteration = 3
    if np.isscalar(speeds) or len(speeds) != 2:
        raise ValueError(f'speeds parameter should be a tuple with two values: {speeds}')
    if speeds[0] > speeds[1]:
        speeds = speeds[1], speeds[0]
    speeds = np.deg2rad(speeds)
    if np.isscalar(steps) or len(steps) != 2:
        raise ValueError(f'steps parameter should be a tuple with two values: {steps}')
    if steps[0] < steps[1]:
        steps = steps[1], steps[0]

    slopeF = np.zeros(cc.nCobras)
    slopeR = np.zeros(cc.nCobras)
    ontF = np.zeros(cc.nCobras)
    ontR = np.zeros(cc.nCobras)
    _ontF = []
    _ontR = []
    _spdF = []
    _spdR = []

    # get the average speeds for onTimeHigh, small step size since it's fast
    logger.info(f'Starting phi on-time search for speed = {np.rad2deg(speeds)}')
    logger.info(f'Initial run, onTime = {onTimeHigh}')
    runDir, duds = makePhiMotorMaps(newXml, repeat=repeat, steps=onTimeHighSteps, phiOnTime=onTimeHigh)
    spdF = np.load(runDir / 'data' / 'phiSpeedFW.npy')
    spdR = np.load(runDir / 'data' / 'phiSpeedRV.npy')

    # assume a typical value for bad cobras, sticky??
    spdF[spdF<speedLow] = speedLow
    spdR[spdR<speedLow] = speedLow

    _ontF.append(np.full(cc.nCobras, onTimeHigh))
    _ontR.append(np.full(cc.nCobras, onTimeHigh))
    _spdF.append(spdF.copy())
    _spdR.append(spdR.copy())

    for (speed, step) in zip(speeds, steps):
        logger.info(f'Search on-time for speed={np.rad2deg(speed)}.')

        # calculate on time
        for c_i in cc.goodIdx:
            ontF[c_i] = phiModel.getOntimeFromData(speed, _spdF[0][c_i], onTimeHigh)
            ontR[c_i] = phiModel.getOntimeFromData(speed, _spdR[0][c_i], onTimeHigh)

        for n in range(iteration):
            ontF[ontF>onTimeHigh] = onTimeHigh
            ontR[ontR>onTimeHigh] = onTimeHigh
            ontF[ontF<onTimeLow] = onTimeLow
            ontR[ontR<onTimeLow] = onTimeLow
            logger.info(f'Run {n+1}/{iteration}, onTime = {np.round([ontF, ontR],4)}')
            runDir, duds = makePhiMotorMaps(newXml, repeat=repeat, steps=step, phiOnTime=[ontF, ontR])
            spdF = np.load(runDir / 'data' / 'phiSpeedFW.npy')
            spdR = np.load(runDir / 'data' / 'phiSpeedRV.npy')
            _ontF.append(ontF.copy())
            _ontR.append(ontR.copy())
            _spdF.append(spdF.copy())
            _spdR.append(spdR.copy())

            # try the same on-time again for bad measuement
            spdF[spdF<=0.0] = speed
            spdR[spdR<=0.0] = speed

            # calculate on time
            for c_i in cc.goodIdx:
                ontF[c_i] = phiModel.getOntimeFromData(speed, spdF[c_i], ontF[c_i])
                ontR[c_i] = phiModel.getOntimeFromData(speed, spdR[c_i], ontR[c_i])

    # try to find best on time, maybe.....
    ontF = searchOnTime(speeds[0], np.array(_spdF), np.array(_ontF))
    ontR = searchOnTime(speeds[0], np.array(_spdR), np.array(_ontR))
    ontF[ontF>onTimeHigh] = onTimeHigh
    ontR[ontR>onTimeHigh] = onTimeHigh

    # build slow motor maps
    logger.info(f'Build slow motor maps, best onTime = {np.round([ontF, ontR],4)}')
    runDir, duds = makePhiMotorMaps(newXml, repeat=repeat, steps=steps[0], phiOnTime=[ontF, ontR], fast=False)
    cc.pfi.loadModel(runDir / 'output' / newXml)

    # for fast motor maps
    ontF = searchOnTime(speeds[1], np.array(_spdF), np.array(_ontF))
    ontR = searchOnTime(speeds[1], np.array(_spdR), np.array(_ontR))
    ontF[ontF>onTimeHigh] = onTimeHigh
    ontR[ontR>onTimeHigh] = onTimeHigh

    # build motor maps
    logger.info(f'Build fast motor maps, best onTime = {np.round([ontF, ontR],4)}')
    runDir, duds = makePhiMotorMaps(newXml, repeat=repeat, steps=steps[1], phiOnTime=[ontF, ontR], fast=True)
    cc.pfi.loadModel(runDir / 'output' / newXml)

    return runDir / 'output' / newXml

def convertXML(newXml):
    """ convert old XML to a new coordinate by taking 'phi homed' images
        assuming the cobra module is in horizontal setup
    """
    cc.connect(False)

    idx = cc.visibleIdx
    idx1 = idx[idx <= cc.camSplit]
    idx2 = idx[idx > cc.camSplit]
    oldPos = cc.calibModel.centers
    newPos = np.zeros(cc.nCobras, dtype=complex)

    # move phi to home and measure new positions
    cc.moveToHome(cc.allCobras, thetaEnable=False, phiEnable=True)
    data, filename, bkgd = cc.cam.expose()
    centroids = np.flip(np.sort_complex(data['x']+data['y']*1j))
    newPos[idx1] = centroids[:len(idx1)]
    newPos[idx2] = centroids[-len(idx2):]

    # calculation tranformation
    offset1, scale1, tilt1, convert1 = cal.transform(oldPos[idx1], newPos[idx1])
    offset2, scale2, tilt2, convert2 = cal.transform(oldPos[idx2], newPos[idx2])

    split = cc.camSplit + 1
    old = cc.calibModel
    new = deepcopy(old)
    new.centers[:split] = convert1(old.centers[:split])
    new.tht0[:split] = (old.tht0[:split]+tilt1) % (2*np.pi)
    new.tht1[:split] = (old.tht1[:split]+tilt1) % (2*np.pi)
    new.L1[:split] = old.L1[:split] * scale1
    new.L2[:split] = old.L2[:split] * scale1
    new.centers[split:] = convert2(old.centers[split:])
    new.tht0[split:] = (old.tht0[split:]+tilt2) % (2*np.pi)
    new.tht1[split:] = (old.tht1[split:]+tilt2) % (2*np.pi)
    new.L1[split:] = old.L1[split:] * scale2
    new.L2[split:] = old.L2[split:] * scale2

    # create a new XML file
    old.updateGeometry(new.centers, new.L1, new.L2)
    old.updateThetaHardStops(new.tht0, new.tht1)

    fn = cc.runManager.outputDir / newXml
    old.createCalibrationFile(fn)
    return fn

def convertXML1(newXml):
    """ convert old XML to a new coordinate by taking 'phi homed' images
        assuming the cobra module is in horizontal setup
    """
    cc.connect(False)

    idx = cc.visibleIdx
    oldPos = cc.calibModel.centers
    newPos = np.zeros(cc.nCobras, dtype=complex)

    # go home and measure new positions
    cc.moveToHome(cc.allCobras, thetaEnable=False, phiEnable=True)
    data, filename, bkgd = cc.cam.expose()
    home = np.array(sorted([(c['x'], c['y']) for c in data], key=lambda t: t[0], reverse=True))
    newPos[idx] = home[:len(idx), 0] + home[:len(idx), 1] * (1j)

    # calculation tranformation
    offset, scale, tilt, convert = cal.transform(oldPos[idx], newPos[idx])

    old = cc.calibModel
    new = deepcopy(old)
    new.centers[:] = convert(old.centers)
    new.tht0[:] = (old.tht0 + tilt) % (2*np.pi)
    new.tht1[:] = (old.tht1 + tilt) % (2*np.pi)
    new.L1[:] = old.L1*scale
    new.L2[:] = old.L2*scale

    # create a new XML file
    old.updateGeometry(new.centers, new.L1, new.L2)
    old.updateThetaHardStops(new.tht0, new.tht1)

    fn = cc.runManager.outputDir / newXml
    old.createCalibrationFile(fn)
    return fn

def phiOntimeScan(cIds=None, speed=None, initOntimes=None, steps=10, totalSteps=6000, repeat=1, scaling=4.0, tolerance=np.deg2rad(1.0)):
    """
    find the on times of phi motors for a given speed

    Parameters
    ----
    cIds : index for the active cobras
    speed : motor travel distance per step
    initOntimes : initial on-times
    steps : how many steps for each move
    totalSteps : total steps
    repeat : how many runs for this process
    scaling : on-time scaling factor
    tolerace : tolerance for hard stops

    Returns
    ----
    A tuple with four elements:
    - dataPath
    - on times
    - angles
    - speeds
    """
    if cc.getMode() != 'phi':
        raise RuntimeError('Switch to phi mode first!!!')
    if not cc.phiInfoIsValid:
        raise RuntimeError('Please set phi Geometry or build a Motor-maps first!!!')
    if cIds is None:
        cIds = cc.goodIdx
    if speed is None:
        speed = cc.constantPhiSpeed
    if initOntimes is None:
        initOntimes = np.array([cc.calibModel.motorOntimeSlowFwd2[cIds], cc.calibModel.motorOntimeSlowRev2[cIds]])
    elif initOntimes.shape != (2, len(cIds)):
        raise RuntimeError("shape of initOntimes must match (2, number of cobras)")

    cc.connect(False)
    dataPath = cc.runManager.dataDir
    tries = int(totalSteps / steps)
    ontimes = np.zeros((len(cIds), repeat, 2, tries+1))
    angles = np.zeros((len(cIds), repeat, 2, tries+1))
    speeds = np.zeros((len(cIds), repeat, 2, tries+1))

    nowDone = np.zeros(cc.nCobras, 'bool')
    notDoneMask = np.zeros(cc.nCobras, 'bool')
    limitAngles = (cc.phiInfo['cwHome'] - cc.phiInfo['ccwHome']) % (np.pi*2)

    logger.info(f'Move phi to CCW home and reset motor scaling')
    cc.pfi.resetMotorScaling()
    cc.moveToHome(cc.allCobras[cIds], thetaEnable=False, phiEnable=True)

    for r in range(repeat):
        nowDone[:] = False
        notDoneMask[:] = False
        notDoneMask[cIds] = True

        cc.cam.resetStack(f'phiOntimeScanFwd{r+1}.fits')
        logger.info(f'=== Forward phi on-time scan #{r+1} ===')

        if r == 0:
            ontimes[:, r, 0, 0] = initOntimes[0]
        else:
            for ci in range(len(cIds)):
                ontimes[ci, r, 0, 0] = cal.calculateOntime(ontimes[ci,r-1,0,0], speed/speeds[ci,r-1,0,0],
                                                           scaling, cc.pfi.phiParameter, cc.pfi.maxPhiOntime)
        angles[:, r, 0, 0] = cc.phiInfo['angle'][cIds]
        oldOntimes = np.copy(cc.calibModel.motorOntimeSlowFwd2)

        for j in range(tries):
            cobras = cc.allCobras[notDoneMask]
            cc.calibModel.motorOntimeSlowFwd2[cIds] = ontimes[:, r, 0, j]
            cc.moveSteps(cobras, 0, steps)
            for ci in range(len(cIds)):
                if notDoneMask[cIds[ci]]:
                    ontimes[ci, r, 0, j] = cc.moveInfo['phiOntime'][cIds[ci]]
                    angles[ci, r, 0, j+1] = cc.phiInfo['angle'][cIds[ci]]
                    speeds[ci, r, 0, j] = cc.moveInfo['movedPhi'][cIds[ci]] / steps
                    if r == 0:
                        ontimes[ci, r, 0, j+1] = cal.calculateOntime(ontimes[ci,r,0,j], speed/speeds[ci,r,0,j],
                                                                     scaling, cc.pfi.phiParameter, cc.pfi.maxPhiOntime)
                    else:
                        k = np.argmin(np.abs(angles[ci,r-1,0,:] - angles[ci,r,0,j+1]))
                        if speeds[ci,r-1,0,k] == 0:
                            k = k - 1
                        ontimes[ci, r, 0, j+1] = cal.calculateOntime(ontimes[ci,r-1,0,k], speed/speeds[ci,r-1,0,k],
                                                                     scaling, cc.pfi.phiParameter, cc.pfi.maxPhiOntime)

            nowDone[cc.phiInfo['angle'] > limitAngles - tolerance] = True
            newlyDone = nowDone & notDoneMask

            if np.any(newlyDone):
                notDoneMask &= ~newlyDone
                logger.info(f'done: {np.where(newlyDone)[0]}, {(notDoneMask == True).sum()} left')
            if not np.any(notDoneMask):
                logger.info(f'all cobras reach CW limits')
                break

        cc.cam.resetStack()
        cc.calibModel.motorOntimeSlowFwd2[:] = oldOntimes
        if np.any(notDoneMask):
            logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                        f'{np.where(notDoneMask)[0]}, ')

        nowDone[:] = False
        notDoneMask[:] = False
        notDoneMask[cIds] = True

        cc.cam.resetStack(f'phiOntimeScanRev{r+1}.fits')
        logger.info(f'=== Reverse phi on-time scan #{r+1} ===')

        if r == 0:
            ontimes[:, r, 1, 0] = initOntimes[1]
        else:
            for ci in range(len(cIds)):
                ontimes[ci, r, 1, 0] = cal.calculateOntime(ontimes[ci,r-1,1,0], -speed/speeds[ci,r-1,1,0],
                                                           scaling, cc.pfi.phiParameter, cc.pfi.maxPhiOntime)
        angles[:, r, 1, 0] = cc.phiInfo['angle'][cIds]
        oldOntimes = np.copy(cc.calibModel.motorOntimeSlowRev2)

        for j in range(tries):
            cobras = cc.allCobras[notDoneMask]
            cc.calibModel.motorOntimeSlowRev2[cIds] = ontimes[:, r, 1, j]
            cc.moveSteps(cobras, 0, -steps)
            for ci in range(len(cIds)):
                if notDoneMask[cIds[ci]]:
                    ontimes[ci, r, 1, j] = cc.moveInfo['phiOntime'][cIds[ci]]
                    angles[ci, r, 1, j+1] = cc.phiInfo['angle'][cIds[ci]]
                    speeds[ci, r, 1, j] = cc.moveInfo['movedPhi'][cIds[ci]] / steps
                    if r == 0:
                        ontimes[ci, r, 1, j+1] = cal.calculateOntime(ontimes[ci,r,1,j], -speed/speeds[ci,r,1,j],
                                                                     scaling, cc.pfi.phiParameter, cc.pfi.maxPhiOntime)
                    else:
                        k = np.argmin(np.abs(angles[ci,r-1,1,:] - angles[ci,r,1,j+1]))
                        if speeds[ci,r-1,1,k] == 0:
                            k = k - 1
                        ontimes[ci, r, 1, j+1] = cal.calculateOntime(ontimes[ci,r-1,1,k], -speed/speeds[ci,r-1,1,k],
                                                                     scaling, cc.pfi.phiParameter, cc.pfi.maxPhiOntime)

            nowDone[cc.phiInfo['angle'] < tolerance] = True
            newlyDone = nowDone & notDoneMask

            if np.any(newlyDone):
                notDoneMask &= ~newlyDone
                logger.info(f'done: {np.where(newlyDone)[0]}, {(notDoneMask == True).sum()} left')
            if not np.any(notDoneMask):
                logger.info(f'all cobras reach CCW limits')
                break

        cc.cam.resetStack()
        cc.calibModel.motorOntimeSlowRev2[:] = oldOntimes
        if np.any(notDoneMask):
            logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                        f'{np.where(notDoneMask)[0]}, ')

    np.save(dataPath / 'ontimes', ontimes)
    np.save(dataPath / 'angles', angles)
    np.save(dataPath / 'speeds', speeds)
    np.save(dataPath / 'cobras', cIds)
    np.save(dataPath / 'parameters', [speed, steps, scaling, tolerance])

    return dataPath, ontimes, angles, speeds

def thetaOntimeScan(cIds=None, speed=None, initOntimes=None, steps=10, totalSteps=10000, repeat=1, scaling=4.0, tolerance=np.deg2rad(1.0)):
    """
    find the on times of theta motors for a given speed

    Parameters
    ----
    cIds : index for the active cobras
    speed : motor travel distance per step
    initOntimes : initial on-times
    steps : how many steps for each move
    totalSteps : total steps
    repeat : how many runs for this process
    scaling : on-time scaling factor
    tolerace : tolerance for hard stops

    Returns
    ----
    A tuple with four elements:
    - dataPath
    - on times
    - angles
    - speeds
    """
    if cc.getMode() != 'theta':
        raise RuntimeError('Switch to theta mode first!!!')
    if not cc.thetaInfoIsValid:
        raise RuntimeError('Please set theta Geometry or build a Motor-maps first!!!')
    if cIds is None:
        cIds = cc.goodIdx
    if speed is None:
        speed = cc.constantThetaSpeed
    if initOntimes is None:
        initOntimes = np.array([cc.calibModel.motorOntimeSlowFwd1[cIds], cc.calibModel.motorOntimeSlowRev1[cIds]])
    elif initOntimes.shape != (2, len(cIds)):
        raise RuntimeError("Shape of initOntimes must match (2, number of cobras)")

    cc.connect(False)
    dataPath = cc.runManager.dataDir
    tries = int(totalSteps / steps)
    ontimes = np.zeros((len(cIds), repeat, 2, tries+1))
    angles = np.zeros((len(cIds), repeat, 2, tries+1))
    speeds = np.zeros((len(cIds), repeat, 2, tries+1))

    nowDone = np.zeros(cc.nCobras, 'bool')
    notDoneMask = np.zeros(cc.nCobras, 'bool')
    limitAngles = (cc.thetaInfo['cwHome'] - cc.thetaInfo['ccwHome'] + np.pi) % (np.pi*2) + np.pi

    logger.info(f'Move theta to CCW home and reset motor scaling')
    cc.pfi.resetMotorScaling()
    cc.moveToHome(cc.allCobras[cIds], thetaEnable=True, phiEnable=False)

    for r in range(repeat):
        nowDone[:] = False
        notDoneMask[:] = False
        notDoneMask[cIds] = True

        cc.cam.resetStack(f'thetaOntimeScanFwd{r+1}.fits')
        logger.info(f'=== Forward theta on-time scan #{r+1} ===')

        if r == 0:
            ontimes[:, r, 0, 0] = initOntimes[0]
        else:
            for ci in range(len(cIds)):
                ontimes[ci, r, 0, 0] = cal.calculateOntime(ontimes[ci,r-1,0,0], speed/speeds[ci,r-1,0,0],
                                                           scaling, cc.pfi.thetaParameter, cc.pfi.maxThetaOntime)
        angles[:, r, 0, 0] = cc.thetaInfo['angle'][cIds]
        oldOntimes = np.copy(cc.calibModel.motorOntimeSlowFwd1)

        for j in range(tries):
            cobras = cc.allCobras[notDoneMask]
            cc.calibModel.motorOntimeSlowFwd1[cIds] = ontimes[:, r, 0, j]
            cc.moveSteps(cobras, steps, 0)
            for ci in range(len(cIds)):
                if notDoneMask[cIds[ci]]:
                    ontimes[ci, r, 0, j] = cc.moveInfo['thetaOntime'][cIds[ci]]
                    angles[ci, r, 0, j+1] = cc.thetaInfo['angle'][cIds[ci]]
                    speeds[ci, r, 0, j] = cc.moveInfo['movedTheta'][cIds[ci]] / steps
                    if r == 0:
                        ontimes[ci, r, 0, j+1] = cal.calculateOntime(ontimes[ci,r,0,j], speed/speeds[ci,r,0,j],
                                                                     scaling, cc.pfi.thetaParameter, cc.pfi.maxThetaOntime)
                    else:
                        k = np.argmin(np.abs(angles[ci,r-1,0,:] - angles[ci,r,0,j+1]))
                        if speeds[ci,r-1,0,k] == 0:
                            k = k - 1
                        ontimes[ci, r, 0, j+1] = cal.calculateOntime(ontimes[ci,r-1,0,k], speed/speeds[ci,r-1,0,k],
                                                                     scaling, cc.pfi.thetaParameter, cc.pfi.maxThetaOntime)

            nowDone[cc.thetaInfo['angle'] > limitAngles - tolerance] = True
            newlyDone = nowDone & notDoneMask

            if np.any(newlyDone):
                notDoneMask &= ~newlyDone
                logger.info(f'done: {np.where(newlyDone)[0]}, {(notDoneMask == True).sum()} left')
            if not np.any(notDoneMask):
                logger.info(f'all cobras reach CW limits')
                break

        cc.cam.resetStack()
        cc.calibModel.motorOntimeSlowFwd1[:] = oldOntimes
        if np.any(notDoneMask):
            logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                        f'{np.where(notDoneMask)[0]}, ')

        nowDone[:] = False
        notDoneMask[:] = False
        notDoneMask[cIds] = True

        cc.cam.resetStack(f'thetaOntimeScanRev{r+1}.fits')
        logger.info(f'=== Reverse theta on-time scan #{r+1} ===')

        if r == 0:
            ontimes[:, r, 1, 0] = initOntimes[1]
        else:
            for ci in range(len(cIds)):
                ontimes[ci, r, 1, 0] = cal.calculateOntime(ontimes[ci,r-1,1,0], -speed/speeds[ci,r-1,1,0],
                                                           scaling, cc.pfi.thetaParameter, cc.pfi.maxThetaOntime)
        angles[:, r, 1, 0] = cc.thetaInfo['angle'][cIds]
        oldOntimes = np.copy(cc.calibModel.motorOntimeSlowRev1)

        for j in range(tries):
            cobras = cc.allCobras[notDoneMask]
            cc.calibModel.motorOntimeSlowRev1[cIds] = ontimes[:, r, 1, j]
            cc.moveSteps(cobras, -steps, 0)
            for ci in range(len(cIds)):
                if notDoneMask[cIds[ci]]:
                    ontimes[ci, r, 1, j] = cc.moveInfo['thetaOntime'][cIds[ci]]
                    angles[ci, r, 1, j+1] = cc.thetaInfo['angle'][cIds[ci]]
                    speeds[ci, r, 1, j] = cc.moveInfo['movedTheta'][cIds[ci]] / steps
                    if r == 0:
                        ontimes[ci, r, 1, j+1] = cal.calculateOntime(ontimes[ci,r,1,j], -speed/speeds[ci,r,1,j],
                                                                     scaling, cc.pfi.thetaParameter, cc.pfi.maxThetaOntime)
                    else:
                        k = np.argmin(np.abs(angles[ci,r-1,1,:] - angles[ci,r,1,j+1]))
                        if speeds[ci,r-1,1,k] == 0:
                            k = k - 1
                        ontimes[ci, r, 1, j+1] = cal.calculateOntime(ontimes[ci,r-1,1,k], -speed/speeds[ci,r-1,1,k],
                                                                     scaling, cc.pfi.thetaParameter, cc.pfi.maxThetaOntime)

            nowDone[cc.thetaInfo['angle'] < tolerance] = True
            newlyDone = nowDone & notDoneMask

            if np.any(newlyDone):
                notDoneMask &= ~newlyDone
                logger.info(f'done: {np.where(newlyDone)[0]}, {(notDoneMask == True).sum()} left')
            if not np.any(notDoneMask):
                logger.info(f'all cobras reach CCW limits')
                break

        cc.cam.resetStack()
        cc.calibModel.motorOntimeSlowRev1[:] = oldOntimes
        if np.any(notDoneMask):
            logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                        f'{np.where(notDoneMask)[0]}, ')

    np.save(dataPath / 'ontimes', ontimes)
    np.save(dataPath / 'angles', angles)
    np.save(dataPath / 'speeds', speeds)
    np.save(dataPath / 'cobras', cIds)
    np.save(dataPath / 'parameters', [speed, steps, scaling, tolerance])

    return dataPath, ontimes, angles, speeds


def convergenceTest2(cIds, runs=3, thetaMargin=np.deg2rad(10.0), phiMargin=np.deg2rad(20.0), tries=8, tolerance=0.1, thetaThreshold=0.2, phiThreshold=0.2):
    """ convergence test """
    cc.connect(False)
    targets = np.zeros((runs, len(cIds), 2))
    moves = np.zeros((runs, len(cIds), tries), dtype=moveDtype)
    positions = np.zeros((runs, len(cIds)), dtype=complex)

    g = 0
    thetaRange = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (np.pi*2) + np.pi)[cIds]
    for i in range(runs):
        thetas = np.random.rand(len(cIds)) * (thetaRange - 2*thetaMargin) + thetaMargin
        phis = np.random.rand(len(cIds))
        phis[(cIds+g)%3==0] = phis[(cIds+g)%3==0] * (np.pi*2/3 - phiMargin) + np.pi/3
        phis[(cIds+g)%3!=0] = phis[(cIds+g)%3!=0] * (np.pi/3 - phiMargin) + phiMargin
        targets[i,:,0] = thetas
        targets[i,:,1] = phis
        positions[i] = cc.pfi.anglesToPositions(cc.allCobras[cIds], thetas, phis)
        g += 1

        logger.info(f'=== Run {i+1}: Convergence test ===')
        cc.pfi.resetMotorScaling(cc.allCobras[cIds])
        dataPath, atThetas, atPhis, moves[i,:,:] = \
            moveThetaPhi2(cIds, thetas, phis, False, True, tolerance, tries, True, thetaThreshold, phiThreshold, False)


    np.save(dataPath / 'positions', positions)
    np.save(dataPath / 'targets', targets)
    np.save(dataPath / 'moves', moves)
    return targets, moves

def convergenceTest3(cIds, runs=3, thetaMargin=np.deg2rad(20.0), phiMargin=np.deg2rad(20.0), tries=8, tolerance=0.1, thetaThreshold=0.2, phiThreshold=0.2):
    """ convergence test """
    cc.connect(False)
    targets = np.zeros((runs, len(cIds), 2))
    moves = np.zeros((runs, len(cIds), tries), dtype=moveDtype)
    positions = np.zeros((runs, len(cIds)), dtype=complex)

    thetaRange = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (np.pi*2) + np.pi)[cIds]
    for i in range(runs):
        thetas = np.random.rand(len(cIds)) * (thetaRange - 2*thetaMargin) + thetaMargin
        phis = np.random.rand(len(cIds)) * (np.pi/3 - phiMargin) + phiMargin
        targets[i,:,0] = thetas
        targets[i,:,1] = phis
        positions[i] = cc.pfi.anglesToPositions(cc.allCobras[cIds], thetas, phis)

        logger.info(f'=== Run {i+1}: Convergence test ===')
        cc.pfi.resetMotorScaling(cc.allCobras[cIds])
        dataPath, atThetas, atPhis, moves[i,:,:] = \
            moveThetaPhi2(cIds, thetas, phis, False, True, tolerance, tries, True, thetaThreshold, phiThreshold, False)


    np.save(dataPath / 'positions', positions)
    np.save(dataPath / 'targets', targets)
    np.save(dataPath / 'moves', moves)
    return targets, moves

def thetaConvergenceTest2(cIds, margin=15.0, runs=50, tries=8, threshold=0.1, tolerance=0.0015):
    """ theta convergence test """
    moves = np.zeros((runs, len(cIds), tries), dtype=move1Dtype)
    cc.connect(False)
#    tolerance = np.deg2rad(tolerance)

#    cc.pfi.resetMotorScaling(cc.allCobras[cIds], 'theta')
#    cc.setScaling(True)

    for i in range(runs):
        if runs > 1:
            angle = np.deg2rad(margin + (360 - 2*margin) * i / (runs - 1))
        else:
            angle = np.deg2rad(180)
        logger.info(f'Run {i+1}: angle={np.rad2deg(angle):.2f} degree')

        cc.pfi.resetMotorScaling(cc.allCobras[cIds], 'theta')
        cc.moveToHome(cc.allCobras[cIds], thetaEnable=True)
        dataPath, diffAngles, moves[i,:,:] = \
            moveThetaAngles2(cIds, angle, False, True, tolerance, tries, threshold, newDir=False)

    np.save(dataPath / 'moves', moves)
    return moves

def phiConvergenceTest2(cIds, margin=15.0, runs=50, tries=8, threshold=0.1, tolerance=0.0015):
    """ phi convergence test """
    moves = np.zeros((runs, len(cIds), tries), dtype=move1Dtype)
    cc.connect(False)
#    tolerance = np.deg2rad(tolerance)

#    cc.pfi.resetMotorScaling(cc.allCobras[cIds], 'phi')
#    cc.setScaling(False)

    for i in range(runs):
        if runs > 1:
            angle = np.deg2rad(margin + (180 - 2*margin) * i / (runs - 1))
        else:
            angle = np.deg2rad(90)
        logger.info(f'Run {i+1}: angle={np.rad2deg(angle):.2f} degree')

        cc.pfi.resetMotorScaling(cc.allCobras[cIds], 'phi')
        cc.moveToHome(cc.allCobras[cIds], phiEnable=True)
        dataPath, diffAngles, moves[i,:,:] = \
            movePhiAngles2(cIds, angle, False, True, tolerance, tries, threshold, newDir=False)

    np.save(dataPath / 'moves', moves)
    return moves

def moveThetaAngles2(cIds, thetas, relative=False, local=True, tolerance=0.0015, tries=6, threshold=0.1, newDir=True):
    """
    move theta arms to the target angles

    Parameters
    ----
    cIds : index for the active cobras
    thetas : angles to move
    relative : angles are offsets from current positions or not
    local : the angles are from the CCW hard stops or in global coordinate
    newDir : create a new directory for data or not

    Returns
    ----
    A tuple with two elements:
    - dataPath
    - errors for the theta angles
    - a numpy array for the moving history
    """
    if cc.getMode() != 'theta':
        raise RuntimeError('Switch to theta mode first!!!')
    if not cc.thetaInfoIsValid:
        raise RuntimeError('Please set theta Geometry or build a Motor-maps first!!!')
    if np.isscalar(thetas):
        thetas = np.full(len(cIds), thetas)
    elif len(thetas) != len(cIds):
        raise RuntimeError("number of theta angles must the match number of cobras")

    if newDir:
        cc.connect(False)
    dataPath = cc.runManager.dataDir
    moves = np.zeros((len(cIds), tries), dtype=move1Dtype)

    nowDone = np.zeros(cc.nCobras, 'bool')
    notDoneMask = np.zeros(cc.nCobras, 'bool')
    targets = np.zeros(cc.nCobras)
    atAngles = np.zeros(cc.nCobras)
    notDoneMask[cIds] = True

    if relative:
        targets[cIds] = thetas + cc.thetaInfo[cIds]['angle']
    elif not local:
        targets[cIds] = (thetas - cc.calibModel.tht0[cIds]) % (np.pi*2)
    else:
        targets[cIds] = thetas

    cc.cam.resetStack(f'thetaStack.fits')
    logger.info(f'Move theta arms to angle={np.round(np.rad2deg(targets[cIds]),2)} degree')

    for j in range(tries):
        cIds2 = np.where(notDoneMask)[0]
        atAngles[notDoneMask], _ = moveToAngles(cIds2, targets[notDoneMask], None, threshold, 0.0, True)

        for ci in range(len(cIds)):
            if notDoneMask[cIds[ci]]:
                moves['position'][ci, j] = cc.cobraInfo['position'][cIds[ci]]
                moves['steps'][ci, j] = cc.moveInfo['thetaSteps'][cIds[ci]]
                moves['angle'][ci, j] = cc.thetaInfo['angle'][cIds[ci]]
                moves['ontime'][ci, j] = cc.moveInfo['thetaOntime'][cIds[ci]]
                moves['fast'][ci, j] = cc.moveInfo['thetaFast'][cIds[ci]]

        diffAngles = cal.absDiffAngle(atAngles, targets)
        nowDone[diffAngles < tolerance] = True
        newlyDone = nowDone & notDoneMask

        if np.any(newlyDone):
            notDoneMask &= ~newlyDone
            logger.info(f'done: {np.where(newlyDone)[0]}, {(notDoneMask == True).sum()} left')
        if not np.any(notDoneMask):
            logger.info(f'all cobras are in positions')
            break

    cc.cam.resetStack()
    if np.any(notDoneMask):
        logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                         f'{np.where(notDoneMask)[0]}, '
                         f'{np.round(np.rad2deg(diffAngles)[notDoneMask], 2)}')

    return dataPath, cal.diffAngle(atAngles, targets)[cIds], moves

def movePhiAngles2(cIds, phis, relative=False, local=True, tolerance=0.0015, tries=6, threshold=0.1, newDir=True):
    """
    move phi arms to the target angles

    Parameters
    ----
    cIds : index for the active cobras
    phis : angles to move
    relative : angles are offsets from current positions or not
    local : the angles are from the CCW hard stops or the theta arms

    Returns
    ----
    A tuple with two elements:
    - dataPath
    - errors for the phi angles
    - a numpy array for the moving history
    """
    if cc.getMode() != 'phi':
        raise RuntimeError('Switch to phi mode first!!!')
    if not cc.phiInfoIsValid:
        raise RuntimeError('Please set phi Geometry or build a Motor-maps first!!!')
    if np.isscalar(phis):
        phis = np.full(len(cIds), phis)
    elif len(phis) != len(cIds):
        raise RuntimeError("number of phi angles must the match number of cobras")

    if newDir:
        cc.connect(False)
    dataPath = cc.runManager.dataDir
    moves = np.zeros((len(cIds), tries), dtype=move1Dtype)

    nowDone = np.zeros(cc.nCobras, 'bool')
    notDoneMask = np.zeros(cc.nCobras, 'bool')
    targets = np.zeros(cc.nCobras)
    atAngles = np.zeros(cc.nCobras)
    notDoneMask[cIds] = True

    if relative:
        targets[cIds] = phis + cc.phiInfo[cIds]['angle']
    elif not local:
        targets[cIds] = phis - cc.calibModel.phiIn[cIds] - np.pi
    else:
        targets[cIds] = phis

    cc.cam.resetStack(f'phiStack.fits')
    logger.info(f'Move phi arms to angle={np.round(np.rad2deg(targets[cIds]),2)} degree')

    for j in range(tries):
        cIds2 = np.where(notDoneMask)[0]
        atThetas, atPhis = moveToAngles(cIds2, None, targets[notDoneMask], 0.0, threshold, True)
        atAngles[notDoneMask] = atPhis

        for ci in range(len(cIds)):
            if notDoneMask[cIds[ci]]:
                moves['position'][ci, j] = cc.cobraInfo['position'][cIds[ci]]
                moves['steps'][ci, j] = cc.moveInfo['phiSteps'][cIds[ci]]
                moves['angle'][ci, j] = cc.phiInfo['angle'][cIds[ci]]
                moves['ontime'][ci, j] = cc.moveInfo['phiOntime'][cIds[ci]]
                moves['fast'][ci, j] = cc.moveInfo['phiFast'][cIds[ci]]

        diffAngles = cal.absDiffAngle(atAngles, targets)
        nowDone[diffAngles < tolerance] = True
        newlyDone = nowDone & notDoneMask

        if np.any(newlyDone):
            notDoneMask &= ~newlyDone
            logger.info(f'done: {np.where(newlyDone)[0]}, {(notDoneMask == True).sum()} left')
        if not np.any(notDoneMask):
            logger.info(f'all cobras are in positions')
            break

    cc.cam.resetStack()
    if np.any(notDoneMask):
        logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                         f'{np.where(notDoneMask)[0]}, '
                         f'{np.round(np.rad2deg(diffAngles)[notDoneMask], 2)}')

    return dataPath, cal.diffAngle(atAngles, targets)[cIds], moves

def moveDeltaAngles(cIds, thetaAngles=None, phiAngles=None, thetaThreshold=0.1, phiThreshold=0.1):
    """ move cobras by the given amount of theta and phi angles """
    cobras = cc.allCobras[cIds]
    if mmTheta is None or mmPhi is None:
        raise RuntimeError('Please set on-time maps first!')

    if cc.cobraInfo['position'][0] == 0.0:
        raise RuntimeError('Last position is unkown! Run moveToHome or setCurrentAngles')

    if cc.mode == cc.thetaMode:
        if phiAngles is not None:
            raise RuntimeError('Move phi arms in theta mode!')
        if not cc.thetaInfoIsValid:
            raise RuntimeError('Please set theta geometry first!')
    elif cc.mode == cc.phiMode:
        if thetaAngles is not None:
            raise RuntimeError('Move theta arms in phi mode!')
        if not cc.phiInfoIsValid:
            raise RuntimeError('Please set phi geometry first!')
    if thetaAngles is None and phiAngles is None:
        logger.info('both theta and phi angles are None, not moving!')
        return

    if thetaAngles is not None:
        if np.isscalar(thetaAngles):
            thetaAngles = np.full(len(cobras), thetaAngles)
        elif len(thetaAngles) != len(cobras):
            raise RuntimeError("number of theta angles must match number of cobras")
    else:
        thetaAngles = np.zeros(len(cobras))
    if phiAngles is not None:
        if np.isscalar(phiAngles):
            phiAngles = np.full(len(cobras), phiAngles)
        elif len(phiAngles) != len(cobras):
            raise RuntimeError("number of phi angles must match number of cobras")
    else:
        phiAngles = np.zeros(len(cobras))

    # get last theta/phi angles
    fromTheta = np.zeros(len(cobras), 'float')
    fromPhi = np.zeros(len(cobras), 'float')

    if cc.mode == cc.thetaMode:
        badFromThetaIdx = cIds[np.isnan(cc.thetaInfo['angle'][cIds])]
        if len(badFromThetaIdx) > 0:
            logger.warn(f'Last theta angle is unknown: {badFromThetaIdx}')

        fromPhi[:] = 0
        for c_i in range(len(cobras)):
            cId = cIds[c_i]
            fromTheta[c_i] = cc.thetaInfo['angle'][cId]
            if np.isnan(fromTheta[c_i]):
                # well, assume in the CCW or CW hard stop to calculate steps
                if thetaAngles[c_i] >= 0:
                    fromTheta[c_i] = 0
                else:
                    fromTheta[c_i] = (cc.calibModel.tht1[cId] - cc.calibModel.tht0[cId] + np.pi) % (np.pi*2) + np.pi

    elif cc.mode == cc.phiMode:
        badFromPhiIdx = cIds[np.isnan(cc.phiInfo['angle'][cIds])]
        if len(badFromPhiIdx) > 0:
            logger.warn(f'Last phi angle is unknown: {badFromPhiIdx}')

        fromTheta[:] = 0
        fromPhi[:] = cc.phiInfo['angle'][cIds]
        for c_i in range(len(cobras)):
            cId = cIds[c_i]
            fromPhi[c_i] = cc.phiInfo['angle'][cId]
            if np.isnan(fromPhi[c_i]):
                # well, assume in the CCW or CW hard stop to calculate steps
                if phiAngles[c_i] >= 0:
                    fromPhi[c_i] = 0
                else:
                    fromPhi[c_i] = (cc.calibModel.phiOut[cId] - cc.calibModel.phiIn[cId]) % (np.pi*2)

    else:
        # normal mode
        badFromThetaIdx = cIds[np.isnan(cc.cobraInfo['thetaAngle'][cIds])]
        if len(badFromThetaIdx) > 0:
            logger.warn(f'Last theta angle is unknown: {badFromThetaIdx}')
        badFromPhiIdx = cIds[np.isnan(cc.cobraInfo['phiAngle'][cIds])]
        if len(badFromPhiIdx) > 0:
            logger.warn(f'Last phi angle is unknown: {badFromPhiIdx}')

        for c_i in range(len(cobras)):
            cId = cIds[c_i]

            fromTheta[c_i] = cc.cobraInfo['thetaAngle'][cId]
            if np.isnan(fromTheta[c_i]):
                # well, assume in the CCW or CW hard stop to calculate steps
                if thetaAngles[c_i] >= 0:
                    fromTheta[c_i] = 0
                else:
                    fromTheta[c_i] = (cc.calibModel.tht1[cId] - cc.calibModel.tht0[cId] + np.pi) % (np.pi*2) + np.pi

            fromPhi[c_i] = cc.cobraInfo['phiAngle'][cId]
            if np.isnan(fromPhi[c_i]):
                # well, assume in the CCW or CW hard stop to calculate steps
                if phiAngles[c_i] >= 0:
                    fromPhi[c_i] = 0
                else:
                    fromPhi[c_i] = (cc.calibModel.phiOut[cId] - cc.calibModel.phiIn[cId]) % (np.pi*2)

    # calculate steps
    thetaSteps, phiSteps = cc.pfi.moveThetaPhi(cobras, thetaAngles, phiAngles, fromTheta, fromPhi, True, True, doRun=False)


    # set on-times for cobras that is close to target
    oldOntimes = np.copy(cc.calibModel.motorOntimeFwd1), np.copy(cc.calibModel.motorOntimeFwd2), \
                 np.copy(cc.calibModel.motorOntimeRev1), np.copy(cc.calibModel.motorOntimeRev2)

    for c_i in range(len(cobras)):
        cId = cIds[c_i]

        if thetaAngles[c_i] != 0 and abs(thetaAngles[c_i]) < thetaThreshold:
            angle = fromTheta[c_i] + thetaAngles[c_i] / 2.0

            if thetaAngles[c_i] > 0:
                idx = np.nanargmin(abs(mmTheta[cId,0]['angle'] - angle))
                cc.calibModel.motorOntimeFwd1[cId] = mmTheta[cId,0,idx]['ontime']
                thetaSteps[c_i] = int(thetaAngles[c_i] / mmTheta[cId,0,idx]['speed'])
            else:
                idx = np.nanargmin(abs(mmTheta[cId,1]['angle'] - angle))
                cc.calibModel.motorOntimeRev1[cId] = mmTheta[cId,1,idx]['ontime']
                thetaSteps[c_i] = -int(thetaAngles[c_i] / mmTheta[cId,1,idx]['speed'])

        if phiAngles[c_i] != 0 and abs(phiAngles[c_i]) < phiThreshold:
            angle = fromPhi[c_i] + phiAngles[c_i] / 2.0

            if phiAngles[c_i] > 0:
                idx = np.nanargmin(abs(mmPhi[cId,0]['angle'] - angle))
                cc.calibModel.motorOntimeFwd2[cId] = mmPhi[cId,0,idx]['ontime']
                phiSteps[c_i] = int(phiAngles[c_i] / mmPhi[cId,0,idx]['speed'])
            else:
                idx = np.nanargmin(abs(mmPhi[cId,1]['angle'] - angle))
                cc.calibModel.motorOntimeRev2[cId] = mmPhi[cId,1,idx]['ontime']
                phiSteps[c_i] = -int(phiAngles[c_i] / mmPhi[cId,1,idx]['speed'])

    # send move command
    cc.moveSteps(cobras, thetaSteps, phiSteps, True, True, thetaAngles, phiAngles)
    cc.calibModel.motorOntimeFwd1, cc.calibModel.motorOntimeFwd2, cc.calibModel.motorOntimeRev1, \
        cc.calibModel.motorOntimeRev2 = oldOntimes

    return cc.moveInfo['movedTheta'][cIds], cc.moveInfo['movedPhi'][cIds]

def moveThetaPhi2(cIds, thetas, phis, relative=False, local=True, tolerance=0.1, tries=6, homed=False, thetaThreshold=0.1, phiThreshold=0.1, newDir=True):
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
    homed : go home first or not, if true, move in the safe way
    newDir : create a new directory for data or not

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

    if newDir:
        cc.connect(False)
    dataPath = cc.runManager.dataDir
    moves = np.zeros((len(cIds), tries), dtype=moveDtype)

    nowDone = np.zeros(cc.nCobras, 'bool')
    notDoneMask = np.zeros(cc.nCobras, 'bool')
    targets = np.zeros(cc.nCobras, 'complex')
    targetThetas = np.zeros(cc.nCobras)
    targetPhis = np.zeros(cc.nCobras)
    atThetas = np.zeros(cc.nCobras)
    atPhis = np.zeros(cc.nCobras)
    notDoneMask[cIds] = True

    if relative:
        targetThetas[cIds] = thetas + cc.thetaInfo[cIds]['angle']
        targetPhis[cIds] = phis + cc.phiInfo[cIds]['angle']
    elif not local:
        targetThetas[cIds] = (thetas - cc.calibModel.tht0[cIds]) % (np.pi*2)
        targetPhis[cIds] = phis - cc.calibModel.phiIn[cIds] - np.pi
    else:
        targetThetas[cIds] = thetas
        targetPhis[cIds] = phis
    targets = cc.pfi.anglesToPositions(cc.allCobras, targetThetas, targetPhis)

    cc.cam.resetStack(f'Stack.fits')
    logger.info(f'Move theta arms to angle={np.round(np.rad2deg(targetThetas[cIds]),2)} degree')
    logger.info(f'Move phi arms to angle={np.round(np.rad2deg(targetPhis[cIds]),2)} degree')

    if homed:
        # go home for safe movement
        cobras = cc.allCobras[cIds]
        logger.info(f'Move theta arms CW and phi arms CCW to the hard stops')
        cc.moveToHome(cobras, thetaEnable=True, phiEnable=True, thetaCCW=False)

    for j in range(tries):
        cobras = cc.allCobras[notDoneMask]
        cIds2 = np.where(notDoneMask)[0]
        atThetas[notDoneMask], atPhis[notDoneMask] = \
            moveToAngles(cIds2, targetThetas[notDoneMask], targetPhis[notDoneMask], thetaThreshold, phiThreshold, True)

        atPositions = cc.pfi.anglesToPositions(cc.allCobras, atThetas, atPhis)
        distances = np.abs(atPositions - targets)
        nowDone[distances < tolerance] = True
        newlyDone = nowDone & notDoneMask

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

    cc.cam.resetStack()
    if np.any(notDoneMask):
        logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                         f'{np.where(notDoneMask)[0]}, '
                         f'{np.round(distances[notDoneMask],2)}')

    return dataPath, atThetas[cIds], atPhis[cIds], moves

def moveToAngles(cIds, thetaAngles=None, phiAngles=None, thetaThreshold=0.1, phiThreshold=0.1, local=True):
    """ move cobras to the given theta and phi angles
        If local is True, both theta and phi angles are measured from CCW hard stops,
        otherwise theta angles are in global coordinate and phi angles are
        measured from the phi arms.
    """
    cobras = cc.allCobras[cIds]
    if cc.mode == cc.thetaMode and phiAngles is not None:
        raise RuntimeError('Move phi arms in theta mode!')
    elif cc.mode == cc.phiMode and thetaAngles is not None:
        raise RuntimeError('Move theta arms in phi mode!')
    if thetaAngles is None and phiAngles is None:
        cc.logger.info('both theta and phi angles are None, not moving!')
        return
    if not local and cc.mode != cc.normalMode:
        raise RuntimeError('In theta/phi mode (local) must be True!')

    if thetaAngles is not None:
        if np.isscalar(thetaAngles):
            thetaAngles = np.full(len(cobras), thetaAngles)
        elif len(thetaAngles) != len(cobras):
            raise RuntimeError("number of theta angles must match number of cobras")
    if phiAngles is not None:
        if np.isscalar(phiAngles):
            phiAngles = np.full(len(cobras), phiAngles)
        elif len(phiAngles) != len(cobras):
            raise RuntimeError("number of phi angles must match number of cobras")

    # calculate theta and phi moving amount
    if thetaAngles is not None:
        if not local:
            toTheta = (thetaAngles - cc.calibModel.tht0[cIds]) % (np.pi*2)
        else:
            toTheta = thetaAngles

        if cc.mode == cc.thetaMode:
            fromTheta = cc.thetaInfo['angle'][cIds]
        else:
            fromTheta = cc.cobraInfo['thetaAngle'][cIds]

        deltaTheta = toTheta - fromTheta
        badThetaIdx = np.where(np.isnan(deltaTheta))[0]
        if len(badThetaIdx) > 0:
            logger.warn(f'Last theta angle is unknown, not moving: {cIds[badThetaIdx]}')
            deltaTheta[badThetaIdx] = 0
    else:
        deltaTheta = None

    if phiAngles is not None:
        if not local:
            toPhi = phiAngles - cc.calibModel.phiIn[cIds] - np.pi
        else:
            toPhi = phiAngles

        if cc.mode == cc.phiMode:
            fromPhi = cc.phiInfo['angle'][cIds]
        else:
            fromPhi = cc.cobraInfo['phiAngle'][cIds]

        deltaPhi = toPhi - fromPhi
        badPhiIdx = np.where(np.isnan(deltaPhi))[0]
        if len(badPhiIdx) > 0:
            logger.warn(f'Last phi angle is unknown, not moving: {cIds[badPhiIdx]}')
            deltaPhi[badPhiIdx] = 0
    else:
        deltaPhi = None

    # send the command
    moveDeltaAngles(cIds, deltaTheta, deltaPhi, thetaThreshold, phiThreshold)
    if local:
        if cc.mode == cc.thetaMode:
            return cc.thetaInfo['angle'][cIds], np.zeros(len(cobras))
        elif cc.mode == cc.phiMode:
            return np.zeros(len(cobras)), cc.phiInfo['angle'][cIds]
        else:
            return cc.cobraInfo['thetaAngle'][cIds], cc.cobraInfo['phiAngle'][cIds]
    else:
        return ((cc.cobraInfo['thetaAngle'][cIds] + cc.calibModel.tht0[cIds]) % (np.pi*2),
                cc.cobraInfo['phiAngle'][cIds] + cc.calibModel.phiIn[cIds] + np.pi)

