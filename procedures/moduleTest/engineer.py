#from procedures.moduleTest.cobraCoach import CobraCoach as cc
from procedures.moduleTest import calculus as cal
import numpy as np
import logging
from copy import deepcopy
import pathlib
from procedures.moduleTest.speedModel import SpeedModel
from procedures.moduleTest.trajectory import Trajectories

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
mmDtype = np.dtype(dict(names=['angle', 'ontime', 'speed'], formats=['f4', 'f4', 'f4']))

cc = None
mmTheta = None
mmPhi = None


def findXML(path):
    '''
    Returns the XML files in a directory
    '''

    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, '*.xml'):
                result.append(os.path.join(root, name))
    return result


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

def setConstantOntimeMode(maxSteps=2000):
    """ using constant on-time motor maps """
    cc.constantSpeedMode = False
    cc.maxTotalSteps = maxSteps

def setConstantSpeedMode(maxSegments=10, maxSteps=100):
    """ using constant speed motor maps """
    cc.constantSpeedMode = True
    cc.maxSegments = maxSegments
    cc.maxStepsPerSeg = maxSteps

def setConstantSpeedMaps(mmTheta, mmPhi, mmThetaSlow, mmPhiSlow):
    cc.mmTheta = mmTheta
    cc.mmPhi = mmPhi
    cc.mmThetaSlow = mmThetaSlow
    cc.mmPhiSlow = mmPhiSlow

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
        targets[cIds] = (thetas - cc.thetaInfo[cIds]['ccwHome']) % (np.pi*2)
    else:
        targets[cIds] = thetas

    cc.camResetStack(f'thetaStack.fits')
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

    cc.camResetStack()
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

    cc.camResetStack(f'phiStack.fits')
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

    cc.camResetStack()
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
    if not isinstance(thetaFast, bool) and len(cobras) != len(thetaFast):
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

def convergenceTest(cIds, runs=8,
                    thetaMargin=np.deg2rad(15.0), phiMargin=np.deg2rad(15.0),
                    thetaOffset=0, phiAngle=np.pi*5/6,
                    tries=8, tolerance=0.2, threshold=3.0,
                    newDir=False, twoSteps=False):
    """ convergence test, all theta arms in the same global direction  """
    cc.connect(False)
    targets = np.zeros((runs, len(cIds), 2))
    moves = np.zeros((runs, len(cIds), tries), dtype=moveDtype)
    positions = np.zeros((runs, len(cIds)), dtype=complex)
    thetaRange = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (np.pi*2) + np.pi)[cIds]
    phiRange = ((cc.calibModel.phiOut - cc.calibModel.phiIn) % (np.pi*2))[cIds]

    for i in range(runs):
        thetas = (thetaOffset + np.pi*2*i/runs - cc.calibModel.tht0[cIds]) % (np.pi*2)
        thetas[thetas < thetaMargin] += np.pi*2
        tooBig = thetas > thetaRange - thetaMargin
        thetas[tooBig] = thetaRange[tooBig] - thetaMargin

        phis = - cc.calibModel.phiIn[cIds] - np.pi + phiAngle
        phis[phis < 0] = 0
        tooBig = phis > phiRange - phiMargin
        phis[tooBig] = phiRange[tooBig] - phiMargin
        targets[i,:,0] = thetas
        targets[i,:,1] = phis
        positions[i] = cc.pfi.anglesToPositions(cc.allCobras[cIds], thetas, phis)

        logger.info(f'=== Run {i+1}: Convergence test ===')
        cc.pfi.resetMotorScaling(cc.allCobras[cIds])

        if twoSteps:
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

            _useScaling, _maxSegments, _maxTotalSteps = cc.useScaling, cc.maxSegments, cc.maxTotalSteps
            cc.useScaling, cc.maxSegments, cc.maxTotalSteps = False, _maxSegments * 2, _maxTotalSteps * 2
            dataPath, atThetas, atPhis, moves[i,:,:2] = \
                moveThetaPhi(cIds, thetasVia, phisVia, False, True, tolerance, 2, True,
                             newDir, True, True, threshold)

            cc.useScaling, cc.maxSegments, cc.maxTotalSteps = _useScaling, _maxSegments, _maxTotalSteps
            dataPath, atThetas, atPhis, moves[i,:,2:] = \
                moveThetaPhi(cIds, thetas, phis, False, True, tolerance, tries-2, False,
                             False, False, True, threshold)

        else:
            dataPath, atThetas, atPhis, moves[i,:,:] = \
                moveThetaPhi(cIds, thetas, phis, False, True, tolerance, tries, True,
                             newDir, False, True, threshold)

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
    cc.pfi.loadModel([pathlib.Path(f'{runDir}/output/{newXml}')])

    # for fast on time
    ontF = searchOnTime(speeds[1], np.array(_spdF), np.array(_ontF))
    ontR = searchOnTime(speeds[1], np.array(_spdR), np.array(_ontR))
    ontF[ontF>onTimeHigh] = onTimeHigh
    ontR[ontR>onTimeHigh] = onTimeHigh

    # build motor maps
    logger.info(f'Build fast motor maps, best onTime = {np.round([ontF, ontR],4)}')
    runDir, duds = makeThetaMotorMaps(newXml, repeat=repeat, steps=steps[1], thetaOnTime=[ontF, ontR], fast=True)
    cc.pfi.loadModel([pathlib.Path(f'{runDir}/output/{newXml}')])

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
    cc.pfi.loadModel([pathlib.Path(f'{runDir}/output/{newXml}')])

    # for fast motor maps
    ontF = searchOnTime(speeds[1], np.array(_spdF), np.array(_ontF))
    ontR = searchOnTime(speeds[1], np.array(_spdR), np.array(_ontR))
    ontF[ontF>onTimeHigh] = onTimeHigh
    ontR[ontR>onTimeHigh] = onTimeHigh

    # build motor maps
    logger.info(f'Build fast motor maps, best onTime = {np.round([ontF, ontR],4)}')
    runDir, duds = makePhiMotorMaps(newXml, repeat=repeat, steps=steps[1], phiOnTime=[ontF, ontR], fast=True)
    cc.pfi.loadModel([pathlib.Path(f'{runDir}/output/{newXml}')])

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

def convertXML2(newXml, homePhi=True):
    """ update old XML to a new coordinate by taking 'phi homed' images
        assuming the shift of cobra bench is small
    """
    cc.connect(False)

    idx = cc.visibleIdx
    oldPos = cc.calibModel.centers
    newPos = np.zeros(cc.nCobras, dtype=complex)

    # go home and measure new positions
    if homePhi:
        cc.moveToHome(cc.allCobras, thetaEnable=False, phiEnable=True)
    newPos[idx] = cc.exposeAndExtractPositions()

    # calculation tranformation
    #offset, scale, tilt, convert = cal.transform(oldPos[idx], newPos[idx])

    afCoeff = cal.tranformAffine(oldPos[idx], newPos[idx])

    offset = afCoeff[0, 2]+afCoeff[1, 2]*1j
    scale = np.sqrt(afCoeff[0, 0]**2+afCoeff[0, 1]**2)
    tilt = np.rad2deg(np.arctan2(afCoeff[1, 0]/np.sqrt(afCoeff[0, 0]**2+afCoeff[0, 1]**2),
                                  afCoeff[1, 1]/np.sqrt(afCoeff[1, 0]**2+afCoeff[1, 1]**2)))

    ori=np.array([oldPos[goodIdx].real,oldPos[goodIdx].imag]).T

    afCor=cv2.transform(np.array([ori]),afCoeff)
    newcenters= afCor[0,:,0]+afCor[0,:,1]*1j
    
    old = cc.calibModel
    new = deepcopy(old)
    #new.centers[:] = convert(old.centers)
    new.centers[:]=newcenters

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
    positions = np.zeros((len(cIds), repeat, 2, tries), 'complex')

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

        cc.camResetStack(f'phiOntimeScanFwd{r+1}.fits')
        logger.info(f'=== Forward phi on-time scan #{r+1} ===')

        if r == 0:
            ontimes[:, r, 0, 0] = initOntimes[0]
        else:
            for ci in range(len(cIds)):
                ontimes[ci, r, 0, 0] = cal.calculateOntime(ontimes[ci,r-1,0,0], speed/speeds[ci,r-1,0,0],
                                                           scaling, cc.pfi.phiParameter, cc.pfi.maxPhiOntime)
        angles[:, r, 0, 0] = cc.phiInfo['angle'][cIds]
        oldOntimes = np.copy(cc.calibModel.motorOntimeSlowFwd2)

        # move away from the hard stops first
        cobras = cc.allCobras[notDoneMask]
        cc.moveSteps(cobras, 0, 30)

        for j in range(tries):
            cobras = cc.allCobras[notDoneMask]
            cc.calibModel.motorOntimeSlowFwd2[cIds] = ontimes[:, r, 0, j]
            cc.moveSteps(cobras, 0, steps)
            positions[:, r, 0, j] = cc.cobraInfo['position'][cIds]
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

#            nowDone[cc.phiInfo['angle'] > limitAngles - tolerance] = True
            if j > 10:
                nowDone[cIds] = angles[:, r, 0, j+1] < angles[:, r, 0, j]
            else:
                nowDone[:] = False
            newlyDone = nowDone & notDoneMask

            if np.any(newlyDone):
                notDoneMask &= ~newlyDone
                logger.info(f'done: {np.where(newlyDone)[0]}, {(notDoneMask == True).sum()} left')
            if not np.any(notDoneMask):
                logger.info(f'all cobras reach CW limits')
                break

        cc.camResetStack()
        cc.calibModel.motorOntimeSlowFwd2[:] = oldOntimes
        if np.any(notDoneMask):
            logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                        f'{np.where(notDoneMask)[0]}, ')

        nowDone[:] = False
        notDoneMask[:] = False
        notDoneMask[cIds] = True

        cc.camResetStack(f'phiOntimeScanRev{r+1}.fits')
        logger.info(f'=== Reverse phi on-time scan #{r+1} ===')

        if r == 0:
            ontimes[:, r, 1, 0] = initOntimes[1]
        else:
            for ci in range(len(cIds)):
                ontimes[ci, r, 1, 0] = cal.calculateOntime(ontimes[ci,r-1,1,0], -speed/speeds[ci,r-1,1,0],
                                                           scaling, cc.pfi.phiParameter, cc.pfi.maxPhiOntime)
        angles[:, r, 1, 0] = cc.phiInfo['angle'][cIds]
        oldOntimes = np.copy(cc.calibModel.motorOntimeSlowRev2)

        # move away from the hard stops first
        cobras = cc.allCobras[notDoneMask]
        cc.moveSteps(cobras, 0, -30)

        for j in range(tries):
            cobras = cc.allCobras[notDoneMask]
            cc.calibModel.motorOntimeSlowRev2[cIds] = ontimes[:, r, 1, j]
            cc.moveSteps(cobras, 0, -steps)
            positions[:, r, 1, j] = cc.cobraInfo['position'][cIds]
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

#            nowDone[cc.phiInfo['angle'] < tolerance] = True
            if j > 10:
                nowDone[cIds] = angles[:, r, 1, j+1] > angles[:, r, 1, j]
            else:
                nowDone[:] = False
            newlyDone = nowDone & notDoneMask

            if np.any(newlyDone):
                notDoneMask &= ~newlyDone
                logger.info(f'done: {np.where(newlyDone)[0]}, {(notDoneMask == True).sum()} left')
            if not np.any(notDoneMask):
                logger.info(f'all cobras reach CCW limits')
                break

        cc.camResetStack()
        cc.calibModel.motorOntimeSlowRev2[:] = oldOntimes
        if np.any(notDoneMask):
            logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                        f'{np.where(notDoneMask)[0]}, ')

    np.save(dataPath / 'ontimes', ontimes)
    np.save(dataPath / 'angles', angles)
    np.save(dataPath / 'speeds', speeds)
    np.save(dataPath / 'cobras', cIds)
    np.save(dataPath / 'positions', positions)
    np.save(dataPath / 'parameters', [speed, steps, scaling, tolerance])

    # build on-time maps, using only the first run
    mm = np.full((angles.shape[0],2,angles.shape[3]-1), np.nan, dtype=mmDtype)
    lim = 0.05
    smooth_len = 11

    for i in range(angles.shape[0]):
        for j in range(2):
            if j == 0:
                nz = np.where(angles[i,0,j] > angles[i,0,j,0]+lim)[0]
            else:
                nz = np.where(angles[i,0,j,1:] < angles[i,0,j,1]-lim)[0]

            if len(nz) > 0:
                lower = nz[0] + j
            else:
                logger.warn(f'sticky at the beginning: {i}, {j}')
                continue

            if j == 0:
                nz = np.where(angles[i,0,j,:-1] > np.max(angles[i,0,j,:-1])-lim)[0]
            else:
                nz = np.where(angles[i,0,j,1:] < np.min(angles[i,0,j,1:])+lim)[0]

            if len(nz) > 0:
                upper = nz[0] + j
            else:
                logger.warn(f'sticky at the end: {i}, {j}')
                continue

            if upper - lower < smooth_len:
                logger.warn(f'moving range is too small: {i}, {j}')
                continue

            mm[i,j,:upper-lower]['angle'] = angles[i,0,j,lower:upper]
            mm[i,j,:upper-lower]['ontime'] = cal.smooth(ontimes[i,0,j,lower:upper])
            mm[i,j,:upper-lower]['speed'] = cal.smooth(speeds[i,0,j,lower:upper])

    mmOut = np.full((cc.nCobras,2,angles.shape[3]-1), np.nan, dtype=mmDtype)
    mmOut[cIds] = mm
    np.save(dataPath / 'phiOntimeMap', mmOut)

    return dataPath, ontimes, angles, speeds

def thetaOntimeScan(cIds=None, speed=None, initOntimes=None,
                    steps=10, totalSteps=10000,
                    repeat=1, scaling=4.0,
                    tolerance=np.deg2rad(1.0)):
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
    positions = np.zeros((len(cIds), repeat, 2, tries), 'complex')

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

        cc.camResetStack(f'thetaOntimeScanFwd{r+1}.fits')
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
            positions[:, r, 0, j] = cc.cobraInfo['position'][cIds]
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

#            nowDone[cc.thetaInfo['angle'] > limitAngles - tolerance] = True
            if j > 10:
                nowDone[cIds] = angles[:, r, 0, j+1] < angles[:, r, 0, j]
            else:
                nowDone[:] = False
            newlyDone = nowDone & notDoneMask

            if np.any(newlyDone):
                notDoneMask &= ~newlyDone
                logger.info(f'done: {np.where(newlyDone)[0]}, {(notDoneMask == True).sum()} left')
            if not np.any(notDoneMask):
                logger.info(f'all cobras reach CW limits')
                break

        cc.camResetStack()
        cc.calibModel.motorOntimeSlowFwd1[:] = oldOntimes
        if np.any(notDoneMask):
            logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                        f'{np.where(notDoneMask)[0]}, ')

        nowDone[:] = False
        notDoneMask[:] = False
        notDoneMask[cIds] = True

        cc.camResetStack(f'thetaOntimeScanRev{r+1}.fits')
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
            positions[:, r, 1, j] = cc.cobraInfo['position'][cIds]
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

#            nowDone[cc.thetaInfo['angle'] < tolerance] = True
            if j > 10:
                nowDone[cIds] = angles[:, r, 1, j+1] > angles[:, r, 1, j]
            else:
                nowDone[:] = False
            newlyDone = nowDone & notDoneMask

            if np.any(newlyDone):
                notDoneMask &= ~newlyDone
                logger.info(f'done: {np.where(newlyDone)[0]}, {(notDoneMask == True).sum()} left')
            if not np.any(notDoneMask):
                logger.info(f'all cobras reach CCW limits')
                break

        cc.camResetStack()
        cc.calibModel.motorOntimeSlowRev1[:] = oldOntimes
        if np.any(notDoneMask):
            logger.warn(f'{(notDoneMask == True).sum()} cobras did not finish: '
                        f'{np.where(notDoneMask)[0]}, ')

    np.save(dataPath / 'ontimes', ontimes)
    np.save(dataPath / 'angles', angles)
    np.save(dataPath / 'speeds', speeds)
    np.save(dataPath / 'cobras', cIds)
    np.save(dataPath / 'positions', positions)
    np.save(dataPath / 'parameters', [speed, steps, scaling, tolerance])

    # build on-time maps, using only the first run
    mm = np.full((angles.shape[0],2,angles.shape[3]-1), np.nan, dtype=mmDtype)
    lim = 0.05
    smooth_len = 11

    for i in range(angles.shape[0]):
        for j in range(2):
            if j == 0:
                nz = np.where(angles[i,0,j] > angles[i,0,j,0]+lim)[0]
            else:
                nz = np.where(angles[i,0,j,1:] < angles[i,0,j,1]-lim)[0]

            if len(nz) > 0:
                lower = nz[0] + j
            else:
                logger.warn(f'sticky at the beginning: {i}, {j}')
                continue

            if j == 0:
                nz = np.where(angles[i,0,j,:-1] > np.max(angles[i,0,j,:-1])-lim)[0]
            else:
                nz = np.where(angles[i,0,j,1:] < np.min(angles[i,0,j,1:])+lim)[0]

            if len(nz) > 0:
                upper = nz[0] + j
            else:
                logger.warn(f'sticky at the end: {i}, {j}')
                continue

            if upper - lower < smooth_len:
                logger.warn(f'moving range is too small: {i}, {j}')
                continue

            mm[i,j,:upper-lower]['angle'] = angles[i,0,j,lower:upper]
            mm[i,j,:upper-lower]['ontime'] = cal.smooth(ontimes[i,0,j,lower:upper])
            mm[i,j,:upper-lower]['speed'] = cal.smooth(speeds[i,0,j,lower:upper])

    mmOut = np.full((cc.nCobras,2,angles.shape[3]-1), np.nan, dtype=mmDtype)
    mmOut[cIds] = mm
    np.save(dataPath / 'thetaOntimeMap', mmOut)

    return dataPath, ontimes, angles, speeds

def convergenceTestX(cIds, runs=3, thetaMargin=np.deg2rad(15.0), phiMargin=np.deg2rad(15.0), 
        tries=8, tolerance=0.1, threshold=3.0, newDir=False, twoSteps=False):
    
    """ convergence test, even and odd cobras move toward each other in a single module """
    if tries < 4:
        raise ValueError("tries parameter should be larger than 4")

    cc.connect(False)
    targets = np.zeros((runs, len(cIds), 2))
    moves = np.zeros((runs, len(cIds), tries), dtype=moveDtype)
    positions = np.zeros((runs, len(cIds)), dtype=complex)
    thetaRange = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (np.pi*2) + np.pi)[cIds]
    phiRange = ((cc.calibModel.phiOut - cc.calibModel.phiIn) % (np.pi*2))[cIds]

    for i in range(runs):
        angle = np.pi/4 - np.pi/6 * i / runs
        thetas = (np.pi/2 + angle - cc.calibModel.tht0[cIds]) % (np.pi*2)
        thetas[cIds%2==1] = (thetas[cIds%2==1] + np.pi) % (np.pi*2)
        thetas[thetas < thetaMargin] += np.pi*2
        tooBig = thetas > thetaRange - thetaMargin
        thetas[tooBig] = thetaRange[tooBig] - thetaMargin
        phis = np.pi - angle*2 - cc.calibModel.phiIn[cIds] - np.pi
        phis[phis < 0] = 0
        tooBig = phis > phiRange - phiMargin
        phis[tooBig] = phiRange[tooBig] - phiMargin
        targets[i,:,0] = thetas
        targets[i,:,1] = phis
        positions[i] = cc.pfi.anglesToPositions(cc.allCobras[cIds], thetas, phis)

        logger.info(f'=== Run {i+1}: Convergence test ===')
        cc.pfi.resetMotorScaling(cc.allCobras[cIds])

        if twoSteps:
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

            _useScaling, _maxSegments, _maxTotalSteps = cc.useScaling, cc.maxSegments, cc.maxTotalSteps
            cc.useScaling, cc.maxSegments, cc.maxTotalSteps = False, _maxSegments * 2, _maxTotalSteps * 2
            dataPath, atThetas, atPhis, moves[i,:,:2] = \
                moveThetaPhi(cIds, thetasVia, phisVia, False, True, tolerance, 2, True, newDir, True, True, threshold)

            cc.useScaling, cc.maxSegments, cc.maxTotalSteps = _useScaling, _maxSegments, _maxTotalSteps
            dataPath, atThetas, atPhis, moves[i,:,2:] = \
                moveThetaPhi(cIds, thetas, phis, False, True, tolerance, tries-2, False, False, False, True, threshold)

        else:
            dataPath, atThetas, atPhis, moves[i,:,:] = \
                moveThetaPhi(cIds, thetas, phis, False, True, tolerance, tries, True, newDir, False, True, threshold)

    np.save(dataPath / 'positions', positions)
    np.save(dataPath / 'targets', targets)
    np.save(dataPath / 'moves', moves)
    return targets, moves

def moveToSafePosition(cIds, tolerance=0.2, tries=10, homed=True, newDir=False, threshold=20.0, thetaMargin=np.deg2rad(15.0)):
    # move theta arms away from the bench center and phi arms to 60 degree
    ydir = np.angle(cc.calibModel.centers[1] - cc.calibModel.centers[55])
    thetas = np.full(len(cIds), ydir)
    thetas[cIds<798] += np.pi*2/3
    thetas[cIds>=1596] -= np.pi*2/3
    thetas = thetas % (np.pi*2)

    cc.pfi.resetMotorScaling()
    dataPath, thetas, phis, moves = moveThetaPhi(cIds, thetas, np.pi/3, False, False, tolerance, tries, homed, newDir, True, True, threshold, thetaMargin)
    
    # Define safe location to be phi = 80-degree outward
    #dataPath, thetas, phis, moves = moveThetaPhi(cIds, thetas, np.pi*(4/9), False, False, tolerance, tries, homed, newDir, True, True, threshold, thetaMargin)

    

def convergenceTest2(cIds, runs=8, thetaMargin=np.deg2rad(15.0), phiMargin=np.deg2rad(15.0), thetaOffset=0, phiAngle=(np.pi*5/6, np.pi/3, np.pi/4), tries=8, tolerance=0.2, threshold=20.0, newDir=False, twoSteps=False):
    """ convergence test, all theta arms in the same global direction. Three non-interfere groups use different phi angles  """
    cc.connect(False)
    targets = np.zeros((runs, len(cIds), 2))
    moves = np.zeros((runs, len(cIds), tries), dtype=moveDtype)
    positions = np.zeros((runs, len(cIds)), dtype=complex)
    thetaRange = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (np.pi*2) + np.pi)[cIds]
    phiRange = ((cc.calibModel.phiOut - cc.calibModel.phiIn) % (np.pi*2))[cIds]
    cm = cIds % 57
    cf = cIds % 798
    ffIdx = np.where((cm == 0) | (cm == 2) | ((cf < 57) & (cf % 4 == 2)))[0]

    for i in range(runs):
        thetas = (thetaOffset + np.pi*2*i/runs - cc.calibModel.tht0[cIds]) % (np.pi*2)
        thetas[thetas < thetaMargin] += np.pi*2
        tooBig = thetas > thetaRange - thetaMargin
        thetas[tooBig] = thetaRange[tooBig] - thetaMargin

        phis = np.zeros(len(cIds))
        for j in range(3):
            groupIdx = (cIds + cIds // 57 + cIds // 798 + i) % 3 == j
            phis[groupIdx] = -cc.calibModel.phiIn[cIds[groupIdx]] - np.pi + phiAngle[j]
        phis[phis < 0] = 0
        tooBig = phis > phiRange - phiMargin
        phis[tooBig] = phiRange[tooBig] - phiMargin

        if len(ffIdx) > 0:
            phis[ffIdx] = np.min((phis[ffIdx], -cc.calibModel.phiIn[cIds[ffIdx]] - np.pi*2/3), axis=0)

        targets[i,:,0] = thetas
        targets[i,:,1] = phis
        positions[i] = cc.pfi.anglesToPositions(cc.allCobras[cIds], thetas, phis)

        logger.info(f'=== Run {i+1}: Convergence test ===')
        cc.pfi.resetMotorScaling(cc.allCobras[cIds])

        if twoSteps:
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

            _useScaling, _maxSegments, _maxTotalSteps = cc.useScaling, cc.maxSegments, cc.maxTotalSteps
            cc.useScaling, cc.maxSegments, cc.maxTotalSteps = False, _maxSegments * 2, _maxTotalSteps * 2
            dataPath, atThetas, atPhis, moves[i,:,:2] = \
                moveThetaPhi(cIds, thetasVia, phisVia, False, True, tolerance, 2, True, newDir, True, True, threshold)

            cc.useScaling, cc.maxSegments, cc.maxTotalSteps = _useScaling, _maxSegments, _maxTotalSteps
            dataPath, atThetas, atPhis, moves[i,:,2:] = \
                moveThetaPhi(cIds, thetas, phis, False, True, tolerance, tries-2, False, False, False, True, threshold)

        else:
            dataPath, atThetas, atPhis, moves[i,:,:] = \
                moveThetaPhi(cIds, thetas, phis, False, True, tolerance, tries, True, newDir, False, True, threshold)

    np.save(dataPath / 'positions', positions)
    np.save(dataPath / 'targets', targets)
    np.save(dataPath / 'moves', moves)
    return targets, moves

def convergenceTestX2(cIds, runs=3, thetaMargin=np.deg2rad(15.0), phiMargin=np.deg2rad(15.0), tries=8, tolerance=0.2, threshold=20.0, newDir=True, twoSteps=False):
    """ convergence test, even and odd cobras move toward each other in a single module """
    if tries < 4:
        raise ValueError("tries parameter should be larger than 4")

    cc.connect(False)
    targets = np.zeros((runs, len(cIds), 2))
    moves = np.zeros((runs, len(cIds), tries), dtype=moveDtype)
    positions = np.zeros((runs, len(cIds)), dtype=complex)
    thetaRange = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (np.pi*2) + np.pi)[cIds]
    phiRange = ((cc.calibModel.phiOut - cc.calibModel.phiIn) % (np.pi*2))[cIds]
    ydir = np.angle(cc.calibModel.centers[1] - cc.calibModel.centers[55])

    for i in range(runs):
        angle = np.pi/4 - np.pi/6 * i / runs
        thetas = ydir + np.pi/2 + angle - cc.calibModel.tht0[cIds]
        thetas[cIds>=798] -= np.pi*2/3
        thetas[cIds>=1596] -= np.pi*2/3
        thetas[cIds%2==0] = thetas[cIds%2==0] % (np.pi*2)
        thetas[cIds%2==1] = (thetas[cIds%2==1] + np.pi) % (np.pi*2)
        thetas[thetas < thetaMargin] += np.pi*2
        tooBig = thetas > thetaRange - thetaMargin
        thetas[tooBig] = thetaRange[tooBig] - thetaMargin
        phis = np.pi - angle*2 - cc.calibModel.phiIn[cIds] - np.pi
        phis[phis < 0] = 0
        tooBig = phis > phiRange - phiMargin
        phis[tooBig] = phiRange[tooBig] - phiMargin
        targets[i,:,0] = thetas
        targets[i,:,1] = phis
        positions[i] = cc.pfi.anglesToPositions(cc.allCobras[cIds], thetas, phis)

        logger.info(f'=== Run {i+1}: Convergence test ===')
        cc.pfi.resetMotorScaling(cc.allCobras[cIds])

        if twoSteps:
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

            _useScaling, _maxSegments, _maxTotalSteps = cc.useScaling, cc.maxSegments, cc.maxTotalSteps
            cc.useScaling, cc.maxSegments, cc.maxTotalSteps = False, _maxSegments * 2, _maxTotalSteps * 2
            dataPath, atThetas, atPhis, moves[i,:,:2] = \
                moveThetaPhi(cIds, thetasVia, phisVia, False, True, tolerance, 2, True, newDir, True, True, threshold)

            cc.useScaling, cc.maxSegments, cc.maxTotalSteps = _useScaling, _maxSegments, _maxTotalSteps
            dataPath, atThetas, atPhis, moves[i,:,2:] = \
                moveThetaPhi(cIds, thetas, phis, False, True, tolerance, tries-2, False, False, False, True, threshold)

        else:
            dataPath, atThetas, atPhis, moves[i,:,:] = \
                moveThetaPhi(cIds, thetas, phis, False, True, tolerance, tries, True, newDir, False, True, threshold)

    np.save(dataPath / 'positions', positions)
    np.save(dataPath / 'targets', targets)
    np.save(dataPath / 'moves', moves)
    return targets, moves

def createTrajectory(cIds, thetas, phis, tries=8, twoSteps=False, threshold=20.0, timeStep=10):
    """ create trajectory objects for given targets

        cIds: the cobra index for generating trajectory
        thetas, phis: the target angles
        tried: the number of maximum movements/iterations
        twoSteps: using one-step or two-steps strategy
        threshold: the threshold for using slow or fast motor maps
        timeStep: timeStep parameter for the trajectory
    """
    if tries < 4:
        raise ValueError("tries parameter should be larger than 4")

    targets = np.zeros((len(cIds), 2))
    moves = np.zeros((len(cIds), tries), dtype=moveDtype)
    positions = np.zeros(len(cIds), dtype=complex)
    thetaRange = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (np.pi*2) + np.pi)[cIds]
    phiRange = ((cc.calibModel.phiOut - cc.calibModel.phiIn) % (np.pi*2))[cIds]

    targets[:,0] = thetas
    targets[:,1] = phis
    positions = cc.pfi.anglesToPositions(cc.allCobras[cIds], thetas, phis)

    if not cc.trajectoryMode:
        logger.info(f'switch cobraCoach to trajectoryMode mode')
        cc.trajectoryMode = True
        toggleFlag = True
    else:
        toggleFlag = False
    cc.trajectory = Trajectories(cc.nCobras, timeStep)

    tolerance = 0.1
    if twoSteps:
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

        _useScaling, _maxSegments, _maxTotalSteps = cc.useScaling, cc.maxSegments, cc.maxTotalSteps
        cc.useScaling, cc.maxSegments, cc.maxTotalSteps = False, _maxSegments * 2, _maxTotalSteps * 2
        dataPath, atThetas, atPhis, moves[:,:2] = \
            moveThetaPhi(cIds, thetasVia, phisVia, False, True, tolerance, 2, True, False, True, True, threshold)

        cc.useScaling, cc.maxSegments, cc.maxTotalSteps = _useScaling, _maxSegments, _maxTotalSteps
        dataPath, atThetas, atPhis, moves[:,2:] = \
            moveThetaPhi(cIds, thetas, phis, False, True, tolerance, tries-2, False, False, False, True, threshold)

    else:
        dataPath, atThetas, atPhis, moves[:,:] = \
            moveThetaPhi(cIds, thetas, phis, False, True, tolerance, tries, True, False, False, True, threshold)

    if toggleFlag:
        cc.trajectoryMode = False
    return cc.trajectory, moves


def buildMotorMapFromRunId(arm=None, runDir=None):
    '''
        Build motor map from given runDir.   This is used when we need to re-calculate the  
    '''
    pass
