from importlib import reload
import logging
import enum

import numpy as np

import ruamel_yaml

import pfs.utils.fiberids
reload(pfs.utils.fiberids)
fiberIds = pfs.utils.fiberids.FiberIds()

class CobraStatus(enum.IntEnum):
    COBRA_OK           = 0x0001  # a synthetic summary bit: 1 for good, 0 for bad.
    COBRA_INVISIBLE    = 0x0002  # 1 if the fiber is not visible
    COBRA_BROKEN_THETA = 0x0004  # 1 if the phi motor do not work
    COBRA_BROKEN_PHI   = 0x0008  # 1 if the theta motor does not work

    COBRA_BROKEN_MOTOR_MASK = COBRA_BROKEN_THETA | COBRA_BROKEN_PHI
    COBRA_BROKEN_THETA_MASK = COBRA_BROKEN_THETA | COBRA_INVISIBLE

class Motor(object):
    def __init__(self, limits, fwdMap=None, revMap=None, ontimeScales=None):
        self.angle = np.nan
        self.lastMove = 0
        self.lastOntime = np.nan
        self.limits = limits
        self.ontimeScales = dict(fwd=1.0, rev=1.0)
        self.maps = dict(fwd=None, rev=None)
        self.setMaps(fwdMap=fwdMap,revMap=revMap)

    def useableStatus(self):
        return (not (self.status & (CobraStatus.COBRA_INVISIBLE | CobraStatus.COBRA_BROKEN_THETA)),
                not (self.status & (CobraStatus.COBRA_INVISIBLE | CobraStatus.COBRA_BROKEN_PHI)))

    def setPosition(self, angle=np.nan):
        if np.isfinite(angle):
            if angle < self.limits[0] or angle > self.limits[1]:
                pass
                # logging.warn('')

        self.angle = angle

    def registerMove(self, steps, ontime):
        if steps is not None and steps != 0:
            self.lastMove = move
            self.lastOntime = ontime

    def setMaps(self, fwdMap=None, revMap=None):
        if fwdMap is not None:
            self.maps['fwd'] = fwdMap
        if revMap is not None:
            self.maps['rev'] = revMap

    def scaleMotorOntime(self, direction, scale, doReset=False):
        """ Declare that we want an ontime to be scaled after interpolation.

        If there is an existing scaling, the new scaling is applied
        _that_: we are expecting to be told that the last effective
        move neeeded adjustment.

        Args
        ----
        direction : {'fwd', 'rev'}
           Which motor map to adjust
        scale : `float`
           Scaling to apply to the theta motor's ontime
        doReset : `bool`
           Whether to replace the existing scaling or adjust it.
        """

        if not doReset:
            existingScale = self.ontimeScales[direction]
        else:
            existingScale = 1.0

        newScale = existingScale * scale
        if newScale < 0.5:
            self.logger.warn(f'clipping scale adjustment from {newScale} to 0.5')
            newScale = 0.5
        if newScale > 2.0:
            self.logger.warn(f'clipping scale adjustment from {newScale} to 2.0')
            newScale = 2.0

        self.ontimeScales[direction] = newScale

    def adjustOntime(self, direction, ontime):
        """ Apply dynamic scaling to ontime. """
        scale = self.ontimeScales[direction]

        newOntime = ontime*scale
        if newOntime > self.maxThetaOntime:
            newOntime = self.maxThetaOntime
            self.ontimeScales[direction] = newOntime/ontime
            self.logger.warn(f'clipping {mapId} ontime to {newOntime} and '
                             f'scale {scale:0.2f} to {cobraState.motorScales[mapId]}')

        return newOntime

     def calculateSteps(self, direction, toAngle, fromAngle=None, map=None):
        if map is None:
            map = self.maps[direction]
        if fromAngle is None:
            fromAngle = self.angle

        steps, rawOntime = map.calculateSteps(fromAngle, toAngle),
        ontime = self.adjustOntime(direction, rawOntime))

        return steps, ontime

class ThetaMotor(Motor):
    defaultMaxOntime = 0.1
    defaultMinOntime = 0.01

class PhiMotor(Motor):
    defaultMaxOntime = 0.08
    defaultMinOntime = 0.01

class Cobra(object):
    validDirections = {'fwd', 'rev'}
    validMotors = {'theta', 'phi'}
    yaml_tag = '!Cobra'

    def __init__(self, cobraId, butler=None):
        self.logger = logging.getLogger('cobra')
        self.maps = dict(theta=dict(fwd=dict(),
                                    rev=dict()),
                         phi=dict(fwd=dict(),
                                  rev=dict()))

        self.butler = butler
        self.initFromParts(cobraId=cobraId)

    @property
    def butler(self):
        if self._butler is None:
            import pfs.utils.butler
            reload(pfs.utils.butler)

            butler = pfs.utils.butler.Butler()
            self._butler = butler

        return self._butler

    @butler.setter
    def butler(self, newButler):
        self._butler = newButler

    def __str__(self):
        return f"Cobra({self.cobraId}, module={self.moduleNum}, modCobra={self.cobraInModule})"

    def initFromParts(self, *,
                      cobraId=None,
                      serial=None,
                      status=None,
                      center=None,
                      thetaLimits=None, phiLimits=None,
                      L1=None, L2=None,
                      thetaMotorFrequency=None, phiMotorFrequency=None):
        self.cobraId = cobraId
        self.serial = serial
        self.status = None if status is None else CobraStatus(status)
        if center is None or isinstance(center, complex):
            self.center = center
        else:
            self.center = complex(center[0], center[1])
        self.L1 = L1
        self.L2 = L2

        thetaLimits = None if thetaLimits is None else np.deg2rad(np.array(thetaLimits))
        phiLimits = None if phiLimits is None else np.deg2rad(np.array(phiLimits))
        self.thetaMotor = ThetaMotor(limits=thetaLimits, frequency=thetaMotorFrequency,
                                     fwdMap=self.getMap(motor='theta', dir='fwd'),
                                     revMap=self.getMap(motor='theta', dir='rev'))
        self.phiMotor = PhiMotor(limits=phiLimits, frequency=phiMotorFrequency,
                                 fwdMap=self.getMap(motor='phi', dir='fwd'),
                                 revMap=self.getMap(motor='phi', dir='rev'))

        self.moduleNum, self.cobraInModule = fiberIds.moduleNumsForCobra(cobraId)

    @property
    def moduleName(self):
        return f'SC{self.moduleNum:02d}'

    def getMap(self, *,
               motor=None, direction=None,
               mapName=''):

        if direction not in self.validDirections:
            raise ValueError(f'direction ({direction}) not in {self.validDirections}')
        if motor not in self.validMotors:
            raise ValueError(f'motor ({motor}) not in {self.validMotors}')

        try:
            map = self.maps[motor][direction][mapName]
        except KeyError:
            map = self._loadMap(motor=motor, direction=direction, mapName=mapName)

        return map

    def _loadMap(self, *,
                 motor=None, direction=None,
                 mapName=None):
        """ Add/overwrite map in cache. """

        idDict = locals().copy()
        idDict['moduleName'] = self.moduleName
        idDict['cobraInModule'] = self.cobraInModule
        map = self.butler.get('motorMap', idDict)
        self.maps[motor][direction][mapName] = map

        return map

    def getExpectedPosition(self, thetaAngle, phiAngle):
        pass

    def calcMovesToThetaPhi(self, thetaAngle, phiAngle):
        pass

    def xyToThetaPhi(self, x, y):
        pass

    def thetaPhiToXY(self, theta, phi):
        pass

    def calculateSteps(self, startTht, deltaTht, startPhi, deltaPhi,
                       thetaFast=False, phiFast=False):
        """ Modified from ics_cobraOps MotorMapGroup.py
        Calculates the total number of motor steps required to move the
        cobra the given theta and phi delta angles from
        CCW hard stops

        Parameters
        ----------
        startTht: object
            A numpy array with the starting theta angle position.
        deltaTht: object
            A numpy array with the theta delta offsets relative to the starting
            theta positions.
        startPhi: object
            A numpy array with the starting phi angle positions.
        deltaPhi: object
            A numpy array with the phi delta offsets relative to the starting
            phi positions.
        thetaFast: object
            A boolean value or array for fast/slow theta motor movement.
        phiFast: object
            A boolean value or array for fast/slow phi motor movement.

        Returns
        -------
        tuple
            A python tuple with the total number of motor steps for the theta
            and phi angles.

        """

        # Get the integrated step maps for the theta angle
        thetaDir = 'fwd' if deltaTht >= 0 else 'rev'
        phiDir = 'fwd' if deltaPhi >= 0 else 'rev'

        self.logger.debug(f'start={startPhi[:3]}, delta={deltaPhi[:3]} move={nPhiSteps[:3]}')
        return (nThtSteps, nPhiSteps)

    def recordMove(self, thetaSteps, phiSteps, thetaOntime, phiOntime):
        self.thetaMotor.registerMove(thetaSteps, thetaOntime)
        self.phiMotor.registerMove(phiSteps, phiOntime)

    @staticmethod
    def to_yaml(dumper, self):
        output = dict(cobraId=int(self.cobraId),
                      serial=int(self.serial),
                      status=int(self.status),
                      center=[round(float(self.center.real), 4),
                              round(float(self.center.imag), 4)],
                      thetaLimits=[round(float(f),4) for f in np.rad2deg(self.thetaLimits)],
                      phiLimits=[round(float(f),4) for f in np.rad2deg(self.phiLimits)],
                      L1=round(float(self.L1), 4), L2=round(float(self.L2), 4),
                      thetaMotorFrequency=round(float(self.thetaMotorFrequency), 4),
                      phiMotorFrequency=round(float(self.phiMotorFrequency), 4))

        return dumper.represent_mapping(self.yaml_tag, output)

    @staticmethod
    def from_yaml(loader, node):
        node_map = loader.construct_mapping(node, deep=True)
        cob = Cobra(node_map['cobraId'])
        cob.initFromParts(**node_map)

        cob.logger.warn(f'returning {cob}')
        return cob

    def dump(self, path):
        with open(path, 'wt') as output:
            yaml.dump(self, output)

yaml = ruamel_yaml.YAML(typ='safe')
yaml.register_class(Cobra)

def load(path):
    with open(path, 'rt') as input:
        o = yaml.load(input)
    return o
