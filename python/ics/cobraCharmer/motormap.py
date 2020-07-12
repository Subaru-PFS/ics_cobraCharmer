import numpy as np
import ruamel_yaml

class MotorMap(object):
    validDirections = {'fwd', 'rev'}
    validMotors = {'theta', 'phi'}
    yaml_tag = "!MotorMap"

    def __init__(self, name, cobraId=None, motor=None, direction=None, angles=None, steps=None, ontimes=None):
        """Encapsulate a motor map: currently a mapping between angles and steps&ontimes. """
        self.name = name
        self._initFromParts(cobraId=cobraId,
                            motor=motor, direction=direction,
                            angles=angles, steps=steps, ontimes=ontimes)

    def _initFromParts(self, cobraId, motor, direction, angles, steps, ontimes):
        if motor not in self.validMotors:
            raise ValueError(f'motor ({motor}) must be one of {self.validMotors}')
        if direction not in self.validDirections:
            raise ValueError(f'direction ({direction}) must be one of {self.validDirections}')
        if len(angles) != len(steps):
            raise ValueError('angles and steps must have the same shape')

        self.cobraId = cobraId
        self.motor = motor
        self.direction = direction
        self.angles = np.array(angles, dtype='f4')
        self.steps = np.array(steps, dtype='i4')

        if np.isscalar(ontimes):
            self.ontimes = np.full(len(self.angles), ontimes, dtype='f4')
        else:
            self.ontimes = np.array(ontimes, dtype='f4')

    def __str__(self):
        return f"{self.__class__.__name__}(motor={self.motor}, name={self.name}, direction={self.direction})"

    def __repr__(self):
        return (f"{self.__class__.__name__}(motor={self.motor}, name={self.name}, direction={self.direction},"
                f" angles={np.round(self.angles,3)}, steps={self.steps.tolist()},"
                f" ontime={np.round(self.ontimes,3)})")

    def updateMap(self, steps, ontime, angles=None):
        """
        """
        raise NotImplementedError()

    def calculateMove(self, fromAngle, toAngle):
        """Return the steps and ontime to move between two angles.

        Args
        ----
        fromAngle : `float` radians
        toAngle : `float` radians
          The two angles to move between.

        Returns
        -------
        moves : [ (int steps, float ontime) ]

        """
        dAngle = toAngle - fromAngle
        if dAngle < 0 and self.direction != 'rev' or dAngle > 0 and self.direction  != 'fwd':
            raise ValueError(f"{self}: wrong direction for from={fromAngle} to={toAngle}")

        # Calculate the total number of motor steps
        stepRange = np.interp([fromAngle, toAngle],
                              self.angles, self.steps)

        if not np.all(np.isfinite(stepRange)):
            raise ValueError(f"{self} angle to step interpolation out of range: "
                             f"from:{fromAngle} to={toAngle}")
        steps = np.rint(stepRange[1] - stepRange[0]).astype('i4')

        # We want to use np.simps or some other integration, then get the average?
        ontime = self.ontime[0]

        return [(steps, ontime)]

    @staticmethod
    def to_yaml(dumper, self):
        output = dict(name=self.name,
                      cobraId=int(self.cobraId),
                      motor=self.motor,
                      direction=self.direction,
                      angles=self.angles.round(4).tolist(),
                      steps=self.steps.tolist(),
                      ontimes=self.ontimes.round(4).tolist())

        return dumper.represent_mapping(self.yaml_tag, output)

    @staticmethod
    def from_yaml(loader, node):
        node_map = loader.construct_mapping(node, deep=True)
        return MotorMap(**node_map)

    def dump(self, path):
        with open(path, 'wt') as output:
            yaml.dump(self, output)

yaml = ruamel_yaml.YAML(typ='safe')
yaml.register_class(MotorMap)

def load(path):
    with open(path, 'rt') as input:
        o = yaml.load(input)
    return o
