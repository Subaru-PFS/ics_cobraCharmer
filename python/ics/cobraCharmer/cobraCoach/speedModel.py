import logging
import numpy as np
from scipy import optimize

class SpeedModel():
    def __init__(self, p0=1.0, p1=0.08, p2=2, logLevel=logging.INFO):
        self.logger = logging.getLogger('speedModel')
        self.logger.setLevel(logLevel)
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

    def toSpeed(self, ontime, p0=None, p1=None, p2=None):
        """ convert ontime to speed """
        if p0 is None:
            p0 = self.p0
        if p1 is None:
            p1 = self.p1
        if p2 is None:
            p2 = self.p2
        return speedFunc(ontime, p0, p1, p2)

    def toOntime(self, speed, p0=None, p1=None, p2=None):
        """ convert speed to ontime """
        if p0 is None:
            p0 = self.p0
        if p1 is None:
            p1 = self.p1
        if p2 is None:
            p2 = self.p2
        return invSpeedFunc(speed, p0, p1, p2)

    def scalingSpeed(self, ontime, scaling):
        """ calculate ontime by speed scaling factor """
        a0 = self.toSpeed(ontime)
        return self.toOntime(a0*scaling)

    def getOntimeFromData(self, speed, lastSpeed, lastOntime):
        """ calculate ontime for a given speed from last data
            only update the p0 parameter
        """
        p0 = lastSpeed / self.toSpeed(lastOntime, p0=1)
        return self.toOntime(speed, p0=p0)

    def buildModel(self, speed_data, ontime_data):
        """ build speed model from data """
        s = speed_data
        t = ontime_data

        try:
            params, params_cov = optimize.curve_fit(speedFunc, t, s, p0=[10, 0.06])
            if params[0] < 0 or params[1] < 0:
                # remove some slow data and try again
                self.logger.warn(f'Build model failed, try again')
                s[t==np.max(t)] = np.max(s[t==np.max(t)])
                params, params_cov = optimize.curve_fit(speedFunc, t, s, p0=[10, 0.06])
            if params[0] < 0 or params[1] < 0:
                raise
            self.p0 = params[0]
            self.p1 = params[1]
            return False
        except:
            self.logger.warn(f'Building model failed!!!')
            return True


def speedFunc(x, p0, p1, p2=2):
    """ map ontime to speed """
    return p0 * (np.power(np.power(x, p2) + np.power(p1, p2), 1.0/p2) - p1)

def invSpeedFunc(x, p0, p1, p2=2):
    """ map speed to ontime """
    return np.power(np.power(x/p0+p1, p2) - np.power(p1, p2), 1.0/p2)
