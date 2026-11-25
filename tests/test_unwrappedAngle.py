"""calculus.unwrappedAngle: a measured theta is placed where its value allows, given the cobra range.

The cases are cobras of convergences on 2026-11-24 (fastRoach 156944, blind-move calibration runs).
"""
import numpy as np

from ics.cobraCharmer.cobraCoach import calculus


def unwrapped(angleDeg, steps, fromDeg, toDeg, rangeDeg=None):
    """unwrappedAngle in degrees."""
    out = calculus.unwrappedAngle(np.deg2rad(angleDeg), steps, fromAngle=np.deg2rad(fromDeg),
                                  toAngle=np.deg2rad(toDeg),
                                  thetaRange=None if rangeDeg is None else np.deg2rad(rangeDeg))
    return np.rad2deg(out)


def test_angleFarFromTheStopsIsKept():
    """Cobra 1654: 343.6 deg after an under-travelled CCW move from its CW home, aimed at 90 deg."""
    assert np.isclose(unwrapped(343.6, -2637, 380.6, 90.1), 343.6 - 360)       # without the range
    assert np.isclose(unwrapped(343.6, -2637, 380.6, 90.1, rangeDeg=380.6), 343.6)


def test_angleFarFromTheStopsIsNotPushedPastTheCwStop():
    """Cobra 1198-like: 62.6 deg after a CW move from 70 deg, on a 379.2 deg range."""
    assert np.isclose(unwrapped(62.6, 100, 70.0, 75.0), 62.6 + 360)           # without the range
    assert np.isclose(unwrapped(62.6, 100, 70.0, 75.0, rangeDeg=379.2), 62.6)


def test_angleNearAStopKeepsTheRules():
    """Cobra 839, held at its CW stop 5.7 deg past its calibrated 379.6 deg: unchanged by the range."""
    expected = unwrapped(385.3, -2000, 385.2, 269.7)
    assert np.isclose(expected, 385.3)
    assert np.isclose(unwrapped(385.3, -2000, 385.2, 269.7, rangeDeg=379.6), expected)
