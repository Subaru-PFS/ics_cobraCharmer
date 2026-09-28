"""targetValidation.py — is a commanded target actually reachable and safe?

One implementation, shared by the targeting software and by fps at move time.  They
must agree: targeting guarantees a design is in spec, but fps applies a last-minute
tweak for proper motion and parallax immediately before moving, so what is finally
commanded is not quite what was validated upstream.  Re-checking at run time is only
meaningful if it asks exactly the same question, with exactly the same geometry.

Everything here is stateless and needs only a calibModel: no PFI, no connection, no
CobraCoach.  The angle solve lives here, and PFI.positionsToAngles is a thin wrapper
over it, so the two cannot drift apart.

Every array is (nCobras,) and indexed by cobra index throughout -- never a subset,
never a list of indices into a subset.  Subsetting is where this kind of code goes
wrong: a mask over good cobras and a mask over all cobras look identical and cannot
be told apart by a reader or a test.  Callers subset at the very end, if at all.
"""
import numpy as np

# ── flags ─────────────────────────────────────────────────────────────────────
# Kinematic flags, describing the angle solve itself.  Module level so
# anglesFromPositions and PFI share one definition rather than each declaring a copy.
SOLUTION_OK = 0x0001  # 1 if the solution is valid
IN_OVERLAPPING_REGION = 0x0002  # 1 if the position in overlapping region
PHI_NEGATIVE = 0x0004  # 1 if phi angle is negative(phi CCW limit < 0)
PHI_BEYOND_PI = 0x0008  # 1 if phi angle is beyond PI(phi CW limit > PI)
TOO_CLOSE_TO_CENTER = 0x0010  # 1 if the position is too close to the center
TOO_FAR_FROM_CENTER = 0x0020  # 1 if the position is too far from the center

# Validation flags: the per-cobra verdict, as bits so a caller can both compose masks
# and report a breakdown.  0 means the target is good.
# Deliberately disjoint from the kinematic bits above.  The two sets share a module and
# a return value, so overlapping values would let a caller mix vocabularies and get a
# plausible wrong answer.  Disjoint by construction instead.
NOT_FINITE = 0x0100  # NaN in the design's pfiNominal
FIDUCIAL = 0x0200  # the arm would interfere with a fiducial fiber

# There is no out-of-reach bit.  anglesFromPositions abandons the solve the moment a
# target falls outside the annulus, so SOLUTION_OK-clear IS out of reach -- verified
# identical over 95760 targets.  Use isInvalid() rather than testing a mask, because
# "no solution" is the ABSENCE of a bit and cannot be expressed as one.

FLAG_NAMES = {NOT_FINITE: 'non-finite target', FIDUCIAL: 'fiducial interference',
              TOO_CLOSE_TO_CENTER: 'too close to centre',
              TOO_FAR_FROM_CENTER: 'too far from centre',
              IN_OVERLAPPING_REGION: 'theta overlap'}


# ── functions ─────────────────────────────────────────────────────────────────
def anglesFromPositions(calibModel, cIdx, positions):
    """Convert fiber positions to theta, phi angles from the CCW limit.

    Solves the two-link geometry from a calibModel alone: no PFI, no connection.

    Parameters
    ----------
    calibModel : PFIDesign
        Cobra calibration model.
    cIdx : int or ndarray of int
        Cobra indices (0-based) the positions correspond to, broadcast against
        positions: a scalar solves many positions for one cobra, and cIdx[:, None]
        against (n, m) positions solves m positions for each of n cobras.
    positions : ndarray of complex
        Fiber positions.

    Returns
    -------
    (tht, phi, flags) : each of shape broadcast(cIdx, positions).shape + (2,)
        Both possible solutions per position, with the flag bits below describing each.
        A NaN position is flagged SOLUTION_OK with NaN angles; a cobra with L1 or L2 of
        zero gets no flag and NaN angles.
    """
    # Per-cobra arrays keep the shape of cIdx and broadcast in the arithmetic
    cIdx = np.asarray(cIdx)
    positions = np.broadcast_to(positions, np.broadcast_shapes(cIdx.shape, np.shape(positions)))

    # Calculate the cobras rotation angles applying the law of cosines
    relativePositions = positions - calibModel.centers[cIdx]
    distance = np.abs(relativePositions)
    L1 = calibModel.L1[cIdx]
    L2 = calibModel.L2[cIdx]
    rMin, rMax = _reachAnnulus(L1, L2, calibModel.phiIn[cIdx], calibModel.phiOut[cIdx])
    phiIn = calibModel.phiIn[cIdx] + np.pi
    phiOut = calibModel.phiOut[cIdx] + np.pi
    tht0 = calibModel.tht0[cIdx]
    overlapEnd = (calibModel.tht1[cIdx] - tht0) % (2 * np.pi)
    spotAngle = np.angle(relativePositions)

    # NaN distances fail both reach tests and fall through to the solve
    good = (L1 != 0) & (L2 != 0)
    tooFar = good & (distance > rMax)
    tooClose = good & ~tooFar & (distance < rMin)
    solved = good & ~tooFar & ~tooClose

    with np.errstate(invalid='ignore', divide='ignore'):
        ang1 = np.arccos((L1 ** 2 + L2 ** 2 - distance ** 2) / (2 * L1 * L2))
        ang2 = np.arccos((L1 ** 2 + distance ** 2 - L2 ** 2) / (2 * L1 * distance))

    # Regular solution, phi between 0 and pi, no checking for phi hard stops.  Out of
    # reach, theta points at the spot and phi is fully extended (too far) or folded (too
    # close), where theta is undetermined.
    thtRegular = np.where(tooFar | tooClose, spotAngle - tht0, spotAngle + ang2 - tht0) % (2 * np.pi)
    thtRegular = np.where(good, thtRegular, np.nan)
    phiRegular = np.select([tooFar, tooClose, solved], [np.pi, 0, ang1 - phiIn], np.nan)

    # Additional solution, on the other side of the phi arc
    phiNegative = solved & (ang1 > 0) & (ang1 <= np.pi / 2)
    phiBeyondPi = solved & (ang1 > np.pi / 2) & (ang1 < np.pi)
    thtOther = np.where(phiNegative | phiBeyondPi, (spotAngle - ang2 - tht0) % (2 * np.pi), np.nan)
    phiOther = np.select([phiNegative, phiBeyondPi], [-ang1 - phiIn, 2 * np.pi - ang1 - phiIn], np.nan)

    flagsRegular = np.zeros(positions.shape, dtype='u2')
    flagsRegular[tooFar] |= TOO_FAR_FROM_CENTER
    flagsRegular[tooClose] |= TOO_CLOSE_TO_CENTER
    flagsRegular[solved] |= SOLUTION_OK
    flagsRegular[thtRegular <= overlapEnd] |= IN_OVERLAPPING_REGION

    flagsOther = np.zeros(positions.shape, dtype='u2')
    flagsOther[phiNegative] |= PHI_NEGATIVE
    flagsOther[phiBeyondPi] |= PHI_BEYOND_PI
    # Only if the phi hard stops allow it
    flagsOther[phiNegative & (phiIn <= -ang1)] |= SOLUTION_OK
    flagsOther[phiBeyondPi & (phiOut >= 2 * np.pi - ang1)] |= SOLUTION_OK
    flagsOther[thtOther <= overlapEnd] |= IN_OVERLAPPING_REGION

    # The two solutions go along a new last axis
    tht = np.stack([thtRegular, thtOther], axis=-1)
    phi = np.stack([phiRegular, phiOther], axis=-1)
    flags = np.stack([flagsRegular, flagsOther], axis=-1)

    return tht, phi, flags


def isInvalid(flags):
    """A target was requested for this cobra and cannot be commanded.

    NOT_FINITE is excluded on purpose: a target that was never given a position is
    not a target the cobra failed to reach, and conflating them makes an ordinary
    design look out of spec.

    Parameters
    ----------
    flags : ndarray of uint16, (nCobras,)
        Per-cobra verdict from validateTargets.

    Returns
    -------
    ndarray of bool, (nCobras,)
        True where the cobra has a target it cannot be commanded to.
    """
    hasTarget = (flags & NOT_FINITE) == 0
    return hasTarget & (((flags & SOLUTION_OK) == 0) | ((flags & FIDUCIAL) != 0))


def loadFidAvoidance():
    """Per-cobra fiducial avoidance limits, one row per cobra in cobra_id order.

    Columns theta_limit_1, theta_limit_2 and max_phi_angle; NaN means the cobra is
    unconstrained.  Read from the butler; needs no calibModel.

    Raises whatever the butler raises: the table ships with pfs_instdata, so its
    absence is a broken installation and the underlying error says which.

    Returns
    -------
    DataFrame
        Columns cobra_id, theta_limit_1, theta_limit_2, max_phi_angle.
    """
    # Imported here, not at module scope: the reach test needs no butler at all, and
    # pfs.utils is not otherwise a dependency of this package, so the reach test stays
    # usable without pfs_utils set up.
    from pfs.utils import butler

    return butler.Butler().get('cobraInterference')


def thetaRange(calibModel):
    """Angular travel of each theta stage, CCW hard stop to CW hard stop, in radians.

    Slightly more than 2*pi -- the two hard stops overlap by ~20 deg by design, so a
    target within that wedge is reachable at two local angles.  The +pi/-pi framing is what makes the result land in [pi, 3*pi) instead
    of wrapping the overlap back to near zero, which is the whole point: writing the
    modulo without it silently turns a 380 deg range into a 20 deg one.

    Parameters
    ----------
    calibModel : PFIDesign
        Cobra calibration model.

    Returns
    -------
    ndarray of float, (nCobras,)
        Theta travel per cobra in radians, in [pi, 3*pi).
    """
    return (calibModel.tht1 - calibModel.tht0 + np.pi) % (np.pi * 2) + np.pi


def reachAnnulus(calibModel):
    """Inner and outer radius each cobra's fiber can reach, from its centre.

    Derived from the phi hard stops rather than from |L1-L2| and L1+L2, which assume
    phi can reach -pi and 0 exactly.  It cannot: phiIn and phiOut are per-cobra and
    the arm is correspondingly less able to fold or extend.  Using the loose bound
    passes targets the cobra cannot actually reach -- on the current model it is wrong
    by more than 0.1 mm for 364 cobras.

    No safety margin is applied; see applySafetyMargin.

    Parameters
    ----------
    calibModel : PFIDesign
        Cobra calibration model.

    Returns
    -------
    rMin, rMax : ndarray of float, (nCobras,)
        Inner and outer reach radius per cobra, in mm.
    """
    return _reachAnnulus(calibModel.L1, calibModel.L2, calibModel.phiIn, calibModel.phiOut)


def _reachAnnulus(L1, L2, phiIn, phiOut):
    """Inner and outer reach radius from the arm lengths and phi hard stops, elementwise."""
    rMin = np.abs(L1 + L2 * np.exp(1j * np.maximum(-np.pi, phiIn)))
    rMax = np.abs(L1 + L2 * np.exp(1j * np.minimum(phiOut, 0)))
    return rMin, rMax


def applySafetyMargin(rMin, rMax, safetyMargin, maximumDistance=np.inf):
    """Shrink a reach annulus to what an assignment run will actually accept.

    The margin is added to rMin and taken off rMax; maximumDistance caps rMax on top
    of that, so one call gives the effective annulus.

    Scalars or arrays.  A margin large enough to invert the annulus is not rejected:
    rMin then exceeds rMax and nothing is accepted.

    Parameters
    ----------
    rMin, rMax : float or ndarray
        Reach annulus to shrink, as returned by reachAnnulus.
    safetyMargin : float
        Millimetres to shave off each end.
    maximumDistance : float, optional
        Hard cap on rMax, applied after the margin.  No cap by default.

    Returns
    -------
    rMin, rMax : same shape as the inputs
        The effective annulus.
    """
    return rMin + safetyMargin, np.minimum(rMax - safetyMargin, maximumDistance)


def validateTargets(calibModel, targets, skipFiducialInterferenceCheck=False):
    """Per-cobra verdict on commanded target positions.

    Parameters
    ----------
    calibModel : PFIDesign
        Cobra calibration model; also the source of truth for which cobras are broken.
    targets : ndarray of complex, (nCobras,)
        Commanded position per cobra, NaN where the design assigns none.
    skipFiducialInterferenceCheck : bool, optional
        Leave FIDUCIAL unset without testing it.  The bit is then absent because it
        was never measured, not because the arm is clear.

    Returns
    -------
    flags : ndarray uint16, (nCobras,)
        0 where the target is good, otherwise a bitwise-or of the constants above.
    """
    targets = np.asarray(targets)
    nCobras = len(calibModel.centers)
    if targets.shape != (nCobras,):
        raise ValueError(f'targets must have shape ({nCobras},), got {targets.shape}')

    flags = np.zeros(nCobras, dtype='u2')

    notFinite = np.isnan(targets.real) | np.isnan(targets.imag)
    flags[notFinite] |= NOT_FINITE

    # Only targets with a position are judged: having none is not a validation
    # failure, it is reported as NOT_FINITE and left at that.
    idx = np.flatnonzero(~notFinite)
    if len(idx) == 0:
        return flags

    # Reach comes straight from the solve, so the verdict cannot disagree with
    # SOLUTION_OK.  Bounds are asymmetric on purpose -- the accepted range is
    # (rMin, rMax] -- because that is what the targeting side does: TargetSelector
    # uses `distances > rMin` and query_ball_point is inclusive at rMax.
    _, _, solFlags = anglesFromPositions(calibModel, idx, targets[idx])
    flags[idx] |= solFlags[:, 0].astype(flags.dtype)

    if not skipFiducialInterferenceCheck:
        fidAvoidance = loadFidAvoidance()
        flags[fiducialInterference(calibModel, targets, fidAvoidance)] |= FIDUCIAL

    return flags


def fiducialInterference(calibModel, targets, fidAvoidance):
    """Cobras whose arm would interfere with a fiducial fiber at its target.

    A cobra interferes when its phi exceeds the row's maximum AND either the theta
    arm or the fiber tip lies inside the forbidden theta window.  The window wraps:
    `limit1 > limit2` means it spans 0 degrees.  Rows with no limits (the "no
    collision" entries) constrain nothing.

    Parameters
    ----------
    calibModel : PFIDesign
        Cobra calibration model.
    targets : ndarray of complex, (nCobras,)
        Commanded position per cobra, NaN where the design assigns none.
    fidAvoidance : DataFrame
        Per-cobra fiducial avoidance limits, from loadFidAvoidance.

    Returns
    -------
    ndarray of bool, (nCobras,)
        True where the arm would interfere with a fiducial fiber.
    """
    targets = np.asarray(targets)
    nCobras = len(calibModel.centers)

    assigned = ~(np.isnan(targets.real) | np.isnan(targets.imag))
    thetas = np.full(nCobras, np.nan)
    phis = np.full(nCobras, np.nan)
    if assigned.any():
        idx = np.flatnonzero(assigned)
        tht, phi, _ = anglesFromPositions(calibModel, idx, targets[idx])
        thetas[idx], phis[idx] = tht[:, 0], phi[:, 0]

    return fiducialInterferenceFromAngles(calibModel, thetas, phis, fidAvoidance)


def fiducialInterferenceFromAngles(calibModel, thetas, phis, fidAvoidance):
    """As fiducialInterference, but from local angles that are already known.

    Where the test is implemented.  Working from angles skips the solve, and avoids
    resolving a position back onto the other branch in the overlapping region.

    Parameters
    ----------
    calibModel : PFIDesign
        Cobra calibration model.
    thetas, phis : ndarray of float, (nCobras,)
        Local angles in radians, NaN where the cobra has no target.
    fidAvoidance : DataFrame or None
        Per-cobra fiducial avoidance limits.  Nothing interferes when None.

    Returns
    -------
    ndarray of bool, (nCobras,)
        True where the arm would interfere with a fiducial fiber.
    """
    thetas = np.asarray(thetas, dtype=float)
    phis = np.asarray(phis, dtype=float)
    nCobras = len(calibModel.centers)
    out = np.zeros(nCobras, dtype=bool)

    assigned = np.flatnonzero(np.isfinite(thetas) & np.isfinite(phis))
    if len(assigned) == 0 or fidAvoidance is None:
        return out

    lo = fidAvoidance.theta_limit_1.to_numpy()[assigned]
    hi = fidAvoidance.theta_limit_2.to_numpy()[assigned]
    maxPhi = fidAvoidance.max_phi_angle.to_numpy()[assigned]
    constrained = ~(np.isnan(lo) | np.isnan(hi) | np.isnan(maxPhi))
    if not constrained.any():
        return out

    thetaDeg = np.rad2deg((thetas[assigned] + calibModel.tht0[assigned]) % (2 * np.pi))
    # No modulo on phi.  Local phi legitimately exceeds 180 degrees -- 2354 of 2394
    # cobras have a span wider than that -- and folding it would turn 188.9 into 8.9,
    # which passes a 60-161 degree limit and silently clears the test at full
    # extension, exactly where interference is most likely.
    phiDeg = np.rad2deg(phis[assigned])

    # Angle of the fiber tip about the cobra centre.  Checked as well as theta because
    # the elbow can clear a fiducial while the tip does not.
    ang1 = calibModel.tht0[assigned] + np.deg2rad(thetaDeg)
    ang2 = ang1 + np.deg2rad(phiDeg) + calibModel.phiIn[assigned]
    endDeg = np.rad2deg(np.angle(calibModel.L1[assigned] * np.exp(1j * ang1)
                                 + calibModel.L2[assigned] * np.exp(1j * ang2))
                        - calibModel.tht0[assigned]) % 360

    hit = constrained & (phiDeg > maxPhi) & (_inWindow(thetaDeg, lo, hi)
                                             | _inWindow(endDeg, lo, hi))
    out[assigned[hit]] = True
    return out


def _inWindow(angleDeg, lo, hi):
    """Whether each angle falls inside [lo, hi], elementwise.

    The window wraps through zero when lo > hi, which is not an edge case: several
    cobras have forbidden ranges straddling 0 degrees.

    Parameters
    ----------
    angleDeg : ndarray of float
        Angles to test, in degrees.
    lo, hi : ndarray of float
        Window bounds in degrees, same shape as angleDeg.

    Returns
    -------
    ndarray of bool
        True where the angle lies inside the window.
    """
    wrapped = lo > hi
    return np.where(wrapped,
                    (angleDeg >= lo) | (angleDeg <= hi),
                    (angleDeg >= lo) & (angleDeg <= hi))


def summarise(flags, isScience=None):
    """Counts per failure mode, for logging and for the out-of-spec decision.

    isScience restricts the tally to cobras carrying a science-class target: an
    cobra with no position failing validation is not a design defect.

    Parameters
    ----------
    flags : ndarray of uint16, (nCobras,)
        Per-cobra verdict from validateTargets.
    isScience : ndarray of bool, (nCobras,), optional
        Restrict the tally to these cobras.  All of them when None.

    Returns
    -------
    dict
        One entry per FLAG_NAMES bit, plus 'invalid' and 'checked'.
    """
    mask = np.ones(len(flags), dtype=bool) if isScience is None else np.asarray(isScience)
    counts = {name: int(np.sum(((flags & bit) != 0) & mask))
              for bit, name in FLAG_NAMES.items()}
    counts['invalid'] = int(np.sum(isInvalid(flags) & mask))
    counts['checked'] = int(mask.sum())
    return counts
