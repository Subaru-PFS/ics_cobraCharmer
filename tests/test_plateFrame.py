"""The cobra plate: plateFrame.fitPlate, pfiDesign FIBER_REFERENCE, CobraCoach._updatePlate and the
targets of engineer.moveThetaPhi.

Needs pfs_instdata (PFS_INSTDATA_DIR) for the calibration XML; the anchors are set on the model.
"""
import types
import xml.etree.ElementTree as ElementTree

import numpy as np
import pytest

from ics.cobraCharmer import pfi as pfiControl
from ics.cobraCharmer import pfiDesign, plateFrame
from ics.cobraCharmer.cobraCoach import cobraCoach, engineer

PLATE = plateFrame.PlateFit(0.040 - 0.010j, (1 - 100e-6) * np.exp(1j * np.deg2rad(12 / 3600)), 20, 5.0)


def anchorReference(n=28, seed=0):
    rng = np.random.default_rng(seed)
    return 180 * np.sqrt(rng.uniform(0, 1, n)) * np.exp(1j * rng.uniform(0, 2 * np.pi, n))


# ── plateFrame ──────────────────────────────────────────────────────────────────

def test_fitRecoversTheSimilarity():
    rng = np.random.default_rng(0)
    reference = anchorReference()
    noise = 0.005 * (rng.normal(size=len(reference)) + 1j * rng.normal(size=len(reference)))
    measured = PLATE.apply(reference) + noise
    measured[3] += 0.08                       # an anchor off by 80 um: clipped
    measured[7] += 3.0                        # a wrong spot: rejected by the median match
    measured[11] = np.nan                     # not measured
    fit = plateFrame.fitPlate(measured, reference)
    assert fit.nUsed == len(reference) - 3
    edge = 200.0 * np.exp(1j * np.linspace(0, 2 * np.pi, 16))
    assert np.max(np.abs(fit.apply(edge) - PLATE.apply(edge))) < 0.008
    assert fit.rmsUm < 10


def test_fitIsMadeOnTheAnchorsKept():
    reference = anchorReference(20, seed=1)
    measured = PLATE.apply(reference)
    measured[:4] += 0.025                     # inside the clip at first, outside once the others agree
    fit = plateFrame.fitPlate(measured, reference)
    kept = np.abs(fit.apply(reference) - measured) < plateFrame.CLIP_MM
    assert fit.nUsed == kept.sum()
    assert fit.rmsUm == pytest.approx(1e3 * np.sqrt(np.mean(np.abs(fit.apply(reference) - measured)[kept] ** 2)))


def test_fitRefusedWithFewAnchors():
    reference = np.arange(9) * (10 + 10j)
    assert plateFrame.fitPlate(reference + 0.01, reference) is None
    reference = np.arange(12) * (10 + 10j)
    measured = reference + 0.01
    measured[:3] = np.nan
    assert plateFrame.fitPlate(measured, reference) is None


def test_identity():
    z = np.array([1 + 2j, -3.5 + 0.25j])
    assert np.array_equal(plateFrame.IDENTITY.apply(z), z)


# ── pfiDesign ───────────────────────────────────────────────────────────────────

def twoCobraXml(tmp_path, reference=None):
    """A calibration XML of the first two cobras of the default one; the first given a FIBER_REFERENCE."""
    model = pfiDesign.PFIDesign.loadPfi('ALL', None)
    root = ElementTree.Element('ARM_DATA')
    for container in model.dataContainers[:2]:
        root.append(container)
    if reference is not None:
        node = ElementTree.SubElement(root.findall('ARM_DATA_CONTAINER')[0], 'FIBER_REFERENCE')
        for tag, text in reference.items():
            ElementTree.SubElement(node, tag).text = text
    path = tmp_path / 'two.xml'
    ElementTree.ElementTree(root).write(path)
    return path


def test_pfiDesignReadsFiberReference(tmp_path):
    model = pfiDesign.PFIDesign(twoCobraXml(tmp_path, dict(Global_pos_x='1.5', Global_pos_y='-2.25')))
    assert model.fiberReference[0] == 1.5 - 2.25j
    assert np.isnan(model.fiberReference[1])


def test_pfiDesignWithoutFiberReference(tmp_path):
    assert np.isnan(pfiDesign.PFIDesign(twoCobraXml(tmp_path)).fiberReference).all()


def test_malformedFiberReference(tmp_path):
    with pytest.raises(ValueError, match='FIBER_REFERENCE'):
        pfiDesign.PFIDesign(twoCobraXml(tmp_path, dict(Global_pos_x='1.0')))


def test_newCentresDropTheFiberReference(tmp_path):
    model = pfiDesign.PFIDesign(twoCobraXml(tmp_path, dict(Global_pos_x='1.5', Global_pos_y='-2.25')))
    model.updateGeometry(centers=model.centers + 0.001)
    assert np.isnan(model.fiberReference).all()
    assert all(c.find('FIBER_REFERENCE') is None for c in model.dataContainers)
    model.updateGeometry(thetaArms=model.L1)                   # arms only: nothing more to drop


# ── CobraCoach ──────────────────────────────────────────────────────────────────

class Source:
    """A plate source handing out the plates of a list, one per frame, and recording its calls."""

    def __init__(self, plates):
        self.plates = list(plates)
        self.calls = []

    def __call__(self, frameNum, measured):
        self.calls.append((frameNum, measured.copy()))
        return self.plates[min(len(self.calls), len(self.plates)) - 1]


@pytest.fixture(scope='module')
def coachModule(tmp_path_factory):
    cc = cobraCoach.CobraCoach('localhost', loadModel=False, rootDir=tmp_path_factory.mktemp('runs'))
    cc.loadModel()
    cc.pfi = pfiControl.PFI(doConnect=False, doLoadModel=False)
    cc.pfi.calibModel = cc.calibModel
    ok = np.flatnonzero((cc.calibModel.status & cc.calibModel.COBRA_OK_MASK) == 0)
    cc.calibModel.fiberReference[ok[:20]] = cc.calibModel.centers[ok[:20]] + 0.1
    return cc


@pytest.fixture
def coach(coachModule):
    cc = coachModule
    xml = cc.xmlCenters
    cc._resetPlate()
    cc.plateSource = None
    engineer.setCobraCoach(cc)
    yield cc
    cc.calibModel.centers[:] = xml
    cc.plateSource = None
    cc._resetPlate()


def frame(cc, n=1):
    """n frames through _updatePlate, every cobra detected at its centre."""
    for _ in range(n):
        cc._updatePlate(1234500, cc.calibModel.centers.copy(), np.ones(cc.nCobras, bool))


def test_noSourceLeavesTheCentres(coach):
    before = coach.calibModel.centers.copy()
    frame(coach)
    assert np.array_equal(coach.calibModel.centers, before)


def test_centresFollowThePlate(coach):
    cc = coach
    cc.plateSource = Source([plateFrame.IDENTITY, PLATE, None])
    frame(cc)
    assert np.array_equal(cc.calibModel.centers, cc.xmlCenters)
    frame(cc)
    assert np.allclose(cc.calibModel.centers, PLATE.apply(cc.xmlCenters), rtol=0, atol=1e-12)
    assert cc.plate is PLATE
    frame(cc)                                                       # None: back to the XML centres
    assert np.array_equal(cc.calibModel.centers, cc.xmlCenters)


def test_undetectedCobrasAreNaN(coach):
    cc = coach
    source = Source([PLATE])
    cc.plateSource = source
    detected = np.ones(cc.nCobras, bool)
    detected[5] = False
    cc._updatePlate(1234500, cc.calibModel.centers.copy(), detected)
    measured = source.calls[0][1]
    assert np.isnan(measured[5]) and np.isfinite(np.delete(measured, 5)).all()


def test_failingSourceKeepsTheCentres(coach):
    cc = coach
    cc.plateSource = Source([PLATE])
    frame(cc)

    def broken(frameNum, measured):
        raise RuntimeError('no database')

    cc.plateSource = broken
    frame(cc)
    assert np.allclose(cc.calibModel.centers, PLATE.apply(cc.xmlCenters), rtol=0, atol=1e-12)


def test_recentredCobrasAreFoldedIntoTheXmlFrame(coach):
    cc = coach
    source = Source([PLATE, plateFrame.IDENTITY])
    cc.plateSource = source
    frame(cc)
    xml = cc.xmlCenters
    cc.calibModel.centers[7] += 0.5                                 # a recentring, measured under PLATE
    recentred = cc.calibModel.centers[7]
    frame(cc)                                                       # the identity: back to the XML frame
    assert len(source.calls) == 2                                   # still asked, still recorded
    assert cc.xmlCenters[7] == pytest.approx((recentred - PLATE.a) / PLATE.b, abs=1e-12)
    assert np.array_equal(np.delete(cc.xmlCenters, 7), np.delete(xml, 7))
    assert np.array_equal(cc.calibModel.centers, cc.xmlCenters)


def test_replacedModelIsTakenAsTheXml(coach):
    cc = coach
    cc.plateSource = Source([PLATE])
    frame(cc)
    model, pfiModel = cc.calibModel, cc.pfi.calibModel
    replacement = pfiDesign.PFIDesign.loadPfi('ALL', None)
    replacement.fiberReference[:] = model.fiberReference
    try:
        cc.calibModel = cc.pfi.calibModel = replacement
        fresh = replacement.centers.copy()
        frame(cc)
        assert np.array_equal(cc.xmlCenters, fresh)
        assert np.allclose(replacement.centers, PLATE.apply(fresh), rtol=0, atol=1e-12)
    finally:
        cc.calibModel, cc.pfi.calibModel = model, pfiModel
        cc._resetPlate()


def test_plateJumpRebasesTheLastAngles(coach):
    """After the plate moves, the last positions give the same angles as before the move would
    under the new centres: the jump is not counted as motion."""
    cc = coach
    cIds = okCobras(cc, 23)
    thetas, phis = np.zeros(cc.nCobras), np.zeros(cc.nCobras)
    thetas[cIds], phis[cIds] = 2.0, 1.0
    cc.cobraInfo['position'][:] = 0
    cc.cobraInfo['position'][cIds] = cc.pfi.anglesToPositions(cc.allCobras, thetas, phis)[cIds]
    cc.cobraInfo['thetaAngle'][:], cc.cobraInfo['phiAngle'][:] = thetas, phis
    cc.plateSource = Source([PLATE])
    frame(cc)
    expect, expectPhi, _ = cc.pfi.positionsToAngles(cc.allCobras[cIds],
                                                    cc.cobraInfo['position'][cIds].astype(complex))
    assert np.allclose(cc.cobraInfo['thetaAngle'][cIds], expect[:, 0], atol=1e-5)
    assert np.allclose(cc.cobraInfo['phiAngle'][cIds], expectPhi[:, 0], atol=1e-5)
    assert np.max(np.abs(cc.cobraInfo['thetaAngle'][cIds] - 2.0)) > 1e-4             # they did change


def test_noAnchorsNoPlate(coach):
    cc = coach
    saved = cc.calibModel.fiberReference.copy()
    cc.calibModel.fiberReference[:] = np.nan
    cc.plateSource = Source([PLATE])
    try:
        frame(cc)
        assert np.array_equal(cc.calibModel.centers, cc.xmlCenters)
        assert not cc.plateSource.calls
    finally:
        cc.calibModel.fiberReference[:] = saved


def test_anchorsGoneResetThePlate(coach):
    """New centres drop the anchors: the plate goes back to the identity, the centres stay."""
    cc = coach
    saved = cc.calibModel.fiberReference.copy()
    cc.plateSource = Source([PLATE])
    frame(cc)
    try:
        cc.calibModel.updateGeometry(centers=cc.calibModel.centers + 0.01)
        recentred = cc.calibModel.centers.copy()
        frame(cc)
        assert cc.plate is plateFrame.IDENTITY
        assert np.array_equal(cc.calibModel.centers, recentred)
        assert np.array_equal(cc.xmlCenters, recentred)
    finally:
        cc.calibModel.fiberReference[:] = saved


# ── engineer ────────────────────────────────────────────────────────────────────

def okCobras(cc, step):
    cIds = np.arange(0, cc.nCobras, step)
    return cIds[(cc.calibModel.status[cIds] & cc.calibModel.COBRA_OK_MASK) > 0]


def test_targetsReSolvedFromTheirXY(coach):
    cc = coach
    cIds = okCobras(cc, 13)
    thetas, phis = np.zeros(cc.nCobras), np.zeros(cc.nCobras)
    thetas[cIds], phis[cIds] = 2.5, 1.2
    xy = cc.pfi.anglesToPositions(cc.allCobras, thetas, phis)
    cc.plateSource = Source([PLATE])
    frame(cc)
    nDone, nKept, dTheta, dPhi = engineer.resolveFromXY(cIds, thetas, phis, xy)
    again = cc.pfi.anglesToPositions(cc.allCobras, thetas, phis)
    assert nDone == len(cIds) and nKept == 0
    assert np.max(np.abs(again[cIds] - xy[cIds])) < 1e-6
    assert dTheta > 1e-3


def test_noValidSolutionKeepsTheAngles(coach):
    cc = coach
    cIds = okCobras(cc, 97)
    thetas, phis = np.full(cc.nCobras, 2.0), np.full(cc.nCobras, 1.0)
    far = cc.calibModel.centers + 10.0                              # out of reach
    nDone, nKept, _, _ = engineer.resolveFromXY(cIds, thetas, phis, far)
    assert nDone == 0 and nKept == len(cIds)
    assert (thetas == 2.0).all() and (phis == 1.0).all()


def test_thetaOutOfRangeKeepsTheAngles(coach):
    cc = coach
    cIds = okCobras(cc, 13)
    thetas, phis = np.zeros(cc.nCobras), np.zeros(cc.nCobras)
    thetas[cIds], phis[cIds] = 0.0005, 1.2                          # at the CCW hard stop
    xy = cc.pfi.anglesToPositions(cc.allCobras, thetas, phis)
    cc.plateSource = Source([plateFrame.PlateFit(-0.2 + 0j, 1 + 0j, 0, 0.0)])
    frame(cc)
    before = thetas.copy()
    _, nKept, _, _ = engineer.resolveFromXY(cIds, thetas, phis, xy)
    assert (thetas[cIds] >= 0).all()
    assert nKept > 0 and np.array_equal(thetas[cIds][thetas[cIds] == before[cIds]],
                                        before[cIds][thetas[cIds] == before[cIds]])


def mover(cc, monkeypatch, tmp_path, commanded=None):
    """moveToAngles replaced: each move takes one frame (the plate source), then lands exactly."""

    def move(cobras, thetaAngles, phiAngles, thetaFast, phiFast, local):
        idx = np.array([cc.calibModel.findCobraByModuleAndPositioner(c.module, c.cobraNum) for c in cobras])
        if commanded is not None:
            commanded.append(dict(zip(idx, zip(thetaAngles, phiAngles))))
        frame(cc)                                                   # the frame of this move sets the plate
        at = cc.cobraInfo['position'].copy()
        at[idx] = cc.pfi.anglesToPositions(cc.allCobras[idx], thetaAngles, phiAngles)
        cc.cobraInfo['position'] = at
        cc.cobraInfo['detected'] = True
        return thetaAngles, phiAngles

    monkeypatch.setattr(cc, 'moveToAngles', move)
    monkeypatch.setattr(cc, 'camResetStack', lambda *a, **k: None)
    monkeypatch.setattr(cc, 'runManager', types.SimpleNamespace(dataDir=tmp_path))


def test_moveThetaPhiUnderAMovingPlate(coach, monkeypatch, tmp_path):
    """The plate moves with the first frame: sky targets end at their xy, re-solved; plate targets keep
    their angles and move with it.  Positions are single precision, so within 0.1 um."""
    cc = coach
    cIds = okCobras(cc, 17)
    sky = np.arange(len(cIds)) % 2 == 0
    thetas, phis = np.full(len(cIds), 2.5), np.full(len(cIds), 1.2)
    full = np.zeros(cc.nCobras), np.zeros(cc.nCobras)
    full[0][cIds], full[1][cIds] = thetas, phis
    xy0 = cc.pfi.anglesToPositions(cc.allCobras, *full)
    cc.plateSource = Source([PLATE])
    commanded = []
    mover(cc, monkeypatch, tmp_path, commanded)
    skyTargets = np.where(sky, xy0[cIds], np.nan)
    engineer.moveThetaPhi(cIds, thetas, phis, tolerance=1e-4, tries=6, newDir=False, skyTargets=skyTargets)

    for i, c in enumerate(cIds):
        if not sky[i]:
            assert all(m[c] == (thetas[i], phis[i]) for m in commanded if c in m)
    last = cc.cobraInfo['position']
    assert np.max(np.abs(last[cIds[sky]] - xy0[cIds[sky]])) < 1e-4                 # sky: at their xy
    moved = cc.pfi.anglesToPositions(cc.allCobras, *full)                           # plate: moved with it
    assert np.max(np.abs(last[cIds[~sky]] - moved[cIds[~sky]])) < 1e-4
    assert np.min(np.abs(moved[cIds[~sky]] - xy0[cIds[~sky]])) > 0.02


def test_skyTargetsDoNotDependOnTheCentresOfTheCall(coach, monkeypatch, tmp_path):
    """Angles solved under one plate, frames under another before the call (the first moves of a
    two-step convergence): the cobras still end at the design xy once the plate is known."""
    cc = coach
    cIds = okCobras(cc, 17)
    cc.plateSource = Source([PLATE])
    frame(cc)                                                       # the previous visit left PLATE
    full = np.zeros(cc.nCobras), np.zeros(cc.nCobras)
    full[0][cIds], full[1][cIds] = 2.5, 1.2
    design = cc.pfi.anglesToPositions(cc.allCobras, *full)
    th, ph, _ = cc.pfi.positionsToAngles(cc.allCobras, design)     # what fps solves, under PLATE
    cc.plateSource = Source([plateFrame.IDENTITY, plateFrame.IDENTITY, PLATE])
    frame(cc, 2)                                                    # two frames of the new visit
    mover(cc, monkeypatch, tmp_path)
    engineer.moveThetaPhi(cIds, th[cIds, 0], ph[cIds, 0], tolerance=1e-4, tries=4, newDir=False,
                          skyTargets=design[cIds])
    assert np.max(np.abs(cc.cobraInfo['position'][cIds] - design[cIds])) < 1e-4


def test_skyTargetWithARampRefused(coach, monkeypatch, tmp_path):
    cc = coach
    cIds = okCobras(cc, 97)
    mover(cc, monkeypatch, tmp_path)
    ramp = np.zeros((3, len(cIds)))
    ramp[1, 0] = 0.1
    with pytest.raises(RuntimeError, match='ramp'):
        engineer.moveThetaPhi(cIds, 2.0, 1.0, tries=3, newDir=False, phiRamp=ramp,
                              skyTargets=np.full(len(cIds), 1 + 1j))
