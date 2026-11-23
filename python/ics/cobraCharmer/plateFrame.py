"""Position of the cobra plate in one MCS frame, from the anchor cobras.

The cobra plate moves against the fiducial frame the MCS-to-PFI transform is fitted on: it shifts by
tens of microns, scales with temperature and turns slightly with the rotator.  Anchor cobras are broken
cobras whose fibre stays put on the plate; the calibration XML records where each one is in the frame of
its cobra centres (`pfiDesign.PFIDesign.fiberReference`).  Their measured positions give the similarity
that carries that reference frame into the frame of the measurement, and with it every cobra centre.
"""
from dataclasses import dataclass

import numpy as np

MATCH_MM = 0.15
"""Largest distance of an anchor from where the median offset of all anchors puts it, for the first fit."""
CLIP_MM = 0.03
"""Largest residual of an anchor kept in the fit."""
MIN_ANCHORS = 10
"""Fewest anchors a fit is made with."""


@dataclass(frozen=True)
class PlateFit:
    """Similarity carrying reference positions into a frame: position = a + b * reference.

    Attributes
    ----------
    a : `complex`
        Shift, mm.
    b : `complex`
        Rotation and scale.
    nUsed : `int`
        Anchors kept in the fit.
    rmsUm : `float`
        Rms residual of the anchors kept, microns.
    """
    a: complex
    b: complex
    nUsed: int
    rmsUm: float

    def apply(self, z):
        """Reference positions (complex mm, any shape) carried into the frame."""
        return self.a + self.b * np.asarray(z)

    def describe(self):
        """One line for a log: shift [um], rotation [arcsec], scale [ppm], anchors and rms."""
        return (f'shift=({1e3 * self.a.real:+.1f},{1e3 * self.a.imag:+.1f}) um '
                f'rotation={np.rad2deg(np.angle(self.b)) * 3600:+.1f} arcsec scale={1e6 * (abs(self.b) - 1):+.1f} ppm '
                f'anchors={self.nUsed} rms={self.rmsUm:.1f} um')


IDENTITY = PlateFit(0j, 1 + 0j, 0, float('nan'))
"""The plate where the XML centres put it: no shift, rotation or scale."""


def fitPlate(measured, reference, matchMm=MATCH_MM, clipMm=CLIP_MM, minAnchors=MIN_ANCHORS):
    """Similarity carrying the anchors' reference positions onto their measured ones.

    The anchors within `matchMm` of the median offset are fitted, then refitted without those further
    than `clipMm` from the fit until the anchors kept no longer change (five times at most); the fit
    returned is made on the anchors kept.

    Parameters
    ----------
    measured, reference : `numpy.ndarray` of `complex`, (nAnchors,)
        Positions in mm; NaN where an anchor was not measured.

    Returns
    -------
    `PlateFit`, or None when fewer than `minAnchors` anchors remain.
    """
    measured, reference = np.asarray(measured, dtype=complex), np.asarray(reference, dtype=complex)
    finite = np.isfinite(measured) & np.isfinite(reference)
    if finite.sum() < minAnchors:
        return None
    offset = measured[finite] - reference[finite]
    median = np.median(offset.real) + 1j * np.median(offset.imag)
    use = finite & (np.abs(np.where(finite, measured - reference, np.inf) - median) < matchMm)

    def fit(sel):
        design = np.c_[np.ones(sel.sum()), reference[sel]]
        (a, b), *_ = np.linalg.lstsq(design, measured[sel], rcond=None)
        return a, b, np.where(finite, np.abs(a + b * reference - measured), np.inf)

    for _ in range(5):
        if use.sum() < minAnchors:
            return None
        a, b, resid = fit(use)
        kept = finite & (resid < clipMm)
        if np.array_equal(kept, use):
            break
        use = kept
    if use.sum() < minAnchors:
        return None
    a, b, resid = fit(use)
    rms = float(np.sqrt(np.mean(resid[use] ** 2)))
    return PlateFit(complex(a), complex(b), int(use.sum()), 1e3 * rms)
