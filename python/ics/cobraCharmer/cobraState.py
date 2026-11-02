import contextlib

motorScales = dict()


def mapId(cobraId, motor, direction):
    """ Return a hashable key for (cobra, motor, direction) """

    if direction not in {'ccw', 'cw'}:
        raise ValueError(f"invalid direction: {direction}")
    if motor not in {'theta', 'phi'}:
        raise ValueError(f"invalid motor: {motor}")

    return (cobraId, motor, direction)


@contextlib.contextmanager
def unscaled(cobraIds, motor='phi'):
    """ Drive these cobras' motor at its nominal on-time, scale 1.0, inside the block.

    Both directions are overridden.  On exit, including when the block raises, every
    scale is put back to its value on entry, so a scale learned or clipped inside the
    block is discarded.  Learning itself is not switched off; `CobraCoach.phiScaling`
    gates it.

    Args
    ----
    cobraIds : iterable of int
        Cobra indices into the calibration product, as taken by `mapId`.
    motor : {'theta', 'phi'}
        The motor to run unscaled.
    """
    keys = [mapId(int(cobraId), motor, direction)
            for cobraId in cobraIds for direction in ('ccw', 'cw')]
    saved = {key: motorScales[key] for key in keys if key in motorScales}

    for key in keys:
        motorScales[key] = 1.0
    try:
        yield
    finally:
        for key in keys:
            if key in saved:
                motorScales[key] = saved[key]
            else:
                motorScales.pop(key, None)
