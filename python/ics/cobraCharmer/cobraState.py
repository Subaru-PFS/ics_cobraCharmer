motorScales = dict()


def mapId(cobraId, motor, direction):
    """ Return a hashable key for (cobra, motor, direction) """

    if direction not in {'ccw', 'cw'}:
        raise ValueError(f"invalid direction: {direction}")
    if motor not in {'theta', 'phi'}:
        raise ValueError(f"invalid motor: {motor}")

    return (cobraId, motor, direction)
