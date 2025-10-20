import logging

import numpy as np


def measureSpots(centers, dataSet, positions, names=None, disp=None,
                 trackCenters=True, sigma=10.0, trackRadius=20.0):

    """ Measure a set of spots at known positions."""

    if any(positions != sorted(positions)):
        raise ValueError("positions must be sorted.")

    if names is None:
        names = dataSet.namelist.keys()
    if len(positions) != len(names):
        raise ValueError('number of positions (%d) must match the number of names (%d)' %
                         (len(positions), len(names)))

    nCobras = len(centers)
    res = np.zeros(len(positions),
                   dtype=dict(names=('name', 'pos', 'centers'),
                              formats=(np.object, np.float32, f'{nCobras}c8')))

    # We _start_ by matching the measured centers against the provided centers. If
    # the trackCenters flag is set, we then match to the previous position of the cobra.
    nearestCenters = centers
    radii = None
    for i in range(len(positions)):
        res[i]['name'] = names[i]
        res[i]['pos'] = positions[i]
        cs, _ = dataSet.spots(names[i], sigma=sigma, disp=disp)

        if len(cs) != len(centers):
            raise RuntimeError('in %s, number of spots (%d) != number of cobras (%d)' %
                               (names[i], len(cs), len(centers)))

        spots = np.array([c['x']+c['y']*(1j) for c in cs])
        idx = lazyIdentification(nearestCenters, spots, radii=radii)
        nomatch_w = (idx == -1)
        if nomatch_w.sum() > 0:
            logging.warning(f'failed to match spots {np.where(nomatch_w)[0]} for {names[i]} at {positions[i]} steps')
        if trackCenters:
            nearestCenters = spots[idx]
            radii = trackRadius

        res[i]['centers'][:] = spots[idx]

    return res

def makeSpotRows(datasetName, imageName, phi, theta, spots):
    pass

def getObjects(im, sigma=None, doBackground=None):
    spotter = Spotter(sigma=sigma, doBackground=doBackground)
    objects, _, _ = spotter.getObjects(im)

    return objects
