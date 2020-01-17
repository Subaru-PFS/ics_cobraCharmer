import numpy as np

import fitsio
import sep

def getObjects(im, sigma=5.0):
    data = im.astype('f4')
    bkg = sep.Background(data)
    data_sub = data - bkg

    mn = np.mean(data_sub)
    std = np.std(data_sub)
    thresh = sigma * std
    objects = sep.extract(data_sub, thresh=thresh)

    return objects, data_sub, bkg

def spots(name, sigma=5.0, doTrim=True, disp=None):
    im = fitsio.read(name)
    objects, imSub, _ = getObjects(im, sigma=sigma)

    if disp is not None:
        disp.set('frame clear')
        disp.set_np2arr(imSub)
        disp.set('regions color green')
        for o in objects:
            disp.set(f"regions command {{point {o['x']} {o['y']}}}")

    if doTrim:
        # CIT Only -- wrap this, CPL.
        w = (objects['y'] < (objects['x'] + 500)) & (objects['y'] > (objects['x'] - 500))
        objects = objects[w]

        if disp is not None:
            disp.set('regions color red')
            for o in objects:
                disp.set(f"regions command {{circle {o['x']} {o['y']} 10}}")

    return objects, imSub
