import numpy as np

# return the tranformation parameters and a function that can convert origPoints to newPoints


def simpleTransform(origPoints, newPoints):
    origCenter = np.mean(origPoints)
    newCenter = np.mean(newPoints)
    origVectors = origPoints - origCenter
    newVectors = newPoints - newCenter
    scale = np.sum(np.abs(newVectors)) / np.sum(np.abs(origVectors))
    diffAngles = ((np.angle(newVectors) - np.angle(origVectors)) + np.pi) % (2*np.pi) - np.pi
    tilt = np.sum(diffAngles * np.abs(origVectors)) / np.sum(np.abs(origVectors))
    offset = -origCenter * scale * np.exp(tilt * (1j)) + newCenter

    def tr(x):
        return x * scale * np.exp(tilt * (1j)) + offset
    return offset, scale, tilt, tr


def skTransform(origPoints, newPoints):
    import skimage.transform as tf

    tfa = tf.SimilarityTransform()
    tfa.estimate(origPoints, newPoints)

    return tfa.translation, tfa.scale, tfa.rotation, tfa


def laydown(points):
    """ Remove any gross angle, which lets us order the points by increasing x """
    fit = np.polyfit(points[:, 0], points[:, 1], 1)
    ang = np.arctan(fit[0])
    rotmat = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    flat = points @ rotmat

    idx = np.argsort(flat[:, 0])
    return idx, flat


makeTransform = simpleTransform
