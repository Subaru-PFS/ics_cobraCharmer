def cameraFactory(name=None):
    try:
        return cameraFactory.__camera
    except:
        if name is None:
            raise ValueError('camera type must be specified')
        if name == 'cit':
            import citcam as citcam
            cameraFactory.__camera = citcam
        elif name == 'asrd':
            import ics.cobraCharmer.asrdCam as asrdCam
            cameraFactory.__camera = asrdCam

        return cameraFactory.__camera

class cameraProto(object):
    def expose(self, exptime=None, name=None, cameraArgs=None):
        """ Returns im, filename """
        raise NotImplementedError()


    def spots(self, name):
        raise NotImplementedError()
