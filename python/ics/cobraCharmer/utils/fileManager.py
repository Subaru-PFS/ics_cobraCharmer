import os
import time

class ProcedureDirectory(object):
    def __init__(self, moduleName, experimentName=None, rootDir="/data/pfs/cobras", doCreate=True):
        """ Add a new experiment directory

        Args
        ----
        moduleName : str
          "SCnn" or "SpareN" or "PFI"

        experimentName : str
           If set, appended to the datestamp

        rootName : str
           The directory under which everything goes.
        """

        rootDir = os.path.normpath(rootDir)

        datestamp = time.strftime("%Y%m%d_%H%M%S")
        if experimentName is not None:
            datestamp = f'{datestamp}_{experimentName}'

        self.dirName = os.path.join(rootDir, moduleName, datestamp)
        if doCreate:
            self._createDirs()
        else:
            self.dirName = None

    @classmethod
    def loadFromPath(cls, path):
        if not os.path.isdir(path):
            raise ValueError('path must be a directory')

        instance = cls('', doCreate=False)
        instance.dirName = path

        return instance

    def _createDirs(self):
        os.makedirs(self.dirName, mode=0o2775)

        for d in (self.logDir, self.imageDir, self.outputDir):
            os.makedirs(d, mode=0o2775)

    @property
    def rootDir(self):
        return self.dirName

    @property
    def logDir(self):
        return os.path.join(self.dirName, 'logs')

    @property
    def imageDir(self):
        return os.path.join(self.dirName, 'images')

    @property
    def outputDir(self):
        return os.path.join(self.dirName, 'output')
    xmlDir = outputDir
