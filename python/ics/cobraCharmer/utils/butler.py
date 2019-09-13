import logging
import os
import pathlib
import time

import yaml

logger = logging.getLogger('butler')

dataRoot = pathlib.Path("/data/MCS")

class RunTree(object):
    def __init__(self, rootDir=None, doCreate=True):
        """ Create and track "runs": sequences of exposures taken as a unit, plus some output.

        Args
        ----
        rootName : str
           The directory under which everything goes. default=/data/MCS
        doCreate: : bool
           Whether to create a run now.
        """

        if rootDir is None:
            rootDir = dataRoot
        self.rootDir = pathlib.Path(rootDir)

        self.logger = logger
        self.logger.setLevel(logging.DEBUG)

        if not self.rootDir.is_dir():
            raise RuntimeError(f'{self.rootDir} is not an existing directory')

        self.runDir = None
        if doCreate:
            self.newRun()

    def newRun(self):
        """ Create a new run directory, plus children. """

        day = time.strftime('%Y%m%d')
        todayDirs = sorted(self.rootDir.glob(f'{day}_[0-9][0-9][0-9]'))
        if len(todayDirs) == 0:
            self.runDir = self.rootDir / f'{day}_000'
        else:
            _, lastRev = todayDirs[-1].name.split('_')
            nextRev = int(lastRev, base=10) + 1
            self.runDir = self.rootDir / f'{day}_{nextRev:03d}'

        self.runDir.mkdir(mode=0o2775, parents=True)
        for d in self.allDirs:
            d.mkdir(mode=0x2775)

        self.logger.warn('newRun: %s', self.runDir)
        return self.runDir

    @property
    def logDir(self):
        return self.runDir / 'logs'

    @property
    def dataDir(self):
        return self.runDir / 'data'

    @property
    def outputDir(self):
        return self.runDir / 'output'
    xmlDir = outputDir

    @property
    def allDirs(self):
        return self.logDir, self.dataDir, self.outputDir

def pathForRun(run):
    runPath = dataRoot / run
    return runPath

def spotsForRun(run):
    runPath = pathForRun(run)
    return runPath / 'output/' / 'spots.npz'

def _instDataDir():
    """ Return the directory under which instrument configuration is stored. """

    instDataRoot = os.environ['PFS_INSTDATA_DIR']
    if not instDataRoot:
        raise ValueError("PFS_INSTDATA_DIR environment variable must be set!")
    return pathlib.Path(instDataRoot)

def configPathForPfi(version=None, rootDir=None):
    """ Return the pathname for a PFI config file.

    Args
    ----
    version : str
       An identifying string. If not set, the "latest" one.
       PFI.yaml or $PFI_$version.yaml
    rootDir : str/Path
       If set, where to look for the config file.
       By default $PFS_INSTDATA_DIR/data/pfi/

    """

    if rootDir is None:
        instDataRoot = _instDataDir()
        rootDir = instDataRoot / 'data' / 'pfi'

    if version is None:
        fname = 'PFI.yaml'
    else:
        fname = f'PFI_{version}.yaml'

    return rootDir / fname

def configPathForModule(module, version=None, rootDir=None):
    """ Return the pathname for a module config file.

    Args
    ----
    version : str
       An identifying string. If not set, the "latest" one.
       SC03.yaml or SCO3_$version.yaml
    rootDir : str/Path
       If set, where to look for the config file.
       By default $PFS_INSTDATA_DIR/data/pfi/modules/$module

    """

    if rootDir is None:
        instDataRoot = _instDataDir()
        rootDir = instDataRoot / 'data' / 'pfi' / 'modules'

    if version is None:
        fname = 'PFI.yaml'
    else:
        fname = f'PFI_{version}.yaml'

    return rootDir / fname

def modulesForPfi(version=None, rootDir=None):
    """ Return the list of active modules in the PFI.

    See modulesPathForPfi for more info.
    """
    yamlPath = configPathForPfi(version=version, rootDir=rootDir)
    with open(yamlPath, mode='rt') as yamlFile:
        config = yaml.load(yamlFile)

    return [c.strip() for c in config['modules']]

def mapPathForModule(moduleName, version=None, rootDir=None):
    """ Return the pathname for a module's map file.

    Args
    ----
    moduleName : str
       Something like "SC42" or "Spare2"
    version : str
       An identifying string. If not set, the "latest" one.
       $moduleName.xml or $moduleName_$version.xml
    rootDir : str/Path
       If set, where to look for the map files.
       By default $PFS_INSTDATA_DIR/data/pfi/modules/$moduleName/

    """

    moduleName = moduleName.strip()
    if rootDir is None:
        instDataRoot = _instDataDir()
        rootDir = instDataRoot / 'data' / 'pfi' / 'modules' / moduleName

    if version is None:
        fname = f'{moduleName}.xml'
    else:
        fname = f'{moduleName}_{version}.xml'

    return rootDir / fname

def mapForModule(moduleName, version=None):
    """ Return the content of the given module's map.

    See mapPathForModule for docs
    """

    mapPath = mapPathForModule(moduleName, version=version)

    with open(mapPath, mode='rt') as mapFile:
        content = mapFile.read()
        return content
