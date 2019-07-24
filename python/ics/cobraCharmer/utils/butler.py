import os
import pathlib

import yaml

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
        instDataRoot = os.environ['PFS_INSTDATA_DIR']
        rootDir = pathlib.Path(instDataRoot) / 'data' / 'pfi'

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
        instDataRoot = os.environ['PFS_INSTDATA_DIR']
        rootDir = pathlib.Path(instDataRoot) / 'data' / 'pfi' / 'modules' / moduleName

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
