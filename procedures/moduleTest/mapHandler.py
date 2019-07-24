import os
import pathlib

def mapNameForModule(moduleName, version=None):
    instDataPath = os.environ['PFS_INSTDATA_DIR']
    root = pathlib.Path(instDataPath) / 'data' / 'pfi' / 'modules' / moduleName

    if version is None:
        fname = f'{moduleName}.xml'
    else:
        fname = f'{moduleName}_{version}.xml'

    return root / fname

def mapForModule(moduleName, version=None):
    instDataPath = os.environ['PFS_INSTDATA_DIR']
    root = pathlib.Path(instDataPath) / 'data' / 'pfi' / 'modules' / moduleName

    if version is None:
        glob = f'{moduleName}.xml'
    else:
        glob = f'{moduleName}_{version}.xml'

    files = root.glob(glob)
    return root, list(files)
