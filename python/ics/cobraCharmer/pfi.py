from importlib import reload
import logging

import numpy as np

import pfs.utils.fiberids

from . import cobra

reload(pfs.utils.fiberids)
reload(cobra)
fiberIds = pfs.utils.fiberids.FiberIds()

class PFI(object):
    COBRAS_PER_MODULE = 57

    def __init__(self, butler=None, doLoad=True, version=''):
        self.logger = logging.getLogger('pfi')

        self.modules = set()
        self.cobras = dict()

        if butler is None:
            import pfs.utils.butler
            reload(pfs.utils.butler)

            butler = pfs.utils.butler.Butler()
        self.butler = butler

        self.version = version

        if doLoad:
            self.loadCobras()

    def __str__(self):
        return f"PFI(ncobras={len(self.cobras)}, modules={sorted(self.modules)})"

    def loadCobras(self, configPath=None):
        if configPath is None:
            configPath = self.butler.getPath('pfi', version=self.version)

        cfg = self.butler.getFromPath('pfi', configPath)
        self.modules = set(cfg['modules'])

        for m in sorted(self.modules):
            idDict = dict(moduleName=m)
            for c_i in range(1, self.COBRAS_PER_MODULE+1):
                idDict['cobraInModule'] = c_i
                cobra = self.butler.get('cobraGeometry', idDict)
                self.cobras[cobra.cobraId] = cobra
