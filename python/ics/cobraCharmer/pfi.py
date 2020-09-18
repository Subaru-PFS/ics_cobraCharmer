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
        return f"PFI(version={self.version} ncobras={len(self.cobras)}, modules={self.modules})"

    def loadCobras(self, configPath=None):
        """
        Load all the cobras defined in the pfi.yaml config file.

        The `pfi.yaml` file has a `modules` list, binding named
        cobra modules ("SC01", etc) to module ids in the PFI focal plane.

        For some better tracability and error checking, we save both
        the module name and the module slot number in the cobra
        object. If we detect that a module has been moved, update its
        cobras and persist them.

        PFI slots are filled consecutively, starting at slot 1. If a
        slot's module name is "None", do not fill that slot, and
        continue with the next one.
        """

        if configPath is None:
            configPath = self.butler.getPath('pfi', version=self.version)

        cfg = self.butler.getFromPath('pfi', configPath)
        self.modules = cfg['modules']

        for m_i, m in enumerate(self.modules):
            moduleNum = m_i + 1
            if m is None or m == "None":
                self.logger.warn(f'skipping empty module slot {moduleNum}')
                continue
            self.logger.info(f'loading module {m} in PFI slot {moduleNum}')

            idDict = dict(moduleName=m)
            assert m is not None
            for c_i in range(1, self.COBRAS_PER_MODULE+1):
                idDict['cobraInModule'] = c_i
                cobra = self.butler.get('cobraGeometry', idDict)
                lastModuleNum = cobra.setModuleNum(moduleNum)
                if lastModuleNum != moduleNum:
                    if c_i == 1:
                        logging.warn(f'moving module {cobra.moduleName} from slot {lastModuleNum} to {moduleNum}')
                    cobra.setModuleNum(moduleNum)
                    assert cobra.moduleName is not None
                    self.butler.put(cobra, 'cobraGeometry', dict(moduleName=cobra.moduleName,
                                                                 cobraInModule=cobra.cobraNum))
                self.cobras[cobra.cobraId] = cobra
