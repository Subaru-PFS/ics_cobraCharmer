from importlib import reload
import logging

from pfs.utils import fiberids

class FPGA(object):
    def __init__(self, fpgaHost='localhost', logDir=None, doConnect=True, debug=False):
        """ Initialize a PFI class
        Args:
           fpgaHost    - fpga device
           logDir      - directory name for logs
           doConnect   - do connection or not
        """

        self.logger = logging.getLogger('fpga')
        self.logger.setLevel(logging.INFO)
        if logDir is not None:
            log.setupLogPaths(logDir)

        self.protoLogger = fpgaLogger.FPGAProtocolLogger(logRoot=logDir)

