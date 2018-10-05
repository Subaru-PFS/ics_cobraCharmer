import logging
import struct
import multiprocessing

from .convert import get_freq, conv_temp, conv_volt, conv_current
from . import fpgaProtocol as proto

logging.basicConfig(format="%(asctime)s.%(msecs)03d %(levelno)s %(name)-10s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")

def loggerProcess(queue):
    """ The logger process itself. Simply loops on queue input.

    The process launcher is responsible for killing us.
    """

    logger = FPGAProtocolLogger()
    while True:
        item = queue.get()
        logger.log(item)

def launchLogger():
    """ Fire up a logging process.

    Returns
    -------
    loggerProcess : a multiprocessing.Process

    The caller is responsible for calling loggerProcess.stop()
    """

    q = multiprocessing.Queue
    loggerProcess = multiprocessing.Process(target=loggerProcess, args=(q))
    loggerProcess.start()

    return loggerProcess

class FPGAProtocolLogger(object):
    """
    A logger for the PFI FPGA protocol.

    The FPGA implements six commands: move, calibrate, setFrequencies,
    getHousekeeping, reset, and diagnostic. We accept all of these,
    neatly log the contents, and generate fake-ish responses when
    required.

    The commands are binary encoded with a 6-10 byte header and an
    optional body. For cross-reference with the ics_mps_prod and
    ics_cobraCharmer packages, I'll use the original canonical names.
    """

    errors = {0:"OK",
              1:"Timeout",
              2:"Bad execution",
              3:"Bad checksum",
              4:"Invalid Motor #",
              5:"Invalid Board #",
              6:"Invalid Frequency",
              7:"Invalid Runtime",
              8:"Invalid Interleave",
              9:"Too many sub-commands"}

    def __init__(self, debug=False):
        """ Parse both MSIM logs and binary FPGA data, and log it nicely.

        Args
        ----
        debug (bool) : Whether to be chatty.
        """

        self.logger = logging.getLogger('fpga')
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.cmdHandlers = {proto.RUN_CMD: self.runHandler,
                            proto.CAL_CMD: self.calHandler,
                            proto.SETFREQ_CMD: self.setFreqHandler,
                            proto.HOUSEKEEPING_CMD: self.housekeepingHandler,
                            proto.POWER_CMD: self.powerHandler,
                            proto.DIAG_CMD: self.diagHandler,
                            proto.FLUSH_CMD: self.flushHandler,
                            proto.EXIT_CMD: self.exitHandler,}

    def log(self, item):
        """ Accept something passed in from the main process.

        Args
        ----
        item : bytearray or pair of bytearrays
          If one bytearray, it is a a return packet: a "TLM"
          If two, it is a header and data
        """
        if isinstance(item, bytearray):
            self.logTLM(item)
        else:
            header, data = item
            self.logCommand(header, data)

    def logCommand(self, header, data, ts=None):
        try:
            cmd = int(header[0])
            handler = self.cmdHandlers[cmd]
            handler(header, data)
        except Exception as e:
            self.logger.error(f"unexpected or unhandled cmd {header}: {e}")

    def logTlm(self, tlm, ts=None, hkData=None):
        try:
            if hkData is not None:
                self.hkTlmHandler(tlm, hkData)
            else:
                tlmType = int(tlm[0])
                if tlmType == proto.DIAG_CMD:
                    self.diagTlmHandler(tlm)
                else:
                    self.mainTlmHandler(tlm)
        except Exception as e:
            self.logger.error(f"unexpected or unhandled response for {tlm}: {e}")

    def runHandler(self, header, data):
        """ Log a RUN command """

        expectedCmd = proto.RUN_CMD

        cmd, cmdNum, nCobras, timeLimit, interleave, CRC = struct.unpack('>BBHHHH', header)
        if cmd != expectedCmd:
            raise RuntimeError(f"incorrect command type; expected {expectedCmd}, got {cmd}")
        self.logger.info(f"CMD run cmdNum= {cmdNum} nCobras= {nCobras} timeLimit= {timeLimit} interleave= {interleave}")

        expectedSize = nCobras * proto.RUN_ARM_SIZE
        if len(data) != expectedSize:
            raise RuntimeError(f"data size wrong; expected {expectedSize} got {len(data)}")

        dirName = {False:' cw', True:'ccw'}
        for c_i in range(nCobras):
            (flags,
             thetaOntime, thetaSteps, thetaOfftime,
             phiOntime, phiSteps, phiOfftime) = struct.unpack('>HHHHHHH',
                                                              data[c_i*proto.RUN_ARM_SIZE:
                                                                   (c_i + 1)*proto.RUN_ARM_SIZE])
            thetaDir = bool(flags & 1)
            phiDir = bool(flags & 2)
            thetaEnable = bool(flags & 4)
            phiEnable = bool(flags & 8)
            boardId = (flags >> 4) & 0x7f
            cobraId = (flags >> 11) & 0x1f

            self.logger.info('  run cobra: %2d %2d  Theta: %d %s  %5d %5.1f %5.1f  Phi: %d %s  %5d %5.1f %5.1f' %
                             (boardId, cobraId,
                              thetaEnable, dirName[thetaDir], thetaSteps,
                              thetaOntime, thetaOfftime,
                              phiEnable, dirName[phiDir], phiSteps,
                              phiOntime, phiOfftime))

    def calHandler(self, header, data):
        """ Look for a complete CALibrate command and process it. """

        expectedCmd = proto.CAL_CMD

        cmd, cmdNum, nCobras, timeLimit, CRC = struct.unpack('>BBHHH', header)
        if cmd != expectedCmd:
            raise RuntimeError(f"incorrect command type; expected {expectedCmd}, got {cmd}")
        self.logger.info(f"CMD cal cmdNum= {cmdNum} nCobras= {nCobras} timeLimit= {timeLimit}")

        expectedSize = nCobras * proto.CAL_ARM_SIZE
        if len(data) != expectedSize:
            raise RuntimeError(f"data size wrong; expected {expectedSize} got {len(data)}")

        dirName = {False:' cw', True:'ccw'}
        for c_i in range(nCobras):
            (flags, thetaRangeLo, thetaRangeHi,
             phiRangeLo, phiRangeHi) = struct.unpack('>HHHHH',
                                                     data[c_i*proto.CAL_ARM_SIZE:
                                                          (c_i + 1)*proto.CAL_ARM_SIZE])
            thetaDir = bool(flags & 1)
            phiDir = bool(flags & 2)
            setTheta = bool(flags & 4)
            setPhi = bool(flags & 8)
            boardId = (flags >> 4) & 0x7f
            cobraId = (flags >> 11) & 0x1f

            self.logger.info(' cal cobra: %2d %2d  Theta: %d %s %0.2f %0.2f  Phi: %d %s %0.2f %0.2f' %
                             (boardId, cobraId,
                              setTheta, dirName[thetaDir], get_freq(thetaRangeLo), get_freq(thetaRangeHi),
                              setPhi, dirName[phiDir], get_freq(phiRangeLo), get_freq(phiRangeHi)))

    def setFreqHandler(self, header, data):
        """ Look for a complete SET FREQUENCY command and process it. """

        expectedCmd = proto.SETFREQ_CMD

        cmd, cmdNum, nCobras, timeLimit, CRC = struct.unpack('>BBHHH', header)
        if cmd != expectedCmd:
            raise RuntimeError(f"incorrect command type; expected {expectedCmd}, got {cmd}")
        self.logger.info(f"CMD set cmdNum= {cmdNum} nCobras= {nCobras} timeLimit= {timeLimit}")

        expectedSize = nCobras * proto.SETFREQ_ARM_SIZE
        if len(data) != expectedSize:
            raise RuntimeError(f"data size wrong; expected {expectedSize} got {len(data)}")

        for c_i in range(nCobras):
            flags, thetaPeriod, phiPeriod = struct.unpack('>HHH',
                                                          data[c_i*proto.SETFREQ_ARM_SIZE:
                                                               (c_i + 1)*proto.SETFREQ_ARM_SIZE])
            setTheta = bool(flags & 1)
            setPhi = bool(flags & 2)
            boardId = (flags >> 4) & 0x7f
            cobraId = (flags >> 11) & 0x1f

            self.logger.info('  set cobra: %2d %2d  Theta: %d %0.2f  Phi: %d %0.2f' %
                             (boardId, cobraId,
                              setTheta, get_freq(thetaPeriod),
                              setPhi, get_freq(phiPeriod)))

    def housekeepingHandler(self, header, data):
        """ HK command """

        expectedCmd = proto.HOUSEKEEPING_CMD

        cmd, cmdNum, boardNumber, timeLimit, CRC = struct.unpack('>BBHHH', header)
        if cmd != expectedCmd:
            raise RuntimeError(f"incorrect command type; expected {expectedCmd}, got {cmd}")
        self.logger.info(f"CMD hk cmdNum= {cmdNum} board= {boardNumber} timeLimit= {timeLimit}")

    def powerHandler(self, header, data):
        """ POWER command """

        expectedCmd = proto.POWER_CMD

        cmd, cmdNum, sectorPower, sectorReset, timeLimit, CRC = struct.unpack('>BBHHHH', header)
        if cmd != expectedCmd:
            raise RuntimeError(f"incorrect command type; expected {expectedCmd}, got {cmd}")
        self.logger.info(f"CMD pow cmdNum= {cmdNum} timeLimit= {timeLimit}")
        self.logger.info("    sectorPower=0x%02x sectorReset=0x%02x" % (sectorPower, sectorReset))

    def diagHandler(self, header, data):
        """ DIAG command """

        expectedCmd = proto.DIAG_CMD

        cmd, cmdNum, timeLimit, CRC = struct.unpack('>BBHH', header)
        if cmd != expectedCmd:
            raise RuntimeError(f"incorrect command type; expected {expectedCmd}, got {cmd}")
        self.logger.info(f"CMD diag cmdNum= {cmdNum} timeLimit= {timeLimit}")

    def exitHandler(self, header, data):
        """ EXIT command, just for the simulator. """

        expectedCmd = proto.EXIT_CMD

        cmd, cmdNum, timeLimit, CRC = struct.unpack('>BBHH', header)
        if cmd != expectedCmd:
            raise RuntimeError(f"incorrect command type; expected {expectedCmd}, got {cmd}")
        self.logger.info(f"CMD exit cmdNum= {cmdNum} timeLimit= {timeLimit}")

    def flushHandler(self, header, data):
        """ FLUSH command, just for the simulator. """

        expectedCmd = proto.FLUSH_CMD

        cmd, cmdNum, timeLimit, CRC = struct.unpack('>BBHH', header)
        if cmd != expectedCmd:
            raise RuntimeError(f"incorrect command type; expected {expectedCmd}, got {cmd}")
        self.logger.info(f"CMD diag cmdNum= {cmdNum} timeLimit= {timeLimit}")

    def mainTlmHandler(self, tlm):
        """ Log a reply ("TLM") for most commands. """

        cmd, cmdNum, errorCode, detail = tlm
        errorString = self.errors.get(int(errorCode), f"UNKNOWN ERROR CODE {errorCode}")

        self.logger.info(f"TLM {cmd} cmdNum= {cmdNum} error= {errorCode} {detail} {errorString}")

    def hkTlmHandler(self, tlm, tlmData):
        """ Log a reply ("TLM") for a housekeeping command. """

        cmd, cmdNum, boardNumber, temp1, temp2 = [int(i) for i in tlm]

        self.logger.info(f"TLM hk {cmd} cmdNum= {cmdNum} temps= {conv_temp(temp1):5.2f} {conv_temp(temp2):5.2f}")

        for cobra_i, i in enumerate(range(0, len(tlmData), 4)):
            mot1freq, mot1current, mot2freq, mot2current = [int(i) for i in tlmData[i:i+4]]
            self.logger.info("  hk  cobra %2d %2d  Theta: %0.2f %0.2f  Phi: %0.2f %0.2f" %
                             (boardNumber, cobra_i + 1,
                              get_freq(mot1freq), conv_current(mot1current),
                              get_freq(mot2freq), conv_current(mot2current)))

    def diagTlmHandler(self, tlm, tlmData):
        """ Log a reply ("TLM") for a DIAG commands. """

        cmd, cmdNum, *inventory, errorCode, detail = tlm
        errorString = self.errors.get(int(errorCode), f"UNKNOWN ERROR CODE {errorCode}")

        self.logger.info(f"TLM DIAG {cmd} cmdNum= {cmdNum} inventory= {inventory} error= {errorCode} {detail} {errorString}")
