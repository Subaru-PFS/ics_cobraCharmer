import argparse
import logging
import struct
import asyncio
import time

from . import convert
from . import fpgaProtocol as proto
from .fpgaLogger import FPGAProtocolLogger

logging.basicConfig(format="%(asctime)s.%(msecs)03d %(levelno)s %(name)-10s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")

class IncompleteDataError(Exception):
    pass

class ChecksumError(Exception):
    pass

class FPGAProtocol(asyncio.Protocol):
    """
    A simulator for the PFI FPGA board.

    The FPGA implements six commands: move, calibrate, setFrequencies,
    getHousekeeping, reset, and diagnostic. We accept all of these,
    neatly log the contents, and generate fake-ish responses when
    required.

    The commands are binary encoded with a 6-10 byte header and an
    optional body. For cross-reference with the ics_mps_prod and
    ics_cobraCharmer packages, I'll use the original canonical names.


    """

    major = 1
    minor = 2

    def __init__(self, fpga=None, debug=False, boards=None):
        """ Accept a new connection. Print out commands and optionally forwar to real FPGA.

        Args:
          fpga (str) : if set, the IP address/name of a real FPGA. We then forward all complete commands.
                       NOT YET IMPLEMENTED.
          debug (bool) : Whether to be chatty.
        """

        self.logger = logging.getLogger('fpga')
        self.ioLogger = logging.getLogger('fpgaIO')
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.ioLogger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.fpga = fpga
        self.fpgaLogger = FPGAProtocolLogger(debug=True)

        if boards is None:
            boards = [14,14,14,14,14,14]
        self.boards = boards

        self.resetBuffer()
        self.pleaseFinishLoop = False

    def connection_made(self, transport):
        peername = transport.get_extra_info('peername')
        self.ioLogger.info(f'FPGA connection from {peername}')
        self.transport = transport
        self.connectTime = time.time()

    def connection_lost(self, exc):
        self.ioLogger.warn('FPGA connection lost')

    def data_received(self, data):
        """ The asyncio call with new data. We buffer until a complete command has been accepted. """

        self.ioLogger.debug("received %d bytes, with %d already buffered",
                            len(data), len(self.data))
        self.data.extend(data)
        self.processAllCommands()

        if self.pleaseFinishLoop:
            asyncio.get_event_loop().stop()

    def resetBuffer(self):
        self.ioLogger.info('clearing buffer')
        self.data = bytearray()

    def processAllCommands(self):
        """ Look for complete commands in .data and handle them all. """

        while len(self.data) >= 2:
            self.cmdCode, self.cmdNum = self.data[:2]
            self.ioLogger.debug("command %d, %d, %d bytes",
                                self.cmdCode, self.cmdNum, len(self.data))
            try:
                self.processOneCmd()
            except IncompleteDataError:
                self.ioLogger.debug('not enough data for one command (%d bytes). Waiting.', len(self.data))
                return
            except KeyError:
                raise RuntimeError(f"unknown call: {self.data[0]}")
            self.ioLogger.debug('command %d,%d handled; %d bytes in buffer',
                                self.cmdCode, self.cmdNum, len(self.data))

    def processOneCmd(self):
        """ Remove a complete command from .data and process it. """

        header, data, self.data = self.splitCommand(self.data)
        cmd = int(header[0])

        self.fpgaLogger.logCommand(header, data)

        # Only two command return anything interesting.
        if cmd == proto.HOUSEKEEPING_CMD:
            self.housekeepingHandler(header, data)
            return
        if cmd == proto.DIAG_CMD:
            self.diagHandler()
            return
        if cmd == proto.POWER_CMD:
            self.powerHandler(header)
            return
        if cmd == proto.ADMIN_CMD:
            self.adminHandler(header, data)
            return

        if cmd == proto.EXIT_CMD:
            self.pleaseFinishLoop = True
        elif cmd == proto.FLUSH_CMD:
            self.resetBuffer()

        # Acknowledge receipt of the command
        self.respond()

        # For some commands, declare that command is done. Could pause here.
        if cmd in {proto.RUN_CMD, proto.CAL_CMD, proto.SETFREQ_CMD}:
            self.respond()

    def splitCommand(self, buf):
        """ Look for a complete command and split it off.

        Args
        ----
        buf : bytearray

        Returns
        -------
        header : bytearray
          the command header
        data : bytearray or None
          the command data, if any
        leftovers : bytearray
          any unconsumed data in buf

        Raises
        ------
        IncompleteDataError, if there is not enough data to complete a single command.
        ChecksumError, if the checksum is not valid.
        """

        try:
            cmd = int(buf[0])
        except IndexError:
            raise IncompleteDataError()

        headerSize, itemSize = proto.cmds[cmd]
        if len(buf) < headerSize:
            raise IncompleteDataError()

        header, allData = buf[:headerSize], buf[headerSize:]

        if itemSize == 0:
            nCobras = 0
        else:
            _, cmdNum, nCobras = struct.unpack('>BBH', header[:4])

        dataLen = nCobras*itemSize
        if len(allData) < dataLen:
            raise IncompleteDataError()

        # empty bytearray if there is no per-arm data
        data = allData[:dataLen]

        # checksum and leftovers
        cmdLen = headerSize + dataLen
        checksum = (sum(buf[:cmdLen]) - header[-2] - header[-1]) & 0xffff
        hdrChecksum = (header[-2] << 8) + header[-1]
        if (checksum + hdrChecksum) != 65536:
            raise ChecksumError(f"cmd={cmd} cmdNum={cmdNum} checksum={checksum}/{hdrChecksum} header={header} data={data}")

        return header, data, buf[cmdLen:]

    def XXcalHandler(self):
        """ Look for a complete CALibrate command and process it. """

        if len(self.data) < proto.CAL_HEADER_SIZE:
            raise IncompleteDataError()

        dirName = {False:' cw', True:'ccw'}

        nCobras, timeLimit, CRC = struct.unpack('>HHH',
                                                self.data[2:proto.CAL_HEADER_SIZE])
        self.ioLogger.debug(f"CAL header: nCobras={nCobras}")

        if len(self.data) < proto.CAL_HEADER_SIZE + nCobras * proto.CAL_ARM_SIZE:
            raise IncompleteDataError()

        splitAt = proto.CAL_HEADER_SIZE + nCobras*proto.CAL_ARM_SIZE
        calData, self.data = self.data[proto.CAL_HEADER_SIZE:splitAt], self.data[splitAt:]

        self.logger.info('CMD: cal (%d cobras)' % (nCobras))
        for c_i in range(nCobras):
            (flags, thetaRangeLo, thetaRangeHi,
             phiRangeLo, phiRangeHi) = struct.unpack('>HHHHH',
                                                     calData[c_i*proto.CAL_ARM_SIZE:
                                                             (c_i + 1)*proto.CAL_ARM_SIZE])
            thetaDir = bool(flags & 1)
            phiDir = bool(flags & 2)
            setTheta = bool(flags & 4)
            setPhi = bool(flags & 8)
            boardId = (flags >> 4) & 0x7f
            cobraId = (flags >> 11) & 0x1f

            self.logger.info('    cobra: %2d %2d  Theta: %d %s %0.2f %0.2f  Phi: %d %s %0.2f %0.2f' %
                             (boardId, cobraId,
                              setTheta, dirName[thetaDir],
                              convert.get_freq(thetaRangeLo), convert.get_freq(thetaRangeHi),
                              setPhi, dirName[phiDir],
                              convert.get_freq(phiRangeLo), convert.get_freq(phiRangeHi)))
        self.respond()
        self.respond()

    def housekeepingHandler(self, header, data):
        self.respond()

        _, _, boardNum, timeLimit, CRC = struct.unpack('>BBHHH', header)
        nCobras = 29

        temp1 = convert.tempToAdc(23.1);
        temp2 = convert.tempToAdc(24.0);
        v = convert.voltToAdc(9.89)
        self.logger.debug(f'temps=0x{temp1:x},0x{temp2:x} volts=0x{v:x}')

        mot = struct.pack('>%s' % ('H'*(4*nCobras)),
                          *([convert.get_per(65.0), 30000,
                             convert.get_per(100.0), 30000] * nCobras))
        TLMheader = struct.pack('>BBHHHHH', self.cmdCode, self.cmdNum, 0,
                                boardNum&0x7f,
                                temp1, temp2, v)
        TLM = TLMheader + mot
        self._respond(TLM)

    def diagHandler(self):
        TLM = struct.pack('>BBBBBBBBHH', self.cmdCode, self.cmdNum, *self.boards, 0, 0)
        self._respond(TLM)

    def adminHandler(self, header, data):
        try:
            _, _, debugLevel, _, timeout, CRC = struct.unpack('>BBBBHH', header)
        except:
            self.logger.warn('admin unpack WTF: %f', header)
        self.logger.info('admin debugLevel=%d', debugLevel)

        uptime = int(time.time() - self.connectTime)
        TLM = struct.pack('>BBBBLHH', self.cmdCode, self.cmdNum,
                          self.major, self.minor, uptime, 0, 0)
        self._respond(TLM)

    def powerHandler(self, header):
        """ Look for a complete RST command and process it. """

        _, _, sectorPower, sectorReset, timeout, CRC = struct.unpack('>BBBBHH', header)

        self.respond()

    def _respond(self, TLM):
        self.fpgaLogger.logTlm(TLM)
        self.transport.write(TLM)

    def respond(self, respCode=0, respDetail=0):
        TLM = struct.pack('>BBHH', self.cmdCode, self.cmdNum, respCode, respDetail)
        self._respond(TLM)

def main():
    parser = argparse.ArgumentParser('fpgaSim')
    parser.add_argument('--host', nargs='?', type=str, default='localhost',
                        help='host address to listen on')
    parser.add_argument('--port', nargs='?', type=int, default=4001,
                        help='host port to listen on')
    parser.add_argument('--fpga', nargs='?', type=str, default='',
                        help='address to forward to')
    parser.add_argument('--debug', action='store_true',
                        help='be chattier')
    args = parser.parse_args()

    logging.getLogger('asyncio').setLevel(logging.DEBUG if args.debug else logging.INFO)
    logging.getLogger('').setLevel(logging.DEBUG if args.debug else logging.INFO)
    logging.warning(f'starting asyncio with {args.debug}')

    loop = asyncio.get_event_loop()
    loop.set_debug(args.debug)
    logging.info('launching server')
    fpgaCmdHandler = loop.create_server(lambda: FPGAProtocol(args.fpga, args.debug), args.host, args.port)
    server = loop.run_until_complete(fpgaCmdHandler)
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        logging.info('finishing with server')
        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.close()

    logging.info('finished with server')

if __name__ == "__main__":
    main()
