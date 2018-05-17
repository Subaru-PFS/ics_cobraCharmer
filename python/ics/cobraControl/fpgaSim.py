import argparse
import logging
import struct
import asyncio

from .convert import get_freq

logging.basicConfig(format="%(asctime)s.%(msecs)04d %(levelno)s %(name)-10s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")

class IncompleteDataError(Exception):
    pass

class FPGAProtocol(asyncio.Protocol):
    RUN_HEADER_SIZE = 10
    RUN_ARM_SIZE = 14
    POWER_HEADER_SIZE = 8
    SETFREQ_HEADER_SIZE = 8
    SETFREQ_ARM_SIZE = 6
    CAL_HEADER_SIZE = 8
    CAL_ARM_SIZE = 10

    def __init__(self, fpga, debug):
        self.logger = logging.getLogger('fpga')
        self.ioLogger = logging.getLogger('fpgaIO')
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.ioLogger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.fpga = fpga

        self.handlers = {1: self.runHandler,
                         2: self.calHandler,
                         3: self.setFreqHandler,
                         4: self.housekeepingHandler,
                         5: self.powerHandler,
                         6: self.diagHandler,
                         7: self.exitHandler,}

        self.resetBuffer()
        self.pleaseFinishLoop = False

    def connection_made(self, transport):
        peername = transport.get_extra_info('peername')
        self.ioLogger.info(f'FPGA connection from {peername}')
        self.transport = transport

    def connection_lost(self, exc):
        self.ioLogger.warn('FPGA connection lost')

    def data_received(self, data):
        """ The asyncio call with new data. We buffer until a complete command has been accepted. """

        self.ioLogger.debug("received %d bytes, with %d already buffered",
                            len(data), len(self.data))
        self.data.extend(data)
        self.handle()

        if self.pleaseFinishLoop:
            asyncio.get_event_loop().stop()

    def resetBuffer(self):
        self.ioLogger.debug('clearing buffer')
        self.data = bytearray()

    def handle(self):
        """ Look for complete commands in .data and handle them all. """

        while len(self.data) >= 2:
            self.cmdHeader = self.data[:2]
            self.cmdCode, self.cmdNum = self.cmdHeader
            self.ioLogger.debug("command %d, %d, %d bytes",
                                self.cmdCode, self.cmdNum, len(self.data))
            try:
                self.handlers[self.cmdCode]()
            except IncompleteDataError:
                self.ioLogger.info('not enough data for one command (%d bytes). Waiting.', len(self.data))
                return
            except KeyError:
                raise RuntimeError(f"unknown call: {self.data[0]}")
            self.ioLogger.debug('command %d,%d handled; %d bytes in buffer',
                                self.cmdCode, self.cmdNum, len(self.data))

    def runHandler(self):
        """ Look for a complete RUN command and process it. """

        if len(self.data) < self.RUN_HEADER_SIZE:
            raise IncompleteDataError()

        nCobras, timeLimit, interleave, CRC = struct.unpack('>HHHH',
                                                            self.data[2:self.RUN_HEADER_SIZE])
        self.ioLogger.debug(f"run header: nCobras={nCobras}")

        if len(self.data) < self.RUN_HEADER_SIZE + nCobras * self.RUN_ARM_SIZE:
            raise IncompleteDataError()

        splitAt = self.RUN_HEADER_SIZE + nCobras*self.RUN_ARM_SIZE
        runData, self.data = self.data[self.RUN_HEADER_SIZE:splitAt], self.data[splitAt:]

        dirName = {0:'ccw', 1:' cw'}
        self.logger.info('CMD: run (%d cobras)' % (nCobras))
        for c_i in range(nCobras):
            (flags,
             thetaOntime, thetaSteps, thetaOfftime,
             phiOntime, phiSteps, phiOfftime) = struct.unpack('>HHHHHHH',
                                                              runData[c_i*self.RUN_ARM_SIZE:
                                                                      (c_i + 1)*self.RUN_ARM_SIZE])
            thetaDir = bool(flags & 1)
            phiDir = bool(flags & 2)
            thetaEnable = bool(flags & 4)
            phiEnable = bool(flags & 8)
            boardId = (flags >> 4) & 0x7f
            cobraId = (flags >> 11) & 0x1f

            self.logger.info('    cobra= %2d %2d   Theta= %d %s  %3d %5.1f %5.1f   Phi= %d %s  %3d %5.1f %5.1f' %
                             (boardId, cobraId,
                              thetaEnable, dirName[thetaDir], thetaSteps,
                              thetaOntime, thetaOfftime,
                              phiEnable, dirName[phiDir], phiSteps,
                              phiOntime, phiOfftime))
        self._respond()

    def calHandler(self):
        """ Look for a complete CALibrate command and process it. """

        if len(self.data) < self.CAL_HEADER_SIZE:
            raise IncompleteDataError()

        nCobras, timeLimit, CRC = struct.unpack('>HHH',
                                                self.data[2:self.CAL_HEADER_SIZE])
        self.ioLogger.debug(f"CAL header: nCobras={nCobras}")

        if len(self.data) < self.CAL_HEADER_SIZE + nCobras * self.CAL_ARM_SIZE:
            raise IncompleteDataError()

        splitAt = self.CAL_HEADER_SIZE + nCobras*self.CAL_ARM_SIZE
        calData, self.data = self.data[self.CAL_HEADER_SIZE:splitAt], self.data[splitAt:]

        self.ioLogger.debug("cal data: %d cobras",
                            len(calData)/self.CAL_ARM_SIZE)
        self._respond()

    def setFreqHandler(self):
        """ Look for a complete SET FREQUENCY command and process it. """

        if len(self.data) < self.SETFREQ_HEADER_SIZE:
            raise IncompleteDataError()

        nCobras, timeLimit, CRC = struct.unpack('>HHH',
                                                self.data[2:self.SETFREQ_HEADER_SIZE])
        self.ioLogger.debug(f"SET header: nCobras={nCobras}")

        if len(self.data) < self.SETFREQ_HEADER_SIZE + nCobras * self.SETFREQ_ARM_SIZE:
            raise IncompleteDataError()

        splitAt = self.SETFREQ_HEADER_SIZE + nCobras*self.SETFREQ_ARM_SIZE
        setData, self.data = self.data[self.SETFREQ_HEADER_SIZE:splitAt], self.data[splitAt:]

        self.logger.info('CMD: setFreq (%d cobras)' % (nCobras))
        for c_i in range(nCobras):
            flags, thetaPeriod, phiPeriod = struct.unpack('>HHH',
                                                          setData[c_i*self.SETFREQ_ARM_SIZE:
                                                                  (c_i + 1)*self.SETFREQ_ARM_SIZE])
            setTheta = bool(flags & 1)
            setPhi = bool(flags & 2)
            boardId = (flags >> 4) & 0x7f
            cobraId = (flags >> 11) & 0x1f

            self.logger.info('    cobra %2d, %2d Theta: %d, 0x%04x (%0.2f) Phi: %d, 0x%04x (%0.2f)' %
                             (boardId, cobraId,
                              setTheta, thetaPeriod,
                              get_freq(thetaPeriod),
                              setPhi, phiPeriod,
                              get_freq(phiPeriod)))
        self._respond()

    def housekeepingHandler(self):
        self._respond()

    def powerHandler(self):
        """ Look for a complete RST command and process it. """

        if len(self.data) < self.POWER_HEADER_SIZE:
            raise IncompleteDataError()

        data, self.data = self.data[2:self.POWER_HEADER_SIZE], self.data[self.POWER_HEADER_SIZE:]

        sectorPower, sectorReset, timeout, CRC = struct.unpack('>BBHH', data)
        self.logger.info('CMD: power')
        self.logger.info('    sectorPower=0x%02x sectorReset=0x%02x' % (sectorPower, sectorReset))

        self._respond()

    def diagHandler(self):
        self._respond()

    def exitHandler(self):
        self.pleaseFinishLoop = True
        self.resetBuffer()

    def _respond(self):
        noError = bytes([self.cmdCode, self.cmdNum, 0,0,0,0])
        self.transport.write(noError)
        self.transport.write(noError)

def main():
    parser = argparse.ArgumentParser('fpgaSim')
    parser.add_argument('--host', nargs='?', type=str, default='localhost',
                        help='host address to listen on')
    parser.add_argument('--port', nargs='?', type=int, default=4001,
                        help='host port to listen on')
    parser.add_argument('--fpga', nargs='?', type=str, default='',
                        help='address to forward to')
    parser.add_argument('--debug', nargs='?', type=bool,
                        help='be chattier')
    args = parser.parse_args()

    logging.getLogger('asyncio').setLevel(logging.DEBUG if args.debug else logging.INFO)

    loop = asyncio.get_event_loop()
    loop.set_debug(args.debug)
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
