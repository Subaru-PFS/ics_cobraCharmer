import array
import socket
import struct
import time

from ics.cobraCharmer.hexprint import arr2Hex

# TODO:
#  Replace all array with bytearray
#  Have timeouts mean something, packet-by-packet.


def bytesAsString(msg, type=''):
    if type == 'h' or type == 'B':
        s = arr2Hex(msg, seperator='')
        length = len(s)
        newstr = ''
        for char in range(1, length+1):
            newstr += s[char-1]
            if(char%16 == 0):
                newstr += '\t'
            if(char%64 == 0 and char != length):
                newstr += '\n'
        s = newstr
    elif type == 's':
        s = msg.tostring()
    else:
        s = ''
    return s


class Sock:
    def __init__(self):
        self._s = socket.socket()

    def connect(self, ip, port, logger=None, protoLogger=None):
        self._s = socket.socket()
        self.logger = logger
        self.protoLogger = protoLogger
        if logger is not None:
            logger.log("(ETH)Connecting...")

        self._s.settimeout(4)
        self._s.connect((ip, port))
        self._s.settimeout(30)

        if logger is not None:
            logger.log("(ETH)Connection Made. %s:%s" % (ip, port))

    def send(self, msg, logger=None, type=None):
        type = 'B'
        # msg is a byteArray
        if logger is None:
            logger = self.logger
        if logger is not None:
            s = bytesAsString(msg, type)
            logger.log("(ETH)Sent msg on socket.(%s)"%s)
        if self.protoLogger is not None:
            self.protoLogger.logSend(msg.tobytes())

        self._s.send(msg)

    def recv(self, tgt_len, logger=None, type=None):
        type = 'B'
        if logger is None:
            logger = self.logger

        if logger is not None:
            logger.debug("(ETH)Looking for %d bytes)" % (tgt_len))

        # msg is a byteArray
        remaining = tgt_len
        msg = array.array('B')

        # The FPGA? or something? occasionally times out replies, but manually retrying does get the reply.
        # Wireshark shows that at CIT, at least, the reply can take over a minute.
        maxRetries = 3
        retry = 0
        while remaining > 0:
            t0 = time.time()
            try:
                chunk = self._s.recv(remaining)
                if logger is not None:
                    logger.debug("(ETH)Rcvd %d bytes on socket:%s" % (len(chunk), chunk))
            except socket.timeout:
                t1 = time.time()
                errMsg = "timed out (%0.3f s) waiting for %d bytes. Received %d bytes: %s" % (t1-t0, tgt_len,
                                                                                              tgt_len-remaining,
                                                                                              msg)
                if retry >= maxRetries:
                    raise RuntimeError(errMsg)
                else:
                    if logger is not None:
                        logger.logger.error(errMsg)
                    else:
                        print(errMsg)
                    retry += 1
                    chunk = bytes()
                    continue

            msg.frombytes(chunk)
            remaining -= len(chunk)

        if logger is not None:
            s = bytesAsString(msg, type)
            logger.log("(ETH)Rcvd msg on socket.(%s)"%s)
        if self.protoLogger is not None:
            self.protoLogger.logRecv(msg.tobytes())

        return msg

    def close(self, logger=None):
        if logger is None:
            logger = self.logger

        self._s.close()

        if logger is not None:
            logger.log("(ETH)Connection Closed.")


sock = Sock()
