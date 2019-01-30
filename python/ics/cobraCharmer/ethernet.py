import array
import socket
import struct

from .hexprint import arr2Hex

# TODO:
#  Replace all array with bytearray
#  Have timeouts mean something, packet-by-packet.

def bytesAsString(msg, type=''):
    if type == 'h' or type == 'B':
        s = arr2Hex(msg, seperator='')
        length = len(s)
        newstr = ''
        for char in range(1,length+1):
            newstr += s[char-1] 
            if( char%16 == 0 ):
                newstr += '\t'
            if( char%64 == 0 and char!=length):
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
        if logger is None:
            from .log import eth_hex_logger
            logger = eth_hex_logger
        self.logger = logger
        self.protoLogger = protoLogger
        if logger is not None:
            logger.log("(ETH)Connecting...")

        self._s.settimeout(4)
        self._s.connect((ip, port))
        self._s.settimeout(20)

        if logger is not None:
            logger.log("(ETH)Connection Made. %s:%s" %(ip,port))

    def send(self, msg, logger=None, type=None):
        type='B'
        # msg is a byteArray
        if logger is None:
            logger = self.logger
        if logger is not None:
            s = bytesAsString(msg, type)
            logger.log("(ETH)Sent msg on socket.\n(%s)"%s)
        if self.ioLogger is not None:
            self.ioLogger.logSend(msg.tobytes())

        self._s.send(msg)

    def recv(self, tgt_len, logger=None, type=None):
        type='B'
        if logger is None:
            logger = self.logger

            # msg is a byteArray
        remaining = tgt_len
        msg = array.array('B')
        while remaining > 0:
            chunk = self._s.recv(remaining)
            msg.frombytes(chunk)
            remaining -= len(chunk)

        if logger is not None:
            s = bytesAsString(msg, type)
            logger.log("(ETH)Rcvd msg on socket.\n(%s)"%s)
        if self.ioLogger is not None:
            self.ioLogger.logRecv(msg.tobytes())

        return msg

    def close(self, logger=None):
        if logger is None:
            logger = self.logger

        self._s.close()

        if logger is not None:
            logger.log("(ETH)Connection Closed.")


sock = Sock()
