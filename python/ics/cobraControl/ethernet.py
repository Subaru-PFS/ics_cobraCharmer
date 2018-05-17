import array
import socket
import struct

from .hexprint import arr2Hex


def bytesAsString(msg, type=''):
    if type == 'h':
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

    def connect(self, ip, port, logger=None):
        self._s = socket.socket()
        self.logger = logger
        if logger is not None:
            logger.log("(ETH)Connecting...")

        self._s.settimeout(4)
        self._s.connect((ip, port))
        self._s.settimeout(30)

        if logger is not None:
            logger.log("(ETH)Connection Made. %s:%s \n" %(ip,port))

    def send(self, msg, logger=None, type=''):
        # msg is a byteArray
        if logger is None:
            logger = self.logger
        if logger is not None:
            s = bytesAsString(msg, type)
            self.logger.log("(ETH)Sent msg on socket.\n(%s)"%s)

        self._s.send(msg)

    def recv(self, tgt_len, logger=None, type=''):
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

        return msg

    def recv_byte(self, logger=None, type='h'):
        if logger is None:
            logger = self.logger

        d = self._s.recv(1)
        by = struct.unpack('B', d)[0]

        if logger is not None:
            s = bytesAsString(msg, type)
            logger.log("(ETH)Rcvd byte on socket.(%s)" %s)

        return by

    def close(self, logger=None):
        if logger is None:
            logger = self.logger

        self._s.close()

        if logger is not None:
            logger.log("(ETH)Connection Closed.")


sock = Sock()
