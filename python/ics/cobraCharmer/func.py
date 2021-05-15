from importlib import reload

import math
import os.path
import csv
import time

import numpy as np

from . import cmds
reload(cmds)

from . import ethernet
from .log import full_log, medium_log, short_log, eth_hex_logger
from .convert import *
from .cmds import *
from .fpgaState import fpgaState

CCW_DIR = ('ccw','ccw')
CW_DIR = ('cw','cw')
EN_BOTH = (True, True)
EN_M0 = (True, False)
EN_M1 = (False, True)

NCOBRAS_MOD = 57
NCOBRAS_BRD = 29
NMODULES = 42
NBOARDS = NMODULES*2

# Classes----------------------------------------------------------------------
class Cobra:
    def __init__(self, module, cobraNum):
        if module < 1 or module > NMODULES:
            raise ValueError(f'invalid module number (%s): must be 1..{NMODULES}')
        if cobraNum < 1 or cobraNum > NCOBRAS_MOD:
            raise ValueError(f'invalid cobra number (%s): must be 1..{NCOBRAS_MOD}')

        self.module = module
        self.cobraNum = cobraNum

        # The FPGA uses board numbers, splitting odd and even cobras.
        # Module, board, and cobra numbering is 1-indexed
        #
        self.board = (module - 1)*2 + 1
        if (cobraNum-1)%2 == 1:
            self.board += 1
        self.cobra = (cobraNum-1)//2 + 1

        self.p = None # Params

    def __str__(self):
        return (f"Cobra(module={self.module} board={self.board} "
                f"cobraNum={self.cobraNum} cobra={self.cobra}")

    def stringify(self):
        s = '(C%s,%s ' % (self.board,self.cobra) + self.p.stringify() + ')'
        return s
    def typeMatch(self, type):
        if self.p is None:
            short_log.log("None params type")
            return False
        if self.p.type != type:
            short_log.log("Wrong params type")
            return False
        return True
        
class CalParams:    
    def __init__(self, m0=(230,280), m1=(140,190), en=EN_BOTH, dir=CCW_DIR ):
        self.type = 'Cal'
        self.dir = dir
        self.en = en
        self.m0Range = sorted(m0)
        self.m1Range = sorted(m1)
    def stringify(self):
        s = ''
        s += 'm0:%s:%s' %(self.dir[0], str(self.m0Range)) if self.en[0] else ''
        s += '  ' if self.en[0] and self.en[1] else ''
        s += 'm1:%s:%s' %(self.dir[1], str(self.m1Range)) if self.en[1] else ''
        return s
    def toList(self, board, cnum):
        assert (board>=1 and board<=NBOARDS  and cnum>=1 and cnum<=NCOBRAS_BRD), f'{board},{cnum}'

        c = 0x0000 | (self.dir[0]=='ccw') | ((self.dir[1]=='ccw')<<1) | \
                (self.en[0]<<2) | (self.en[1]<<3) | (board<<4) | (cnum<<11)
                
        p = [c>>8, c%256, self.m0Range[0]>>8, self.m0Range[0]%256, \
                self.m0Range[1]>>8, self.m0Range[1]%256, self.m1Range[0]>>8, \
                self.m1Range[0]%256, self.m1Range[1]>>8,  self.m1Range[1]%256]
        return p
        
class HkParams:
    def __init__(self, m0=(60.0,70.0), m1=(100.0,115.0), \
                    temps=(16.0,31.0), cur=(0.25,1.2), volt=(9.5,10.5)
                ):
        self.type = 'Hk'
        self.trange = temps
        self.vrange = volt
        self._names = ["M0 Freq", "M1 Freq", "M0 Current", "M1 Current"]
        self._ranges = [ m0, m1, cur, cur]
    def chk(self, per0, per1, cur0, cur1, en_log=True):
        # Check the inputted values against expected
        err = 0
        vals = [get_freq(per0), get_freq(per1), \
                conv_current(cur0), conv_current(cur1) \
        ]
        for i in range(4):
            if not inrange(vals[i], self._ranges[i]):
                short_log.log(
                    "Error! %s %.3f not in %s" \
                    %(self._names[i], vals[i], self._ranges[i]), \
                    enable= en_log
                )
                # give a unique error value
                err = (i+1)
        return err


class RunParams:
    def __init__(self, pu=(300,300), st=(10,10), sl=(0,0), en=EN_BOTH, dir=CCW_DIR, ):
        self.type = 'Run'
        self.pulses = pu
        self.steps = st
        self.sleeps = sl
        self.dir = dir
        self.en = en
    def stringify(self):
        s = ''
        s += '%suS x%s d%s %s' %(str(self.pulses), str(self.steps), \
                str(self.sleeps), str(self.dir) )
        return s
    def toList(self, board, cnum):
        assert (board>=1 and board<=NBOARDS  and cnum>=1 and cnum<=NCOBRAS_BRD), f'{board},{cnum}'
        c = 0x0000 | (self.dir[0]=='ccw') | ((self.dir[1]=='ccw')<<1) | \
                (self.en[0]<<2) | (self.en[1]<<3) | (board<<4) | (cnum<<11)
        p = [c>>8, c%256, self.pulses[0]>>8, self.pulses[0]%256, self.steps[0]>>8, \
                self.steps[0]%256, self.sleeps[0]>>8, self.sleeps[0]%256, \
                self.pulses[1]>>8, self.pulses[1]%256, self.steps[1]>>8, \
                self.steps[1]%256, self.sleeps[1]>>8, self.sleeps[1]%256 ]
        return p
        
class SetParams:
    def __init__(self, p0=250, p1=165, en=EN_BOTH):
        self.type = 'Set'
        self.m0Per = p0
        self.m1Per = p1
        self.en = en
    def stringify(self):
        s = ''
        s += 'm0:%s m1:%s' %(str(self.m0Per),str(self.m1Per))
        return s
    def toList(self, board, cnum):
        assert (board>=1 and board<=NBOARDS  and cnum>=1 and cnum<=NCOBRAS_BRD), f'{board},{cnum}'

        c = 0x0000 | (self.en[0]) | (self.en[1]<<1) | \
                (board<<4) | (cnum<<11)
        p = [c>>8, c%256, self.m0Per>>8, self.m0Per%256, \
                self.m1Per>>8, self.m1Per%256 ]
        return p

# Support Functions for Classes------------------------------------------------
def inrange( val, minmax=(0.0,1.0) ):
    return (val >= minmax[0]) and (val <= minmax[1])
    
def cobras2Str(cobras, perLine=3):
    i = 0
    s = ''
    for c in cobras:
        s += '\n' if (i%perLine == 0) and i>0 else ''
        s += c.stringify() + ' '
        i += 1
    return s
    
def cobrasAreType(cobras, type='Hk'):
    for c in cobras:
        if( not c.typeMatch(type) ):
            return False
    return True


def calibrate(cobras, thetaLow=60.4, thetaHigh=70.3, phiLow=94.4, phiHigh=108.2, clockwise=True):
    """ calibrate a set of cobras.

    Args:
    thetaLow, thetaHigh -
    phiLow, phiHigh -

    """

    spin = CW_DIR if clockwise else CCW_DIR
    for c in cobras:
        c.p = CalParams(m0=(convert.get_per(thetaLow),convert.get_per(thetaHigh)),
                        m1=(convert.get_per(phiLow), convert.get_per(phiHigh)),
                        en=(True,True), dir=spin)

    err = CAL(cobras)
    if err:
        raise RuntimeError("calibration failed")

def setFreq(cobras, thetaPeriods, phiPeriods):
    """ set the frequencies for a set of cobras.

    Args:


    """

    if (len(cobras) != len(thetaPeriods) or
        len(cobras) != len(phiPeriods)):
        raise ValueError("length of all arguments must match")

    for c_i, c in enumerate(cobras):
        enable = thetaPeriods[c_i] != 0, phiPeriods[c_i] != 0

        c.p = SetParams(p0=thetaPeriods[c_i],
                        p1=phiPeriods[c_i],
                        en=enable)

    err = SET(cobras)
    if err:
        raise RuntimeError("set frequency failed")

def run(cobras, thetaSteps, phiSteps, thetaPeriods=None, phiPeriods=None, dirs=None):
    """ Moves the given cobras

    Args:
       theta ([int]): steps to move the theta motors
       phi ([int]): steps to move the theta motors

    """

    if np.isscalar(thetaSteps):
        thetaSteps = [thetaSteps]*len(cobras)
    if np.isscalar(phiSteps):
        phiSteps = [phiSteps]*len(cobras)

    if (len(cobras) != len(thetaSteps) or
        len(cobras) != len(phiSteps)):
        raise ValueError("length of all arguments must match")

    for c_i, c in enumerate(cobras):
        enable = thetaSteps[c_i] != 0, phiSteps[c_i] != 0
        c.p = RunParams(pu=(thetaPeriods[c_i], phiPeriods[c_i]),
                        st=(thetaSteps[c_i], phiSteps[c_i]),
                        en=enable, dir=dirs)

    err = RUN(cobras)
    if err:
        raise RuntimeError("run failed")


# Test Functions-------------------------------------------------------------
def POW(sectors=0x3f):
    short_log.log("--- POWER ---")
    
    sectors_off = []
    for i in range(0,6):
        if not sectors & (0x01 << i):
            sectors_off.append(i)
    
    medium_log.log("Sectors Without Power: %s" %sectors_off)
    
    cmd = CMD_pow(sectors, 0)
    ethernet.sock.send(cmd, eth_hex_logger, 'h')
    resp = ethernet.sock.recv(TLM_LEN, eth_hex_logger, 'h')
    error = tlm_chk(resp)
    return error
    
def RST(sectors=0x3f):
    short_log.log("--- RESET ---")

    sectors_reseting = []
    for i in range(0,6):
        if sectors & (0x01 << i):
            sectors_reseting.append(i)
    
    medium_log.log("Sectors Reseting: %s" %sectors_reseting)
    
    cmd = CMD_pow(0, sectors)
    ethernet.sock.send(cmd, eth_hex_logger, 'h')
    resp = ethernet.sock.recv(TLM_LEN, eth_hex_logger, 'h')
    error = tlm_chk(resp)
    return error
    
def DIA():
    short_log.log("--- DIAGNOSTIC INFO ---")
    cmd = CMD_dia()
    ethernet.sock.send(cmd, eth_hex_logger, 'h')
    resp = ethernet.sock.recv(DIA_TLM_LEN, eth_hex_logger, 'h')
    
    boards_per_sector = [
        int(resp[2]), int(resp[3]), int(resp[4]), int(resp[5]), int(resp[6]), int(resp[7])
    ]
    short_log.log("Board Counts: %s" %(boards_per_sector) )
    return boards_per_sector
    
def ADMIN(debugLevel=0):
    cmd = CMD_admin(debugLevel=debugLevel)
    ethernet.sock.send(cmd, eth_hex_logger, 'h')
    resp = ethernet.sock.recv(ADMIN_TLM_LEN, eth_hex_logger, 'h')

    error = int(resp[8])
    version = f"{resp[2]}.{resp[3]}"
    uptime = int(resp[4]) << 24 | int(resp[5]) << 16 | int(resp[6]) << 8 | int(resp[7])

    short_log.log("Admin: version=%s, uptime=%d" % (version, uptime))
    return error, version, uptime/1000

def HK(cobras, export=0, feedback=False, updateModel=None):
    board = cobras[0].board
    nCobras = NCOBRAS_BRD
    nBoardCobras = nCobras if (board%2 == 1) else nCobras-1

    if nBoardCobras != len(cobras) or not all([(c.board == board) for c in cobras]):
        raise ValueError("Can only fetch housekeeping for one single board.")

    short_log.log("--- ISSUE HK & VERIFY (brd:%d) ---" %board)      
    
    cmd = CMD_hk(board, timeout=2000)
    ethernet.sock.send(cmd, eth_hex_logger, 'h')
    
    resp = ethernet.sock.recv(TLM_LEN, eth_hex_logger, 'h')
    er1 = tlm_chk(resp)

    tlmLen = HK_TLM_HDR_LEN + nCobras*8
    resp = ethernet.sock.recv(tlmLen, eth_hex_logger, 'h')
    if feedback:
        er2, t1, t2, v, f1, c1, f2, c2 = hk_chk(resp, cobras, export, feedback,
                                                updateModel=updateModel)
        return er1 or er2, t1, t2, v, f1, c1, f2, c2
    else:
        er2 = hk_chk(resp, cobras, export,
                     updateModel=updateModel)
        return er1 or er2

def CAL( cobras, timeout=0 ):
    if not cobrasAreType(cobras, 'Cal'):
        return True # error
    board = cobras[0].board
    short_log.log("--- ISSUE CAL & VERIFY (%d) ---" %board)
    full_log.log( cobras2Str(cobras) )
    
    # Get Timeout by finding largest freq range to test
    if timeout == 0:
        fRngCob = []
        for c in cobras:
            f0rng = abs(c.p.m0Range[0] - c.p.m0Range[1] )
            f1rng = abs(c.p.m1Range[0] - c.p.m1Range[1] )
            fRngCob.append( max(f0rng, f1rng) )
        tPerCobra = int(7.0*max(fRngCob)) # assume <=7ms per freq
        timeout = math.ceil( 1200 + tPerCobra*len(cobras) )
        
    medium_log.log("Timeout:%d" %timeout)
    
    payload = []
    for c in cobras:
        payload += c.p.toList(c.board, c.cobra)

    cmd = CMD_cal(payload, cmds=len(cobras), timeout=timeout)
    ethernet.sock.send(cmd, eth_hex_logger, 'h')
    
    error = False
    for i in range(2):
        resp = ethernet.sock.recv(TLM_LEN, eth_hex_logger, 'h')
        error |= tlm_chk(resp)
    return error

def RUN( cobras, timeout=0, inter=0 ):
    if not cobrasAreType(cobras, 'Run'):
        return True # error
    board = cobras[0].board
    short_log.log("--- ISSUE RUN & VERIFY (%d) ---" %board)
    full_log.log( cobras2Str(cobras) )

    # Get timeout by finding longest runtime
    if timeout == 0:
        tCob = []
        for c in cobras:
            t0 = (c.p.steps[0]+c.p.sleeps[0])*c.p.pulses[0]
            t1 = (c.p.steps[1]+c.p.sleeps[1])*c.p.pulses[1]
            tCob.append( max(t0,t1) )
        timeout = math.ceil( 1000 + 20*len(cobras)*(max( tCob )/1000) )
        timeout = min(timeout, 2**16-1)
    # Get interleave by finding longest pulsetime
    if inter == 0:
        puCob = []
        for c in cobras:
            puCob.append( max(c.p.pulses) )
        # * 9/15ths for max 10% duty cycle
        inter = math.ceil( max(puCob)*9/15 )
    
    medium_log.log("Timeout:%d, inter:%d" %(timeout,inter) )

    payload = []
    fpgaState.clearMoves()
    for c in cobras:
        payload += c.p.toList(c.board, c.cobra)
        fpgaState.runCobra(c)

    cmd = cmds.CMD_run(payload, cmds=len(cobras), timeout=timeout, inter=inter)
        
    ethernet.sock.send(cmd, eth_hex_logger, 'h')

    error = False
    for i in range(2):
        resp = ethernet.sock.recv(TLM_LEN, eth_hex_logger, 'h')
        error |= tlm_chk(resp)
    return error
    
def SET( cobras ):
    if not cobrasAreType(cobras, 'Set'):
        return True # error

    payload = []
    for c in cobras:
        payload += c.p.toList(c.board, c.cobra)

    cmd = CMD_set(payload, cmds=len(cobras), timeout=2000)
    ethernet.sock.send(cmd, eth_hex_logger, 'h')

    error = False
    for i in range(2):
        resp = ethernet.sock.recv(TLM_LEN, eth_hex_logger, 'h')
        error |= tlm_chk(resp)
    return error

def EXIT():
    short_log.log("--- EXIT FPGA ---")
    cmd = CMD_exit()
    ethernet.sock.send(cmd, eth_hex_logger, 'h')

    return True

# CMD Response Parsing Functions-----------------------------------------------
def tlm_chk(data):
    error = False
    code = int(data[2] << 8) + int(data[3])
    mess = int(data[4] << 8) + int(data[5])

    medium_log.log("%s tlm rx'd. (Code:%d) (Message:%d)" \
            %(CMD_NAMES[data[0]], code, mess))
    #Error Logging
    if( code != 0 ):
        short_log.log("Error! Error code %d." %code)
        error = True
    return error

def hk_chk(data, cobras, export=False, feedback=False, updateModel=None):
    """ Consume a housedkeeping response.

    Args
    ----
    data : (byte)array
      the entire HK TLM packet.
    cobras : list of Cobras
      the identities of the cobras (module, board, cobra)
    feedback : bool
      return HK data if True
    updateModel : PfiDesign or None
      if set, design to update
    """

    error = False

    op = data[0]
    code = int(data[2] << 8) + int(data[3])
    b = int(data[4]<<8) + int(data[5])
    raw_t1 = int(data[6]<<8) + int(data[7]) 
    raw_t2 = int(data[8]<<8) + int(data[9])
    raw_v = int(data[10]<<8) + int(data[11])

    t1 = conv_temp( raw_t1 )
    t2 = conv_temp( raw_t2 )
    v = conv_volt( raw_v )

    medium_log.log("%s data tlm rx'd. (Brd:%d) (code:0x%04x) (Temps:%.1fC,%.1fC) (Voltage:%.3fV)" \
            %(CMD_NAMES[op], b, code, t1, t2, v))

    # Write board number, temps, and voltage to a new .csv file in `log` folder
    if export:
        file = time.strftime("%Y%m%d-%H%M%S.csv")
        path_to_file = os.path.join(os.getcwd(), "log", file)
        with open(path_to_file, "w", newline="") as csvfile:
            filewriter = csv.writer(csvfile, delimiter=",", quotechar="|")
            filewriter.writerow([b, t1, t2, v])

    #Error Logging
    vrange = [9.7, 10.2]
    trange = [-10, 35]
    if code != 0:
        short_log.log("Error! Error Code %d." %code)
        error = True
    if not inrange(v, vrange):
        short_log.log("Error! Voltage %.3fV outside %s." %(v, vrange))
        error = True
    if not (inrange(t1, trange) and inrange(t2, trange)):
        short_log.log("Error! Temps %d,%dC outside %s." %(t1, t2, trange) )
        error = True

    # Error Logging Payload
    i = HK_TLM_HDR_LEN

    freq1 = np.zeros(len(cobras))
    current1 = np.zeros(len(cobras))
    freq2 = np.zeros(len(cobras))
    current2 = np.zeros(len(cobras))
    hkParams = HkParams()
    for k, c in enumerate(cobras):
        # Note that this loop conveniently skips the 29th cobra on the
        # 2nd board. That is the unconnected one for which we actually
        # get a (dummy) reading for.
        p1 = int(data[i]<<8) + int(data[i+1])
        c1 = int(data[i+2]<<8) + int(data[i+3])
        p2 = int(data[i+4]<<8) + int(data[i+5])
        c2 = int(data[i+6]<<8) + int(data[i+7])
        i += 8

        freq1[k] = get_freq(p1)
        current1[k] = conv_current(c1)
        freq2[k] = get_freq(p2)
        current2[k] = conv_current(c2)

        if updateModel is not None:
            updateModel.updateMotorFrequency(theta=[get_freq(p1)*1000],
                                             phi=[get_freq(p2)*1000],
                                             moduleId=c.module,
                                             cobraId=c.cobraNum)

        logtxt = "%d 3.4mm(%.1fKhz,%.3fAmps) 2.4mm(%.1fKhz,%.3fAmps)" \
                %(c.cobra, get_freq(p1), conv_current(c1), \
                get_freq(p2), conv_current(c2))
        
        medium_log.log(logtxt)

	# Write motor data to .csv file
        if export:
            with open(path_to_file, "a", newline="") as csvfile:
                filewriter = csv.writer(csvfile, delimiter=",", quotechar="|")
                filewriter.writerow([c.cobra, get_freq(p1), conv_current(c1), \
                    get_freq(p2), conv_current(c2)])

        error |= hkParams.chk(p1, p2, c1, c2, en_log= not error)

    if not feedback:
        return error
    else:
        return error, t1, t2, v, freq1, current1, freq2, current2
