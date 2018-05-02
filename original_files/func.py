import math

from cmds import *
from ethernet import sock
from log import full_log, medium_log, short_log
from convert import *

CCW_DIR = ('ccw','ccw')
CW_DIR = ('cw','cw')
EN_BOTH = (True, True)
EN_M0 = (True, False)
EN_M1 = (False, True)

#
eth_hex_logger = None

# Classes----------------------------------------------------------------------
class Cobra:
    def __init__(self, board, num):
        self.board = board
        self.cobra = num
        self.p = None #Params
    def stringify(self):
        s = '(C%s '%self.cobra + self.p.stringify() + ')'
        return s
    def typeMatch(self, type):
        if self.p is None:
            return False
            short_log.log("None params type")
        if self.p.type != type:
            return False
            short_log.log("Wrong params type")
        return True
        
class CalParams:    
    def __init__(self, m0=(230,280), m1=(140,190), en=EN_BOTH, dir=CCW_DIR ):
        self.type = 'Cal'
        self.dir = dir
        self.en = en
        self.m0Range = m0
        self.m1Range = m1
    def stringify(self):
        s = ''
        s += 'm0:%s:%s' %(self.dir[0], str(self.m0Range)) if self.en[0] else ''
        s += '  ' if self.en[0] and self.en[1] else ''
        s += 'm1:%s:%s' %(self.dir[1], str(self.m1Range)) if self.en[1] else ''
        return s
    def toList(self, board, cnum):
        c = 0x0000 | (self.dir[0]=='ccw') | ((self.dir[1]=='ccw')<<1) | \
                (self.en[0]<<2) | (self.en[1]<<3) | (board<<4) | (cnum%30<<11)
                
        p = [c>>8, c%256, self.m0Range[0]>>8, self.m0Range[0]%256, \
                self.m0Range[1]>>8, self.m0Range[1]%256, self.m1Range[0]>>8, \
                self.m1Range[0]%256, self.m1Range[1]>>8,  self.m1Range[1]%256]
        return p
        
class HkParams:
    def __init__(self, m0=(60.0,70.0), m1=(94.0,110.0), \
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
        c = 0x0000 | (self.dir[0]=='ccw') | ((self.dir[1]=='ccw')<<1) | \
                (self.en[0]<<2) | (self.en[1]<<3) | (board<<4) | (cnum%30<<11)
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
        c = 0x0000 | (self.en[0]) | (self.en[1]<<1) | \
                (board<<4) | (cnum%30<<11)
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
 

# Test Functions-------------------------------------------------------------
def POW(sec_pwr=255):
    short_log.log("--- POWER ---")
    
    sectors_off = []
    for i in range(0,6):
        if not sec_pwr & (0x01 << i):
            sectors_off.append(i)
    
    medium_log.log("Sectors Without Power: %s" %sectors_off)
    
    cmd = CMD_pow(sec_pwr, 0)
    sock.send(cmd, eth_hex_logger, 'h')
    resp = sock.recv(TLM_LEN, eth_hex_logger, 'h')
    error = tlm_chk(resp)
    return error
    
def RST():
    short_log.log("--- RESET ---")
    sec_rst = 255
    
    sectors_reseting = []
    for i in range(0,6):
        if sec_rst & (0x01 << i):
            sectors_reseting.append(i)
    
    medium_log.log("Sectors Reseting: %s" %sectors_reseting)
    
    cmd = CMD_pow(255, sec_rst)
    sock.send(cmd, eth_hex_logger, 'h')
    resp = sock.recv(TLM_LEN, eth_hex_logger, 'h')
    error = tlm_chk(resp)
    return error
    
def DIA():
    short_log.log("--- DIAGNOSTIC INFO ---")
    cmd = CMD_dia()
    sock.send(cmd, eth_hex_logger, 'h')
    resp = sock.recv(DIA_TLM_LEN, eth_hex_logger, 'h')
    
    boards_per_sector = [
        int(resp[2]), int(resp[3]), int(resp[4]), int(resp[5]), int(resp[6]), int(resp[7])
    ]
    short_log.log("Board Counts: %s" %(boards_per_sector) )
    return boards_per_sector
    
    
def HK(cobras):
    if not cobrasAreType(cobras, 'Hk'):
        return True # error
    board = cobras[0].board
    
    short_log.log("--- ISSUE HK & VERIFY (brd:%d) ---" %board)      
    
    cmd = CMD_hk(board, timeout=2000)
    sock.send(cmd, eth_hex_logger, 'h')
    
    resp = sock.recv(TLM_LEN, eth_hex_logger, 'h')
    er1 = tlm_chk(resp)
    
    resp = sock.recv(HK_TLM_LEN, eth_hex_logger, 'h')
    er2 = hk_chk(resp, cobras)
    
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
            f0rng = (c.p.m0Range[1] - c.p.m0Range[0] )
            f1rng = (c.p.m1Range[1] - c.p.m1Range[0] )
            fRngCob.append( max(f0rng, f1rng) )
        tPerCobra = int(7.0*max(fRngCob)) # assume <=7ms per freq
        timeout = math.ceil( 1200 + tPerCobra*len(cobras) )
        
    medium_log.log("Timeout:%d" %timeout)
    
    payload = []
    for c in cobras:
        payload += c.p.toList(c.board, c.cobra)

    cmd = CMD_cal(payload, cmds=len(cobras), timeout=timeout)
    sock.send(cmd, eth_hex_logger, 'h')
    
    error = False
    for i in range(2):
        resp = sock.recv(TLM_LEN, eth_hex_logger, 'h')
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
        
    # Get interleave by finding longest pulsetime
    if inter == 0:
        puCob = []
        for c in cobras:
            puCob.append( max(c.p.pulses) )
        # * 9/15ths for max 10% duty cycle
        inter = math.ceil( max(puCob)*9/15 )
    
    medium_log.log("Timeout:%d, inter:%d" %(timeout,inter) )
        
    payload = []
    for c in cobras:
        payload += c.p.toList(c.board, c.cobra)
        
    cmd = CMD_run(payload, cmds=len(cobras), timeout=timeout, inter=inter)
    sock.send(cmd, eth_hex_logger, 'h')
    
    error = False
    for i in range(2):
        resp = sock.recv(TLM_LEN, eth_hex_logger, 'h')
        error |= tlm_chk(resp)
    return error
    
def SET( cobras ):
    if not cobrasAreType(cobras, 'Set'):
        return True # error
    board = cobras[0].board
    
    short_log.log("--- ISSUE SETFREQ & VERIFY (brd:%d) ---" %board)
    
    payload = []
    for c in cobras:
        payload += c.p.toList(c.board, c.cobra)

    cmd = CMD_set(payload, cmds=len(cobras), timeout=2000)
    sock.send(cmd, eth_hex_logger, 'h')
    
    error = False
    for i in range(2):
        resp = sock.recv(TLM_LEN, eth_hex_logger, 'h')
        error |= tlm_chk(resp)
    return error


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
    
def hk_chk(data, cobras):
    error = False
    trange = cobras[0].p.trange
    vrange = cobras[0].p.vrange
    
    op = data[0]
    code = int(data[2] << 8) + int(data[3])
    b = int(data[4]<<8) + int(data[5])
    raw_t1 = int(data[6]<<8) + int(data[7]) 
    raw_t2 = int(data[8]<<8) + int(data[9])
    raw_v = int(data[10]<<8) + int(data[11])
    
    t1 = conv_temp( raw_t1 )
    t2 = conv_temp( raw_t2 )
    v = conv_volt( raw_v )
    
    medium_log.log("%s data tlm rx'd. (Brd:%d) (Temps:%.1fC,%.1fC) (Voltage:%.3fV)" \
            %(CMD_NAMES[op], b, t1, t2, v))
            
    #Error Logging
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
    for c in cobras:
        p1 = int(data[i]<<8) + int(data[i+1])
        c1 = int(data[i+2]<<8) + int(data[i+3])
        p2 = int(data[i+4]<<8) + int(data[i+5])
        c2 = int(data[i+6]<<8) + int(data[i+7])
        i += 8
        
        logtxt = "%d 3.4mm(%.1fKhz,%.3fAmps) 2.4mm(%.1fKhz,%.3fAmps)" \
                %(c.cobra, get_freq(p1), conv_current(c1), \
                get_freq(p2), conv_current(c2))
        
        medium_log.log(logtxt)

        error |= c.p.chk(p1, p2, c1, c2, en_log= not error)
    return error
    
