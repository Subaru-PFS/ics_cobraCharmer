from array import array

N_BOARDS = 84
COBRAS_PER_BOARD = 29
OPCODES = {'run': 1, 'cal': 2, 'set': 3, 'hk': 4, 'pow': 5, 'dia': 6, 'admin': 7}
CMD_NAMES = {
    1: 'Run', 2: 'Calibrate', 3: 'SetFrequency', 4: 'HouseKeeping',
    5: 'Power', 6: 'Diagnostic', 7: 'Admin', 0: 'Invalid_CMD'
}

TLM_LEN = 6
HK_TLM_HDR_LEN = 12
HK_TLM_LEN = HK_TLM_HDR_LEN + 8*COBRAS_PER_BOARD
DIA_TLM_LEN = 12
ADMIN_TLM_LEN = 12


def cmd_gen(hdr_vals, payload=[]):
    t = hdr_vals + [0, 0] + payload
    cmd = array('B', t)
    ck = chksum(cmd)
    ckLoc = len(hdr_vals)
    cmd[ckLoc:ckLoc+2] = array('B', [ck >> 8, ck%256])
    return cmd


def chksum(CMD):
    res = 0
    for b in CMD:
        res = (res + b)%65536
    res = (65536 - res) & 0xffff
    return res

# Functions that return ByteArrays for a specific CMD---------------------


def CMD_run(payload, count=0, cmds=0, timeout=100, inter=500):
    hdr_vals = [OPCODES['run'], count, cmds >> 8, cmds%256,
                timeout >> 8, timeout%256, inter >> 8, inter%256]
    return cmd_gen(hdr_vals, payload)


def CMD_cal(payload, count=0, cmds=0, timeout=100):
    hdr_vals = [OPCODES['cal'], count, cmds >> 8, cmds%256,
                timeout >> 8, timeout%256]
    return cmd_gen(hdr_vals, payload)


def CMD_set(payload, count=0, cmds=0, timeout=100):
    hdr_vals = [OPCODES['set'], count, cmds >> 8, cmds%256,
                timeout >> 8, timeout%256]
    return cmd_gen(hdr_vals, payload)


def CMD_hk(board=0, count=0, timeout=100):
    hdr_vals = [OPCODES['hk'], count, board >> 8, board%256,
                timeout >> 8, timeout%256]
    return cmd_gen(hdr_vals)


def CMD_pow(pwr_secs=255, rst_secs=255, count=0):
    hdr_vals = [OPCODES['pow'], count, pwr_secs%256, rst_secs%256,
                0, 100]
    return cmd_gen(hdr_vals)


def CMD_dia(count=0):
    hdr_vals = [OPCODES['dia'], count, 0, 100]
    return cmd_gen(hdr_vals)


def CMD_exit(count=0):
    hdr_vals = [OPCODES['exit'], count]
    return cmd_gen(hdr_vals)


def CMD_admin(debugLevel=0, count=0):
    hdr_vals = [OPCODES['admin'], count, debugLevel & 0xff, 0, 0, 100]
    return cmd_gen(hdr_vals)
