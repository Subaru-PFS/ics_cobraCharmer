""" take msim fpgalog_txt files on stdin, generate our logs on output.

Bugs:
 - MSIM timestamps are not retained.
 - We do not do anything with replies ("TLM"s) yet.

"""

from . import fpgaLogger, fpgaProtocol

fpgaLog = fpgaLogger.FPGAProtocolLogger()

def dispatchCmd(cmdStr):
    cmd, ts, data = cmdStr.split(',')
    data = bytes.fromhex(data)
    headerSize, _ = fpgaProtocol.cmds[int(cmd)]

    header, data = data[:headerSize], data[headerSize:]
    fpgaLog.logCommand(header, data, ts=ts)

def dispatchTlm(tlmStr):
    try:
        header, data = tlmStr[1:-1].split('][')
        header = header.split(',')
        data = data.split(',')
        # There is a stray ',' at the end
        if data[-1] == '':
            data = data[:-1]
    except ValueError:
        header = tlmStr[1:-1].split(',')
        data = None

    tlmType, cmd, ts, *header = header
    fpgaLog.logTlm(header, ts=ts, hkData=data)

def nextLine(f):
    for line in f:
        yield line

def main(f):
    for l in nextLine(f):
        l = l.strip()
        if l == '':
            continue
        if l.startswith('['):
            dispatchTlm(l)
        else:
            dispatchCmd(l)

if __name__ == "__main__":
    import sys

    main(sys.stdin)

    # with open("/data/cobras/TEST_RESULTS/Spare1/18_01_10_14_31_35_TargetRun/fpga_log.txt", "rt") as f:
    #    main(f)


sample = """
3,3918714170.000000,030100384e208ae2082300fd0099101300fb009a102300fc0099181300fd0098182300fc0097201301000097202300fc0098281300fa0099282300fc0097301300fe0095302300fd0098381300fe0097382300fc0098401300fa0097402300fc009a481300fc0098482300fe0096501300fd0098502300fa0097581300fb0097582300fe0099601300fc0099602300fa0098681300fd0097682301000096701300fd0099702300fd0094781300fc0097782300fc0098801300fb0093802300fb0096881300fb0098882300fd0098901300fd0096902300fa0098981300ff0097982300fa0098a01300ff0099a02300fe0096a81300fc0097a82301000098b01300fd0097b02301010099b81300fa0097b82300fc0098c01300fa0097c02300fe0098c81300fc0096c82300f80096d01300fa0097d02300f90097d81300f80099d82300fb0099e01300fa009ae02300f70095e81300fc009a
[TLM1,3,3918714233.000000,3,1,0,0]

[TLM1,3,3918714389.000000,3,1,0,0]

4,3918714514.000000,040200019c40ff1d
[TLM1,4,3918714576.000000,4,2,0,0]

[TLM_HK,4,3918714732.000000,4,2,1,47640,47720][253,0,153,0,252,0,153,0,252,0,151,0,252,0,152,0,252,0,151,0,253,0,152,0,252,0,152,0,252,0,154,0,254,0,150,0,250,0,151,0,254,0,153,0,250,0,152,0,256,0,150,0,253,0,148,0,252,0,152,0,251,0,150,0,253,0,152,0,250,0,152,0,250,0,152,0,254,0,150,0,256,0,152,0,257,0,153,0,252,0,152,0,254,0,152,0,248,0,150,0,249,0,151,0,251,0,153,0,247,0,149,0,]

4,3918714826.000000,040300029c40ff1b
[TLM1,4,3918714888.000000,4,3,0,0]

[TLM_HK,4,3918715044.000000,4,3,2,47532,47710][253,0,148,0,251,0,154,0,253,0,152,0,256,0,151,0,250,0,153,0,254,0,149,0,254,0,151,0,250,0,151,0,252,0,152,0,253,0,152,0,251,0,151,0,252,0,153,0,253,0,151,0,253,0,153,0,252,0,151,0,251,0,147,0,251,0,152,0,253,0,150,0,255,0,151,0,255,0,153,0,252,0,151,0,253,0,151,0,250,0,151,0,250,0,151,0,252,0,150,0,250,0,151,0,248,0,153,0,250,0,154,0,252,0,154,0,]

1,3918720722.000000,01040038fffe009cb6e6082a000000000000001e17700000101b000000000000001a17700000102b000000000000001a17700000181b000000000000001e17700000182b000000000000001b17700000201b000000000000002217700000202b000000000000001b17700000281b000000000000001717700000282b000000000000002417700000301a000000000000001f17700000302b000000000000001d17700000381b000000000000002917700000382b000000000000001a17700000401b000000000000002617700000402b000000000000001f17700000481b000000000000002b17700000482a000000000000001e17700000501b000000000000001e17700000502a000000000000001417700000581a000000000000002317700000582b000000000000001717700000601b000000000000003917700000602b000000000000002717700000681a000000000000002317700000682a000000000000002817700000701a000000000000001f17700000702a000000000000002717700000781a000000000000001d17700000782b000000000000001517700000801b000000000000002417700000802b000000000000002917700000881a000000000000002817700000882a000000000000001817700000901a000000000000001b17700000902a000000000000001717700000981a000000000000001f17700000982a000000000000001a17700000a01b000000000000002117700000a02b000000000000001e17700000a81b000000000000001b17700000a82b000000000000002217700000b01b000000000000001a17700000b02a000000000000001f17700000b81a000000000000001d17700000b82a000000000000002317700000c01b000000000000002a17700000c02a000000000000001a17700000c81b000000000000001e17700000c82b000000000000001517700000d01a000000000000001817700000d02b000000000000001417700000d81a000000000000002317700000d82b000000000000002117700000e01a000000000000002e17700000e02a000000000000001a17700000e81b000000000000003117700000
[TLM1,1,3918720816.000000,1,4,0,0]

[TLM1,1,3918735777.000000,1,4,0,0]
"""