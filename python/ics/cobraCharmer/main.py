from ics.cobraCharmer.ethernet import sock
from ics.cobraCharmer.log import dia_log, short_log, LOGS
from ics.cobraCharmer.func import *

ctrl_ip = '127.0.0.1'
lcl_ip = '127.0.0.1'
port = 4001


def setup(ip=ctrl_ip):
    # Setup the logs
    for i in LOGS:
        i.setup()

    # Socket connect
    sock.connect(ctrl_ip, port, short_log)


def closure():
    # close the logs
    for i in LOGS:
        i.close()

    # socket close
    sock.close(short_log)


'''
def main():
    setup()
    
    results = []
    halted = False
    # Run appropriate tests, but quit if a HALT is returned
    for i in range(NUM_TESTS):
        if( RUN_TEST[i] ):
            results.append( TESTS[i]() )
            if( results[-1] == HALT ):
                # Issue diagnosis on halt
                dia_log.log("Halted on %s!" %TEST_NAMES[i])
                halted = True
                break
            
    # If no halt, count number of test results that err'd
    if(not halted):
        errors = sum(results)
        # Then issue diagnosis
        if(errors):
            for i in range(len(results)):
                if( results[i] == ERROR):
                    dia_log.log( TEST_NAMES[i] + "Test Failed!" )
        else:
            dia_log.log("All Tests ran successfully!")
        
        
    #Close Socket
    sock.close(short_log)

    # Close the logs
    for i in LOGS:
        i.close() 

'''
