from ethernet import sock
#from tests import *
from log import dia_log, short_log, LOGS
from func import *

ctrl_ip = '128.149.77.24'
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