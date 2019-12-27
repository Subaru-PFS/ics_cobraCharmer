import sys

ADC_VREF = 2.5
NO_CONVERT = False

def adc_val_to_voltage(adc_val):
    # voltage range on ADC is 0-2VREF
    # returns the analog voltage into the ADC (ignoring noise/railing)
    if adc_val < 0 or adc_val > 65535:
        print("Invalid Adc Digital value!")
        
    return (adc_val / 65535.0) * ADC_VREF   
    
def _fpgaVoltToAdc(v):
    return int(v/ADC_VREF * 65536)

def conv_temp(adc_val):
    ''' Converts Raw ADC value to Celcius '''
    adc_volts = adc_val_to_voltage(adc_val)
    if NO_CONVERT:
        return adc_val
    #vref = 2.81
    #return (((adc_volts/15) + vref)*100) - 273

    return ((adc_volts*1000)/5.99)-273.15

def tempToAdc(t):
    v = (t + 273.15) * 5.99/1000
    return _fpgaVoltToAdc(v)

def conv_volt(adc_val):
    ''' Converts Raw ADC value to Volts '''
    r1 = 820   # changed from 835 by Mitsuko 9/13/2017
    r2 = 162   # changed from 165 by Mitsuko 9/13/2017
    adc_volts = adc_val_to_voltage(adc_val)
    if NO_CONVERT:
        return adc_val

    return adc_volts * (r1+r2)/r2
    #return adc_val

def voltToAdc(voltage):
    r1 = 820   # changed from 835 by Mitsuko 9/13/2017
    r2 = 162   # changed from 165 by Mitsuko 9/13/2017

    t = r2*voltage/(r1+r2)
    return _fpgaVoltToAdc(t)

def conv_current(adc_val):
    ''' Returns current in Amps '''
    adc_volts = adc_val_to_voltage(adc_val)
    if NO_CONVERT:
        return adc_val
    av = 100.0
    rsense = 0.020
    vsense = adc_volts / av
    return vsense/rsense

def get_freq( per ):
    ''' Converts a period value to KHz '''
    # per is number of 16Mhz periods
    freq = (16e3 / per) if (per>=1) else 0
    return freq

def get_per( freq ):
    ''' Converts a frequency in Khz to number of 60ns periods '''
    per = int(round(16e3 / (freq)))
    return per

def swapBytes(u):
    return (u&0xff)<<8 | (u&0xff00)>>8
