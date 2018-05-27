def num2Hex(number, byteSize=1):
    # Number to a string of Hex.  ie 180 to '00B4'
    # Format does the job
    str = format(number, 'x')
    # But may need padding in string depending on intended size
    while len(str) < byteSize*2: 
        str = '0' + str
    return str

def arr2Hex(arr, seperator=','):
    # Takes ByteArray and makes it a string of Hex.
    str = ''
    size = len(arr)
    for i in range(0,size):
        str = str +  num2Hex(arr[i],1)
        str = str + seperator if i != (size-1) else str
    return str