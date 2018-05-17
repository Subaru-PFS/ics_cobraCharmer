import logging
import os

LOG_TYPES = ["Short", "Medium", "Full", "Dia"]
NUM_LOGS = 4
LOGS = [] # Filled in after declaration


class Logger:
    def __init__(self, name, filePath, fileName='default.log'):
        self.filePath = filePath
        self.fileName = fileName
        self.name = name
        self.form = '%(message)s'
        self.__l = logging.getLogger( self.name )
        
    def setFileName(self, fileName):
        self.fileName = fileName
    def getFileName(self):
        return self.fileName
    def getFilePath(self):
        return self.filePath + chr(92) + self.fileName
        
    def setup(self, propagate=True):
        self.__fileHandler = logging.FileHandler(
            ( self.getFilePath() ), mode='w'
            )
        self.__streamHandler = logging.StreamHandler()
        self.__formatter = logging.Formatter( self.form )
        
        self.__fileHandler.setFormatter( self.__formatter )
        self.__streamHandler.setFormatter( self.__formatter )

        self.__l.setLevel( logging.INFO )
        self.__l.addHandler( self.__fileHandler )
        self.__l.addHandler( self.__streamHandler )
        self.__l.propagate = propagate
        
    def log(self, text, enable=True):
        if(enable):
            self.__l.info( text )
        
    def close(self):
        x = list( self.__l.handlers )
        for i in x:
            self.__l.removeHandler(i)
            i.flush()
            i.close()


logPath = os.path.dirname(os.path.abspath(__file__)) + r'\log'
full_log = Logger('log', logPath, 'full.log')
dia_log = Logger('log.dia', logPath, 'dia.log')
medium_log = Logger('log.medium', logPath, 'med.log')
short_log = Logger('log.medium.short', logPath, 'short.log')


LOGS = [short_log, medium_log, full_log, dia_log]
