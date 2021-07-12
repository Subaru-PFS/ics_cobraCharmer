import logging
import os

LOG_TYPES = ["Short", "Medium", "Full", "Dia"]
NUM_LOGS = 4
LOGS = []  # Filled in after declaration

logging.basicConfig(format="%(asctime)s.%(msecs)03d %(levelno)s %(name)-10s %(filename)s:%(lineno)s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")


class Logger:
    def __init__(self, name, filePath, fileName='default.log'):
        self.name = name
        self.filePath = filePath
        self.fileName = fileName
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def setFileName(self, fileName):
        self.fileName = fileName

    def getFileName(self):
        return self.fileName

    def getFilePath(self):
        return os.path.join(self.filePath, self.fileName)

    def setup(self, propagate=True):
        pass

    def debug(self, text):
        self.logger.debug(text)

    def log(self, text, enable=True):
        if(enable):
            self.logger.info(text)
        else:
            self.logger.debug(text)

    def close(self):
        pass

    def setLogpath(self, logPath, logLevel=None):
        self.logPath = logPath
        if logPath is not None:
            self.logger.handlers = []
            self.logger.addHandler(logging.FileHandler(os.path.join(logPath, f'{self.name}.log')))
        if logLevel is not None:
            self.logger.setLevel(logLevel)

    @classmethod
    def getLogger(cls, name='logger', debug=False):
        logger = logging.getLogger(name)
        return logger


logPath = '.'
full_log = Logger('log.full', logPath, 'full.log')
dia_log = Logger('log.dia', logPath, 'dia.log')
medium_log = Logger('log.medium', logPath, 'med.log')
short_log = Logger('log.short', logPath, 'short.log')
eth_hex_logger = Logger('log.eth', logPath, 'eth.log')

LOGS = [short_log, medium_log, full_log, dia_log]


def setupLogPaths(logPath):
    for l in LOGS:
        l.setLogpath(logPath)
