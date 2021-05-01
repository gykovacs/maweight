import logging
import datetime

class VerboseLoggingMixin:
    def __init__(self, name="", level=3):
        self.logger= logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        #logging.basicConfig(format='%(asctime)s %(message)s')
        ch= logging.StreamHandler()
        formatter= logging.Formatter('%(asctime)s %(message)s')
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)
        
        self.logging_level=level

    def set_verbosity_level(self, level):
        if level == 'NOTSET' or level == 5:
            self.logging_level= level
        elif level == 'DEBUG' or level == 4:
            self.logging_level= level
        elif level == 'INFO' or level == 3:
            self.logging_level= level
        elif level == 'WARNING' or level == 2:
            self.logging_level= level
        elif level == 'ERROR' or level == 1:
            self.logging_level= level
        elif level == 'CRITICAL' or level == 0:
            self.logging_level= level
        else:
            self.logging_level= level
    
    def verbose_logging(self, level, message):
        if level <= self.logging_level:
            #self.logger.info(message)
            print(datetime.datetime.utcnow(), message)
