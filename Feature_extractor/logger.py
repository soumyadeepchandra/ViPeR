
import os
import sys
import logging


def setup_logging(output_dir=None, level='info', bypass=False):
    
    logger_level = logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(logger_level)

    format_ = logging.Formatter(
        "%(asctime)s %(levelname)s %(process)d %(filename)s %(lineno)3d: "
        "%(message)s",
        datefmt="%d/%m/%y %H:%M:%S"
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logger_level)
    ch.setFormatter(format_)
    logger.addHandler(ch)

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        filename = os.path.join(output_dir, 'logs.log')
        fh = logging.FileHandler(filename)
        fh.setLevel(logger_level)
        fh.setFormatter(format_)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with specified name
    """
    return logging.getLogger(name)
