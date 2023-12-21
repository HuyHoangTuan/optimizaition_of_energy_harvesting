from datetime import datetime
import logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
file_name = f"res/log/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
file_handler = logging.FileHandler(file_name)
logger.addHandler(file_handler)

LOG_TYPE = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40
}


def _log(_type, module_name, *args):
    logger.log(_type, f"[{module_name}]: {' '.join(map(str, args))}")


def debug(module_name, *args):
    _log(LOG_TYPE["DEBUG"], module_name, *args)


def info(module_name, *args):
    _log(LOG_TYPE["INFO"], module_name, *args)


def warning(module_name, *args):
    _log(LOG_TYPE["WARNING"], module_name, *args)


def error(module_name, *args):
    _log(LOG_TYPE["ERROR"], module_name, *args)

def delete_log():
    import os
    os.remove(file_name)

__all__ = ["debug", "info", "warning", "error", "delete_log"]