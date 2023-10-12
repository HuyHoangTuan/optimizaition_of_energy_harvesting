from datetime import datetime

LOG_TYPE = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR"
}


def _log(_type, module_name):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{_type}] [{module_name}]: ", end="")


def debug(module_name, *args):
    _log(LOG_TYPE["DEBUG"], module_name)
    print(*args)


def info(module_name, *args):
    _log(LOG_TYPE["INFO"], module_name)
    print(*args)


def warning(module_name, *args):
    _log(LOG_TYPE["WARNING"], module_name)
    print(*args)


def error(module_name, *args):
    _log(LOG_TYPE["ERROR"], module_name)
    print(*args)

__all__ = ["debug", "info", "warning", "error"]