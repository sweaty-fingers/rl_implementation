import logging
from utilities.managers import logger_decorator

@logger_decorator
def add_log(msg, level="debug", logger=None):
    """
    log 메시지 출력
    """
    log_levels = ["debug", "info", "warning", "error", "critical"]
    if logger is None:
        print(msg)
    else:
        if level in log_levels:
            getattr(logger, level)(msg)
        else:
            print(f"Log level should be in {log_levels}")
            
@logger_decorator
def set_log_level(level, logger=None):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    logger.setLevel(numeric_level)
    print(f"Log level set to: {level}")