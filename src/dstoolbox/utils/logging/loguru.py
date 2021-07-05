"""loguru-related functions."""
import sys

from loguru import logger

log_levels = ["ERROR", "WARNING", "INFO", "DEBUG", "TRACE"]

LOGURU_FORMAT_FULL = str(
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <7}</level> | "
    "<cyan>{name:.10}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)

LOGURU_FORMAT_SHORT = str(
    "<level>{level: <7}</level> | "
    "<cyan>{name:.10}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)

# default log levels
default_log_levels = {
    "": "DEBUG",
    "dstoolbox.sklearn.transformers": "INFO",
}


def set_up_logger(stdout_log_level="INFO", log_file=None):
    # redirect stdout to logger, set log level to TRACE for file output always and to a
    # level commensurate with the environment for stdout (seen in Jenkins in production)
    # unless overridden via command line argument
    logger.remove()
    if log_file:
        logger.add(sys.__stdout__, level=stdout_log_level, format=LOGURU_FORMAT_SHORT)
        logger.add(log_file, level="TRACE", format=LOGURU_FORMAT_FULL)
    else:
        logger.add(sys.__stdout__, level=stdout_log_level, format=LOGURU_FORMAT_FULL)


class StreamToLogger:
    """Log stream like stdout/stderr to logger."""

    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self):
        pass


def showwarning(message, *args, **kwargs):
    """Route warnings to logger.

    To do this add the following to the top of the script:

    .. code-block:: python

        import warnings
        import dstoolbox.utils.logging.loguru as log

        showwarning_ = warnings.showwarning
        warnings.showwarning = log.showwarning
    """
    logger.warning(message)
