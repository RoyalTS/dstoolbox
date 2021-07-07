"""loguru-related functions."""
import sys
from typing import TYPE_CHECKING

AnyPath = None
if TYPE_CHECKING:
    from _typeshed import AnyPath

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


def set_up_logger(stdout_log_level="INFO", log_file: AnyPath = None) -> None:
    """Set up the loguru logger.

    - Redirect stdout to logger and output logger back to stdout, with log-level INFO unless otherwise specified
    - if a log_file is passed, set up a logger to that file with log-level TRACE

    Parameters
    ----------
    stdout_log_level : str, optional
        log level for stdout output, by default "INFO"
    log_file : AnyPath, optional
        path to the log file, by default None
    """
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
