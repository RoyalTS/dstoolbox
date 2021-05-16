"""loguru-related functions."""
from loguru import logger

LOGURU_FORMAT = str(
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <7}</level> | "
    "<cyan>{name:.10}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)

# default log levels
default_log_levels = {
    "": "DEBUG",
    "toolbox.sklearn.transformers": "INFO",
    "toolbox.data": "INFO",
}


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
        import toolbox.utils.logging.loguru as log

        showwarning_ = warnings.showwarning
        warnings.showwarning = log.showwarning
    """
    logger.warning(message)
