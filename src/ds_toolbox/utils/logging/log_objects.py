"""Helper functions for logging various objects."""
from loguru import logger


def log_colspec(colspec):
    """Log a column_name : dtype column specification dict."""
    logger.info(f"The DataFrame has dimension {len(colspec)}.")
    logger.debug("It contains the following columns:")
    for col, dtype in colspec.items():
        logger.debug(f"- {col} ({dtype})")
