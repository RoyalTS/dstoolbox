"""Utility functions for file handling."""

import datetime
import pathlib
from typing import Union


def _modified_at(
    f: pathlib.Path,
    return_type: str = "datetime",
) -> Union[datetime.datetime, datetime.date]:
    """Return modified at of a file as date or datetime.

    Parameters
    ----------
    f : pathlib.Path
        path to file
    return_type : str
        "datetime" or "date"

    Returns
    -------
    "datetime" or "date"
    """
    modified_at = datetime.datetime.fromtimestamp(f.stat().st_mtime)

    if return_type == "date":
        return modified_at.date()
    else:
        return modified_at
