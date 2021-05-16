"""Utility functions for formatting."""
from typing import Union


def millify(n: Union[float, int], decimals: int) -> str:
    """Format a number in a human-readable way.

    pilfered from https://stackoverflow.com/a/3155023/1291563

    Parameters
    ----------
    n : float or int
        Number to be formatted
    decimals : int
        Number of decimals

    Returns
    -------
    str
        formatted number
    """
    import math

    millnames = ["", "k", "M", "B", "T"]

    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )

    return ("{val:." + str(decimals) + "f}{unit}").format(
        val=n / 10 ** (3 * millidx),
        unit=millnames[millidx],
    )
