"""Various "dirty" values."""

from typing import List, Union

import numpy as np


def common_missing_fills(variable_type: str) -> Union[List[int], List[str]]:
    """Return a list of common missing stand-ins.

    Returns
    -------
    List[int]

    Parameters
    ----------
    variable_type : str
        What type of variable to return the missing stand-ins for, either
        "numeric" or "string".

    Returns
    -------
    list
        list of strings if variable_type is "string" and of ints if
        variable_type is "numeric".
    """
    if variable_type == "numeric":
        common_missing_fill_vals_numeric = [-1]

        # repetitions of "9"s
        mc_cains = [int("9" * i) for i in range(2, 5)]
        mc_cains = mc_cains + [-i for i in mc_cains]
        common_missing_fill_vals_numeric = common_missing_fill_vals_numeric + mc_cains

        return common_missing_fill_vals_numeric

    elif variable_type == "string":
        common_missing_fill_vals_str = ["NA", "missing", "N/A", "NaN"]

        return common_missing_fill_vals_str

    else:
        raise NotImplementedError


def overflow_suspects() -> List[int]:
    """Return a list of integers that might be indicative of overflows having occurred.

    Returns
    -------
    List[int]
    """
    bit_sizes = [8, 16, 32]
    signed_unsigned = ["", "u"]

    overflow_suspects = []
    for sign in signed_unsigned:
        for size in bit_sizes:
            for pos in [-1, 1]:
                overflow_suspects.append(pos * np.iinfo(f"{sign}int{size}").min)
                overflow_suspects.append(pos * np.iinfo(f"{sign}int{size}").max)
    overflow_suspects = [s for s in overflow_suspects if s != 0]

    return overflow_suspects
