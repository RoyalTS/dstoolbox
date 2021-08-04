import typing
from collections.abc import Iterable
from itertools import tee

import pandas as pd


def _pairwise(iterable: Iterable) -> list:
    """Returns all pairs of successive elements of a list:

    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def check_lengths_all_equal(*args):
    """Check the lengths of all passed objects is the same."""
    lengths = [len(x) for x in args]
    if not len(set(lengths)) == 1:
        raise ValueError(
            f"Passed Series and DataFrames don't all have the same lenghts: {lengths}",
        )
    else:
        return lengths[0]


def check_indices_all_equal(*args):
    """Check the indices of all passed pandas Series or DataFrames are equal."""
    pairs = _pairwise(args)
    pairs_equal = [x.index.equals(y.index) for x, y in pairs]
    if not all(pairs_equal):
        unequal_pairs = [pair for ix, pair in enumerate(pairs) if not pairs_equal[ix]]
        raise ValueError(
            f"Passed Series and DataFrames don't all have indentical indices. :{unequal_pairs[0]}",
        )


def check_columns_present(df: pd.DataFrame, column_names: typing.List[str]) -> None:
    """Raise ValueError if the passed DataFrame does not contain the passsed columns.

    Parameters
    ----------
    df : pd.DataFrame
        pd.DataFrame
    column_names : List[str]
        column name

    Raises
    ------
    ValueError
        raised if at least one of column_names is not in df
    """
    columns_missing = set(column_names) - set(df.columns.tolist())
    if columns_missing:
        raise ValueError(
            f"DataFrame must contain '{column_names}'"
            " but is missing {columns_missing}",
        )
