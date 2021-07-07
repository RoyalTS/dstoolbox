"""Functions for performing sanity checks on data frames."""
from typing import Optional

import pandas as pd

from ..pandas.data_munging import mixed_domain
from .dirty_values import common_missing_fills


def check_no_objects(df: pd.DataFrame) -> bool:
    """Check the DataFrame contains no object columns."""
    return ~(df.dtypes == "object").any()


def mixed_domain_columns(df: pd.DataFrame) -> list:
    """Find the names of columns with mixed domain."""
    has_mixed_domain = df.select_dtypes(
        exclude=["object", "category"],
    ).apply(mixed_domain)

    return has_mixed_domain[has_mixed_domain].index.tolist()


def missing_suspect_columns(
    df: pd.DataFrame,
    missing_suspect_values: Optional[list] = None,
) -> list:
    """Find the names of columns which contain values that might actually be missings.

    Will look for that values contained in dirty_values.common_missing_fills as well as
    any `missing_suspect_values` passed

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame
    missing_suspect_values : Optional[list], optional
        additional values that should be considered missing suspects, by default None

    Returns
    -------
    list
        list of column names in df
    """
    if missing_suspect_values is None:
        missing_suspect_values = []

    fills_string = common_missing_fills(variable_type="string") + missing_suspect_values

    # returns a DataFrame withn the original columns and the matched missings
    # as rows, with shares of the data that have this value in the cells
    missing_string = df.select_dtypes(include=["object", "category"]).apply(
        lambda x: (x[x.isin(fills_string)].value_counts() / len(x)),
    )

    fills_numeric = common_missing_fills(variable_type="numeric")

    missing_numerics = df.select_dtypes(exclude=["object", "category"]).apply(
        lambda x: (x[x.isin(fills_numeric)].value_counts() / len(x)),
    )

    missing_all = pd.concat([missing_numerics, missing_string])
    missing_all = missing_all[missing_all.columns[missing_all.sum() > 0]]
    missing_all = missing_all[~missing_all.isna().all(axis=1)]

    return missing_all
