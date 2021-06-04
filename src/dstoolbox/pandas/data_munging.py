"""Functions for data munging."""
from typing import List

import pandas as pd
from loguru import logger
from itertools import tee
from collections.abc import Iterable


def mixed_domain(series: pd.Series) -> pd.Series:
    """Determine mixed domain.

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    boolean
        True if the Series contains both positive and negative values, False otherwise.
    """
    return (series < 0).any() & (series > 0).any()


def find_signed_numeric_cols(X: pd.DataFrame, domain: str = "both") -> List[str]:
    """Find the names of signed numeric columns.

    Return those columns from a pandas DataFrame whose values are either:
    - all non-negative ("positive")
    - all negative ("negative")
    - a mix of the two ("both")

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame
    domain : str
        one of "positive", "negative" or "both", by default "both"

    Returns
    -------
    list
        containing the names of the columns from X
    """
    numerics = X.select_dtypes(["integer", "float"])

    non_negatives = (numerics >= 0).all(axis=0)
    negatives = (numerics < 0).all(axis=0)

    if domain == "positive":
        signed_cols = non_negatives
    elif domain == "negative":
        signed_cols = negatives
    else:
        signed_cols = ~(non_negatives | negatives)

    return signed_cols[signed_cols].index.tolist()


def find_duplicated_columns(df: pd.DataFrame) -> List[str]:
    """Find columns with identical values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame

    Returns
    -------
    list
        containing the names of columns whose values are identical to those in
        another column in df.
    """
    dupes = []

    columns = df.columns

    for i in range(len(columns)):
        col1 = df.iloc[:, i]
        for j in range(i + 1, len(columns)):
            col2 = df.iloc[:, j]
            # break early if dtypes aren't the same (helps deal with
            # categorical dtypes)
            if col1.dtype is not col2.dtype:
                break
            # otherwise compare values
            if col1.equals(col2):
                dupes.append(columns[i])
                logger.debug(f"Column {columns[i]} identical to {columns[j]} ")
                break

    return dupes


def concatenate_categorical_dfs(
    dfs: List[pd.DataFrame],
    **concat_kwargs,
) -> pd.DataFrame:
    """Concatenate pandas DataFrames while preserving categorical columns.

    N.B.: This changes the categories in-place for the input DataFrames

    Parameters
    ----------
    dfs : List[pandas.DataFrame]

    Returns
    -------
    pandas.DataFrame
        as returned by pandas.concat()
    """
    from pandas.api.types import union_categoricals

    categorical_cols = set.intersection(
        *[set(df.select_dtypes(include=["category"]).columns) for df in dfs],
    )

    # Iterate on categorical columns common to all dfs
    for col in categorical_cols:
        # Generate the union of the categories across dfs for this column
        uc = union_categoricals([df[col] for df in dfs])
        # Expand category set for all DataFrames
        for df in dfs:
            df[col] = pd.Categorical(
                df[col].values,
                categories=uc.categories,
            )

    return pd.concat(dfs, **concat_kwargs)


def flatten_column_index(columns: pd.core.indexes.multi.MultiIndex) -> list:
    """Flatten a pandas DataFrame's multi-index.

    Parameters
    ----------
    columns : pandas.core.indexes.multi.MultiIndex
        column index as returned by pandas.DataFrame.columns

    Returns
    -------
    list
        the flattened index
    """
    if not type(columns) == pd.core.indexes.multi.MultiIndex:
        raise ValueError("columns must be pd.core.indexes.multi.MultiIndex")

    return ["_".join(tuple(map(str, t))).rstrip("_") for t in columns]


def is_booleanish(series: pd.Series) -> bool:
    """Test if a pandas.Series is a boolean with NA (and therefore is carried as object).

    Parameters
    ----------
    series : pd.Series
        pandas Series

    Returns
    -------
    bool
        whether series is a boolean Series containing NAs
    """
    if series.dtype != "object":
        return False
    else:
        # FIXME? This is a bit imprecise: None != np.nan
        return set(series) == set([True, False, None])


def weekdays_as_category(ser: pd.Series) -> pd.Series:
    """Convert a series of weekday strings into an ordered category.

    Parameters
    ----------
    ser : pd.Series
        Series of uppercase weekday names

    Returns
    -------
    pd.Series
        Series of dtype category
    """
    import calendar

    if pd.api.types.is_object_dtype(ser):
        ser = ser.astype("category")

    if ser.cat.categories.str.isupper().all():
        cats = [day.upper() for day in calendar.day_name]
    else:
        cats = [day for day in calendar.day_name]

    ser = ser.cat.reorder_categories(
        new_categories=cats,
        ordered=True,
    )

    return ser


def floor_to_week_start(ser: pd.Series) -> pd.Series:
    """Floor a datetime Series to the date of the start of the week

    Parameters
    ----------
    dt : pd.Series[datetime64[ns]]
        datetime series

    Returns
    -------
    pd.Series
        datetime series
    """
    return ser.dt.floor("D") - pd.to_timedelta(ser.dt.dayofweek, unit="d")


def decimal_time_of_day(ser: pd.Series) -> pd.Series:
    """Calculate the time of day in 24-hour format with minutes as decimals

    That is, e.g., 12:30 will become 12.5

    Parameters
    ----------
    ser : pd.Series
        datetime series

    Returns
    -------
    pd.Series
        float series
    """
    return (ser.dt.hour * 3600 + ser.dt.minute * 60 + ser.dt.second) / 3600


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
            f"Passed Series and DataFrames don't all have the same lenghts: {lengths}"
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
            f"Passed Series and DataFrames don't all have indentical indices. :{unequal_pairs[0]}"
        )