"""Functions for data munging."""
from collections.abc import Iterable
from itertools import tee
from typing import List

import numpy as np
import pandas as pd
from loguru import logger


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
        *(set(df.select_dtypes(include=["category"]).columns) for df in dfs),
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
    import warnings

    if series.dtype != "object":
        return False
    if series.isna().all():
        warnings.warn("Series is completely NA. Can't tell if booleanish")
        return False
    else:
        # FIXME? This is a bit imprecise: None != np.nan
        return set(series) == {True, False, None}


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

    # Full names
    current_cats = set(ser.cat.categories.str.upper())
    if current_cats.issubset({d.upper() for d in calendar.day_name}):
        day_names = list(calendar.day_name)
    # Abbreviated names
    elif current_cats.issubset({d.upper() for d in calendar.day_abbr}):
        day_names = list(calendar.day_abbr)
    else:
        raise ValueError(f"Unrecognized day names: {ser.cat.categories}")

    # If the source was upper-case, upper-case the target as well
    if ser.cat.categories.str.isupper().all():
        day_names = [day.upper() for day in day_names]

    ser = ser.cat.set_categories(
        new_categories=day_names,
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


def calculate_midpoints(ser: pd.Series) -> pd.Series:
    """Calculate the midpoints between the numbers in a pandas.Series
    (implicitly adding on a 0 at the start)

    Parameters
    ----------
    ser : pd.Series
        float series

    Returns
    -------
    pd.Series
        float series
    """
    ser_cum = ser.cumsum()
    ser_cum_lag = ser_cum.shift(1).fillna(0)
    return ser_cum_lag + ser / 2


def group_rare_categories(
    ser: pd.Series,
    prob: float = None,
    cum_prob: float = None,
) -> pd.Series:
    """Group all categories whose share of the Series amount to fewer than cum_prob into an "Other" category

    Parameters
    ----------
    ser : pd.Series
        pandas.Series
    prob : float
        absolute probability threshold below which categories will be grouped into "Other"
    cumprob : float
        cumulative probability threshold below which categories will be grouped into "Other"

    Returns
    -------
    pd.Series
        pandas.Series
    """
    if not prob and not cum_prob:
        raise ValueError("Either prob or cum_prob must be passed")
    if prob and not (0.0 <= prob <= 1.0):
        raise ValueError("prob must be between 0 and 1")
    if cum_prob and not (0.0 <= cum_prob <= 1.0):
        raise ValueError("cum_prob must be between 0 and 1")

    ser_out = ser.copy()

    ser_out = ser_out.cat.add_categories("Other")

    # Get the relative frequencies of the categories
    frequencies = ser_out.value_counts(normalize=True)

    if prob:
        # replace each category with its relative frequency and compare against threshold
        # if comparison is True, replace with Other
        ser_out = ser_out.mask(
            ser_out.map(frequencies).astype(np.float64) < prob,
            "Other",
        )
    if cum_prob:
        decum_frequencies = 1 - frequencies.cumsum()

        ser_out = ser_out.mask(
            ser_out.map(decum_frequencies).astype(np.float64) < prob,
            "Other",
        )

    ser_out = ser_out.cat.remove_unused_categories()

    return ser_out


def percentile(n):
    """Return the nth percentile and give the resulting series a pretty name."""

    def percentile_(x):
        return x.quantile(n)

    percentile_.__name__ = f"percentile_{n * 100:02.0f}"
    return percentile_
