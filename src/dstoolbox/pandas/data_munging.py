"""Functions for data munging."""
from argparse import ArgumentError
from typing import List

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
