"""Function to describe a DataFrame."""
import numpy as np
import pandas as pd
from sparklines import sparklines


def sparkline_str(x: np.array, bins=20) -> str:
    """Create a sparkline string that represents the distribution of a numpy array.

    Parameters
    ----------
    x : np.array
        The array to plot

    Returns
    -------
    str
        sparkline string
    """
    bins = np.histogram(x, bins=bins, range=[np.nanmin(x), np.nanmax(x)])[0]

    sl = "".join(sparklines(bins))

    return sl


sparkline_str.__name__ = "sparkline"


def share_na(x):
    """Calculate the share of NAs in a pandas.Series."""
    return x.isna().sum() / len(x)


share_na.__name__ = "share_NA"


def share_zeros(x):
    """Calculate the share of 0s in a pandas.Series."""
    return (x == 0).sum() / len(x)


def share_mode(x):
    """Calculate the share of the modal value in a pandas.Series."""
    return x.value_counts().head(1).iloc[0] / len(x)


def describe(df: pd.DataFrame, dtypes: str = "numeric") -> pd.DataFrame:
    """Describe the contents of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    dtypes : str, optional
        either "numeric" or "object", by default "numeric"

    Returns
    -------
    pd.DataFrame
    """
    if dtypes == "numeric":
        df_numeric = df.select_dtypes(exclude=["object", "category"])

        # need to split this into three parts: simple aggregations, quantiles
        # and the sparkline
        basics = df_numeric.agg(
            [
                share_na,
                share_zeros,
                "mean",
                "std",
                "min",
                "max",
            ],
        ).T

        quantiles = {
            "0.05": "q05",
            "0.5": "median",
            "0.95": "q95",
        }
        quantile_cols = df_numeric.quantile(q=[float(q) for q in quantiles.keys()]).T

        sparkline_col = df_numeric.apply(sparkline_str, raw=True)
        sparkline_col = sparkline_col.to_frame(name="sparkline")

        combined = pd.concat([basics, quantile_cols, sparkline_col], axis=1)

        combined.columns = combined.columns.astype(str)
        combined = combined.rename(columns=quantiles)

        format_dict = {
            "sparkline": "{}",
            "share_NA": "{:.2%}",
            "share_zeros": "{:.2%}",
            "mean": "{0:,.2f}",
            "std": "{0:,.2f}",
            "min": "{0:,.2f}",
            "q05": "{0:,.2f}",
            "median": "{0:,.2f}",
            "q95": "{0:,.2f}",
            "max": "{0:,.2f}",
        }

        combined = combined[list(format_dict.keys())]

        styled = combined.style

    elif dtypes == "object":
        df_object = df.select_dtypes(include=["object", "category"])
        counts_and_shares = df_object.agg(
            [
                share_na,
                "nunique",
                share_mode,
            ],
        ).T
        vals = df_object.agg("mode").T.rename(columns={0: "mode"})

        combined = pd.concat([counts_and_shares, vals], axis=1)

        combined.columns = combined.columns.astype(str)
        combined[["nunique"]] = combined[["nunique"]].astype(np.int64)

        format_dict = {
            "share_NA": "{:.2%}",
            "nunique": "{:d}",
            "mode": "{}",
            "share_mode": "{:.2%}",
        }

        combined = combined[list(format_dict.keys())]

        styled = combined.style.bar(
            color=["#CCCCCC"],
            vmin=0,
            vmax=1,
            subset=["share_mode"],
        )

    styled = styled.bar(
        color="#d65f5f",
        vmin=0,
        vmax=1,
        subset=["share_NA"],
    ).format(format_dict)

    return styled
