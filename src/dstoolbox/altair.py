import typing

import altair as alt
from loguru import logger


def sample_for_altair(df):
    """Sample a pandas DataFrame for altair plotting if necessary."""
    if len(df) > 5000:
        logger.warning(f"Sampling down dataset from {len(df)} to 5000 rows")
        return df.sample(5000)
    else:
        return df


def date_axis(
    date_var: str,
    interval: str = "month",
    domain: typing.List[str] = None,
) -> alt.Axis:
    """Generate a date axis for Altair plots.

    Parameters
    ----------
    date_var : str
        variable name for the date variable
    interval : str, optional
        the tick interval for the date axis. Default is "month"
    domain : List[str], optional
        domain for the date axis, a list of two date strings in YYYY-MM-DD format.
        Default is None

    Returns
    -------
    alt.Axis
    """
    intervals = ["day", "week", "month"]
    if interval not in intervals:
        raise ValueError(f'interval must be one of {", ".join(intervals)}')

    # day formatting
    if interval == "day":
        scale_args = dict(nice={"interval": "day", "step": 1})
        axis_args = dict(format="%d.%m.")

    # calendar week formatting
    elif interval == "week":
        axis_args = dict(format="W%W")
        scale_args = dict(nice={"interval": "day", "step": 7})

    # day/month formatting
    elif interval == "month":
        axis_args = dict(format="%e. %b")
        scale_args = dict(nice={"interval": "month", "step": 1})

    if domain:
        scale_args["domain"] = domain

    return alt.X(
        date_var,
        title=None,
        axis=alt.Axis(**axis_args),
        scale=alt.Scale(**scale_args),
    )
