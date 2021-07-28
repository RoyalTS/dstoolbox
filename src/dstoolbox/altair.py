import altair as alt
from loguru import logger


def sample_for_altair(df):
    """Sample a pandas DataFrame for altair plotting if necessary."""
    if len(df) > 5000:
        logger.warning(f"Sampling down dataset from {len(df)} to 5000 rows")
        return df.sample(5000)
    else:
        return df


def date_axis(date_var: str, interval: str = "month") -> alt.Axis:
    """Generate a date axis for Altair plots.

    Parameters
    ----------
    date_var : str
        variable name for the date variable
    interval : str, optional
        the tick interval for the date axis. Default is "month"

    Returns
    -------
    alt.Axis
    """
    # calendar week formatting
    if interval == "week":
        return alt.X(
            date_var,
            title=None,
            axis=alt.Axis(format="W%W"),
            scale=alt.Scale(nice={"interval": "day", "step": 7}),
        )
    # day/month formatting
    elif interval == "month":
        return alt.X(
            date_var,
            title=None,
            axis=alt.Axis(format="%e. %b"),
            scale=alt.Scale(nice={"interval": "month", "step": 1}),
        )
