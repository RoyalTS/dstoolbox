"""Simple plots."""

import altair as alt
import pandas as pd


def brush_histogram(df: pd.DataFrame, var: str, maxbins: int = 50):
    """Create a brush histogram.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame
    var : str
        the variable to plot
    maxbins : int, optional
        maximal number of bins, by default 50

    Returns
    -------
    alt.Chart
    """
    brush = alt.selection_interval(encodings=["x"])

    base = (
        alt.Chart(df)
        .transform_joinaggregate(total="count(*)")
        .transform_calculate(pct="1 / datum.total")
        .mark_bar()
        .encode(y="sum(pct):Q")
        .properties(width=600, height=100)
    )

    return alt.vconcat(
        base.encode(
            alt.X(
                f"{var}:Q",
                bin=alt.Bin(maxbins=maxbins, extent=brush),
                scale=alt.Scale(domain=brush),
            ),
        ),
        base.encode(
            alt.X(
                f"{var}:Q",
                bin=alt.Bin(maxbins=maxbins),
            ),
        ).add_selection(brush),
    )
