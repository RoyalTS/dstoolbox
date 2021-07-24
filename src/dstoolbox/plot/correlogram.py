"""Interactive correlogram.

Shamelessly pilfered from https://towardsdatascience.com/altair-plot-deconstruction-visualizing-the-correlation-structure-of-weather-data-38fb5668c5b1

"""
import re

import altair as alt
import numpy as np
import pandas as pd


def compute_2d_histogram(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    density=True,
) -> pd.DataFrame:
    """Compute 2d histogram for a pair of numeric columns."""

    # numpy can only deal with column pairs without NAs
    notnull = ~df[[var1, var2]].isna().any(axis=1)
    H, xedges, yedges = np.histogram2d(
        df.loc[notnull, var1],
        df.loc[notnull, var2],
        density=density,
    )
    H[H == 0] = np.nan

    # Create a nice variable that shows the bin boundaries
    xedges = pd.Series([f"{num:.4g}" for num in xedges])
    xedges = (
        pd.DataFrame({"a": xedges.shift(), "b": xedges})
        .dropna()
        .agg(" - ".join, axis=1)
    )
    yedges = pd.Series([f"{num:.4g}" for num in yedges])
    yedges = (
        pd.DataFrame({"a": yedges.shift(), "b": yedges})
        .dropna()
        .agg(" - ".join, axis=1)
    )

    # Cast to long format using melt
    res = (
        pd.DataFrame(H, index=yedges, columns=xedges)
        .reset_index()
        .melt(id_vars="index")
        .rename(columns={"index": "value2", "value": "count", "variable": "value"})
    )

    # Also add the raw left boundary of the bin as a column, will be used to sort the axis labels later
    res["raw_left_value"] = (
        res["value"].str.split(" - ").map(lambda x: x[0]).astype(float)
    )
    res["raw_left_value2"] = (
        res["value2"].str.split(" - ").map(lambda x: x[0]).astype(float)
    )
    res["variable"] = var1
    res["variable2"] = var2
    return res.dropna()  # Drop all combinations for which no values where found


def compute_tidy_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all pairwise correlations between numeric columns of a DataFrame and return a tidy DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    correlations = (
        df.corr()
        .stack()
        .reset_index()  # The stacking results in an index on the correlation values, we need the index as normal columns for Altair
        .rename(
            columns={0: "correlation", "level_0": "variable", "level_1": "variable2"},
        )
    )
    return correlations


def interactive_correlogram(df: pd.DataFrame) -> alt.Chart:
    """Create an interactive correlogram for the numeric columns of df."""
    correlations = compute_tidy_correlations(df)
    correlations["correlation_label"] = correlations["correlation"].map(
        lambda x: re.sub("0(?=[.])", "", f"{-x:.1f}"),
    )

    numeric_columns = df.select_dtypes(np.number).columns.tolist()

    plot_data_2dbinned = pd.concat(
        [
            compute_2d_histogram(df, var1, var2)
            for var1 in numeric_columns
            for var2 in numeric_columns
        ],
    )

    # Define selector
    var_sel_cor = alt.selection_single(
        fields=["variable", "variable2"],
        clear=False,
        init={"variable": numeric_columns[0], "variable2": numeric_columns[1]},
    )

    # Define correlation heatmap
    base = alt.Chart(correlations).encode(x="variable:O", y="variable2:O")

    text = base.mark_text().encode(
        text="correlation_label",
        color=alt.condition(
            alt.datum.correlation > 0.5,
            alt.value("white"),
            alt.value("black"),
        ),
    )

    cor_plot = (
        base.mark_rect()
        .encode(
            color=alt.condition(
                var_sel_cor,
                alt.value("pink"),
                alt.Color("correlation:Q", legend=None),
            ),
        )
        .add_selection(var_sel_cor)
    )

    # Define 2d binned histogram plot
    scat_plot = (
        alt.Chart(plot_data_2dbinned)
        .transform_filter(var_sel_cor)
        .mark_rect()
        .encode(
            alt.X("value:N", sort=alt.EncodingSortField(field="raw_left_value")),
            alt.Y(
                "value2:N",
                sort=alt.EncodingSortField(field="raw_left_value2", order="descending"),
            ),
            alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
        )
    )

    # Combine all plots. hconcat plots both side-by-side
    return alt.hconcat(
        (cor_plot + text).properties(width=400, height=400),
        scat_plot.properties(width=400, height=400),
    ).resolve_scale(color="independent")
