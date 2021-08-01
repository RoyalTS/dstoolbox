"""Functions for plotting."""
from typing import List, Optional, Tuple, Union

import altair as alt
import numpy as np
import pandas as pd
from loguru import logger

from dstoolbox.pandas.data_munging import group_rare_categories
from dstoolbox.utils.formatting import millify


def _create_string_labels(categories):
    """Create unique string labels for pandas Interval series.

    Can be used to create pretty labels for the results of pandas.qcut()"""
    for decimals in range(5):
        string_labels = [
            f"({millify(i.left, decimals=decimals)}, "
            f"{millify(i.right, decimals=decimals)}]"
            for i in categories
        ]
        # test for uniqueness
        if len(set(string_labels)) == len(string_labels):
            break

    return string_labels


def _determine_axis_scale(ser: pd.Series) -> Tuple[str, int]:
    """Automatically determine axis scale and tick count for an altair plot as
    a function of the number of orders of magnitude spanned by the variable.

    If the variable spans two or fewer orders of magnitude, a linear scale is chosen.
    If it spans more two a log scale is chosen.

    Parameters
    ----------
    ser : pd.Series
        Series

    Returns
    -------
    axis_scale : str
        either "linear" or "log"
    tick_count : int
        number of ticks
    """
    non_zeros = ser[ser != 0]
    # np.ptp returns the range ("peak to peak")
    range = np.ptp(non_zeros)
    orders_of_magnitude_range = int(np.log10(range))

    if orders_of_magnitude_range > 2:
        axis_scale = "log"
        tick_count = orders_of_magnitude_range

        # remove the zeros that bust the axis under log transform
        ser = ser.replace({0: np.nan})
    else:
        axis_scale = "linear"
        tick_count = 10

    return axis_scale, tick_count


def _compute_bin_edges(
    x: np.array,
    n_bins: int,
    clip_quantiles: Optional[List[float]],
) -> np.array:
    """Compute the bin edges for binning x.

    Parameters
    ----------
    x : np.array
        the array to bin
    n_bins : int
        the number of bins
    clip_quantiles : Optional[List[float]]
        what quantiles to clip x at on the left and right. If the quantiles
        passed are identical to maximum or minimum the clipping is ignored. If
        clipped a bin each contains all values between [min, clip_left] and
        [clip_max, max]

    Returns
    -------
    np.array
        array of bin edges
    """
    if clip_quantiles and x.min() < x.quantile(clip_quantiles[0]):
        clipped_left = True
        bins_min = x.quantile(clip_quantiles[0])
        n_bins = n_bins - 1
    else:
        clipped_left = False
        bins_min = x.min()

    if clip_quantiles and x.quantile(clip_quantiles[1]) < x.max():
        clipped_right = True
        bins_max = x.quantile(clip_quantiles[1])
        n_bins = n_bins - 1
    else:
        clipped_right = False
        bins_max = x.max()

    bin_edges = np.linspace(bins_min, bins_max, num=n_bins + 1)

    if clip_quantiles and clipped_left:
        bin_edges = np.concatenate((np.array([x.min()]), bin_edges))
    if clip_quantiles and clipped_right:
        bin_edges = np.concatenate((bin_edges, np.array([x.max()])))

    bin_edges[0] = bin_edges[0] - 0.001
    bin_edges[-1] = bin_edges[-1] + 0.001

    return bin_edges


def plot_feature(
    df: pd.DataFrame,
    feature: str,
    target_var: str = None,
    bins: Union[str, List[float]] = "auto",
    n_bins: int = 20,
    clip_quantiles: list = (0.01, 0.99),
    drop_na: bool = False,
    lower_var: str = "share",
    axis_scale: str = "linear",
    split_var: Optional[str] = None,
    descriptions: Optional[pd.DataFrame] = None,
):
    """Plot the relationship between a feature and the target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        the dataset
    feature : str
        feature variable name
    target_var : str, optional
        target variable name if any
    bins: Union[str, List[float]]
        whether to create "equidistant" bins or bin by "quantile" or to
        determine the best method "auto"matically. Alternatively, pass a list
        of bin edges to bin manually
    n_bins : int, optional
        number of bins to bin any numeric variable into, by default 20
    clip_quantiles : list
        What quantiles to clip the distribution at
    drop_na : bool, optional
        ignore missings, by default False
    lower_var : str, optional
        either "share" or "count", by default "share"
    axis_scale : str
        passed to altair.Scale's type argument
    split_var : str, optional
        name of the variable to split plots

    Returns
    -------
    alt.Chart
        Altair chart

    """
    feature_orig = feature
    df = df.copy()

    # save the original dtype (for later use)
    feature_dtype = df[feature].dtype
    unique_values = df[feature].nunique(dropna=False)

    explicit_bins = isinstance(bins, list)
    num_as_cat = (not explicit_bins) & (unique_values < n_bins)

    if pd.api.types.is_object_dtype(df[feature]):
        var_type = "nominal"
        tick_min_step = 1
        sort_order = None

    elif pd.api.types.is_bool_dtype(df[feature]):
        var_type = "nominal"
        tick_min_step = 1
        sort_order = None

    elif pd.api.types.is_categorical_dtype(df[feature]):
        var_type = "nominal"
        tick_min_step = 1

        if df[feature].nunique() > 100:
            logger.info(
                "Variable has more than than 100 categories. Grouping "
                f'categories that amount to less than 0.1% into "Other" category',
            )
            df[feature] = group_rare_categories(df[feature], cum_prob=0.001)

        # unless the categorical is ordered already, order by frequency
        if not df[feature].cat.ordered:
            sort_order = df[feature].value_counts().index.tolist()
        else:
            sort_order = df[feature].cat.categories.tolist()

    elif pd.api.types.is_numeric_dtype(df[feature]) & num_as_cat:
        var_type = "quantitative"
        tick_min_step = 1
        sort_order = None

    elif pd.api.types.is_numeric_dtype(df[feature]):
        var_type = "ordinal"
        tick_min_step = 1

        if num_as_cat:
            binned = df[feature].astype("category")
        else:
            # bin a numerical feature and give it pretty labels
            if (not explicit_bins) & (bins == "auto"):
                if target_var:
                    bins = "quantiles"
                else:
                    bins = "equidistant"

            if explicit_bins:
                bin_edges = bins

                binned = pd.cut(
                    df[feature],
                    bins=bin_edges,
                    duplicates="drop",
                )
            elif bins == "equidistant":
                bin_edges = _compute_bin_edges(df[feature], n_bins, clip_quantiles)

                binned = pd.cut(
                    df[feature],
                    bins=bin_edges,
                    duplicates="drop",
                )
            elif bins == "quantiles":
                binned = pd.qcut(
                    df[feature],
                    n_bins,
                    duplicates="drop",
                )
            else:
                raise ValueError(f"bins '{bins}' not recognized")

        # up the number of decimals in category labels until they become unique
        string_labels = _create_string_labels(binned.cat.categories)

        binned = binned.cat.set_categories(
            new_categories=string_labels,
            ordered=True,
            rename=True,
        )

        # write back to DataFrame
        df[f"{feature}_binned"] = binned
        feature = f"{feature}_binned"

        sort_order = string_labels

    else:
        var_type = None
        tick_min_step = 1
        sort_order = None

    # Fill in an explicit missing category
    if not drop_na and df[feature].isna().any():
        # need to cast from boolean to category to be able to display explicit missings
        if pd.api.types.is_bool_dtype(df[feature]):
            df[feature] = df[feature].astype("category")
        if pd.api.types.is_categorical_dtype(df[feature]):
            df[feature] = df[feature].cat.add_categories("NA")
        df[feature] = df[feature].fillna("NA")

    # if target_var:
    # calculate mean, standard deviation and count of target on a feature
    # (x split_var) grouping
    grouping_vars = [feature]

    df_lower = df.groupby(feature).agg(count=(feature, "count")).reset_index()
    df_lower["share"] = df_lower["count"] / df_lower["count"].sum()

    # plot!
    common_x = {
        "shorthand": feature,
        "type": var_type,
        "sort": sort_order,
        "scale": alt.Scale(zero=False),
    }

    if axis_scale == "auto":
        axis_scale, tick_count = _determine_axis_scale(df_lower[feature])
    else:
        tick_count = 10

    y_axis_lower = {
        "count": {
            "axis": alt.Axis(
                title="Count",
                # tickCount=tick_count,
            ),
            "type": "quantitative",
        },
        "share": {
            "axis": alt.Axis(
                title="Share",
                format="%",
                tickCount=tick_count,
            ),
            "type": "quantitative",
        },
    }

    lower_bars = (
        alt.Chart(df_lower)
        .transform_calculate(bar_label=f'format(datum.{lower_var},".4f") + " %"')
        .encode(
            x=alt.X(
                **common_x,
                axis=alt.Axis(
                    tickMinStep=tick_min_step,
                    grid=False,
                ),
            ),
            y=alt.Y(
                lower_var,
                **y_axis_lower[lower_var],
                scale=alt.Scale(type=axis_scale),
            ),
            text=alt.condition(alt.datum.share < 0.01, "bar_label:N", alt.value("")),
        )
        .mark_bar()
    )

    lower_text = lower_bars.mark_text(
        align="left",
        baseline="middle",
        # Nudges text up so it doesn't appear on top of the bar
        dx=3,
        dy=0,
        angle=270,
    )

    lower = (lower_bars + lower_text).properties(height=100)

    if not target_var:
        complete_chart = lower

    else:
        if split_var:
            grouping_vars.append(split_var)
            color = {"color": alt.Color(split_var, type="nominal")}
        else:
            color = {}

        df_upper = (
            df.groupby(grouping_vars)[target_var]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        df_upper["stderr"] = df_upper["std"] / np.sqrt(df_upper["count"])

        base_upper = alt.Chart(df_upper).encode(
            x=alt.X(
                **common_x,
                axis=alt.Axis(
                    title=None,
                    labels=False,
                    tickMinStep=tick_min_step,
                    grid=False,
                ),
            ),
            y=alt.Y(
                "mean:Q",
                axis=alt.Axis(title=f"Mean {target_var}"),
                scale=alt.Scale(
                    domain=(
                        # max((df_upper["mean"] - df_upper["stderr"].fillna(0)).min(), 0),
                        0,
                        # fillna in case the stderr degenerate
                        min((df_upper["mean"] + df_upper["stderr"].fillna(0)).max(), 1),
                    ),
                ),
            ),
            yError="stderr",
            **color,
        )
        upper = base_upper.mark_point(filled=True) + base_upper.mark_errorbar()

        complete_chart = alt.vconcat(upper, lower)

        # axis can't be shared for nominal variables (Bug in Altair?)
        if var_type == "quantitative":
            complete_chart = complete_chart.resolve_scale(x="shared")

    # main title
    title_dict = {"text": [f"{feature_orig}", f"({feature_dtype})"]}

    # grab description and add as subtitle if available
    if (descriptions is not None) and (descriptions["row"] == feature_orig).any():
        row = descriptions.loc[descriptions["row"] == feature_orig]
        title_dict["subtitle"] = [row["description"].iloc[0]]
        if not row["special"].isna().iloc[0]:
            title_dict["subtitle"].append(row["special"].iloc[0])

    complete_chart = complete_chart.properties(title=title_dict)

    return complete_chart
