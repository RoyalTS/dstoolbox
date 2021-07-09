import altair as alt
import numpy as np
import pandas as pd

from dstoolbox.pandas.data_munging import flatten_column_index


def calibration_plot(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 20,
    ci: bool = True,
    log_axes=False,
    color="blue",
):
    """Plot the calibration of a scikit-learn model

    Parameters
    ----------
    predicted : np.ndarray
        Predicted outcomes
    actual : np.ndarray
        Actual outcomes
    n_bins : int, optional
        Number of bins to display in the plot, by default 20
    ci : bool, optional
        whether or not to show confidence bands around the calibration line, by default True
    log_axes : boolean
        Should the aces be logarithmic
    color: str
        label to be used to color the calibration line

    Returns
    -------
    altair.Chart
        The calibration plot
    """
    # combine predicted probabilities and actual outcomes in a single pandas DataFrame
    pred_v_actual = pd.DataFrame(
        {"actual": actual, "predicted": predicted},
    )

    # bin the predicted probabilities, then calculate means and stds for both predictions and actuals within bins
    pred_v_actual["predicted_binned"] = pd.qcut(
        pred_v_actual["predicted"],
        n_bins,
        duplicates="drop",
    )
    pred_v_actual_agg = pred_v_actual.groupby("predicted_binned").agg(
        {
            "actual": ["mean", "std", "count"],
            "predicted": ["mean", "std"],
        },
    )
    pred_v_actual_agg.columns = flatten_column_index(pred_v_actual_agg.columns)
    pred_v_actual_agg = pred_v_actual_agg.assign(
        actual_stderr=lambda x: x["actual_std"] / x["actual_count"].pow(1.0 / 2),
        actual_ci_lower=lambda x: x["actual_mean"] - 1.96 * x["actual_stderr"],
        actual_ci_upper=lambda x: x["actual_mean"] + 1.96 * x["actual_stderr"],
    )
    pred_v_actual_agg["color"] = color

    # to make the plot a square set the max on each axis to the max of the means
    plot_max = pred_v_actual_agg[["predicted_mean", "actual_mean"]].max().max()
    plot_min = pred_v_actual_agg[["predicted_mean", "actual_mean"]].min().min()

    domain_linear = [0, plot_max]
    domain_log = [min(0.01, plot_min), plot_max]

    if log_axes:
        domain = domain_log
        scale = alt.Scale(domain=domain_log, type="log")
    else:
        domain = domain_linear
        scale = alt.Scale(domain=domain_linear)

    # the Altair base specification on top of which to add line, points and error bars
    base = alt.Chart(pred_v_actual_agg).encode(
        x=alt.X(
            "predicted_mean",
            scale=scale,
            axis=alt.Axis(grid=False, title="Ø predicted probability"),
        ),
        y=alt.Y(
            "actual_mean",
            scale=scale,
            axis=alt.Axis(grid=False, title="Ø actual probability"),
        ),
        color=alt.Color("color", legend=alt.Legend(orient="top", title=None)),
    )

    line = base.mark_line()

    points = base.mark_point()

    # a simple diagonal for comparison
    diag_df = pd.DataFrame({"predicted_mean": domain, "actual_mean": domain})
    diagonal = (
        alt.Chart(diag_df)
        .encode(alt.X("predicted_mean"), alt.Y("actual_mean"))
        .mark_line(color="grey")
    )

    errorbars = base.encode(
        x=alt.X("predicted_mean", scale=scale),
        y=alt.Y("actual_ci_lower", scale=scale),
        y2=alt.Y2("actual_ci_upper"),
    ).mark_area(color="lightgrey", clip=True, opacity=0.2)

    everything = diagonal
    if ci:
        everything += errorbars
    everything = everything + line + points

    return everything
