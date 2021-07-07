import altair as alt
import numpy as np
import pandas as pd

from dstoolbox.pandas.data_munging import flatten_column_index


def calibration_plot(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 20,
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
    log_axes : boolean
        Should the aces be logarithmic
    color: str
        color of the dots and lines

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
            axis=alt.Axis(
                grid=False,
                title="Ø predicted probability",
            ),
        ),
        y=alt.Y(
            "actual_mean",
            scale=scale,
            axis=alt.Axis(
                grid=False,
                title="Ø actual probability",
            ),
        ),
    )

    line = base.mark_line(color=color)

    points = base.mark_point()

    # a simple diagonal for comparison
    diagonal = (
        alt.Chart(
            pd.DataFrame(
                {
                    "predicted_mean": domain,
                    "actual_mean": domain,
                },
            ),
        )
        .encode(
            alt.X("predicted_mean", title=None),
            alt.Y("actual_mean", title=None),
        )
        .mark_line(color="grey")
    )

    # FIXME: for some reason the errorbar does not work with log transforms
    if not log_axes:
        errorbars = (
            base.transform_calculate(
                ymin="datum.actual_mean - datum.actual_std",
                ymax="datum.actual_mean + datum.actual_std",
            )
            .encode(
                x=alt.X("predicted_mean", scale=scale, title=None),
                y=alt.Y("ymin:Q", scale=scale, title=None),
                y2=alt.Y2("ymax:Q", title=None),
            )
            .mark_errorbar(color="lightgrey", clip=True)
        )
        return diagonal + errorbars + line + points

    else:
        return diagonal + line + points
