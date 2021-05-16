"""Process the results of XGBoost models."""

import pandas as pd
import xgboost as xgb


def round_metrics_to_pd(clf: xgb.XGBModel) -> pd.DataFrame:
    """Get round-by-round metrics as a pandas.DataFrame.

    Parameters
    ----------
    clf : sklearn.XGBModel
        An XGBoost model object

    Returns
    -------
    pd.DataFrame
        containing columns 'round', 'metric' and 'value'
    """
    round_metrics = pd.concat(
        [
            pd.DataFrame(v)
            .assign(dataset=k)
            .reset_index()
            .melt(id_vars=["index", "dataset"], var_name="metric")
            .rename(columns={"index": "round"})
            for k, v in clf.evals_result().items()
        ],
        ignore_index=True,
    )

    # logloss -> neg_log_loss
    ll_mask = round_metrics["metric"] == "logloss"
    round_metrics.loc[ll_mask, "value"] = -round_metrics.loc[ll_mask, "value"]
    round_metrics["metric"] = round_metrics["metric"].replace(
        {
            "logloss": "neg_log_loss",
        },
    )
    round_metrics["dataset"] = round_metrics["dataset"].replace(
        {
            "validation_0": "train",
            "validation_1": "test",
        },
    )

    return round_metrics
