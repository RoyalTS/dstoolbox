import numpy as np
import pandas as pd
import shap

# FIXME?: This doesn't seem to return quite the same things as shap.summary_plot()
def shap_importances(model, X: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe containing the features sorted by Shap importance

    Parameters
    ----------
    model : The tree-based model
    X : pd.Dataframe
         training set/test set
    Returns
    -------
    pd.Dataframe
        A pd.DataFrame containing the features sorted by Shap importance
    """

    explainer = shap.Explainer(model)

    shap_values = explainer(X)

    vals_abs = np.abs(shap_values.values)
    # average across cases
    val_mean = np.mean(vals_abs, axis=0)

    feature_importances = pd.DataFrame(
        list(zip(shap_values.feature_names, val_mean)),
        columns=["feature", "importance"],
    ).sort_values(by=["importance"], ascending=False)

    return feature_importances
