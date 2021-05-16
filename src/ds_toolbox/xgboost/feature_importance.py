"""Feature importances for XGBoost models."""
import pandas as pd
from sklearn.inspection import permutation_importance


def xgb_feature_importances(xgb, importance_type=None):
    """Extract feature importances from an XGBClassifier as a pandas DataFrame.

    Importance types are (https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score):

    ‘weight’: the number of times a feature is used to split the data across all trees.
    ‘gain’: the average gain across all splits the feature is used in.
    ‘cover’: the average coverage across all splits the feature is used in.
    ‘total_gain’: the total gain across all splits the feature is used in.
    ‘total_cover’: the total coverage across all splits the feature is used in.
    """
    if not importance_type:
        importance_type = xgb.importance_type

    booster = xgb.get_booster()

    importance_dict = booster.get_score(importance_type=importance_type)
    feats_imp = pd.DataFrame.from_records(
        [(name, importance_dict.get(name, 0)) for name in booster.feature_names],
        columns=["feature", "importance"],
    )

    # sort descending
    feats_imp = feats_imp.sort_values("importance", ascending=False).reset_index(
        drop=True,
    )
    # normalize
    feats_imp["importance"] = feats_imp["importance"] / feats_imp["importance"].sum()

    return feats_imp


def xgb_permutation_importances(xgb, X, y):
    """Permutation importances in XGBoost as sorted pandas DataFrame."""
    permutation_importances = permutation_importance(xgb, X, y)

    permutation_importances_df = (
        pd.DataFrame(
            {
                "feature": xgb.get_booster().feature_names,
                "importance": permutation_importances["importances_mean"],
            },
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return permutation_importances_df
