"""Transform and log GridSearchCV and BayesSearchCV results."""

import numpy as np
import pandas as pd


def bayesopt_results_to_pd(results, scoring):
    """Convert the results of BayesianOptimization to pandas.DataFrame."""
    results_df = pd.DataFrame(results)
    results_df = pd.concat(
        [results_df, results_df["params"].apply(pd.Series)],
        axis=1,
    )
    results_df = results_df.rename(columns={"target": f"mean_test_{scoring}"})
    # enforce integrality
    results_df[["max_depth", "n_estimators"]] = results_df[
        ["max_depth", "n_estimators"]
    ].astype(np.int64)

    return results_df


def log_cv_results(
    cv_results_df,
    con,
    base_features,
    estimator,
    pipeline_flags,
    variable_set,
    optimization_method,
    sample_frac,
):
    """Log results DataFrames to a database.

    Parameters
    ----------
    cv_results_df : pandas.DataFrame
        DataFrame containing results
    con : connection
        connection
    base_features : str
        base feature setting
    estimator : str
        name of the estimator
    pipeline_flags : dict
        dictionary of flags used to parameterize the sklearn pipeline
    variable_set : array
        variables fed into the pipeline
    optimization_method : str
        optimization method
    sample_frac : float
        fraction of the entire training sample
    """
    cv_results_df["base_features"] = base_features
    cv_results_df["estimator"] = estimator
    cv_results_df["pipeline_flags"] = str(pipeline_flags)
    cv_results_df["variable_set"] = str(variable_set)
    cv_results_df["optimization_method"] = optimization_method
    cv_results_df["sample_frac"] = sample_frac
    cv_results_df["created_at"] = pd.Timestamp.now()

    # harmonize some parameter names
    if estimator == "lgbm":
        cv_results_df.columns = cv_results_df.columns.str.replace(
            "min_split_gain",
            "gamma",
        )

    object_cols = cv_results_df.select_dtypes("object").columns
    # convert OrderedDict created by e.g. BayesSearchCV into regular dictionary
    cv_results_df["params"] = cv_results_df["params"].apply(dict)
    cv_results_df[object_cols] = cv_results_df[object_cols].applymap(str)

    cv_results_df.to_sql(
        "results",
        con,
        if_exists="append",
        index_label="iteration",
    )
