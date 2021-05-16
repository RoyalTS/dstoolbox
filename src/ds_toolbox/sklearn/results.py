"""Process the results of cross-validation commands."""

import pandas as pd


def cv_results_to_pd(cv_results, scoring: str) -> pd.DataFrame:
    """Convert the results of GridSearchCV or BayesSearchCV to pandas.DataFrame."""
    results = pd.DataFrame(cv_results)

    results.columns = results.columns.str.replace("_score$", f"_{scoring}")
    results.columns = results.columns.str.replace("final_estimator__", "")

    rank_cols = results.columns[results.columns.str.startswith("rank")]
    results = results.drop(columns=rank_cols)

    return results
