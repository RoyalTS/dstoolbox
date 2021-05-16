"""Utility functions."""

from sklearn.pipeline import Pipeline


def transform_datasets(pipeline, datasets):
    """Transform datasets through an sklearn pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A Pipeline having an estimator as its last element
    datasets : dict of arrays
        each value of the dict containing an (X, y) tuple

    Returns
    -------
    dict of arrays
        with keys identical to those in datasets and each value of the dict
        containing an (X, y) tuple
    """
    # pop off the estimator and fit_transform
    feature_pipeline = Pipeline(pipeline.steps[:-1])

    feature_pipeline.fit(*datasets["train"])

    datasets_transformed = datasets.copy()
    for dat in datasets.keys():
        datasets_transformed[dat][0] = feature_pipeline.transform(datasets[dat][0])

    return datasets_transformed
