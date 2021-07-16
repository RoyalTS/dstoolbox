"""Utility functions."""

import typing

import sklearn.base
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
    feature_pipeline, _ = pop_estimator_off_pipeline(pipeline)

    feature_pipeline.fit(*datasets["train"])

    datasets_transformed = datasets.copy()
    for dat in datasets.keys():
        datasets_transformed[dat][0] = feature_pipeline.transform(datasets[dat][0])

    return datasets_transformed


def pop_estimator_off_pipeline(
    pipeline: sklearn.pipeline.Pipeline,
) -> typing.Tuple[sklearn.pipeline.Pipeline, sklearn.base.BaseEstimator]:
    """Separate a complete sklearn pipeline into the feature pipeline and its final estimator.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A Pipeline having an estimator as its last element

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        A Pipeline having an estimator as its last element
    estimator : sklearn.base.BaseEstimator
        The final estimator
    """
    feature_pipeline = Pipeline(pipeline.steps[:-1])
    estimator = pipeline.steps[-1][1]

    if not isinstance(estimator, sklearn.base.BaseEstimator):
        raise ValueError("The final step in the passed pipeline is not an estimator")

    return feature_pipeline, estimator
