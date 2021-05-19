from collections import defaultdict
from typing import DefaultDict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

from ._sklearn_copies import _check_unknown

__all__ = ["Categorizer", "CategoryGrouper", "CategoryMissingAdder"]


def _check_categorical(X):
    """Check all columns are either object or categorical."""
    correct_dtypes = X.apply(
        lambda x: pd.api.types.is_object_dtype(x)
        | pd.api.types.is_categorical_dtype(x),
        axis="columns",
    )
    if not correct_dtypes.all():
        non_categoricals = correct_dtypes[~correct_dtypes].index.tolist()
        raise ValueError(
            "All columns passed must be either object or categorical dtype. "
            f"The following columns are not: {', '.join(non_categoricals)}.",
        )


class Categorizer(BaseEstimator, TransformerMixin):
    """Baseline-treat categorical variables.

    - turn them into pandas categoricals
    - handle values unseen at training time

    Parameters
    ----------
    handle_unknown : str
        either 'error', 'mark' or 'na'
    """

    def __init__(self, handle_unknown="error") -> None:
        # categories is a dictionary with the name of the columns as keys and
        # lists of the categories as values
        self.categories_: DefaultDict[str, list] = defaultdict(list)
        self.handle_unknown = handle_unknown

    def fit(self, X, y):
        check_X_y(X, y, dtype=None, force_all_finite=False)
        _check_categorical(X)

        # save the categories
        for col in X.columns:
            self.categories_[col] = X[col].unique().tolist()

        return self

    def transform(self, X):

        check_is_fitted(self)
        check_array(X, dtype=None, force_all_finite=False)

        X_copy = X.astype("category").copy()

        for col in X_copy.columns:
            diff = _check_unknown(X[col], self.categories_[col])
            if diff:
                msg = (
                    f"Found {len(diff)} unknown categories in column {col} during fit."
                )
                logger.debug(msg)
                for cat in diff:
                    logger.trace(f"- {cat}")

                if self.handle_unknown == "error":
                    raise ValueError(msg)

                elif self.handle_unknown == "ignore":
                    logger.debug("Leaving them as is")
                    continue

                elif self.handle_unknown == "missing":
                    logger.debug("Converting them to missings")
                    X_copy.loc[X_copy[col].isin(diff), col] = np.nan

                elif self.handle_unknown == "mark":
                    logger.debug('Changing all of them to "(unknown)"')
                    X_copy[col].cat.categories.append("(unknown)")
                    X_copy[X_copy[col].isin(diff), col] = "(unknown)"

        return X_copy


class CategoryGrouper(BaseEstimator, TransformerMixin):
    """A tranformer for combining low count observations in categorical features.

    This transformer will preserve category values that are above certain
    thresholds in absolute or relative frequency, while bucketing together all
    the other values. Category values that fall below _either_ the absolute or
    the relative threshold will be buckedtted into an "other" category

    This will also fix issues where new data may have an unobserved category
    value that the training data did not have.
    """

    def __init__(
        self,
        threshold_absolute=0,
        threshold_relative=0.0,
        ignore_na=True,
        other_value="(other)",
    ):
        """Initialize method.

        Parameters
        ----------
        threshold_absolute : int
            The threshold_absolute to apply the bucketing when categorical
            values drop below that threshold_absolute.
        threshold_relative : float
            The threshold_relative to apply the bucketing when categorical
            values drop below that threshold_relative.
        other_value : str
            string value/label for the residual category
        """
        self.categories_ = defaultdict(list)
        self.threshold_absolute = threshold_absolute
        self.threshold_relative = threshold_relative
        self.ignore_na = ignore_na
        self.other_value = other_value

    def fit(self, X, y=None):
        """Fits transformer over X.

        Builds a dictionary of lists where the lists are category values of the
        column key for preserving, since they meet both threshold_absolute and
        threshold_relative.
        """

        check_X_y(X, y, dtype=None, force_all_finite=False)
        _check_categorical(X)

        for col in X.columns:
            value_counts = (
                X[col].value_counts().rename_axis("value").reset_index(name="count")
            )
            value_counts["share"] = value_counts["count"] / value_counts["count"].sum()

            # value_count mask for the categories to be preserved according to
            # either threshold
            preserve_absolute_mask = value_counts["count"] >= self.threshold_absolute
            preserve_relative_mask = value_counts["share"] >= self.threshold_relative
            preserve_mask = preserve_absolute_mask & preserve_relative_mask

            kill_count = len(value_counts[~preserve_mask])

            if kill_count > 0:
                logger.debug(
                    f"Bucketting {kill_count} values in column {col} into a "
                    f'"{self.other_value}" category either because they occurred fewer '
                    f"than {self.threshold_absolute} times or because their share was "
                    f"less than {self.threshold_relative * 100:.3f}% during fit:",
                )
                for val in value_counts[~preserve_mask].to_dict("records"):
                    logger.trace(
                        f"- {val['value']} ({val['count']} cases / "
                        f"{val['share'] * 100:.3f}%)",
                    )

            # names according to the above masks
            self.categories_[col] = value_counts[preserve_mask]['value'].tolist()

        return self

    def transform(self, X):
        """Transform X with new buckets.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataset to pass to the transformer.

        Returns
        -------
            The transformed X with grouped buckets.
        """

        check_is_fitted(self)
        check_array(X, dtype=None, force_all_finite=False)

        X_copy = X.copy()

        for col in X_copy.columns:
            # boolean mask for the values to be collapsed into the "other" category
            replacement_mask = ~X_copy[col].isin(self.categories_[col])
            if self.ignore_na:
                replacement_mask = replacement_mask & (~X_copy[col].isna())

            if replacement_mask.any():
                # handle categoricals separately
                if pd.api.types.is_categorical_dtype(X_copy[col]):
                    X_copy[col] = X_copy[col].cat.add_categories(self.other_value)
                    X_copy[col][replacement_mask] = self.other_value
                    X_copy[col] = X_copy[col].cat.remove_unused_categories()
                else:
                    X_copy[col][replacement_mask] = self.other_value

        return X_copy


class CategoryMissingAdder(BaseEstimator, TransformerMixin):
    """Add an explicit missing category to category columns."""

    def __init__(self, missing_value="(missing)") -> None:

        self.missing_value = missing_value

    def fit(self, X, y):
        check_X_y(X, y, dtype=None, force_all_finite=False)
        _check_categorical(X)

        return self

    def transform(self, X):
        check_array(X, dtype=None, force_all_finite=False)

        X_copy = X.copy()

        for col in X_copy.columns:
            X_copy[col] = X_copy[col].cat.add_categories(self.missing_value)

        return X_copy
