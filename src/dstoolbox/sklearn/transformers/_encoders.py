from collections import defaultdict
from typing import DefaultDict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

__all__ = [
    "_check_categorical",
    "Categorizer",
    "CategoryGrouper",
    "CategoryMissingAdder",
]


def _check_categorical(X: pd.DataFrame):
    """Check all columns of X are categorical.

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame to check

    Raises
    ------
    ValueError
        if any columns in X are not categorical
    """
    cat_cols = X.apply(
        lambda x: pd.api.types.is_categorical_dtype(x),
        axis="index",
    )
    if not cat_cols.all():
        non_categoricals = cat_cols[~cat_cols].index.tolist()
        raise ValueError(
            "All columns passed must be categorical dtype. "
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
            # if the column is already categorical .astype('category') will leave it
            # unchanged. If it is object, this will convert it to category
            # Then save the categories
            self.categories_[col] = X[col].astype("category").cat.categories

        return self

    def transform(self, X):

        check_is_fitted(self)
        check_array(X, dtype=None, force_all_finite=False)

        X_copy = X.copy()

        for col in X_copy.columns:
            unique_vals = set(X[col].dropna().unique())
            unknown_categories = list(unique_vals - set(self.categories_[col]))
            if unknown_categories:
                msg = f"Found {len(unknown_categories)} unknown categories in column {col} during fit."
                logger.debug(msg)
                for cat in unknown_categories:
                    logger.trace(f"- {cat}")

                if self.handle_unknown == "error":
                    raise ValueError(msg)

                elif self.handle_unknown == "ignore":
                    logger.debug("Leaving them as is")
                    X_copy[col] = X_copy[col].astype(
                        pd.CategoricalDtype(
                            categories=self.categories_[col] + unknown_categories
                        )
                    )

                elif self.handle_unknown == "missing":
                    logger.debug("Converting them to missings")
                    X_copy.loc[X_copy[col].isin(unknown_categories), col] = np.nan
                    X_copy[col] = X_copy[col] = X_copy[col].astype(
                        pd.CategoricalDtype(categories=self.categories_[col])
                    )

                elif self.handle_unknown == "mark":
                    logger.debug('Changing all of them to "(unknown)"')
                    X_copy.loc[X_copy[col].isin(unknown_categories), col] = "(unknown)"
                    X_copy[col] = X_copy[col].astype(
                        pd.CategoricalDtype(
                            categories=self.categories_[col] + ["(unknown)"]
                        )
                    )
            else:
                X_copy[col] = X_copy[col].astype(
                    pd.CategoricalDtype(categories=self.categories_[col])
                )
            categories_not_present = list(set(self.categories_[col]) - unique_vals)
            if categories_not_present:
                logger.debug(
                    f"Categories {categories_not_present} do not occur in the "
                    "data but will nonetheless be added to the categories."
                )

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
        self.categories_to_group_ = defaultdict(list)
        self.threshold_absolute = threshold_absolute
        self.threshold_relative = threshold_relative
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
            self.categories_to_group_[col] = value_counts[~preserve_mask][
                "value"
            ].tolist()
            logger.trace(
                f"Final list of categories to be grouped: {self.categories_to_group_[col]}"
            )

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
            if self.categories_to_group_[col]:
                X_copy[col] = X_copy[col].cat.add_categories(self.other_value)
                for old_category in self.categories_to_group_[col]:
                    X_copy.loc[old_category, col] = self.other_value
                X_copy[col] = X_copy[col].cat.remove_categories(
                    self.categories_to_group_[col]
                )

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
