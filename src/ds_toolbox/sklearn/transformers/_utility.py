from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

__all__ = ["ColumnNameSanitizer", "ConstantRemover"]


class ColumnNameSanitizer(BaseEstimator, TransformerMixin):
    """Clean up column names."""

    def fit(self, X, y=None):
        self.is_fitted_ = True

        return self

    def transform(self, X):
        check_is_fitted(self)

        def repl(m):
            return "__" + m.group(1)

        X.columns = X.columns.str.replace(r"\[(.*)T\.\]", repl=repl)
        X.columns = X.columns.str.replace(" ", repl="_")
        X.columns = X.columns.str.replace(":", repl="_")

        return X


class ConstantRemover(BaseEstimator, TransformerMixin):
    """Remove constant features."""

    def fit(self, X, y):
        is_constant = X.apply(lambda x: x.nunique() <= 1)
        self.constant_columns = is_constant[is_constant].index.tolist()
        self.is_fitted_ = True

        return self

    def transform(self, X):
        check_is_fitted(self)

        logger.info(
            f"Dropping {len(self.constant_columns)} columns "
            "because they were constant in training:",
        )
        for col in self.constant_columns:
            logger.debug(f"- {col}")

        return X.drop(columns=self.constant_columns)
