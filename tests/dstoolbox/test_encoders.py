from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

import dstoolbox.sklearn.transformers as t


@contextmanager
def not_raises(ExpectedException):
    try:
        yield

    except ExpectedException as error:
        raise AssertionError(f"Raised exception {error} when it should not!")

    except Exception as error:
        raise AssertionError(f"An unexpected exception {error} raised.")


X_cat = pd.DataFrame(
    {
        "var1": pd.Categorical(["a"]),
        "var2": pd.Categorical(["b"]),
    },
)

X_cat_object = pd.DataFrame(
    {
        "var1": pd.Categorical(["a"]),
        "var2": pd.Series(["b"], dtype="object"),
    },
)

X_cat_numeric = pd.DataFrame(
    {
        "var1": pd.Categorical(["a"]),
        "var2": pd.Series([1], dtype=np.int64),
    },
)

# test that _check_categorical raises appropriately
@pytest.mark.parametrize(
    "X, expect_raises",
    [
        (X_cat, False),
        (X_cat_object, True),
        (X_cat_numeric, True),
    ],
)
def test_check_categorical_raises(X, expect_raises):
    if expect_raises:
        with pytest.raises(ValueError):
            t._check_categorical(X)
    else:
        with not_raises(ValueError):
            t._check_categorical(X)


X_a = pd.DataFrame(
    {
        "var1": pd.Categorical(["a"]),
    },
)

X_a_na = pd.DataFrame(
    {
        "var1": pd.Categorical(["a", np.nan]),
    },
)

X_a_empty = pd.DataFrame(
    {
        "var1": pd.Categorical([np.nan], categories=["a"]),
    },
)

X_na = pd.DataFrame(
    {
        "var1": pd.Categorical([np.nan]),
    },
)


def fit_transform(transformer, X_fit, X_transform):
    """Utility function for testing fit-transformers."""
    y = np.repeat(1, len(X_fit))

    transformer.fit(X_fit, y)

    return transformer.transform(X_transform)


# Test all manner of different combinations for X_fit and X_transform yield the expected result
@pytest.mark.parametrize(
    "X_fit, X_transform, expected_categories",
    [
        (X_a, X_a, ["a"]),
        (X_a, X_a_na, ["a"]),
        (X_a, X_a_empty, ["a"]),
        (X_a, X_na, ["a"]),
        (X_a_na, X_a, ["a"]),
        (X_a_na, X_na, ["a"]),
        (X_a_empty, X_a, ["a"]),
        (X_a_empty, X_na, ["a"]),
    ],
)
def test_categorizer_no_unknowns(X_fit, X_transform, expected_categories):
    categorizer = t.Categorizer(handle_unknown="error")
    X_trans = fit_transform(categorizer, X_fit, X_transform)

    assert X_trans["var1"].dtype == pd.CategoricalDtype(categories=expected_categories)


X_a_b = pd.DataFrame(
    {
        "var1": pd.Categorical(["a", "a", "b", "b"], categories=["a", "b"]),
    },
)
X_a_group_b = pd.DataFrame(
    {
        "var1": pd.Categorical(["a", "a", "a", "b"], categories=["a", "b"]),
    },
)
X_a_group_b_na = pd.DataFrame(
    {
        "var1": pd.Categorical(["a", "a", np.nan, "b"], categories=["a", "b"]),
    },
)
X_a_na = pd.DataFrame(
    {
        "var1": pd.Categorical(["a", np.nan, np.nan, np.nan], categories=["a", "b"]),
    },
)
X_all_na = pd.DataFrame(
    {
        "var1": pd.Categorical([np.nan, np.nan, np.nan, np.nan], categories=["a", "b"]),
    },
)


@pytest.mark.parametrize(
    "X_fit, X_transform, expected_categories",
    [
        (X_a_b, X_a_b, ["a", "b"]),
        (X_a_b, X_a_group_b, ["a", "b"]),
        (X_a_group_b, X_a_group_b, ["a", "(other)"]),
        (X_a_group_b, X_a_b, ["a", "(other)"]),
        (X_a_group_b_na, X_a_group_b_na, ["a", "(other)"]),
        (X_a_na, X_a_na, ["(other)"]),
        (X_all_na, X_all_na, ["(other)"]),
        (X_a_b, X_all_na, ["a", "b"]),
    ],
)
def test_category_grouper_correct_categories(X_fit, X_transform, expected_categories):
    grouper = t.CategoryGrouper(threshold_absolute=2)

    X_trans = fit_transform(grouper, X_fit, X_transform)

    assert X_trans["var1"].dtype == pd.CategoricalDtype(
        categories=expected_categories,
        ordered=False,
    )
