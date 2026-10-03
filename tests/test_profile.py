"""
Tests for DataFrameCleaner.profile(), column_types(), and profile_dataframe() convenience function.
"""

import numpy as np
import pandas as pd
import pytest
from data_engine import DataFrameCleaner, profile_dataframe


@pytest.fixture
def mixed_df():
    """DataFrame with numeric, categorical, datetime, and missing values."""
    return pd.DataFrame({
        "age": [25.0, 32.0, np.nan, 45.0, 28.0],
        "income": [50000, 65000, 72000, 80000, 55000],
        "city": ["NYC", "LA", "NYC", None, "LA"],
        "joined": pd.to_datetime(["2021-01-01", "2022-06-15", "2020-03-10", "2023-09-05", "2021-11-20"]),
        "active": [True, False, True, True, False],
    })


def test_profile_returns_dataframe(mixed_df):
    cleaner = DataFrameCleaner(verbose=False)
    profile = cleaner.profile(mixed_df)

    assert isinstance(profile, pd.DataFrame)
    assert len(profile) == len(mixed_df.columns)
    assert profile.index.name == "column"


def test_profile_index_matches_columns(mixed_df):
    cleaner = DataFrameCleaner(verbose=False)
    profile = cleaner.profile(mixed_df)

    assert list(profile.index) == list(mixed_df.columns)


def test_profile_expected_columns(mixed_df):
    cleaner = DataFrameCleaner(verbose=False)
    profile = cleaner.profile(mixed_df)

    expected_cols = [
        "dtype", "missing_count", "missing_pct", "unique_count", "unique_pct",
        "mean", "median", "std", "min", "max", "mode",
        "is_numeric", "is_categorical", "is_datetime",
    ]
    for col in expected_cols:
        assert col in profile.columns, f"Expected column '{col}' in profile output"


def test_profile_missing_values(mixed_df):
    cleaner = DataFrameCleaner(verbose=False)
    profile = cleaner.profile(mixed_df)

    # 'age' has 1 NaN, 'city' has 1 NaN
    assert profile.loc["age", "missing_count"] == 1
    assert profile.loc["age", "missing_pct"] == pytest.approx(20.0, abs=0.01)
    assert profile.loc["city", "missing_count"] == 1
    assert profile.loc["income", "missing_count"] == 0


def test_profile_numeric_stats(mixed_df):
    cleaner = DataFrameCleaner(verbose=False)
    profile = cleaner.profile(mixed_df)

    # income: all non-null, well-defined stats
    assert profile.loc["income", "is_numeric"] == True
    assert profile.loc["income", "mean"] == pytest.approx(64400.0, rel=1e-3)
    assert profile.loc["income", "min"] == 50000.0
    assert profile.loc["income", "max"] == 80000.0


def test_profile_categorical_no_numeric_stats(mixed_df):
    cleaner = DataFrameCleaner(verbose=False)
    profile = cleaner.profile(mixed_df)

    assert profile.loc["city", "is_categorical"] == True
    # Pandas coerces None to NaN for numeric columns in a mixed-type DataFrame
    assert pd.isna(profile.loc["city", "mean"])
    assert pd.isna(profile.loc["city", "median"])
    assert pd.isna(profile.loc["city", "std"])


def test_profile_datetime_column(mixed_df):
    cleaner = DataFrameCleaner(verbose=False)
    profile = cleaner.profile(mixed_df)

    assert profile.loc["joined", "is_datetime"] == True
    assert profile.loc["joined", "is_numeric"] == False
    assert profile.loc["joined", "is_categorical"] == False


def test_profile_convenience_function(mixed_df):
    profile = profile_dataframe(mixed_df)

    assert isinstance(profile, pd.DataFrame)
    assert len(profile) == len(mixed_df.columns)


def test_profile_raises_on_non_dataframe():
    cleaner = DataFrameCleaner(verbose=False)
    with pytest.raises(TypeError):
        cleaner.profile([1, 2, 3])


def test_column_types_basic(mixed_df):
    cleaner = DataFrameCleaner(verbose=False)
    types = cleaner.column_types(mixed_df)

    assert isinstance(types, dict)
    assert types["age"] == "numeric"
    assert types["income"] == "numeric"
    assert types["city"] == "categorical"
    assert types["joined"] == "datetime"
    assert types["active"] == "boolean"


def test_column_types_raises_on_non_dataframe():
    cleaner = DataFrameCleaner(verbose=False)
    with pytest.raises(TypeError):
        cleaner.column_types("not a dataframe")


def test_profile_empty_dataframe():
    cleaner = DataFrameCleaner(verbose=False)
    empty_df = pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=str)})
    profile = cleaner.profile(empty_df)

    assert isinstance(profile, pd.DataFrame)
    assert len(profile) == 2
    assert profile.loc["a", "missing_count"] == 0
    assert profile.loc["a", "missing_pct"] == 0.0
