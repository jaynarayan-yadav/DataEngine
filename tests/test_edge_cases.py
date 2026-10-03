"""
Edge case tests for cleanframe.
"""

import numpy as np
import pandas as pd
import pytest
from data_engine import DataFrameCleaner, clean_dataframe


def test_empty_dataframe():
    df = pd.DataFrame()
    cleaner = DataFrameCleaner(verbose=False)
    cleaned = cleaner.clean(df)
    assert cleaned.empty


def test_single_row_dataframe():
    df = pd.DataFrame({
        "a": [10.0],
        "cat": ["A"],
        "dt": ["2023-01-01"]
    })
    cleaner = DataFrameCleaner(verbose=False)
    cleaned = cleaner.clean(df)
    assert len(cleaned) == 1
    assert "dt_year" in cleaned.columns


def test_all_nans_column():
    df = pd.DataFrame({
        "num": [np.nan, np.nan, np.nan],
        "cat": [None, None, None]
    })
    cleaner = DataFrameCleaner(verbose=False)
    cleaned = cleaner.clean(df)
    assert not cleaned.isna().any().any()
    # Default numeric constant 0.0, default categorical "missing"
    assert (cleaned["num"] == 0.0).all()


def test_already_clean_dataframe():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "num2": [10.0, 20.0, 30.0]
    })
    cleaner = DataFrameCleaner(verbose=False)
    cleaned = cleaner.clean(df)
    pd.testing.assert_frame_equal(cleaned, df)


def test_categories_with_special_characters():
    df = pd.DataFrame({
        "category": ["A / B", "C & D", "E - F"]
    })
    cleaner = DataFrameCleaner(encoding_strategy="onehot", verbose=False)
    cleaned = cleaner.clean(df)
    # Checks that column names don't cause issues
    assert any(col.startswith("category_") for col in cleaned.columns)
