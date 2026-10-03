"""
Tests for missing value imputation and handling.
"""

import numpy as np
import pandas as pd
import pytest
from data_engine.steps.missing import MissingValueHandler


def test_numeric_median_imputation():
    df = pd.DataFrame({
        "a": [10.0, 20.0, 30.0, np.nan, 100.0],
    })
    # Median of [10, 20, 30, 100] is 25.0
    handler = MissingValueHandler(numeric_strategy="median")
    res = handler.fit_transform(df)

    assert not res["a"].isna().any()
    assert res.loc[3, "a"] == 25.0


def test_numeric_mean_imputation():
    df = pd.DataFrame({
        "a": [10.0, 20.0, np.nan, 30.0],
    })
    # Mean of [10, 20, 30] is 20.0
    handler = MissingValueHandler(numeric_strategy="mean")
    res = handler.fit_transform(df)

    assert not res["a"].isna().any()
    assert res.loc[2, "a"] == 20.0


def test_categorical_constant_imputation():
    df = pd.DataFrame({
        "cat": ["A", "B", None, "A"],
    })
    handler = MissingValueHandler(categorical_strategy="constant", categorical_fill_value="unknown")
    res = handler.fit_transform(df)

    assert not res["cat"].isna().any()
    assert res.loc[2, "cat"] == "unknown"


def test_categorical_mode_imputation():
    df = pd.DataFrame({
        "cat": ["A", "B", "A", None, "A"],
    })
    handler = MissingValueHandler(categorical_strategy="mode")
    res = handler.fit_transform(df)

    assert not res["cat"].isna().any()
    assert res.loc[3, "cat"] == "A"


def test_column_drop_threshold():
    df = pd.DataFrame({
        "mostly_empty": [1.0, np.nan, np.nan, np.nan],  # 75% missing
        "mostly_full": [1.0, 2.0, 3.0, np.nan],         # 25% missing
    })
    handler = MissingValueHandler(column_threshold=0.5)
    res = handler.fit_transform(df)

    assert "mostly_empty" not in res.columns
    assert "mostly_full" in res.columns


def test_custom_per_column_strategies():
    df = pd.DataFrame({
        "age": [20.0, 40.0, np.nan],
        "city": ["NY", None, "SF"],
        "flag": [1.0, np.nan, 0.0]
    })
    handler = MissingValueHandler(
        custom_strategies={
            "age": "mean",
            "city": "UnknownCity",
            "flag": 0.0
        }
    )
    res = handler.fit_transform(df)

    assert res.loc[2, "age"] == 30.0
    assert res.loc[1, "city"] == "UnknownCity"
    assert res.loc[1, "flag"] == 0.0


def test_drop_rows_strategy():
    df = pd.DataFrame({
        "req": [1.0, np.nan, 3.0],
        "opt": [10.0, 20.0, 30.0]
    })
    handler = MissingValueHandler(numeric_strategy="drop_rows")
    res = handler.fit_transform(df)

    assert len(res) == 2
    assert list(res["req"]) == [1.0, 3.0]
