"""
Tests for outlier detection and handling.
"""

import numpy as np
import pandas as pd
import pytest
from data_engine.steps.outliers import OutlierHandler


def test_iqr_clipping():
    # Regular values around 10-20, with extreme 1000
    df = pd.DataFrame({
        "val": [10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0, 1000.0]
    })
    handler = OutlierHandler(method="iqr", action="clip", iqr_factor=1.5)
    res = handler.fit_transform(df)

    # The extreme value should be capped
    assert res["val"].max() < 1000.0
    assert len(res) == len(df)


def test_iqr_dropping():
    df = pd.DataFrame({
        "val": [10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0, 1000.0]
    })
    handler = OutlierHandler(method="iqr", action="drop", iqr_factor=1.5)
    res = handler.fit_transform(df)

    assert len(res) == 7
    assert 1000.0 not in res["val"].values


def test_iqr_nan():
    df = pd.DataFrame({
        "val": [10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0, 1000.0]
    })
    handler = OutlierHandler(method="iqr", action="nan", iqr_factor=1.5)
    res = handler.fit_transform(df)

    assert len(res) == 8
    assert res["val"].isna().sum() == 1


def test_zscore_clipping():
    np.random.seed(42)
    normal_data = np.random.normal(loc=50, scale=5, size=100).tolist()
    normal_data.append(500.0)  # extreme outlier
    df = pd.DataFrame({"score": normal_data})

    handler = OutlierHandler(method="zscore", action="clip", zscore_threshold=3.0)
    res = handler.fit_transform(df)

    assert res["score"].max() < 500.0
    assert np.isclose(res["score"].max(), 189.1, atol=1.0)
    assert len(res) == 101


def test_exclude_binary_columns():
    df = pd.DataFrame({
        "flag": [0, 1, 0, 1, 1],
        "val": [10.0, 12.0, 14.0, 16.0, 500.0]
    })
    handler = OutlierHandler(method="iqr", action="clip")
    handler.fit(df)

    # 'flag' should not be in bounds because it has <= 2 unique values
    assert "flag" not in handler.bounds_
    assert "val" in handler.bounds_
