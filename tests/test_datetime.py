"""
Tests for datetime parsing and feature extraction.
"""

import pandas as pd
import pytest
from data_engine.steps.datetime import DatetimeExtractor


def test_datetime_auto_detection_and_extraction():
    df = pd.DataFrame({
        "timestamp": [
            "2023-01-15 09:30:00",
            "2023-02-20 14:15:00",
            "2023-03-25 18:45:00",
            "2023-04-30 23:00:00",
        ],
        "other": [1, 2, 3, 4]
    })
    extractor = DatetimeExtractor(
        features=["year", "month", "day", "day_of_week", "is_weekend", "hour"],
        drop_original=True
    )
    res = extractor.fit_transform(df)

    assert "timestamp" not in res.columns
    assert "timestamp_year" in res.columns
    assert "timestamp_month" in res.columns
    assert "timestamp_day" in res.columns
    assert "timestamp_day_of_week" in res.columns
    assert "timestamp_is_weekend" in res.columns
    assert "timestamp_hour" in res.columns

    assert list(res["timestamp_year"]) == [2023, 2023, 2023, 2023]
    assert list(res["timestamp_month"]) == [1, 2, 3, 4]
    assert list(res["timestamp_day"]) == [15, 20, 25, 30]


def test_existing_datetime_dtype():
    dates = pd.to_datetime(["2024-06-01", "2024-06-02"])  # Saturday, Sunday
    df = pd.DataFrame({"dt": dates})
    extractor = DatetimeExtractor(features=["is_weekend"], drop_original=False)
    res = extractor.fit_transform(df)

    assert "dt" in res.columns
    assert "dt_is_weekend" in res.columns
    assert list(res["dt_is_weekend"]) == [1, 1]


def test_explicit_columns():
    df = pd.DataFrame({
        "my_date": ["2022/01/01", "2022/01/02"],
        "text": ["hello", "world"]
    })
    extractor = DatetimeExtractor(columns=["my_date"], features=["year", "quarter"])
    res = extractor.fit_transform(df)

    assert "my_date_year" in res.columns
    assert "my_date_quarter" in res.columns
    assert list(res["my_date_quarter"]) == [1, 1]
