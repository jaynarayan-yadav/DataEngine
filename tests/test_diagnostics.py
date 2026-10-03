"""
Tests for pre-cleaning diagnostic analysis and reporting.
"""

import pandas as pd
import numpy as np
import pytest

from data_engine import DataFrameCleaner, DatasetDiagnostics, PreCleaningReport, diagnose_dataframe


@pytest.fixture
def dirty_sample_df():
    return pd.DataFrame({
        "customer_id": [101, 102, 103, 104, 104, 105, 106, 107, 108, 109, 110, 111],
        "signup_date": [
            "2023-01-15", "2023-02-14", "2023-03-22", "2023-04-05", "2023-04-05",
            "2023-05-18", "2023-06-30", "2023-07-04", "2023-08-19", "2023-09-02",
            "2023-10-10", "2023-11-25"
        ],
        "age": [24.0, 31.0, np.nan, 45.0, 45.0, 29.0, 52.0, 36.0, np.nan, 28.0, 145.0, 33.0],
        "annual_income": [45000, 62000, 58000, 75000, 75000, 51000, 89000, 110000, 38000, 67000, 4500000, 53000],
        "loyalty_tier": ["Silver", "Gold", "Bronze", "Gold", "Gold", np.nan, "Silver", "Platinum", "Bronze", np.nan, "Gold", "Silver"],
        "city": ["NYC", "Chicago", "Boston", "Austin", "Austin", "Chicago", "NYC", "Seattle", "Denver", "Boston", "Austin", "NYC"],
        "churned": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    })


def test_diagnostics_detection(dirty_sample_df):
    cleaner = DataFrameCleaner(target_column="churned")
    report = cleaner.diagnose(dirty_sample_df, display=False)

    assert isinstance(report, PreCleaningReport)
    assert report.initial_shape == (12, 7)
    assert report.duplicate_rows == 1
    assert report.total_missing_cells == 4  # 2 in age, 2 in loyalty_tier
    assert report.total_outliers_detected >= 2  # age (145.0) and annual_income (4500000)
    assert report.has_issues is True
    assert report.total_issue_count >= 7

    # Datetime detection
    dt_names = [c.name for c in report.datetime_candidates]
    assert "signup_date" in dt_names

    # Categorical detection
    cat_names = [c.name for c in report.categorical_candidates]
    assert "loyalty_tier" in cat_names
    assert "city" in cat_names
    assert "churned" not in cat_names  # Excluded target


def test_diagnostics_summary_output(dirty_sample_df):
    report = diagnose_dataframe(dirty_sample_df, display=False)
    text = report.summary()

    assert "DATA ENGINE PRE-CLEANING DIAGNOSIS" in text
    assert "12 rows x 7 columns" in text
    assert "Duplicate Rows:" in text
    assert "MISSING VALUES BREAKDOWN:" in text
    assert "OUTLIERS BREAKDOWN:" in text
    assert "PROPOSED CLEANING BLUEPRINT:" in text

    # Dataframe table conversion
    plan_df = report.to_dataframe()
    assert isinstance(plan_df, pd.DataFrame)
    assert not plan_df.empty
    assert "Step" in plan_df.columns
    assert "Planned Action" in plan_df.columns

    # HTML output for notebooks
    html = report._repr_html_()
    assert "Pre-Cleaning Diagnostic Summary" in html
    assert "Duplicate Rows" in html


def test_diagnostics_clean_data():
    clean_df = pd.DataFrame({
        "num": [10, 20, 30, 40],
        "cat": ["A", "B", "C", "D"]
    })
    # Configure without categorical encoding to simulate clean state
    cleaner = DataFrameCleaner(steps=["duplicates", "missing", "outliers"])
    report = cleaner.diagnose(clean_df, display=False)

    assert report.duplicate_rows == 0
    assert report.total_missing_cells == 0
    assert report.total_outliers_detected == 0
    assert report.total_issue_count == 0


def test_diagnostics_warnings_on_empty_and_constant_columns():
    df = pd.DataFrame({
        "all_nan": [np.nan, np.nan, np.nan],
        "constant": [42, 42, 42],
        "normal": [1, 2, 3]
    })
    report = diagnose_dataframe(df, display=False)
    assert len(report.quality_warnings) >= 2
    warn_text = " ".join(report.quality_warnings)
    assert "all_nan" in warn_text
    assert "constant" in warn_text
