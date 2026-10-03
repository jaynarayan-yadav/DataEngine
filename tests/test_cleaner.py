"""
Tests for DataFrameCleaner main pipeline and notebook integration.
"""

import numpy as np
import pandas as pd
import pytest
from data_engine import DataFrameCleaner, clean_dataframe, DataEngine


@pytest.fixture
def dirty_dataframe():
    """Create a realistic messy dataset."""
    return pd.DataFrame({
        "id": [1, 2, 3, 3, 4, 5, 6, 7],
        "signup_date": [
            "2023-01-10", "2023-01-15", "2023-02-01", "2023-02-01",
            "2023-03-05", "2023-04-12", "2023-05-20", "2023-06-30"
        ],
        "age": [25.0, 32.0, np.nan, np.nan, 45.0, 28.0, 52.0, 150.0],  # NaN and outlier 150
        "city": ["New York", "Chicago", "Boston", "Boston", None, "New York", "Chicago", "Boston"],
        "plan": ["Basic", "Pro", "Basic", "Basic", "Pro", "Enterprise", "Pro", "Basic"],
        "target": [0, 1, 0, 0, 1, 1, 0, 1]
    })


def test_automatic_clean_pipeline(dirty_dataframe):
    cleaner = DataFrameCleaner(target_column="target", verbose=False)
    cleaned = cleaner.clean(dirty_dataframe)

    # 1. Duplicates: row with id=3 was duplicated -> 8 rows down to 7
    assert len(cleaned) == 7

    # 2. Datetime: signup_date parsed and features extracted
    assert "signup_date" not in cleaned.columns
    assert "signup_date_year" in cleaned.columns
    assert "signup_date_month" in cleaned.columns

    # 3. Missing values: age NaN filled with median, city None filled with 'missing'
    assert not cleaned.isna().any().any()

    # 4. Outliers: age 150.0 should be clipped
    assert cleaned["age"].max() < 150.0

    # 5. Encoding: city and plan should be encoded, target retained as-is
    assert "target" in cleaned.columns
    assert "plan" not in cleaned.columns
    assert any(col.startswith("plan_") for col in cleaned.columns)


def test_convenience_function(dirty_dataframe):
    cleaned = clean_dataframe(dirty_dataframe, verbose=False)
    assert isinstance(cleaned, pd.DataFrame)
    assert not cleaned.isna().any().any()


def test_manual_mode_steps_selection(dirty_dataframe):
    # Only run duplicates and missing value handling, skip datetime and encoding
    cleaner = DataFrameCleaner(
        steps=["duplicates", "missing"],
        missing_numeric_strategy="mean",
        verbose=False
    )
    cleaned = cleaner.clean(dirty_dataframe)

    # Duplicates handled
    assert len(cleaned) == 7
    # Missing values imputed
    assert not cleaned["age"].isna().any()
    # Datetime NOT converted
    assert "signup_date" in cleaned.columns
    # Categoricals NOT encoded
    assert "city" in cleaned.columns
    assert "plan" in cleaned.columns


def test_granular_methods_chaining(dirty_dataframe):
    cleaner = DataFrameCleaner(verbose=False)
    
    # Use individual step methods directly
    step1 = cleaner.remove_duplicates(dirty_dataframe)
    step2 = cleaner.impute_missing(step1, numeric_strategy="median")
    step3 = cleaner.treat_outliers(step2, method="iqr", action="clip")

    assert len(step3) == 7
    assert not step3["age"].isna().any()
    assert step3["age"].max() < 150.0


def test_report_and_summary(dirty_dataframe):
    cleaner = DataFrameCleaner(verbose=False)
    cleaner.clean(dirty_dataframe)

    report = cleaner.report
    assert report.duplicates_removed == 1
    assert report.initial_shape == (8, 6)
    assert report.final_shape[0] == 7

    # Test report to_dataframe
    df_actions = report.to_dataframe()
    assert isinstance(df_actions, pd.DataFrame)
    assert not df_actions.empty
    assert "Step" in df_actions.columns

    # Test text summary
    summary_text = cleaner.summary()
    assert "DATA ENGINE CLEANING REPORT" in summary_text
    assert "Duplicate Rows" in summary_text

    # Test HTML representation for Jupyter notebooks
    html_repr = cleaner._repr_html_()
    assert "CleanFrame" in html_repr
    assert "Cleaning Report" in html_repr


def test_fit_and_transform_split(dirty_dataframe):
    train_df = dirty_dataframe.iloc[:5].copy()
    test_df = dirty_dataframe.iloc[5:].copy()

    cleaner = DataFrameCleaner(target_column="target", verbose=False)
    cleaner.fit(train_df)

    cleaned_train = cleaner.transform(train_df)
    cleaned_test = cleaner.transform(test_df)

    assert isinstance(cleaned_train, pd.DataFrame)
    assert isinstance(cleaned_test, pd.DataFrame)
    # Check that columns match between train and test
    assert list(cleaned_train.columns) == list(cleaned_test.columns)
