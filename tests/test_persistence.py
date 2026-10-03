"""
Tests for DataFrameCleaner serialization (save and load) and model persistence.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from data_engine import DataFrameCleaner, load_cleaner


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customer_id": [101, 102, 103, 104, 104],
        "signup_date": ["2023-01-15", "2023-02-14", "2023-03-22", "2023-04-05", "2023-04-05"],
        "age": [24.0, 31.0, np.nan, 45.0, 45.0],
        "city": ["New York", "Chicago", "Boston", "Austin", "Austin"],
        "target": [0, 1, 0, 1, 1]
    })


def test_save_and_load_cleaner(sample_df, tmp_path):
    model_path = tmp_path / "fitted_cleaner.pkl"

    # Fit and transform
    cleaner = DataFrameCleaner(target_column="target", verbose=False)
    transformed_orig = cleaner.fit_transform(sample_df)

    # Save to disk
    saved_path = cleaner.save(model_path)
    assert saved_path.exists()

    # Load from disk using classmethod
    loaded_cleaner = DataFrameCleaner.load(model_path)
    assert loaded_cleaner._is_fitted

    # Transform new data using loaded cleaner
    new_data = sample_df.copy()
    transformed_loaded = loaded_cleaner.transform(new_data)

    assert list(transformed_loaded.columns) == list(transformed_orig.columns)
    assert transformed_loaded.shape[1] == transformed_orig.shape[1]


def test_load_cleaner_convenience_function(sample_df, tmp_path):
    model_path = tmp_path / "cleaner_conv.pkl"
    cleaner = DataFrameCleaner(verbose=False).fit(sample_df)
    cleaner.save(model_path)

    loaded = load_cleaner(model_path)
    assert isinstance(loaded, DataFrameCleaner)
    assert loaded._is_fitted


def test_load_non_existent_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataFrameCleaner.load(tmp_path / "does_not_exist.pkl")
