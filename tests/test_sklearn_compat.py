"""
Tests for scikit-learn parameter convention and estimator compatibility.
"""

import pandas as pd
import pytest

from data_engine import DataFrameCleaner


def test_get_params():
    cleaner = DataFrameCleaner(
        missing_numeric_strategy="mean",
        outliers_action="drop",
        max_one_hot_cardinality=15,
    )
    params = cleaner.get_params()

    assert isinstance(params, dict)
    assert params["missing_numeric_strategy"] == "mean"
    assert params["outliers_action"] == "drop"
    assert params["max_one_hot_cardinality"] == 15
    assert "steps" in params


def test_set_params():
    cleaner = DataFrameCleaner()
    cleaner.set_params(
        missing_numeric_strategy="constant",
        missing_numeric_fill_value=-999.0,
        outliers_method="zscore",
    )

    assert cleaner.config.missing_numeric_strategy == "constant"
    assert cleaner.config.missing_numeric_fill_value == -999.0
    assert cleaner.config.outliers_method == "zscore"


def test_set_params_invalid():
    cleaner = DataFrameCleaner()
    with pytest.raises(ValueError, match="Invalid parameter"):
        cleaner.set_params(non_existent_param=123)
