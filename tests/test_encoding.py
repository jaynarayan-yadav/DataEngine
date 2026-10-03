"""
Tests for categorical encoding.
"""

import pandas as pd
import pytest
from data_engine.steps.encoding import CategoricalEncoder


def test_onehot_encoding():
    df = pd.DataFrame({
        "color": ["red", "blue", "green", "blue"],
        "num": [1, 2, 3, 4]
    })
    encoder = CategoricalEncoder(strategy="onehot")
    res = encoder.fit_transform(df)

    assert "color" not in res.columns
    assert "color_blue" in res.columns
    assert "color_green" in res.columns
    assert "color_red" in res.columns

    assert list(res["color_blue"]) == [0, 1, 0, 1]
    assert list(res["color_red"]) == [1, 0, 0, 0]


def test_label_encoding():
    df = pd.DataFrame({
        "tier": ["silver", "gold", "bronze", "gold"]
    })
    encoder = CategoricalEncoder(strategy="label")
    res = encoder.fit_transform(df)

    # Categories sorted: bronze (0), gold (1), silver (2)
    assert list(res["tier"]) == [2, 1, 0, 1]


def test_frequency_encoding():
    df = pd.DataFrame({
        "cat": ["A", "A", "A", "B"]
    })
    encoder = CategoricalEncoder(strategy="frequency")
    res = encoder.fit_transform(df)

    # A appears 3/4 = 0.75, B appears 1/4 = 0.25
    assert list(res["cat"]) == [0.75, 0.75, 0.75, 0.25]


def test_auto_strategy_cardinality_split():
    df = pd.DataFrame({
        # Low cardinality (2 unique values) -> OneHot
        "gender": ["M", "F", "M", "F"],
        # High cardinality (12 unique values) -> Label when max_one_hot_cardinality=5
        "zipcode": [f"ZIP_{i}" for i in range(4)]
    })
    encoder = CategoricalEncoder(strategy="auto", max_one_hot_cardinality=3)
    res = encoder.fit_transform(df)

    # gender should be onehot
    assert "gender" not in res.columns
    assert "gender_F" in res.columns
    assert "gender_M" in res.columns

    # zipcode has 4 unique values > 3 -> should be label encoded (still named zipcode, but integers)
    assert "zipcode" in res.columns
    assert pd.api.types.is_integer_dtype(res["zipcode"])


def test_target_column_excluded():
    df = pd.DataFrame({
        "feature": ["cat", "dog", "cat"],
        "target": ["yes", "no", "yes"]
    })
    encoder = CategoricalEncoder(strategy="onehot", target_column="target")
    res = encoder.fit_transform(df)

    assert "feature" not in res.columns
    assert "target" in res.columns
    assert list(res["target"]) == ["yes", "no", "yes"]


def test_train_test_alignment():
    train_df = pd.DataFrame({"color": ["red", "blue"]})
    test_df = pd.DataFrame({"color": ["blue", "yellow"]})  # "yellow" is unseen

    encoder = CategoricalEncoder(strategy="onehot")
    encoder.fit(train_df)
    test_res = encoder.transform(test_df)

    # Columns should match train categories
    assert "color_red" in test_res.columns
    assert "color_blue" in test_res.columns
    assert "color_yellow" not in test_res.columns
    # Row 1 is blue (color_blue=1), Row 2 is yellow (color_red=0, color_blue=0)
    assert list(test_res["color_blue"]) == [1, 0]
    assert list(test_res["color_red"]) == [0, 0]
