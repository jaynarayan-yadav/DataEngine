"""
Tests for duplicate row handling.
"""

import pandas as pd
import pytest
from data_engine.steps.duplicates import DuplicateHandler


def test_duplicate_removal_all_columns():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": ["Alice", "Bob", "Bob", "Charlie"],
        "score": [90, 80, 80, 70]
    })
    handler = DuplicateHandler(keep="first")
    res = handler.fit_transform(df)

    assert len(res) == 3
    assert list(res["id"]) == [1, 2, 3]
    assert list(res.index) == [0, 1, 2]


def test_duplicate_removal_subset():
    df = pd.DataFrame({
        "user_id": [1, 1, 2, 3],
        "version": [1, 2, 1, 1],
        "val": [10, 20, 30, 40]
    })
    handler = DuplicateHandler(subset=["user_id"], keep="last")
    res = handler.fit_transform(df)

    assert len(res) == 3
    # Keeps user_id=1 with version 2 because keep="last"
    assert res.loc[res["user_id"] == 1, "version"].iloc[0] == 2


def test_no_duplicates():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["A", "B", "C"]
    })
    handler = DuplicateHandler()
    res = handler.fit_transform(df)
    assert len(res) == 3
