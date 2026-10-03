"""
Tests for interactive confirmation and pre-cleaning user approval.
"""

import pandas as pd
import numpy as np
import pytest

from data_engine import DataFrameCleaner, clean_dataframe, clean_interactive


@pytest.fixture
def dirty_df():
    return pd.DataFrame({
        "id": [1, 2, 2, 3],
        "score": [10.0, np.nan, np.nan, 30.0],
        "city": ["NY", "LA", "LA", "SF"]
    })


def test_interactive_confirm_yes(dirty_df):
    """User answers 'y': cleaning proceeds."""
    prompts_called = []

    def mock_input(prompt: str) -> str:
        prompts_called.append(prompt)
        return "y"

    cleaner = DataFrameCleaner(verbose=False)
    cleaned = cleaner.clean(dirty_df, interactive=True, prompt_fn=mock_input)

    assert len(prompts_called) == 1
    assert "Should CleanFrame proceed with cleaning?" in prompts_called[0]
    assert cleaner.aborted is False
    # Verified that duplicate was dropped (4 rows -> 3 rows) and NaN imputed
    assert len(cleaned) == 3
    assert cleaned["score"].isna().sum() == 0


def test_interactive_confirm_no_aborts(dirty_df):
    """User answers 'n': cleaning is aborted and original data is returned unmodified."""
    prompts_called = []

    def mock_input(prompt: str) -> str:
        prompts_called.append(prompt)
        return "n"

    cleaner = DataFrameCleaner(verbose=False)
    result = cleaner.clean(dirty_df, interactive=True, prompt_fn=mock_input)

    assert len(prompts_called) == 1
    assert cleaner.aborted is True
    # Shape and values remain identical to dirty_df
    assert result.shape == dirty_df.shape
    assert result["score"].isna().sum() == dirty_df["score"].isna().sum()


def test_interactive_custom_steps(dirty_df):
    """User answers 'c' and customizes steps to only 'duplicates,missing'."""
    inputs = ["c", "duplicates,missing"]

    def mock_input(prompt: str) -> str:
        return inputs.pop(0)

    cleaner = DataFrameCleaner(verbose=False)
    result = cleaner.clean(dirty_df, interactive=True, prompt_fn=mock_input)

    assert cleaner.aborted is False
    assert len(result) == 3  # Duplicates handled
    assert result["score"].isna().sum() == 0  # Missing handled
    # Encoding was NOT run, so city remains as string column
    assert "city" in result.columns


def test_clean_interactive_wrapper(dirty_df):
    """clean_interactive helper method."""
    cleaner = DataFrameCleaner(verbose=False)
    result = cleaner.clean_interactive(dirty_df, prompt_fn=lambda p: "y")
    assert cleaner.aborted is False
    assert len(result) == 3


def test_explicit_confirm_flag_bypass(dirty_df):
    """confirm=True or confirm=False skips prompt."""
    cleaner = DataFrameCleaner(verbose=False)

    # confirm=True: cleans directly
    res_yes = cleaner.clean(dirty_df, interactive=True, confirm=True)
    assert cleaner.aborted is False
    assert len(res_yes) == 3

    # confirm=False: aborts directly
    res_no = cleaner.clean(dirty_df, interactive=True, confirm=False)
    assert cleaner.aborted is True
    assert len(res_no) == 4


def test_clean_dataframe_interactive_param(dirty_df):
    """Top-level clean_dataframe function with interactive=True."""
    res = clean_dataframe(dirty_df, interactive=True, prompt_fn=lambda p: "y", verbose=False)
    assert len(res) == 3

    res_aborted = clean_dataframe(dirty_df, interactive=True, prompt_fn=lambda p: "n", verbose=False)
    assert len(res_aborted) == 4
