"""
data_engine: Automated and manual data cleaning and preprocessing for pandas DataFrames.

Key public API:
  DataEngine / DataFrameCleaner – Main cleaner class (auto, manual, fit/transform).
  clean_dataframe()   – One-liner functional wrapper for quick cleaning.
  clean_interactive() – Prompt-based interactive cleaning with confirmation.
  diagnose_dataframe()– Inspect DataFrame health without modifying data.
  load_cleaner()      – Restore a saved, fitted cleaner pipeline from disk.
"""

from __future__ import annotations

import pandas as pd
from typing import Any, Callable, Optional, Union

from .cleaner import DataFrameCleaner, DataEngine
from .config import CleaningConfig
from .diagnostics import DatasetDiagnostics, PreCleaningReport, ColumnDiagnostic
from .report import CleaningReport, CleaningAction
from .logger import get_logger

__version__ = "0.1.0"


def clean_dataframe(
    df: pd.DataFrame,
    interactive: bool = False,
    confirm: Optional[bool] = None,
    prompt_fn: Optional[Callable[[str], str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Convenience function to quickly clean a pandas DataFrame using default or custom settings.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to clean.
    interactive : bool, default False
        Whether to display pre-cleaning diagnostic summary and ask for confirmation before cleaning.
    confirm : bool, optional
        Pre-set confirmation flag. If True, proceeds without prompting. If False, aborts.
    prompt_fn : callable, optional
        Custom input function for user prompting (defaults to built-in input()).
    **kwargs : Any
        Options passed to DataFrameCleaner.
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
        
    Examples
    --------
    >>> import pandas as pd
    >>> import cleanframe as cf
    >>> df = pd.DataFrame({"a": [1, 2, None, 2], "b": ["x", "y", "x", "y"]})
    >>> clean_df = cf.clean_dataframe(df)
    """
    cleaner = DataFrameCleaner(interactive=interactive, prompt_fn=prompt_fn, **kwargs)
    return cleaner.clean(df, interactive=interactive, confirm=confirm, prompt_fn=prompt_fn)


def clean_interactive(
    df: pd.DataFrame,
    prompt_fn: Optional[Callable[[str], str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Interactively clean a DataFrame: displays pre-cleaning diagnosis and prompts for confirmation.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to clean.
    prompt_fn : callable, optional
        Custom input prompt function.
    **kwargs : Any
        Options passed to DataFrameCleaner.
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame if approved, or original DataFrame if aborted.
    """
    cleaner = DataFrameCleaner(interactive=True, prompt_fn=prompt_fn, **kwargs)
    return cleaner.clean_interactive(df, prompt_fn=prompt_fn)


def diagnose_dataframe(
    df: pd.DataFrame,
    display: bool = True,
    **kwargs: Any,
) -> PreCleaningReport:
    """
    Analyze and diagnose data cleanliness issues in a DataFrame before cleaning.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    display : bool, default True
        Whether to print the formatted summary report to stdout.
    **kwargs : Any
        Configuration options for diagnostic inspection.
        
    Returns
    -------
    PreCleaningReport
        Diagnostic report containing dataset metrics and proposed cleaning actions.
    """
    cleaner = DataFrameCleaner(**kwargs)
    return cleaner.diagnose(df, display=display)


def profile_dataframe(
    df: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generate a rich per-column profile table for a DataFrame without modifying it.

    Returns a DataFrame with one row per column and statistics including
    dtype, missing count/%, unique count/%, mean, median, std, min, max, and mode.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to profile.
    **kwargs : Any
        Options passed to DataFrameCleaner.

    Returns
    -------
    pd.DataFrame
        Column-level profile summary.

    Examples
    --------
    >>> import pandas as pd
    >>> import cleanframe as cf
    >>> df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "x"]})
    >>> cf.profile_dataframe(df)
    """
    cleaner = DataFrameCleaner(**kwargs)
    return cleaner.profile(df)


def load_cleaner(filepath: Union[str, Any]) -> DataFrameCleaner:
    """
    Load a serialized DataFrameCleaner pipeline from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to the saved cleaner file.

    Returns
    -------
    DataFrameCleaner
        Fitted cleaner instance ready for transform().
    """
    return DataFrameCleaner.load(filepath)


__all__ = [
    "DataEngine",
    "DataFrameCleaner",
    "CleaningConfig",
    "CleaningReport",
    "CleaningAction",
    "DatasetDiagnostics",
    "PreCleaningReport",
    "ColumnDiagnostic",
    "clean_dataframe",
    "clean_interactive",
    "diagnose_dataframe",
    "profile_dataframe",
    "load_cleaner",
    "get_logger",
    "__version__",
]
