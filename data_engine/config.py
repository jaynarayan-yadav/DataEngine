"""
Configuration dataclasses and options for cleanframe.
Provides strongly typed defaults and customizable settings for every cleaning step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union


@dataclass
class CleaningConfig:
    """
    Configuration options for DataFrameCleaner.
    
    Attributes
    ----------
    steps : list of str
        The sequence of steps to run in automatic mode.
        Default is ["duplicates", "datetime", "missing", "outliers", "encoding"].
    drop_duplicates : bool
        Whether to drop duplicate rows. Default is True.
    duplicates_subset : list of str, optional
        Columns to consider when identifying duplicates. If None, uses all columns.
    duplicates_keep : {"first", "last", False}
        Which duplicate to keep. Default is "first".
    missing_numeric_strategy : {"median", "mean", "mode", "constant", "drop_rows", "none"}
        Strategy for imputing numeric missing values. Default is "median".
    missing_numeric_fill_value : float
        Value to use when missing_numeric_strategy is "constant". Default is 0.0.
    missing_categorical_strategy : {"constant", "mode", "drop_rows", "none"}
        Strategy for imputing categorical missing values. Default is "constant".
    missing_categorical_fill_value : str
        Value to use when missing_categorical_strategy is "constant". Default is "missing".
    missing_column_threshold : float, optional
        Drop columns where missing value ratio exceeds this threshold (e.g. 0.7 for 70%). Default is None.
    missing_custom_strategies : dict, optional
        Per-column imputation strategies or fill values. E.g. {"age": "mean", "zipcode": "mode"}.
    outliers_method : {"iqr", "zscore", "none"}
        Outlier detection method. Default is "iqr".
    outliers_action : {"clip", "drop", "nan", "none"}
        What to do with detected outliers:
        - "clip" / "winsorize": cap to boundary values (preserves dataset size).
        - "drop": drop rows with outliers.
        - "nan": replace outliers with NaN.
        Default is "clip".
    outliers_iqr_factor : float
        Multiplier for IQR range. Default is 1.5.
    outliers_zscore_threshold : float
        Z-score threshold for outlier detection. Default is 3.0.
    outliers_columns : list of str, optional
        Specific numeric columns to apply outlier handling to. If None, auto-detects continuous numeric columns.
    datetime_columns : list of str, optional
        Specific datetime columns to process. If None, auto-detects existing datetime dtypes and date strings.
    datetime_features : list of str
        Features to extract from datetime columns.
        Supported options: "year", "month", "day", "day_of_week", "day_name", "is_weekend", "hour", "quarter".
        Default is ["year", "month", "day", "day_of_week", "is_weekend"].
    datetime_drop_original : bool
        Whether to drop original datetime columns after feature extraction. Default is True.
    encoding_strategy : {"auto", "onehot", "label", "frequency", "none"}
        Categorical encoding strategy:
        - "auto": one-hot encode columns with cardinality <= max_one_hot_cardinality, and label encode others.
        - "onehot": one-hot encode all categorical columns.
        - "label": ordinal/integer encode categorical columns.
        - "frequency": replace categories with their frequency proportion.
        - "none": skip encoding.
        Default is "auto".
    encoding_columns : list of str, optional
        Specific columns to encode. If None, auto-detects categorical/object columns.
    max_one_hot_cardinality : int
        Maximum unique values for a column to be one-hot encoded in "auto" mode. Default is 10.
    encoding_drop_first : bool
        Whether to drop the first category in one-hot encoding (avoids collinearity). Default is False.
    target_column : str, optional
        Target variable name to exclude from encoding or aggressive transformations.
    copy : bool
        Whether to work on a copy of the input DataFrame. Default is True.
    verbose : bool
        Whether to print or log cleaning actions as they happen. Default is True.
    """
    steps: List[str] = field(
        default_factory=lambda: ["duplicates", "datetime", "missing", "outliers", "encoding"]
    )
    
    # Duplicate rows configuration
    drop_duplicates: bool = True
    duplicates_subset: Optional[List[str]] = None
    duplicates_keep: Literal["first", "last", False] = "first"
    
    # Missing values configuration
    missing_numeric_strategy: Literal["median", "mean", "mode", "constant", "drop_rows", "none"] = "median"
    missing_numeric_fill_value: float = 0.0
    missing_categorical_strategy: Literal["constant", "mode", "drop_rows", "none"] = "constant"
    missing_categorical_fill_value: str = "missing"
    missing_column_threshold: Optional[float] = None
    missing_custom_strategies: Optional[Dict[str, Any]] = None
    
    # Outliers configuration
    outliers_method: Literal["iqr", "zscore", "none"] = "iqr"
    outliers_action: Literal["clip", "drop", "nan", "none"] = "clip"
    outliers_iqr_factor: float = 1.5
    outliers_zscore_threshold: float = 3.0
    outliers_columns: Optional[List[str]] = None
    
    # Datetime feature extraction configuration
    datetime_columns: Optional[List[str]] = None
    datetime_features: List[str] = field(
        default_factory=lambda: ["year", "month", "day", "day_of_week", "is_weekend"]
    )
    datetime_drop_original: bool = True
    
    # Categorical encoding configuration
    encoding_strategy: Literal["auto", "onehot", "label", "frequency", "none"] = "auto"
    encoding_columns: Optional[List[str]] = None
    max_one_hot_cardinality: int = 10
    encoding_drop_first: bool = False
    
    # General configuration
    target_column: Optional[str] = None
    copy: bool = True
    verbose: bool = True
    interactive: bool = False
    prompt_fn: Optional[Callable[[str], str]] = None
