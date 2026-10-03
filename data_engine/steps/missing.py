"""
Missing value imputation and handling step.
Handles both numeric and categorical missing values with robust statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
import numpy as np
import pandas as pd

from .base import BaseCleanerStep
from ..logger import CleanerLogger
from ..report import CleaningReport


class MissingValueHandler(BaseCleanerStep):
    """
    Handles missing values in both numeric and categorical columns.
    
    Parameters
    ----------
    numeric_strategy : {"median", "mean", "mode", "constant", "drop_rows", "none"}, default "median"
        Strategy to impute missing numeric values.
    numeric_fill_value : float, default 0.0
        Value used when numeric_strategy is "constant".
    categorical_strategy : {"constant", "mode", "drop_rows", "none"}, default "constant"
        Strategy to impute missing categorical/string values.
    categorical_fill_value : str, default "missing"
        Value used when categorical_strategy is "constant".
    column_threshold : float, optional
        Drop any column where the fraction of missing values exceeds this threshold (e.g., 0.8 for 80%).
    custom_strategies : dict, optional
        Column-specific strategies or direct fill values. E.g.: {"age": "mean", "dept": "unknown"}
    """

    def __init__(
        self,
        numeric_strategy: Literal["median", "mean", "mode", "constant", "drop_rows", "none"] = "median",
        numeric_fill_value: float = 0.0,
        categorical_strategy: Literal["constant", "mode", "drop_rows", "none"] = "constant",
        categorical_fill_value: str = "missing",
        column_threshold: Optional[float] = None,
        custom_strategies: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="missing")
        self.numeric_strategy = numeric_strategy
        self.numeric_fill_value = numeric_fill_value
        self.categorical_strategy = categorical_strategy
        self.categorical_fill_value = categorical_fill_value
        self.column_threshold = column_threshold
        self.custom_strategies = custom_strategies or {}

        # Learned statistics during fit
        self.dropped_columns_: List[str] = []
        self.fill_values_: Dict[str, Any] = {}
        self.drop_row_columns_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "MissingValueHandler":
        self.dropped_columns_ = []
        self.fill_values_ = {}
        self.drop_row_columns_ = []

        total_rows = len(df)
        if total_rows == 0:
            self.is_fitted = True
            return self

        # 1. Identify columns exceeding missing threshold
        if self.column_threshold is not None:
            for col in df.columns:
                missing_ratio = df[col].isna().sum() / total_rows
                if missing_ratio > self.column_threshold:
                    self.dropped_columns_.append(col)

        # 2. Compute fill values for remaining columns
        for col in df.columns:
            if col in self.dropped_columns_:
                continue

            # Check if there's a custom strategy or value
            if col in self.custom_strategies:
                cust = self.custom_strategies[col]
                if cust == "drop_rows":
                    self.drop_row_columns_.append(col)
                elif cust == "median" and pd.api.types.is_numeric_dtype(df[col]):
                    val = df[col].median()
                    self.fill_values_[col] = 0.0 if pd.isna(val) else val
                elif cust == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                    val = df[col].mean()
                    self.fill_values_[col] = 0.0 if pd.isna(val) else val
                elif cust == "mode":
                    mode_vals = df[col].mode(dropna=True)
                    self.fill_values_[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else (
                        0.0 if pd.api.types.is_numeric_dtype(df[col]) else "missing"
                    )
                else:
                    # Direct constant value provided
                    self.fill_values_[col] = cust
                continue

            # Default logic based on dtype
            is_num = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])

            if is_num:
                if self.numeric_strategy == "none":
                    continue
                elif self.numeric_strategy == "drop_rows":
                    self.drop_row_columns_.append(col)
                elif self.numeric_strategy == "median":
                    val = df[col].median()
                    self.fill_values_[col] = 0.0 if pd.isna(val) else val
                elif self.numeric_strategy == "mean":
                    val = df[col].mean()
                    self.fill_values_[col] = 0.0 if pd.isna(val) else val
                elif self.numeric_strategy == "mode":
                    mode_vals = df[col].mode(dropna=True)
                    self.fill_values_[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else 0.0
                elif self.numeric_strategy == "constant":
                    self.fill_values_[col] = self.numeric_fill_value
            else:
                if self.categorical_strategy == "none":
                    continue
                elif self.categorical_strategy == "drop_rows":
                    self.drop_row_columns_.append(col)
                elif self.categorical_strategy == "mode":
                    mode_vals = df[col].mode(dropna=True)
                    self.fill_values_[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else self.categorical_fill_value
                elif self.categorical_strategy == "constant":
                    self.fill_values_[col] = self.categorical_fill_value

        self.is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        logger: Optional[CleanerLogger] = None,
        report: Optional[CleaningReport] = None,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            self.fit(df)

        res = df.copy()

        # 1. Drop columns exceeding missing threshold
        cols_to_drop = [c for c in self.dropped_columns_ if c in res.columns]
        if cols_to_drop:
            res = res.drop(columns=cols_to_drop)
            msg = f"Dropped columns exceeding missing threshold ({self.column_threshold*100:.0f}%): {cols_to_drop}"
            if logger:
                logger.log(msg)
            if report:
                for c in cols_to_drop:
                    report.add_action(
                        step=self.name,
                        action="Drop column",
                        details=f"Missing ratio exceeded threshold ({self.column_threshold})",
                        column=c
                    )

        # 2. Impute missing values with learned fill values
        for col, fill_val in self.fill_values_.items():
            if col not in res.columns:
                continue

            num_missing = int(res[col].isna().sum())
            if num_missing > 0:
                strat = (
                    self.custom_strategies.get(col)
                    if col in self.custom_strategies
                    else (
                        self.numeric_strategy
                        if pd.api.types.is_numeric_dtype(res[col])
                        else self.categorical_strategy
                    )
                )
                
                # Convert categorical/object to proper string if filling with string
                if not pd.api.types.is_numeric_dtype(res[col]):
                    res[col] = res[col].fillna(fill_val).astype(str)
                else:
                    res[col] = res[col].fillna(fill_val)

                msg = f"Imputed {num_missing} missing value(s) in '{col}' using {strat} ({fill_val})"
                if logger:
                    logger.log(msg)
                if report:
                    report.missing_imputations[col] = {
                        "count": num_missing,
                        "strategy": str(strat),
                        "fill_value": fill_val
                    }
                    report.add_action(
                        step=self.name,
                        action=f"Impute {strat}",
                        details=f"Filled with {fill_val}",
                        column=col,
                        count=num_missing
                    )

        # 3. Drop rows for columns set to drop_rows
        active_drop_cols = [c for c in self.drop_row_columns_ if c in res.columns]
        if active_drop_cols:
            before_len = len(res)
            res = res.dropna(subset=active_drop_cols).reset_index(drop=True)
            dropped_rows = before_len - len(res)
            if dropped_rows > 0:
                msg = f"Dropped {dropped_rows} row(s) with missing values in {active_drop_cols}"
                if logger:
                    logger.log(msg)
                if report:
                    report.add_action(
                        step=self.name,
                        action="Drop rows with missing",
                        details=msg,
                        count=dropped_rows
                    )

        return res
