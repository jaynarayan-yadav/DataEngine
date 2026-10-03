"""
Outlier detection and handling step.
Supports IQR (Interquartile Range) and Z-score methods with clipping, dropping, or NaN replacement.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseCleanerStep
from ..logger import CleanerLogger
from ..report import CleaningReport


class OutlierHandler(BaseCleanerStep):
    """
    Detects and handles outliers in numeric columns.
    
    Parameters
    ----------
    method : {"iqr", "zscore", "none"}, default "iqr"
        Method used to define outlier boundaries.
        - "iqr": Tukey's fences using Q1 - k*IQR and Q3 + k*IQR.
        - "zscore": Gaussian distance using mean +/- threshold*std.
    action : {"clip", "drop", "nan", "none"}, default "clip"
        What to do with values exceeding boundaries.
        - "clip": Winsorize / cap values at the calculated boundaries (preserves sample size).
        - "drop": Drop rows where an outlier occurs.
        - "nan": Replace outlier values with NaN.
    iqr_factor : float, default 1.5
        Multiplier for IQR when method is "iqr" (1.5 for mild outliers, 3.0 for extreme).
    zscore_threshold : float, default 3.0
        Number of standard deviations when method is "zscore".
    columns : list of str, optional
        Specific numeric columns to inspect. If None, automatically detects continuous numeric columns.
    exclude_columns : list of str, optional
        Columns to explicitly exclude from outlier inspection (e.g. IDs, targets, binary flags).
    """

    def __init__(
        self,
        method: Literal["iqr", "zscore", "none"] = "iqr",
        action: Literal["clip", "drop", "nan", "none"] = "clip",
        iqr_factor: float = 1.5,
        zscore_threshold: float = 3.0,
        columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
    ):
        super().__init__(name="outliers")
        self.method = method
        self.action = action
        self.iqr_factor = iqr_factor
        self.zscore_threshold = zscore_threshold
        self.columns = columns
        self.exclude_columns = set(exclude_columns or [])

        # Learned boundaries: {col: (lower_bound, upper_bound)}
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def _get_target_columns(self, df: pd.DataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns and c not in self.exclude_columns]
        
        # Auto-detect continuous numeric columns
        target_cols = []
        for col in df.columns:
            if col in self.exclude_columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                # Skip binary/indicator columns with <= 2 unique values
                n_unique = df[col].dropna().nunique()
                if n_unique > 2:
                    target_cols.append(col)
        return target_cols

    def fit(self, df: pd.DataFrame) -> "OutlierHandler":
        self.bounds_ = {}
        if self.method == "none" or self.action == "none" or len(df) == 0:
            self.is_fitted = True
            return self

        target_cols = self._get_target_columns(df)
        for col in target_cols:
            series = df[col].dropna()
            if len(series) < 3:
                continue

            if self.method == "iqr":
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - self.iqr_factor * iqr
                    upper = q3 + self.iqr_factor * iqr
                    self.bounds_[col] = (lower, upper)
            elif self.method == "zscore":
                mean = float(series.mean())
                std = float(series.std())
                if std > 0:
                    lower = mean - self.zscore_threshold * std
                    upper = mean + self.zscore_threshold * std
                    self.bounds_[col] = (lower, upper)

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

        if not self.bounds_ or self.action == "none":
            if logger and self.method != "none" and self.action != "none":
                logger.log("No outlier boundaries identified or applied.")
            return df.copy()

        res = df.copy()

        if self.action == "drop":
            outlier_mask = pd.Series(False, index=res.index)
            for col, (lower, upper) in self.bounds_.items():
                if col in res.columns:
                    col_mask = (res[col] < lower) | (res[col] > upper)
                    col_outliers = int(col_mask.sum())
                    if col_outliers > 0:
                        outlier_mask = outlier_mask | col_mask
                        if report:
                            report.outlier_actions[col] = {
                                "count": col_outliers,
                                "method": self.method,
                                "action": "drop",
                                "lower": lower,
                                "upper": upper
                            }

            dropped_count = int(outlier_mask.sum())
            if dropped_count > 0:
                res = res[~outlier_mask].reset_index(drop=True)
                msg = f"Dropped {dropped_count} row(s) containing outliers across {len(self.bounds_)} columns."
                if logger:
                    logger.log(msg)
                if report:
                    report.add_action(
                        step=self.name,
                        action=f"Drop outliers ({self.method})",
                        details=msg,
                        count=dropped_count
                    )
            return res

        # For "clip" or "nan"
        for col, (lower, upper) in self.bounds_.items():
            if col not in res.columns:
                continue

            below_mask = res[col] < lower
            above_mask = res[col] > upper
            outlier_count = int((below_mask | above_mask).sum())

            if outlier_count > 0:
                if self.action == "clip":
                    res[col] = res[col].clip(lower=lower, upper=upper)
                    action_name = "Clip outliers"
                    detail = f"Capped {outlier_count} values to [{lower:.2f}, {upper:.2f}]"
                elif self.action == "nan":
                    res.loc[below_mask | above_mask, col] = np.nan
                    action_name = "Set outliers to NaN"
                    detail = f"Replaced {outlier_count} values with NaN outside [{lower:.2f}, {upper:.2f}]"
                else:
                    continue

                msg = f"Treated {outlier_count} outlier(s) in '{col}' via {self.method} ({self.action}): [{lower:.2f}, {upper:.2f}]"
                if logger:
                    logger.log(msg)
                if report:
                    report.outlier_actions[col] = {
                        "count": outlier_count,
                        "method": self.method,
                        "action": self.action,
                        "lower": lower,
                        "upper": upper
                    }
                    report.add_action(
                        step=self.name,
                        action=action_name,
                        details=detail,
                        column=col,
                        count=outlier_count
                    )

        return res
