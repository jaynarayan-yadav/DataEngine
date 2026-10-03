"""
Datetime parsing and feature extraction step.
Automatically detects datetime columns and extracts informative calendar and cyclical components.
"""

from __future__ import annotations

from typing import List, Optional, Set
import pandas as pd

from .base import BaseCleanerStep
from ..logger import CleanerLogger
from ..report import CleaningReport


class DatetimeExtractor(BaseCleanerStep):
    """
    Parses datetime columns and extracts calendar features suitable for machine learning.
    
    Parameters
    ----------
    columns : list of str, optional
        Specific datetime columns to process. If None, auto-detects datetime dtypes and date strings.
    features : list of str, default ["year", "month", "day", "day_of_week", "is_weekend"]
        Features to extract. Supported values:
        - "year": Year integer
        - "month": Month (1-12)
        - "day": Day of month (1-31)
        - "day_of_week": Day of week integer (0=Monday, 6=Sunday)
        - "day_name": Day of week string name (e.g. "Monday")
        - "is_weekend": Binary indicator (1 if Saturday/Sunday else 0)
        - "hour": Hour of day (0-23)
        - "minute": Minute of hour (0-59)
        - "quarter": Fiscal quarter (1-4)
    drop_original : bool, default True
        Whether to drop the original datetime column after extraction.
    auto_detect_sample_size : int, default 50
        Number of non-null rows to inspect when auto-detecting datetime strings.
    auto_detect_threshold : float, default 0.8
        Fraction of successfully parsed date strings required to treat an object column as datetime.
    """

    SUPPORTED_FEATURES = {
        "year", "month", "day", "day_of_week", "day_name",
        "is_weekend", "hour", "minute", "quarter"
    }

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        drop_original: bool = True,
        auto_detect_sample_size: int = 50,
        auto_detect_threshold: float = 0.8,
    ):
        super().__init__(name="datetime")
        self.columns = columns
        self.features = features or ["year", "month", "day", "day_of_week", "is_weekend"]
        self.drop_original = drop_original
        self.auto_detect_sample_size = auto_detect_sample_size
        self.auto_detect_threshold = auto_detect_threshold

        # Validate requested features
        invalid = set(self.features) - self.SUPPORTED_FEATURES
        if invalid:
            raise ValueError(f"Unsupported datetime features: {invalid}. Supported: {self.SUPPORTED_FEATURES}")

        self.detected_columns_: List[str] = []

    def _is_datetime_series(self, series: pd.Series) -> bool:
        """Check if a series is datetime or convertible string dates."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # Only inspect object or string series
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            return False

        # Drop NaNs for sampling
        valid = series.dropna()
        if len(valid) == 0:
            return False

        # Take a sample to test
        sample = valid.iloc[: self.auto_detect_sample_size]
        
        # Quick heuristics to skip obviously non-date strings (e.g. pure numbers, very long texts)
        first_val = str(sample.iloc[0]).strip()
        if len(first_val) < 6 or len(first_val) > 35:
            return False

        # Check for common date separators (-, /, :, T, or space)
        if not any(sep in first_val for sep in ("-", "/", ":", "T", " ")):
            return False

        try:
            parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
            success_ratio = parsed.notna().sum() / len(sample)
            return success_ratio >= self.auto_detect_threshold
        except Exception:
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                success_ratio = parsed.notna().sum() / len(sample)
                return success_ratio >= self.auto_detect_threshold
            except Exception:
                return False

    def fit(self, df: pd.DataFrame) -> "DatetimeExtractor":
        self.detected_columns_ = []
        if self.columns is not None:
            self.detected_columns_ = [c for c in self.columns if c in df.columns]
        else:
            for col in df.columns:
                if self._is_datetime_series(df[col]):
                    self.detected_columns_.append(col)

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

        if not self.detected_columns_:
            if logger and self.columns is not None:
                logger.log("No datetime columns found to process.")
            return df.copy()

        res = df.copy()

        for col in self.detected_columns_:
            if col not in res.columns:
                continue

            # Convert to datetime series
            try:
                dt_series = pd.to_datetime(res[col], format="mixed", errors="coerce")
            except Exception:
                try:
                    dt_series = pd.to_datetime(res[col], errors="coerce")
                except Exception as e:
                    if logger:
                        logger.warning(f"Could not convert column '{col}' to datetime: {e}")
                    continue

            extracted_names: List[str] = []

            for feat in self.features:
                new_col_name = f"{col}_{feat}"

                if feat == "year":
                    res[new_col_name] = dt_series.dt.year
                elif feat == "month":
                    res[new_col_name] = dt_series.dt.month
                elif feat == "day":
                    res[new_col_name] = dt_series.dt.day
                elif feat == "day_of_week":
                    res[new_col_name] = dt_series.dt.dayofweek
                elif feat == "day_name":
                    res[new_col_name] = dt_series.dt.day_name()
                elif feat == "is_weekend":
                    res[new_col_name] = (dt_series.dt.dayofweek >= 5).astype(int)
                elif feat == "hour":
                    res[new_col_name] = dt_series.dt.hour
                elif feat == "minute":
                    res[new_col_name] = dt_series.dt.minute
                elif feat == "quarter":
                    res[new_col_name] = dt_series.dt.quarter

                extracted_names.append(new_col_name)

            if self.drop_original:
                res = res.drop(columns=[col])

            msg = (
                f"Extracted {len(extracted_names)} datetime features from '{col}': "
                f"[{', '.join(self.features)}]"
                + (" (dropped original)" if self.drop_original else " (kept original)")
            )
            if logger:
                logger.log(msg)
            if report:
                report.datetime_extracted[col] = extracted_names
                report.add_action(
                    step=self.name,
                    action="Extract datetime features",
                    details=f"Created {len(extracted_names)} features: {', '.join(extracted_names)}",
                    column=col,
                    count=len(extracted_names)
                )

        return res
