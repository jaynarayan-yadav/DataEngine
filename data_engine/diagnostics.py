"""
Pre-cleaning diagnostic analysis and issue inspection for cleanframe.
Analyzes DataFrames before any modifications, presents an inspection summary,
and proposes a cleaning blueprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .config import CleaningConfig
from .steps.datetime import DatetimeExtractor


@dataclass
class ColumnDiagnostic:
    """Diagnostic details for a single column."""
    name: str
    dtype: str
    total_count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    is_numeric: bool
    is_categorical: bool
    is_datetime: bool
    is_constant: bool
    outlier_count: int = 0
    outlier_pct: float = 0.0
    outlier_bounds: Optional[Tuple[float, float]] = None
    planned_missing_action: Optional[str] = None
    planned_outlier_action: Optional[str] = None
    planned_datetime_action: Optional[str] = None
    planned_encoding_action: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class PreCleaningReport:
    """
    Detailed diagnostic report produced BEFORE any cleaning transformations are executed.
    """
    initial_shape: Tuple[int, int]
    memory_usage_str: str
    duplicate_rows: int
    duplicate_pct: float
    total_missing_cells: int
    total_missing_pct: float
    total_outliers_detected: int
    columns_with_missing: List[ColumnDiagnostic] = field(default_factory=list)
    columns_with_outliers: List[ColumnDiagnostic] = field(default_factory=list)
    datetime_candidates: List[ColumnDiagnostic] = field(default_factory=list)
    categorical_candidates: List[ColumnDiagnostic] = field(default_factory=list)
    quality_warnings: List[str] = field(default_factory=list)
    action_plan: List[Dict[str, Any]] = field(default_factory=list)
    config: Optional[CleaningConfig] = None
    column_diagnostics: Dict[str, ColumnDiagnostic] = field(default_factory=dict)

    @property
    def has_issues(self) -> bool:
        """Check if any cleaning issues or transformation candidates were detected."""
        return (
            self.duplicate_rows > 0
            or self.total_missing_cells > 0
            or self.total_outliers_detected > 0
            or len(self.datetime_candidates) > 0
            or len(self.categorical_candidates) > 0
            or len(self.quality_warnings) > 0
        )

    @property
    def total_issue_count(self) -> int:
        """Count total individual data cleanliness issues found."""
        return self.duplicate_rows + self.total_missing_cells + self.total_outliers_detected

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert planned actions to a structured pandas DataFrame.
        """
        if not self.action_plan:
            return pd.DataFrame(columns=["Step", "Target", "Issue Found", "Planned Action", "Details"])
        
        return pd.DataFrame(self.action_plan)

    def summary(self) -> str:
        """
        Generate a formatted, human-readable terminal summary card.
        """
        lines = [
            "===========================================================",
            "             CLEANFRAME PRE-CLEANING DIAGNOSIS             ",
            "===========================================================",
            f"Dataset Dimensions:    {self.initial_shape[0]} rows x {self.initial_shape[1]} columns",
            f"Estimated Memory:      {self.memory_usage_str}",
            f"Duplicate Rows:        {self.duplicate_rows} ({self.duplicate_pct:.1f}% of dataset)",
            f"Total Missing Cells:   {self.total_missing_cells} ({self.total_missing_pct:.1f}% of all cells)",
            f"Total Outliers:        {self.total_outliers_detected} values detected",
            "-----------------------------------------------------------",
        ]

        # 1. Quality warnings
        if self.quality_warnings:
            lines.append("DATA QUALITY WARNINGS:")
            for w in self.quality_warnings:
                lines.append(f"  [!] {w}")
            lines.append("-----------------------------------------------------------")

        # 2. Missing value breakdown
        lines.append("MISSING VALUES BREAKDOWN:")
        if self.columns_with_missing:
            for c in self.columns_with_missing:
                action_text = c.planned_missing_action or "impute"
                lines.append(
                    f"  - {c.name} ({c.dtype}): {c.missing_count} missing ({c.missing_pct:.1f}%) "
                    f"-> Planned: {action_text}"
                )
        else:
            lines.append("  No missing values detected. (100% complete)")

        # 3. Outlier breakdown
        lines.append("\nOUTLIERS BREAKDOWN:")
        if self.columns_with_outliers:
            for c in self.columns_with_outliers:
                bounds_str = f"bounds: [{c.outlier_bounds[0]:.2f}, {c.outlier_bounds[1]:.2f}]" if c.outlier_bounds else ""
                lines.append(
                    f"  - {c.name}: {c.outlier_count} outliers ({c.outlier_pct:.1f}%) {bounds_str} "
                    f"-> Planned: {c.planned_outlier_action}"
                )
        else:
            lines.append("  No significant numeric outliers detected.")

        # 4. Datetime extractions
        lines.append("\nDATETIME EXTRACTION CANDIDATES:")
        if self.datetime_candidates:
            for c in self.datetime_candidates:
                lines.append(
                    f"  - {c.name} ({c.dtype}): recognized date/time format "
                    f"-> Planned: {c.planned_datetime_action}"
                )
        else:
            lines.append("  No datetime columns detected.")

        # 5. Categorical encodings
        lines.append("\nCATEGORICAL ENCODING CANDIDATES:")
        if self.categorical_candidates:
            for c in self.categorical_candidates:
                lines.append(
                    f"  - {c.name} ({c.dtype}): cardinality {c.unique_count} "
                    f"-> Planned: {c.planned_encoding_action}"
                )
        else:
            lines.append("  No categorical columns detected.")

        # 6. Proposed Action Blueprint
        lines.append("-----------------------------------------------------------")
        lines.append("PROPOSED CLEANING BLUEPRINT:")
        if self.action_plan:
            for idx, plan in enumerate(self.action_plan, 1):
                lines.append(f"  {idx}. [{plan['Step'].upper()}] {plan['Target']}: {plan['Planned Action']}")
        else:
            lines.append("  Dataset appears clean! No transformations required.")
        lines.append("===========================================================")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        df_plan = self.to_dataframe()
        plan_html = df_plan.to_html(classes="table table-striped table-hover", index=False)

        dup_color = "#28a745" if self.duplicate_rows == 0 else "#e36209"
        missing_color = "#28a745" if self.total_missing_cells == 0 else "#d73a49"
        outlier_color = "#28a745" if self.total_outliers_detected == 0 else "#6f42c1"

        warnings_html = ""
        if self.quality_warnings:
            items = "".join(f"<li style='margin-bottom: 4px;'>{w}</li>" for w in self.quality_warnings)
            warnings_html = f"""
            <div style="background: #fff5b1; border: 1px solid #e2b714; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; font-size: 13px; color: #735c0f;">
                <b>Data Quality Warnings:</b>
                <ul style="margin: 4px 0 0 16px; padding: 0;">{items}</ul>
            </div>
            """

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 18px; margin: 12px 0; background: #fafbfc;">
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #0366d6; padding-bottom: 8px; margin-bottom: 12px;">
                <h3 style="margin: 0; color: #24292e; display: flex; align-items: center; gap: 8px;">
                    <span style="background: #0366d6; color: white; padding: 3px 8px; border-radius: 4px; font-size: 13px;">CleanFrame</span>
                    Pre-Cleaning Diagnostic Summary
                </h3>
                <span style="font-size: 13px; color: #586069;">
                    <b>{self.initial_shape[0]}</b> rows &times; <b>{self.initial_shape[1]}</b> columns |
                    {self.memory_usage_str}
                </span>
            </div>

            {warnings_html}

            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px;">
                <div style="background: white; border: 1px solid #e1e4e8; border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #586069; font-weight: 600;">Duplicate Rows</div>
                    <div style="font-size: 20px; font-weight: 700; color: {dup_color}; margin-top: 4px;">{self.duplicate_rows}</div>
                    <div style="font-size: 11px; color: #586069;">{self.duplicate_pct:.1f}%</div>
                </div>
                <div style="background: white; border: 1px solid #e1e4e8; border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #586069; font-weight: 600;">Missing Cells</div>
                    <div style="font-size: 20px; font-weight: 700; color: {missing_color}; margin-top: 4px;">{self.total_missing_cells}</div>
                    <div style="font-size: 11px; color: #586069;">{self.total_missing_pct:.1f}%</div>
                </div>
                <div style="background: white; border: 1px solid #e1e4e8; border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #586069; font-weight: 600;">Outliers Detected</div>
                    <div style="font-size: 20px; font-weight: 700; color: {outlier_color}; margin-top: 4px;">{self.total_outliers_detected}</div>
                    <div style="font-size: 11px; color: #586069;">IQR / Z-score</div>
                </div>
                <div style="background: white; border: 1px solid #e1e4e8; border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #586069; font-weight: 600;">Transformations</div>
                    <div style="font-size: 20px; font-weight: 700; color: #0366d6; margin-top: 4px;">{len(self.action_plan)}</div>
                    <div style="font-size: 11px; color: #586069;">Planned steps</div>
                </div>
            </div>

            <div style="font-size: 13px; font-weight: 600; color: #24292e; margin-bottom: 8px;">Proposed Cleaning Actions:</div>
            <div style="max-height: 280px; overflow-y: auto; border: 1px solid #e1e4e8; border-radius: 6px; background: white;">
                {plan_html}
            </div>
        </div>
        """
        return html

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"<PreCleaningReport rows={self.initial_shape[0]}, cols={self.initial_shape[1]}, "
            f"duplicates={self.duplicate_rows}, missing={self.total_missing_cells}, "
            f"outliers={self.total_outliers_detected}, planned_actions={len(self.action_plan)}>"
        )


class DatasetDiagnostics:
    """
    Engine to inspect and diagnose DataFrames before any cleaning steps are executed.
    """

    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()

    def inspect(self, df: pd.DataFrame) -> PreCleaningReport:
        """
        Thoroughly inspect DataFrame and produce a PreCleaningReport.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to diagnose.
            
        Returns
        -------
        PreCleaningReport
            Diagnostic report with planned cleaning actions.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols

        # 1. Memory usage
        mem_bytes = df.memory_usage(deep=True).sum()
        if mem_bytes < 1024:
            mem_str = f"{mem_bytes} B"
        elif mem_bytes < 1024 * 1024:
            mem_str = f"{mem_bytes / 1024:.1f} KB"
        else:
            mem_str = f"{mem_bytes / (1024 * 1024):.2f} MB"

        # 2. Duplicates
        if self.config.drop_duplicates and n_rows > 0:
            dup_mask = df.duplicated(subset=self.config.duplicates_subset, keep=self.config.duplicates_keep)
            dup_count = int(dup_mask.sum())
            dup_pct = (dup_count / n_rows) * 100.0
        else:
            dup_count = 0
            dup_pct = 0.0

        # 3. Overall missing
        total_missing = int(df.isna().sum().sum())
        missing_pct = (total_missing / total_cells * 100.0) if total_cells > 0 else 0.0

        # 4. Column-level diagnostics
        column_diags: Dict[str, ColumnDiagnostic] = {}
        cols_with_missing: List[ColumnDiagnostic] = []
        cols_with_outliers: List[ColumnDiagnostic] = []
        datetime_candidates: List[ColumnDiagnostic] = []
        categorical_candidates: List[ColumnDiagnostic] = []
        warnings: List[str] = []
        action_plan: List[Dict[str, Any]] = []

        # Datetime detector helper
        dt_detector = DatetimeExtractor(
            columns=self.config.datetime_columns,
            features=self.config.datetime_features,
            drop_original=self.config.datetime_drop_original,
        )

        total_outliers = 0

        # Add duplicate row action to plan if applicable
        if dup_count > 0:
            action_plan.append({
                "Step": "Duplicates",
                "Target": "[All rows]" if self.config.duplicates_subset is None else str(self.config.duplicates_subset),
                "Issue Found": f"{dup_count} duplicate row(s) ({dup_pct:.1f}%)",
                "Planned Action": f"Drop duplicates (keep '{self.config.duplicates_keep}')",
                "Details": f"Will remove {dup_count} redundant row(s), leaving {n_rows - dup_count} rows.",
            })

        for col in df.columns:
            series = df[col]
            missing_c = int(series.isna().sum())
            col_missing_pct = (missing_c / n_rows * 100.0) if n_rows > 0 else 0.0
            unique_c = int(series.dropna().nunique())
            dtype_str = str(series.dtype)

            is_num = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            is_cat = (
                isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            )
            is_dt = dt_detector._is_datetime_series(series) if self.config.datetime_columns is None else (col in (self.config.datetime_columns or []))
            if is_dt:
                is_cat = False

            is_const = (unique_c <= 1 and missing_c == 0 and n_rows > 1)
            col_warnings: List[str] = []

            if missing_c == n_rows and n_rows > 0:
                col_warnings.append(f"Column '{col}' is 100% missing values (empty).")
                warnings.append(f"Column '{col}' is entirely empty (100% NaN).")
            elif is_const:
                col_warnings.append(f"Column '{col}' is constant (zero variance, 1 unique value).")
                warnings.append(f"Column '{col}' contains only 1 distinct value ('{series.iloc[0]}').")

            col_diag = ColumnDiagnostic(
                name=col,
                dtype=dtype_str,
                total_count=n_rows,
                missing_count=missing_c,
                missing_pct=col_missing_pct,
                unique_count=unique_c,
                is_numeric=is_num,
                is_categorical=is_cat,
                is_datetime=is_dt,
                is_constant=is_const,
                warnings=col_warnings,
            )

            # Planned datetime action
            if is_dt and "datetime" in self.config.steps:
                feats = self.config.datetime_features
                col_diag.planned_datetime_action = f"Extract {len(feats)} features ({', '.join(feats)})"
                datetime_candidates.append(col_diag)
                action_plan.append({
                    "Step": "Datetime",
                    "Target": col,
                    "Issue Found": f"Raw {dtype_str} datetime strings",
                    "Planned Action": f"Extract calendar features ({', '.join(feats)})",
                    "Details": f"Extracts: {feats}. Drop original: {self.config.datetime_drop_original}.",
                })

            # Planned missing value action
            if missing_c > 0 and "missing" in self.config.steps:
                if self.config.missing_column_threshold is not None and (missing_c / n_rows) > self.config.missing_column_threshold:
                    action_desc = f"Drop column (missing ratio {col_missing_pct:.1f}% > threshold {self.config.missing_column_threshold*100:.0f}%)"
                elif self.config.missing_custom_strategies and col in self.config.missing_custom_strategies:
                    strat = self.config.missing_custom_strategies[col]
                    action_desc = f"Impute via custom strategy/value: '{strat}'"
                elif is_num:
                    if self.config.missing_numeric_strategy == "drop_rows":
                        action_desc = "Drop rows with missing values"
                    elif self.config.missing_numeric_strategy == "constant":
                        action_desc = f"Impute constant ({self.config.missing_numeric_fill_value})"
                    else:
                        action_desc = f"Impute {self.config.missing_numeric_strategy}"
                else:
                    if self.config.missing_categorical_strategy == "drop_rows":
                        action_desc = "Drop rows with missing values"
                    elif self.config.missing_categorical_strategy == "constant":
                        action_desc = f"Impute constant ('{self.config.missing_categorical_fill_value}')"
                    else:
                        action_desc = f"Impute {self.config.missing_categorical_strategy}"

                col_diag.planned_missing_action = action_desc
                cols_with_missing.append(col_diag)
                action_plan.append({
                    "Step": "Missing",
                    "Target": col,
                    "Issue Found": f"{missing_c} missing cells ({col_missing_pct:.1f}%)",
                    "Planned Action": action_desc,
                    "Details": f"Column dtype: {dtype_str}. Strategy applies during transform.",
                })

            # Outlier detection
            if is_num and "outliers" in self.config.steps and self.config.outliers_method != "none" and self.config.outliers_action != "none":
                # Check if excluded
                is_excluded = (
                    col == self.config.target_column
                    or (self.config.outliers_columns is not None and col not in self.config.outliers_columns)
                )
                if not is_excluded and unique_c > 2 and n_rows > 3:
                    valid_vals = series.dropna()
                    if len(valid_vals) > 3:
                        if self.config.outliers_method == "iqr":
                            q1 = float(np.percentile(valid_vals, 25))
                            q3 = float(np.percentile(valid_vals, 75))
                            iqr = q3 - q1
                            lb = q1 - self.config.outliers_iqr_factor * iqr
                            ub = q3 + self.config.outliers_iqr_factor * iqr
                        else:  # zscore
                            m = float(np.mean(valid_vals))
                            s = float(np.std(valid_vals, ddof=1)) if len(valid_vals) > 1 else 0.0
                            lb = m - self.config.outliers_zscore_threshold * s
                            ub = m + self.config.outliers_zscore_threshold * s

                        outlier_mask = (valid_vals < lb) | (valid_vals > ub)
                        outlier_c = int(outlier_mask.sum())
                        if outlier_c > 0:
                            total_outliers += outlier_c
                            col_diag.outlier_count = outlier_c
                            col_diag.outlier_pct = (outlier_c / len(valid_vals)) * 100.0
                            col_diag.outlier_bounds = (lb, ub)
                            col_diag.planned_outlier_action = (
                                f"{self.config.outliers_action} outliers via {self.config.outliers_method} "
                                f"bounds: [{lb:.2f}, {ub:.2f}]"
                            )
                            cols_with_outliers.append(col_diag)
                            action_plan.append({
                                "Step": "Outliers",
                                "Target": col,
                                "Issue Found": f"{outlier_c} outlier(s) ({col_diag.outlier_pct:.1f}%) outside [{lb:.2f}, {ub:.2f}]",
                                "Planned Action": f"{self.config.outliers_action.capitalize()} outliers ({self.config.outliers_method})",
                                "Details": f"Limits: lower {lb:.2f}, upper {ub:.2f}.",
                            })

            # Categorical encoding
            if is_cat and "encoding" in self.config.steps and self.config.encoding_strategy != "none":
                is_excluded = col == self.config.target_column or (
                    self.config.encoding_columns is not None and col not in self.config.encoding_columns
                )
                if not is_excluded:
                    if self.config.encoding_strategy == "auto":
                        enc_desc = (
                            f"One-hot encode ({unique_c} binary cols)"
                            if unique_c <= self.config.max_one_hot_cardinality
                            else f"Label/Ordinal encode (cardinality {unique_c} > {self.config.max_one_hot_cardinality})"
                        )
                    elif self.config.encoding_strategy == "onehot":
                        enc_desc = f"One-hot encode ({unique_c} binary cols)"
                    elif self.config.encoding_strategy == "label":
                        enc_desc = f"Label/Ordinal encode ({unique_c} classes)"
                    elif self.config.encoding_strategy == "frequency":
                        enc_desc = "Frequency proportion encode"
                    else:
                        enc_desc = "No encoding"

                    col_diag.planned_encoding_action = enc_desc
                    categorical_candidates.append(col_diag)
                    action_plan.append({
                        "Step": "Encoding",
                        "Target": col,
                        "Issue Found": f"Categorical/string feature ({unique_c} unique values)",
                        "Planned Action": enc_desc,
                        "Details": f"Dtype: {dtype_str}. Train/test alignment guaranteed.",
                    })

            column_diags[col] = col_diag

        return PreCleaningReport(
            initial_shape=(n_rows, n_cols),
            memory_usage_str=mem_str,
            duplicate_rows=dup_count,
            duplicate_pct=dup_pct,
            total_missing_cells=total_missing,
            total_missing_pct=missing_pct,
            total_outliers_detected=total_outliers,
            columns_with_missing=cols_with_missing,
            columns_with_outliers=cols_with_outliers,
            datetime_candidates=datetime_candidates,
            categorical_candidates=categorical_candidates,
            quality_warnings=warnings,
            action_plan=action_plan,
            config=self.config,
            column_diagnostics=column_diags,
        )
