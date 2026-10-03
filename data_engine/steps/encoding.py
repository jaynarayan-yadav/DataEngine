"""
Categorical encoding step.
Supports automatic strategy selection, one-hot encoding, label/ordinal encoding,
and frequency encoding with strict test-set consistency (no data leakage).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Set
import numpy as np
import pandas as pd

from .base import BaseCleanerStep
from ..logger import CleanerLogger
from ..report import CleaningReport


class CategoricalEncoder(BaseCleanerStep):
    """
    Encodes categorical features for machine learning pipelines.
    
    Parameters
    ----------
    strategy : {"auto", "onehot", "label", "frequency", "none"}, default "auto"
        Encoding strategy to apply.
        - "auto": One-hot encodes columns with unique values <= max_one_hot_cardinality;
                  label encodes columns with higher cardinality.
        - "onehot": One-hot encodes all categorical columns.
        - "label": Assigns an integer index (0, 1, 2, ...) to each unique category.
        - "frequency": Replaces each category with its relative frequency in the dataset.
        - "none": No encoding is performed.
    columns : list of str, optional
        Specific columns to encode. If None, auto-detects categorical/object columns.
    max_one_hot_cardinality : int, default 10
        Maximum unique values for a column to receive one-hot encoding under "auto" mode.
    drop_first : bool, default False
        Whether to drop the first category level in one-hot encoding (useful for linear regression).
    target_column : str, optional
        Target variable name to exclude from encoding.
    """

    def __init__(
        self,
        strategy: Literal["auto", "onehot", "label", "frequency", "none"] = "auto",
        columns: Optional[List[str]] = None,
        max_one_hot_cardinality: int = 10,
        drop_first: bool = False,
        target_column: Optional[str] = None,
    ):
        super().__init__(name="encoding")
        self.strategy = strategy
        self.columns = columns
        self.max_one_hot_cardinality = max_one_hot_cardinality
        self.drop_first = drop_first
        self.target_column = target_column

        # Learned encoding metadata
        self.column_strategies_: Dict[str, str] = {}
        self.onehot_categories_: Dict[str, List[Any]] = {}
        self.label_mappings_: Dict[str, Dict[Any, int]] = {}
        self.frequency_mappings_: Dict[str, Dict[Any, float]] = {}

    def _get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns and c != self.target_column]

        cat_cols = []
        for col in df.columns:
            if col == self.target_column:
                continue
            is_cat = (
                pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])
                or isinstance(df[col].dtype, pd.CategoricalDtype)
                or (hasattr(pd.api.types, "is_bool_dtype") and pd.api.types.is_bool_dtype(df[col]))
            )
            if is_cat:
                cat_cols.append(col)
        return cat_cols

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        self.column_strategies_ = {}
        self.onehot_categories_ = {}
        self.label_mappings_ = {}
        self.frequency_mappings_ = {}

        if self.strategy == "none" or len(df) == 0:
            self.is_fitted = True
            return self

        target_cols = self._get_categorical_columns(df)
        total_rows = len(df)

        for col in target_cols:
            series = df[col].astype(str)
            unique_vals = sorted(series.unique().tolist())
            n_unique = len(unique_vals)

            # Determine strategy for this column
            if self.strategy == "auto":
                col_strat = "onehot" if n_unique <= self.max_one_hot_cardinality else "label"
            else:
                col_strat = self.strategy

            self.column_strategies_[col] = col_strat

            if col_strat == "onehot":
                # If drop_first is True, drop the first category
                cats_to_keep = unique_vals[1:] if (self.drop_first and len(unique_vals) > 1) else unique_vals
                self.onehot_categories_[col] = cats_to_keep
            elif col_strat == "label":
                self.label_mappings_[col] = {val: idx for idx, val in enumerate(unique_vals)}
            elif col_strat == "frequency":
                freqs = series.value_counts(normalize=True).to_dict()
                self.frequency_mappings_[col] = freqs

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

        if not self.column_strategies_ or self.strategy == "none":
            if logger and self.strategy != "none":
                logger.log("No categorical columns identified for encoding.")
            return df.copy()

        res = df.copy()

        for col, col_strat in self.column_strategies_.items():
            if col not in res.columns:
                continue

            series_str = res[col].astype(str)

            if col_strat == "onehot":
                categories = self.onehot_categories_.get(col, [])
                new_col_names = []
                for cat in categories:
                    # Clean column name string
                    clean_cat = str(cat).replace(" ", "_").replace("/", "_").replace("-", "_")
                    new_col_name = f"{col}_{clean_cat}"
                    res[new_col_name] = (series_str == str(cat)).astype(int)
                    new_col_names.append(new_col_name)

                # Drop original column
                res = res.drop(columns=[col])

                msg = (
                    f"One-hot encoded '{col}' into {len(new_col_names)} binary columns "
                    f"(cardinality: {len(categories)})"
                )
                if logger:
                    logger.log(msg)
                if report:
                    report.encoded_columns[col] = {
                        "strategy": "onehot",
                        "new_columns": new_col_names,
                        "details": f"{len(new_col_names)} indicator columns"
                    }
                    report.add_action(
                        step=self.name,
                        action="One-hot encode",
                        details=f"Created {len(new_col_names)} columns: {', '.join(new_col_names[:5])}{'...' if len(new_col_names) > 5 else ''}",
                        column=col,
                        count=len(new_col_names)
                    )

            elif col_strat == "label":
                mapping = self.label_mappings_.get(col, {})
                # Map known categories; unknown categories map to -1
                res[col] = series_str.map(mapping).fillna(-1).astype(int)

                msg = f"Label encoded '{col}' into {len(mapping)} integer levels (0 to {len(mapping)-1})"
                if logger:
                    logger.log(msg)
                if report:
                    report.encoded_columns[col] = {
                        "strategy": "label",
                        "levels": len(mapping),
                        "details": f"{len(mapping)} levels (0..{len(mapping)-1})"
                    }
                    report.add_action(
                        step=self.name,
                        action="Label encode",
                        details=f"Mapped to {len(mapping)} integer values",
                        column=col,
                        count=len(mapping)
                    )

            elif col_strat == "frequency":
                freq_map = self.frequency_mappings_.get(col, {})
                # Map known categories; unknown categories default to 0.0
                res[col] = series_str.map(freq_map).fillna(0.0).astype(float)

                msg = f"Frequency encoded '{col}' using category proportions"
                if logger:
                    logger.log(msg)
                if report:
                    report.encoded_columns[col] = {
                        "strategy": "frequency",
                        "details": f"{len(freq_map)} unique frequency proportions"
                    }
                    report.add_action(
                        step=self.name,
                        action="Frequency encode",
                        details="Replaced with class frequency proportions",
                        column=col
                    )

        return res
