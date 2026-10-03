"""
Duplicate rows removal step.
"""

from __future__ import annotations

from typing import List, Literal, Optional
import pandas as pd

from .base import BaseCleanerStep
from ..logger import CleanerLogger
from ..report import CleaningReport


class DuplicateHandler(BaseCleanerStep):
    """
    Identifies and removes duplicate rows from a DataFrame.
    
    Parameters
    ----------
    subset : list of str, optional
        Column labels to consider for identifying duplicates. If None, all columns are used.
    keep : {"first", "last", False}, default "first"
        Determines which duplicates (if any) to keep.
        - "first": Drop duplicates except for the first occurrence.
        - "last": Drop duplicates except for the last occurrence.
        - False: Drop all duplicates.
    reset_index : bool, default True
        Whether to reset the index to a contiguous 0-based range after dropping duplicates.
    """

    def __init__(
        self,
        subset: Optional[List[str]] = None,
        keep: Literal["first", "last", False] = "first",
        reset_index: bool = True,
    ):
        super().__init__(name="duplicates")
        self.subset = subset
        self.keep = keep
        self.reset_index = reset_index

    def fit(self, df: pd.DataFrame) -> "DuplicateHandler":
        if self.subset is not None:
            missing_cols = [c for c in self.subset if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Subset columns not found in DataFrame: {missing_cols}")
        self.is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        logger: Optional[CleanerLogger] = None,
        report: Optional[CleaningReport] = None,
    ) -> pd.DataFrame:
        initial_rows = len(df)
        
        # Calculate duplicate rows
        dup_mask = df.duplicated(subset=self.subset, keep=self.keep)
        dup_count = int(dup_mask.sum())
        
        if dup_count == 0:
            if logger:
                logger.log("No duplicate rows found.")
            if report:
                report.add_action(
                    step=self.name,
                    action="Check duplicates",
                    details="0 duplicate rows found",
                    count=0
                )
            return df.copy()

        cleaned_df = df.drop_duplicates(
            subset=self.subset,
            keep=self.keep,
            ignore_index=self.reset_index
        )
        final_rows = len(cleaned_df)
        
        msg = (
            f"Dropped {dup_count} duplicate row(s) "
            f"(rows: {initial_rows} -> {final_rows}, kept '{self.keep}')"
        )
        if logger:
            logger.log(msg)
            
        if report:
            report.duplicates_removed += dup_count
            report.add_action(
                step=self.name,
                action="Drop duplicates",
                details=msg,
                count=dup_count
            )

        return cleaned_df
