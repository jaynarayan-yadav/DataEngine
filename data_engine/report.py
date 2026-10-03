"""
Cleaning report and summary generator for cleanframe.
Provides structured reports, pandas summary tables, and HTML representations for Jupyter notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class CleaningAction:
    """A record of a single cleaning operation performed on the DataFrame."""
    step: str
    column: Optional[str]
    action: str
    details: str
    count: int = 0


@dataclass
class CleaningReport:
    """
    Structured summary of all transformations applied by DataFrameCleaner.
    """
    initial_shape: Tuple[int, int] = (0, 0)
    final_shape: Tuple[int, int] = (0, 0)
    initial_missing: int = 0
    final_missing: int = 0
    duplicates_removed: int = 0
    missing_imputations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    outlier_actions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    datetime_extracted: Dict[str, List[str]] = field(default_factory=dict)
    encoded_columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    actions: List[CleaningAction] = field(default_factory=list)

    def add_action(
        self,
        step: str,
        action: str,
        details: str,
        column: Optional[str] = None,
        count: int = 0
    ) -> None:
        """Record a cleaning action."""
        self.actions.append(
            CleaningAction(
                step=step,
                column=column,
                action=action,
                details=details,
                count=count
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert recorded cleaning actions to a pandas DataFrame for inspection.
        """
        if not self.actions:
            return pd.DataFrame(columns=["Step", "Column", "Action", "Count", "Details"])
        
        data = [
            {
                "Step": a.step.capitalize(),
                "Column": a.column if a.column else "[All rows]",
                "Action": a.action,
                "Count / Impact": a.count if a.count > 0 else "-",
                "Details": a.details
            }
            for a in self.actions
        ]
        return pd.DataFrame(data)

    def summary(self) -> str:
        """Return a formatted text summary of the cleaning process."""
        lines = [
            "===========================================================",
            "                  CLEANFRAME CLEANING REPORT               ",
            "===========================================================",
            f"Shape:               {self.initial_shape} -> {self.final_shape}",
            f"Rows Changed:        {self.final_shape[0] - self.initial_shape[0]:+d} rows",
            f"Cols Changed:        {self.final_shape[1] - self.initial_shape[1]:+d} columns",
            f"Duplicate Rows:      {self.duplicates_removed} dropped",
            f"Missing Values:      {self.initial_missing} -> {self.final_missing} (remaining: {self.final_missing})",
            "-----------------------------------------------------------",
        ]

        if self.missing_imputations:
            lines.append("Missing Value Imputations:")
            for col, info in self.missing_imputations.items():
                val_repr = info.get('fill_value')
                val_str = f"'{val_repr}'" if isinstance(val_repr, str) else f"{val_repr}"
                lines.append(
                    f"  - {col}: {info.get('count', 0)} values imputed via {info.get('strategy')} ({val_str})"
                )
        else:
            lines.append("Missing Value Imputations: None")

        if self.outlier_actions:
            lines.append("\nOutlier Handling:")
            for col, info in self.outlier_actions.items():
                lines.append(
                    f"  - {col}: {info.get('count', 0)} values handled via {info.get('method')} ({info.get('action')}) "
                    f"bounds: [{info.get('lower', 0):.2f}, {info.get('upper', 0):.2f}]"
                )
        else:
            lines.append("Outlier Handling: None")

        if self.datetime_extracted:
            lines.append("\nDatetime Feature Extraction:")
            for col, feats in self.datetime_extracted.items():
                lines.append(f"  - {col}: extracted {len(feats)} features ({', '.join(feats)})")
        else:
            lines.append("Datetime Feature Extraction: None")

        if self.encoded_columns:
            lines.append("\nCategorical Encoding:")
            for col, info in self.encoded_columns.items():
                lines.append(
                    f"  - {col}: {info.get('strategy')} encoding ({info.get('details', '')})"
                )
        else:
            lines.append("Categorical Encoding: None")

        lines.append("===========================================================")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebook display."""
        df_actions = self.to_dataframe()
        actions_html = df_actions.to_html(classes="table table-striped table-hover", index=False)

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 18px; margin: 12px 0; background: #fafbfc;">
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #0366d6; padding-bottom: 8px; margin-bottom: 12px;">
                <h3 style="margin: 0; color: #24292e; display: flex; align-items: center; gap: 8px;">
                    <span style="background: #0366d6; color: white; padding: 3px 8px; border-radius: 4px; font-size: 13px;">CleanFrame</span>
                    Cleaning Report
                </h3>
                <span style="font-size: 13px; color: #586069;">
                    <b>{self.initial_shape[0]}</b> &rarr; <b>{self.final_shape[0]}</b> rows |
                    <b>{self.initial_shape[1]}</b> &rarr; <b>{self.final_shape[1]}</b> columns
                </span>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px;">
                <div style="background: white; border: 1px solid #e1e4e8; border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #586069; font-weight: 600;">Duplicates Removed</div>
                    <div style="font-size: 20px; font-weight: 700; color: #28a745; margin-top: 4px;">{self.duplicates_removed}</div>
                </div>
                <div style="background: white; border: 1px solid #e1e4e8; border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #586069; font-weight: 600;">Missing Imputed</div>
                    <div style="font-size: 20px; font-weight: 700; color: #0366d6; margin-top: 4px;">{self.initial_missing - self.final_missing}</div>
                </div>
                <div style="background: white; border: 1px solid #e1e4e8; border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #586069; font-weight: 600;">Outliers Handled</div>
                    <div style="font-size: 20px; font-weight: 700; color: #d73a49; margin-top: 4px;">
                        {sum(v.get('count', 0) for v in self.outlier_actions.values())}
                    </div>
                </div>
                <div style="background: white; border: 1px solid #e1e4e8; border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #586069; font-weight: 600;">Categoricals Encoded</div>
                    <div style="font-size: 20px; font-weight: 700; color: #6f42c1; margin-top: 4px;">{len(self.encoded_columns)}</div>
                </div>
            </div>

            <div style="font-size: 13px; font-weight: 600; color: #24292e; margin-bottom: 8px;">Detailed Operations:</div>
            <div style="max-height: 280px; overflow-y: auto; border: 1px solid #e1e4e8; border-radius: 6px; background: white;">
                {actions_html}
            </div>
        </div>
        """
        return html

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"<CleaningReport rows: {self.initial_shape[0]}->{self.final_shape[0]}, cols: {self.initial_shape[1]}->{self.final_shape[1]}, duplicates_dropped={self.duplicates_removed}>"
