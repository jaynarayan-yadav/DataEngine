"""
Main DataFrameCleaner class for cleanframe.
Provides both high-level automatic cleaning and granular manual step execution.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
import pandas as pd

from .config import CleaningConfig
from .diagnostics import DatasetDiagnostics, PreCleaningReport
from .logger import CleanerLogger, get_logger
from .report import CleaningReport
from .steps.base import BaseCleanerStep
from .steps.duplicates import DuplicateHandler
from .steps.missing import MissingValueHandler
from .steps.outliers import OutlierHandler
from .steps.datetime import DatetimeExtractor
from .steps.encoding import CategoricalEncoder


class DataFrameCleaner:
    """
    Automated and manual data cleaner for pandas DataFrames.
    
    Parameters
    ----------
    config : CleaningConfig, optional
        Complete configuration object. If provided, overrides conflicting kwargs.
    steps : list of str, optional
        Steps to execute in automatic mode. Default is:
        ["duplicates", "datetime", "missing", "outliers", "encoding"]
    drop_duplicates : bool, default True
        Whether to drop duplicate rows.
    duplicates_subset : list of str, optional
        Subset of columns to consider for duplicates.
    duplicates_keep : {"first", "last", False}, default "first"
        Which duplicate occurrence to keep.
    missing_numeric_strategy : {"median", "mean", "mode", "constant", "drop_rows", "none"}, default "median"
        Strategy to impute numeric missing values.
    missing_numeric_fill_value : float, default 0.0
        Constant value for numeric imputation when strategy is "constant".
    missing_categorical_strategy : {"constant", "mode", "drop_rows", "none"}, default "constant"
        Strategy to impute categorical missing values.
    missing_categorical_fill_value : str, default "missing"
        Constant value for categorical imputation when strategy is "constant".
    missing_column_threshold : float, optional
        Drop columns where missing value ratio exceeds this fraction (e.g. 0.8).
    missing_custom_strategies : dict, optional
        Column-specific missing strategies or values, e.g. {"age": "mean", "dept": "HR"}.
    outliers_method : {"iqr", "zscore", "none"}, default "iqr"
        Outlier detection method.
    outliers_action : {"clip", "drop", "nan", "none"}, default "clip"
        What to do with outliers: "clip" (winsorize), "drop", or "nan".
    outliers_iqr_factor : float, default 1.5
        IQR multiplier for outlier detection.
    outliers_zscore_threshold : float, default 3.0
        Z-score threshold for outlier detection.
    outliers_columns : list of str, optional
        Specific numeric columns to inspect for outliers.
    datetime_columns : list of str, optional
        Specific datetime columns to process. If None, auto-detected.
    datetime_features : list of str, default ["year", "month", "day", "day_of_week", "is_weekend"]
        Calendar features to extract from datetime columns.
    datetime_drop_original : bool, default True
        Whether to drop the original datetime column after extraction.
    encoding_strategy : {"auto", "onehot", "label", "frequency", "none"}, default "auto"
        Categorical encoding strategy.
    encoding_columns : list of str, optional
        Specific columns to encode. If None, auto-detects categorical columns.
    max_one_hot_cardinality : int, default 10
        Max unique categories for one-hot encoding in "auto" mode.
    encoding_drop_first : bool, default False
        Whether to drop the first category in one-hot encoding.
    target_column : str, optional
        Target variable name to exclude from encoding and transformations.
    copy : bool, default True
        Whether to operate on a copy of the input DataFrame.
    verbose : bool, default True
        Whether to print informative logs during cleaning.
    """

    def __init__(
        self,
        config: Optional[CleaningConfig] = None,
        steps: Optional[List[str]] = None,
        drop_duplicates: bool = True,
        duplicates_subset: Optional[List[str]] = None,
        duplicates_keep: Literal["first", "last", False] = "first",
        missing_numeric_strategy: Literal["median", "mean", "mode", "constant", "drop_rows", "none"] = "median",
        missing_numeric_fill_value: float = 0.0,
        missing_categorical_strategy: Literal["constant", "mode", "drop_rows", "none"] = "constant",
        missing_categorical_fill_value: str = "missing",
        missing_column_threshold: Optional[float] = None,
        missing_custom_strategies: Optional[Dict[str, Any]] = None,
        outliers_method: Literal["iqr", "zscore", "none"] = "iqr",
        outliers_action: Literal["clip", "drop", "nan", "none"] = "clip",
        outliers_iqr_factor: float = 1.5,
        outliers_zscore_threshold: float = 3.0,
        outliers_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,
        datetime_features: Optional[List[str]] = None,
        datetime_drop_original: bool = True,
        encoding_strategy: Literal["auto", "onehot", "label", "frequency", "none"] = "auto",
        encoding_columns: Optional[List[str]] = None,
        max_one_hot_cardinality: int = 10,
        encoding_drop_first: bool = False,
        target_column: Optional[str] = None,
        copy: bool = True,
        verbose: bool = True,
        interactive: bool = False,
        prompt_fn: Optional[Callable[[str], str]] = None,
    ):
        if config is not None:
            self.config = config
        else:
            dt_feats = datetime_features if datetime_features is not None else ["year", "month", "day", "day_of_week", "is_weekend"]
            step_list = steps if steps is not None else ["duplicates", "datetime", "missing", "outliers", "encoding"]
            self.config = CleaningConfig(
                steps=step_list,
                drop_duplicates=drop_duplicates,
                duplicates_subset=duplicates_subset,
                duplicates_keep=duplicates_keep,
                missing_numeric_strategy=missing_numeric_strategy,
                missing_numeric_fill_value=missing_numeric_fill_value,
                missing_categorical_strategy=missing_categorical_strategy,
                missing_categorical_fill_value=missing_categorical_fill_value,
                missing_column_threshold=missing_column_threshold,
                missing_custom_strategies=missing_custom_strategies,
                outliers_method=outliers_method,
                outliers_action=outliers_action,
                outliers_iqr_factor=outliers_iqr_factor,
                outliers_zscore_threshold=outliers_zscore_threshold,
                outliers_columns=outliers_columns,
                datetime_columns=datetime_columns,
                datetime_features=dt_feats,
                datetime_drop_original=datetime_drop_original,
                encoding_strategy=encoding_strategy,
                encoding_columns=encoding_columns,
                max_one_hot_cardinality=max_one_hot_cardinality,
                encoding_drop_first=encoding_drop_first,
                target_column=target_column,
                copy=copy,
                verbose=verbose,
                interactive=interactive,
                prompt_fn=prompt_fn,
            )

        self.logger = CleanerLogger(verbose=self.config.verbose)
        self.report_ = CleaningReport()
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.pipeline_steps: List[BaseCleanerStep] = []
        self._is_fitted: bool = False
        self.aborted: bool = False

        self.diagnostics_ = DatasetDiagnostics(self.config)
        self.latest_diagnostics: Optional[PreCleaningReport] = None

        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Construct cleaner step objects according to configuration."""
        self.pipeline_steps = []
        step_names = self.config.steps

        for step in step_names:
            step_lower = step.strip().lower()
            if step_lower in ("duplicates", "duplicate", "drop_duplicates"):
                if self.config.drop_duplicates:
                    self.pipeline_steps.append(
                        DuplicateHandler(
                            subset=self.config.duplicates_subset,
                            keep=self.config.duplicates_keep,
                        )
                    )
            elif step_lower in ("datetime", "dates", "datetimes"):
                self.pipeline_steps.append(
                    DatetimeExtractor(
                        columns=self.config.datetime_columns,
                        features=self.config.datetime_features,
                        drop_original=self.config.datetime_drop_original,
                    )
                )
            elif step_lower in ("missing", "impute", "imputation"):
                self.pipeline_steps.append(
                    MissingValueHandler(
                        numeric_strategy=self.config.missing_numeric_strategy,
                        numeric_fill_value=self.config.missing_numeric_fill_value,
                        categorical_strategy=self.config.missing_categorical_strategy,
                        categorical_fill_value=self.config.missing_categorical_fill_value,
                        column_threshold=self.config.missing_column_threshold,
                        custom_strategies=self.config.missing_custom_strategies,
                    )
                )
            elif step_lower in ("outliers", "outlier"):
                if self.config.outliers_method != "none" and self.config.outliers_action != "none":
                    self.pipeline_steps.append(
                        OutlierHandler(
                            method=self.config.outliers_method,
                            action=self.config.outliers_action,
                            iqr_factor=self.config.outliers_iqr_factor,
                            zscore_threshold=self.config.outliers_zscore_threshold,
                            columns=self.config.outliers_columns,
                            exclude_columns=[self.config.target_column] if self.config.target_column else None,
                        )
                    )
            elif step_lower in ("encoding", "encode", "categorical", "categoricals"):
                if self.config.encoding_strategy != "none":
                    self.pipeline_steps.append(
                        CategoricalEncoder(
                            strategy=self.config.encoding_strategy,
                            columns=self.config.encoding_columns,
                            max_one_hot_cardinality=self.config.max_one_hot_cardinality,
                            drop_first=self.config.encoding_drop_first,
                            target_column=self.config.target_column,
                        )
                    )
            else:
                self.logger.warning(f"Unrecognized step name: '{step}'. Skipping.")

    def fit(self, df: pd.DataFrame) -> "DataFrameCleaner":
        """
        Fit all pipeline steps on the input DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training or input DataFrame.
            
        Returns
        -------
        DataFrameCleaner
            Fitted cleaner instance.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        current_df = df.copy() if self.config.copy else df
        for step in self.pipeline_steps:
            step.fit(current_df)
            # Intermediate transform is needed so downstream steps learn on properly transformed dtypes
            current_df = step.transform(current_df, logger=None, report=None)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a DataFrame using parameters learned during fit.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to clean (e.g. test set or new batch).
            
        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("DataFrameCleaner is not fitted yet. Call fit() or clean() first.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        self.report_ = CleaningReport()
        self.report_.initial_shape = df.shape
        self.report_.initial_missing = int(df.isna().sum().sum())
        self.logger.clear()

        self.logger.log(f"Starting cleanup on DataFrame with shape {df.shape}...")

        current_df = df.copy() if self.config.copy else df
        for step in self.pipeline_steps:
            current_df = step.transform(current_df, logger=self.logger, report=self.report_)

        self.report_.final_shape = current_df.shape
        self.report_.final_missing = int(current_df.isna().sum().sum())
        self.cleaned_df = current_df

        self.logger.log(
            f"Cleanup complete! Shape: {self.report_.initial_shape} -> {self.report_.final_shape}"
        )
        return self.cleaned_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit all steps and transform the DataFrame in one call.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to clean.
            
        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame.
        """
        return self.clean(df)

    def diagnose(self, df: pd.DataFrame, display: bool = False) -> PreCleaningReport:
        """
        Inspect and diagnose the input DataFrame before any cleaning transformations.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to inspect.
        display : bool, default False
            Whether to print the formatted diagnostic summary to stdout.
            
        Returns
        -------
        PreCleaningReport
            Complete pre-cleaning diagnostic analysis and proposed cleaning blueprint.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        report = self.diagnostics_.inspect(df)
        self.latest_diagnostics = report
        if display:
            print(report.summary())
        return report

    def preview(self, df: pd.DataFrame, display: bool = True) -> PreCleaningReport:
        """
        Alias for diagnose(df, display=True). Displays the pre-cleaning diagnosis summary.
        """
        return self.diagnose(df, display=display)

    def _prompt_confirmation(
        self,
        report: PreCleaningReport,
        prompt_fn: Optional[Callable[[str], str]] = None,
    ) -> Tuple[bool, Optional[List[str]]]:
        """
        Interactively prompt user to confirm or customize cleaning steps.
        
        Returns
        -------
        Tuple[bool, Optional[List[str]]]
            (proceed, custom_steps)
        """
        fn = prompt_fn or self.config.prompt_fn or input
        prompt_msg = "\n>>> Should CleanFrame proceed with cleaning? [y = Yes, n = No / Abort, c = Customize steps]: "
        try:
            choice = fn(prompt_msg).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False, None

        if choice in ("y", "yes", ""):
            return True, None
        elif choice in ("c", "customize", "custom"):
            steps_help = (
                "\nAvailable cleaning steps:\n"
                "  1. duplicates (drop duplicate rows)\n"
                "  2. datetime   (extract calendar features)\n"
                "  3. missing    (impute missing values)\n"
                "  4. outliers   (detect and treat outliers)\n"
                "  5. encoding   (encode categorical variables)\n"
                "Enter comma-separated step numbers or names (e.g. '1,3' or 'duplicates,missing'): "
            )
            try:
                selected_raw = fn(steps_help).strip()
            except (EOFError, KeyboardInterrupt):
                return False, None

            step_map = {
                "1": "duplicates",
                "2": "datetime",
                "3": "missing",
                "4": "outliers",
                "5": "encoding",
                "duplicates": "duplicates",
                "datetime": "datetime",
                "missing": "missing",
                "outliers": "outliers",
                "encoding": "encoding",
            }
            chosen_steps = []
            for item in selected_raw.split(","):
                key = item.strip().lower()
                if key in step_map:
                    val = step_map[key]
                    if val not in chosen_steps:
                        chosen_steps.append(val)

            if not chosen_steps:
                print("No valid steps entered. Running all steps.")
                return True, None
            return True, chosen_steps
        else:
            return False, None

    def clean_interactive(
        self,
        df: pd.DataFrame,
        prompt_fn: Optional[Callable[[str], str]] = None,
    ) -> pd.DataFrame:
        """
        Interactively clean a DataFrame: displays pre-cleaning diagnosis and prompts for confirmation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to clean.
        prompt_fn : callable, optional
            Custom function for user input (default is built-in input()).
            
        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame if confirmed, or original DataFrame if aborted.
        """
        return self.clean(df, interactive=True, prompt_fn=prompt_fn)

    def clean(
        self,
        df: pd.DataFrame,
        interactive: Optional[bool] = None,
        confirm: Optional[bool] = None,
        prompt_fn: Optional[Callable[[str], str]] = None,
    ) -> pd.DataFrame:
        """
        Primary entrypoint: cleans the input DataFrame with optional pre-cleaning summary and confirmation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to clean.
        interactive : bool, optional
            Whether to display diagnostic summary and ask for user confirmation before cleaning.
            If None, uses self.config.interactive (default False).
        confirm : bool, optional
            Explicit confirmation flag. If True, proceeds without prompting.
            If False, aborts immediately without modifying data.
        prompt_fn : callable, optional
            Function used to prompt user for input (default is input).
            
        Returns
        -------
        pd.DataFrame
            Cleaned and preprocessed DataFrame, or original DataFrame if aborted.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        is_interactive = self.config.interactive if interactive is None else interactive
        custom_steps: Optional[List[str]] = None

        # Pre-cleaning diagnosis
        diag_report = self.diagnose(df, display=is_interactive)

        if is_interactive:
            if confirm is None:
                proceed, custom_steps = self._prompt_confirmation(diag_report, prompt_fn=prompt_fn)
            else:
                proceed = bool(confirm)

            if not proceed:
                self.aborted = True
                self.logger.log("Cleaning aborted by user. Returning original DataFrame without modifications.")
                return df.copy() if self.config.copy else df

        self.aborted = False
        self.report_ = CleaningReport()
        self.report_.initial_shape = df.shape
        self.report_.initial_missing = int(df.isna().sum().sum())
        self.logger.clear()

        self.logger.log(f"Starting cleanup on DataFrame with shape {df.shape}...")

        steps_to_run = self.pipeline_steps
        if custom_steps is not None:
            steps_to_run = [s for s in self.pipeline_steps if s.name in custom_steps]
            self.logger.log(f"Running customized step sequence: {[s.name for s in steps_to_run]}")

        current_df = df.copy() if self.config.copy else df
        for step in steps_to_run:
            current_df = step.fit_transform(current_df, logger=self.logger, report=self.report_)

        self.report_.final_shape = current_df.shape
        self.report_.final_missing = int(current_df.isna().sum().sum())
        self.cleaned_df = current_df
        self._is_fitted = True

        self.logger.log(
            f"Cleanup complete! Resulting shape: {self.report_.final_shape} "
            f"(Rows: {self.report_.final_shape[0] - self.report_.initial_shape[0]:+d}, "
            f"Cols: {self.report_.final_shape[1] - self.report_.initial_shape[1]:+d})"
        )
        return self.cleaned_df


    # -------------------------------------------------------------------------
    # Granular manual steps (can be used individually or via df.pipe(...))
    # -------------------------------------------------------------------------

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: Literal["first", "last", False] = "first",
        reset_index: bool = True,
    ) -> pd.DataFrame:
        """Manually remove duplicate rows."""
        step = DuplicateHandler(subset=subset, keep=keep, reset_index=reset_index)
        return step.fit_transform(df, logger=self.logger, report=self.report_)

    def impute_missing(
        self,
        df: pd.DataFrame,
        numeric_strategy: Literal["median", "mean", "mode", "constant", "drop_rows", "none"] = "median",
        numeric_fill_value: float = 0.0,
        categorical_strategy: Literal["constant", "mode", "drop_rows", "none"] = "constant",
        categorical_fill_value: str = "missing",
        column_threshold: Optional[float] = None,
        custom_strategies: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Manually impute missing values."""
        step = MissingValueHandler(
            numeric_strategy=numeric_strategy,
            numeric_fill_value=numeric_fill_value,
            categorical_strategy=categorical_strategy,
            categorical_fill_value=categorical_fill_value,
            column_threshold=column_threshold,
            custom_strategies=custom_strategies,
        )
        return step.fit_transform(df, logger=self.logger, report=self.report_)

    def treat_outliers(
        self,
        df: pd.DataFrame,
        method: Literal["iqr", "zscore", "none"] = "iqr",
        action: Literal["clip", "drop", "nan", "none"] = "clip",
        iqr_factor: float = 1.5,
        zscore_threshold: float = 3.0,
        columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Manually detect and treat outliers."""
        step = OutlierHandler(
            method=method,
            action=action,
            iqr_factor=iqr_factor,
            zscore_threshold=zscore_threshold,
            columns=columns,
            exclude_columns=exclude_columns,
        )
        return step.fit_transform(df, logger=self.logger, report=self.report_)

    def extract_datetime_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        drop_original: bool = True,
    ) -> pd.DataFrame:
        """Manually parse datetime columns and extract features."""
        step = DatetimeExtractor(
            columns=columns,
            features=features,
            drop_original=drop_original,
        )
        return step.fit_transform(df, logger=self.logger, report=self.report_)

    def encode_categoricals(
        self,
        df: pd.DataFrame,
        strategy: Literal["auto", "onehot", "label", "frequency", "none"] = "auto",
        columns: Optional[List[str]] = None,
        max_one_hot_cardinality: int = 10,
        drop_first: bool = False,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Manually encode categorical columns."""
        step = CategoricalEncoder(
            strategy=strategy,
            columns=columns,
            max_one_hot_cardinality=max_one_hot_cardinality,
            drop_first=drop_first,
            target_column=target_column,
        )
        return step.fit_transform(df, logger=self.logger, report=self.report_)

    # -------------------------------------------------------------------------
    # Reports and Inspection
    # -------------------------------------------------------------------------

    @property
    def report(self) -> CleaningReport:
        """Access the latest CleaningReport."""
        return self.report_

    def profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a rich per-column summary profile of the DataFrame without modifying it.

        Returns a pandas DataFrame with one row per column and the following fields:
        ``dtype``, ``missing_count``, ``missing_pct``, ``unique_count``, ``unique_pct``,
        ``mean``, ``median``, ``std``, ``min``, ``max``, ``mode``, ``is_numeric``,
        ``is_categorical``, ``is_datetime``.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to profile.

        Returns
        -------
        pd.DataFrame
            Column profile summary table.

        Examples
        --------
        >>> cleaner = DataFrameCleaner()
        >>> cleaner.profile(df)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        n_rows = len(df)
        rows = []
        for col in df.columns:
            s = df[col]
            missing_count = int(s.isna().sum())
            missing_pct = round(missing_count / n_rows * 100.0, 2) if n_rows > 0 else 0.0
            unique_count = int(s.dropna().nunique())
            unique_pct = round(unique_count / n_rows * 100.0, 2) if n_rows > 0 else 0.0

            is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            is_dt = pd.api.types.is_datetime64_any_dtype(s)
            is_cat = (
                pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(s)
            ) and not is_dt

            # Numeric stats
            if is_num:
                valid = s.dropna()
                mean_val = round(float(valid.mean()), 4) if len(valid) > 0 else None
                median_val = round(float(valid.median()), 4) if len(valid) > 0 else None
                std_val = round(float(valid.std()), 4) if len(valid) > 1 else None
                min_val = round(float(valid.min()), 4) if len(valid) > 0 else None
                max_val = round(float(valid.max()), 4) if len(valid) > 0 else None
                modes = valid.mode()
                mode_val = round(float(modes.iloc[0]), 4) if len(modes) > 0 else None
            else:
                mean_val = median_val = std_val = min_val = max_val = None
                modes = s.dropna().mode()
                mode_val = str(modes.iloc[0]) if len(modes) > 0 else None

            rows.append({
                "column": col,
                "dtype": str(s.dtype),
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "unique_count": unique_count,
                "unique_pct": unique_pct,
                "mean": mean_val,
                "median": median_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "mode": mode_val,
                "is_numeric": is_num,
                "is_categorical": is_cat,
                "is_datetime": is_dt,
            })

        return pd.DataFrame(rows).set_index("column")

    def column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Return a mapping of column names to their inferred semantic type:
        ``\"numeric\"``, ``\"categorical\"``, ``\"datetime"``, or ``\"boolean\"``.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to inspect.

        Returns
        -------
        dict
            ``{column_name: inferred_type}`` for each column.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        result: Dict[str, str] = {}
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_bool_dtype(s):
                result[col] = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(s):
                result[col] = "datetime"
            elif pd.api.types.is_numeric_dtype(s):
                result[col] = "numeric"
            elif (
                pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)
            ):
                result[col] = "categorical"
            else:
                result[col] = "other"
        return result

    def summary(self) -> str:
        """Print or return the text summary of the latest cleaning run."""
        summary_text = self.report_.summary()
        if self.config.verbose:
            print(summary_text)
        return summary_text

    def _repr_html_(self) -> str:
        """Display styled HTML report in Jupyter notebooks."""
        return self.report_._repr_html_()

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        steps_str = ", ".join(self.config.steps)
        return f"<DataFrameCleaner status='{status}', steps=[{steps_str}]>"

    # -------------------------------------------------------------------------
    # Scikit-Learn Estimator Compatibility
    # -------------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator, complying with scikit-learn convention.
        """
        params = asdict(self.config)
        # Exclude non-serializable callables if default
        return params

    def set_params(self, **params: Any) -> "DataFrameCleaner":
        """
        Set parameters of this estimator, complying with scikit-learn convention.
        """
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for DataFrameCleaner")
        self._build_pipeline()
        return self

    # -------------------------------------------------------------------------
    # Model Persistence (Serialization)
    # -------------------------------------------------------------------------

    def save(self, filepath: Union[str, Path]) -> Path:
        """
        Serialize this DataFrameCleaner instance (including all learned statistics) to disk.

        Parameters
        ----------
        filepath : str or Path
            Destination file path (e.g. 'cleaner.pkl').

        Returns
        -------
        Path
            Path to the saved file.
        """
        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Temporarily detach unpicklable prompt_fn if it's a lambda or non-pickleable
        prompt_fn_backup = self.config.prompt_fn
        try:
            try:
                pickle.dumps(self.config.prompt_fn)
            except Exception:
                self.config.prompt_fn = None
            with open(dest, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            self.config.prompt_fn = prompt_fn_backup
        return dest

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DataFrameCleaner":
        """
        Load a serialized DataFrameCleaner instance from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to the saved cleaner file.

        Returns
        -------
        DataFrameCleaner
            Fitted cleaner instance ready for transform().
        """
        src = Path(filepath)
        if not src.exists():
            raise FileNotFoundError(f"CleanFrame model file not found: {src}")
        with open(src, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a DataFrameCleaner (got {type(obj).__name__})")
        return obj


# Alias for intuitive importing as `from data_engine import DataEngine`
DataEngine = DataFrameCleaner


