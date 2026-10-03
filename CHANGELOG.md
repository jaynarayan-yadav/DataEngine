# Changelog

All notable changes to Data Engine (`data_engine`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `DataEngine` class alias (`from data_engine import DataEngine`), serving as primary entry point alongside `DataFrameCleaner`.
- `profile()` method on cleaner returning a rich per-column statistics table
  (dtype, missing %, unique %, mean, median, std, min, max, mode, type flags).
- `profile_dataframe()` top-level convenience function, mirroring the API style of
  `clean_dataframe()` and `diagnose_dataframe()`.
- `column_types()` helper method to get a `{col: inferred_type}` mapping.
- LICENSE (MIT) file.

### Fixed
- `data_engine/__init__.py`: missing `Union` import fixed for type annotations.

---

## [0.1.0] - 2026-10-03

### Added
- `DataEngine` / `DataFrameCleaner` — main cleaner class with `fit`, `transform`, `fit_transform`, `clean`, `clean_interactive`, `diagnose`, and `preview` methods.
- Full five-step automatic cleaning pipeline:
  - **Duplicates**: drop duplicate rows (subset, keep strategy).
  - **Datetime**: auto-detect and extract calendar features from date columns.
  - **Missing**: median/mean/mode/constant imputation for numeric and categorical columns; column-level threshold-based dropping; per-column custom strategies.
  - **Outliers**: IQR Tukey fences and Z-score with clip, drop, or NaN replacement.
  - **Encoding**: auto one-hot vs. label encoding by cardinality; frequency encoding; consistent train/test column alignment to prevent data leakage.
- Interactive mode: pre-cleaning diagnostic summary printed before any transformation, followed by a `[y / n / c]` prompt to confirm, abort, or customize steps.
- `clean_dataframe()`, `clean_interactive()`, `diagnose_dataframe()`, `profile_dataframe()`, `load_cleaner()` — functional top-level convenience wrappers.
- `CleaningConfig` dataclass with strongly typed, fully documented defaults.
- `CleaningReport` — post-cleaning structured report with `to_dataframe()` and HTML repr.
- `PreCleaningReport` — pre-cleaning diagnostic report with terminal and Jupyter HTML summary.
- `DatasetDiagnostics` — standalone DataFrame inspector producing `PreCleaningReport`.
- Model persistence via `cleaner.save()` / `cleaner.load()` (pickle serialization).
- scikit-learn estimator API: `get_params()`, `set_params()`, compatible with `Pipeline`.
- CLI (`python -m data_engine my_data.csv`) with `--preview`, `--yes`, `--output`, `--target`, `--steps`, `--missing-strategy`, `--outliers-action`, and `--encoding` flags.
- BaseCleanerStep ABC ensuring every step follows `fit` / `transform` semantics.
- 67 automated tests covering all modules, edge cases, interactive flows, CLI, profiling, and scikit-learn compatibility.
- `examples/demo.py` — runnable walkthrough of all features.
- `examples/data_engine_notebook.py` — 14-section comprehensive usage guide script.
