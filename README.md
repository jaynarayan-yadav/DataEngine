# Data Engine ⚙️

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-67%20passed-brightgreen.svg)]()

> **Data Engine** (`data_engine`) is a lightweight, intuitive Python package that automatically diagnoses, cleans, and preprocesses pandas DataFrames for data science and machine learning projects.

Before modifying your dataset, Data Engine can **inspect, profile, and diagnose** data quality issues, present an informative pre-cleaning diagnostic summary with a proposed blueprint, and **ask you whether to proceed or customize steps**.

---

## Key Features

- 🔍 **Pre-Cleaning Diagnosis & Data Profiling**:
  - `profile_dataframe()` / `.profile()`: Per-column summary statistics (missing %, unique count, min/max/mean/std, mode, type mapping).
  - `diagnose_dataframe()`: Audits missing values, duplicates, numeric outliers, datetime strings, and categorical columns.
  - Interactive Jupyter notebook HTML cards and printable terminal summary tables.
- ❓ **Interactive User Confirmation ("Ask me should I clean or not")**:
  - Prompts you with the proposed action plan before applying any transformations (`y` = proceed, `n` = abort, `c` = customize steps).
  - Clean safely: aborting returns your original DataFrame 100% untouched without mutations.
- 🔄 **Duplicate Rows**: Automatically detects and removes duplicates (first, last, or all) across all columns or a specified subset.
- 📅 **Datetime Feature Extraction**: Auto-detects datetime columns and extracts informative calendar features (`year`, `month`, `day`, `day_of_week`, `is_weekend`, `hour`, `quarter`).
- 🩹 **Missing Value Imputation**:
  - **Numeric**: `median` (default, robust to outliers), `mean`, `mode`, `constant`, or `drop_rows`.
  - **Categorical**: `constant` (default `'missing'`), `mode`, or `drop_rows`.
  - **Thresholding**: Option to drop columns with missing ratios exceeding a threshold.
  - **Custom Overrides**: Column-specific strategies or fill values.
- 📈 **Outlier Handling**:
  - **Methods**: IQR (Interquartile Range) Tukey fences or Z-score boundaries.
  - **Actions**: `clip` / winsorization (default, preserves sample size for ML), `drop` rows, or replace with `nan`.
- 🏷️ **Categorical Encoding**:
  - **Smart Auto**: One-hot encodes low-cardinality features and label/frequency encodes high-cardinality features.
  - **Consistent Train/Test Alignment**: Fits category mappings and one-hot levels on training data to prevent data leakage and guarantee column alignment during inference.
- 💻 **Complete Command Line Interface (CLI)**:
  - Run diagnostics and clean CSV, Excel, Parquet, or JSON files directly from your terminal.
- 📊 **Flexible Operating Modes**:
  - **Interactive Mode**: Shows pre-cleaning diagnosis and asks for confirmation.
  - **Automatic Mode**: Full end-to-end cleanup pipeline with zero configuration required.
  - **Manual Mode**: Choose only the steps you want, configure custom strategies, or chain individual methods using `df.pipe()`.

---

## Installation

Install the package directly into your environment:

```bash
pip install -e .
```

Dependencies: `pandas >= 2.0.0`, `numpy >= 1.24.0`.

---

## Quickstart

### 1. Interactive Mode: Inspect & Confirm Before Cleaning

```python
import pandas as pd
from data_engine import clean_interactive

df = pd.read_csv("data.csv")

# Displays pre-cleaning diagnostic summary and asks:
# "Should Data Engine proceed with cleaning? [y/N/customize]:"
clean_df = clean_interactive(df)
```

Or using `DataEngine` (or `DataFrameCleaner`):

```python
from data_engine import DataEngine

cleaner = DataEngine(interactive=True)
clean_df = cleaner.clean(df)
```

If you confirm (`y`), Data Engine executes the planned pipeline and returns the clean DataFrame.  
If you decline (`n`), Data Engine cancels immediately and returns your original data unmodified.  
If you choose customize (`c`), Data Engine lets you select exactly which steps to run (e.g. `1,3` for duplicates and missing values).

---

### 2. Pre-Cleaning Diagnosis Only (No Modifications)

Preview all data quality issues, metrics, and proposed actions without touching your data:

```python
from data_engine import diagnose_dataframe, profile_dataframe

# Rich per-column statistical profile summary table:
profile_df = profile_dataframe(df)

# Formatted diagnostic issue report:
report = diagnose_dataframe(df)

# Or inspect as a pandas DataFrame:
report.to_dataframe()
```

Sample Diagnostic Output:
```text
===========================================================
             DATA ENGINE PRE-CLEANING DIAGNOSIS            
===========================================================
Dataset Dimensions:    12 rows x 7 columns
Estimated Memory:      2.5 KB
Duplicate Rows:        1 (8.3% of dataset)
Total Missing Cells:   4 (4.8% of all cells)
Total Outliers:        2 values detected
-----------------------------------------------------------
MISSING VALUES BREAKDOWN:
  - age (float64): 2 missing (16.7%) -> Planned: Impute median
  - loyalty_tier (str): 2 missing (16.7%) -> Planned: Impute constant ('missing')

OUTLIERS BREAKDOWN:
  - age: 1 outliers (10.0%) bounds: [6.25, 68.25] -> Planned: clip outliers via iqr
  - annual_income: 1 outliers (8.3%) bounds: [4875.00, 141875.00] -> Planned: clip outliers via iqr

DATETIME EXTRACTION CANDIDATES:
  - signup_date (str): recognized date/time format -> Planned: Extract 5 features

CATEGORICAL ENCODING CANDIDATES:
  - loyalty_tier (str): cardinality 4 -> Planned: One-hot encode (4 binary cols)
  - city (str): cardinality 6 -> Planned: One-hot encode (6 binary cols)
-----------------------------------------------------------
PROPOSED CLEANING BLUEPRINT:
  1. [DUPLICATES] [All rows]: Drop duplicates (keep 'first')
  2. [DATETIME] signup_date: Extract calendar features (year, month, day, day_of_week, is_weekend)
  3. [MISSING] age: Impute median
  4. [OUTLIERS] age: Clip outliers (iqr)
  5. [OUTLIERS] annual_income: Clip outliers (iqr)
  6. [MISSING] loyalty_tier: Impute constant ('missing')
  7. [ENCODING] loyalty_tier: One-hot encode (4 binary cols)
  8. [ENCODING] city: One-hot encode (6 binary cols)
===========================================================
```

---

### 3. Automatic 1-Line Cleanup (Non-Interactive)

```python
import pandas as pd
from data_engine import DataEngine

cleaner = DataEngine()
clean_df = cleaner.clean(df)
```

Or using the functional wrapper:

```python
from data_engine import clean_dataframe

clean_df = clean_dataframe(df)
```

---

## Command Line Interface (CLI)

Data Engine includes a command-line tool to inspect and clean dataset files directly from your shell:

### 1. Interactive Inspection & Cleaning
```bash
python -m data_engine my_dataset.csv
```
Displays the pre-cleaning diagnosis, asks `Should Data Engine proceed with cleaning? [y/n/c]:`, and asks for the destination file path.

### 2. Preview Only (No Changes)
```bash
python -m data_engine my_dataset.csv --preview
```

### 3. Automatic Cleaning & Direct Export
```bash
python -m data_engine my_dataset.csv --yes --output cleaned_dataset.csv
```

### 4. Protect Target Column from Transformation
```bash
python -m data_engine my_dataset.csv --target churned --output cleaned.csv
```

---

## Jupyter Notebook Usage

In a Jupyter notebook, simply run `cleaner` or `cleaner.report` in a cell to view an interactive visual summary card:

```python
cleaner = DataEngine(target_column="target")
clean_df = cleaner.clean(df)

# Displays styled summary card with metrics:
cleaner
```

You can also inspect the operations as a standard pandas DataFrame:

```python
cleaner.report.to_dataframe()
```

| Step | Column | Action | Count / Impact | Details |
|---|---|---|---|---|
| Duplicates | [All rows] | Drop duplicates | 1 | Dropped 1 duplicate row(s) |
| Datetime | signup_date | Extract datetime features | 5 | Created 5 features: year, month, day... |
| Missing | age | Impute median | 2 | Filled with 33.0 |
| Missing | loyalty_tier | Impute constant | 2 | Filled with 'missing' |
| Outliers | age | Clip outliers | 1 | Capped 1 values to [14.25, 56.25] |
| Encoding | city | One-hot encode | 5 | Created 5 indicator columns |

---

## Operating Modes

### 1. Manual Mode (Select Specific Steps & Custom Strategies)

Pick only the steps you want:

```python
cleaner = DataEngine(
    steps=["duplicates", "missing"],
    missing_numeric_strategy="mean",
    missing_categorical_strategy="mode",
    verbose=True
)
clean_df = cleaner.clean(df)
```

Customize outlier detection and encoding parameters:

```python
cleaner = DataEngine(
    outliers_method="zscore",
    outliers_action="clip",
    outliers_zscore_threshold=2.5,
    datetime_features=["year", "month", "day", "is_weekend"],
    encoding_strategy="auto",
    max_one_hot_cardinality=5
)
clean_df = cleaner.clean(df)
```

---

### 2. Granular Step Chaining (`df.pipe`)

Every step is also available as an individual method for pandas method chaining:

```python
cleaner = DataEngine(verbose=False)

clean_df = (
    df
    .pipe(cleaner.remove_duplicates, keep="first")
    .pipe(cleaner.impute_missing, numeric_strategy="median", categorical_fill_value="Unknown")
    .pipe(cleaner.treat_outliers, method="iqr", action="clip", iqr_factor=1.5)
    .pipe(cleaner.extract_datetime_features, features=["year", "month", "day", "is_weekend"])
    .pipe(cleaner.encode_categoricals, strategy="auto", max_one_hot_cardinality=8)
)
```

---

### 3. Machine Learning: Train / Test Split (No Data Leakage)

Data Engine follows scikit-learn transformer conventions (`fit`, `transform`, `fit_transform`). This guarantees that statistics learned on the training split (medians, one-hot category levels, outlier bounds) are applied consistently to the test or inference split:

```python
from sklearn.model_selection import train_test_split
from data_engine import DataEngine

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

cleaner = DataEngine(target_column="label")

# Fit on train set ONLY
cleaner.fit(train_df)

# Transform both splits using learned parameters
X_train = cleaner.transform(train_df)
X_test = cleaner.transform(test_df)

# Guaranteed identical feature columns
assert list(X_train.columns) == list(X_test.columns)
```

---

### 4. Model Persistence: Save & Load for Production

Save the entire fitted pipeline with all learned parameters (medians, categories, outlier fences) to disk and load it in any production API:

```python
from data_engine import DataEngine, load_cleaner

# Fit on training data
cleaner = DataEngine(target_column="churned")
cleaner.fit(train_df)

# Save to disk
cleaner.save("cleaner_pipeline.pkl")

# Load in production inference service
prod_cleaner = load_cleaner("cleaner_pipeline.pkl")
clean_batch = prod_cleaner.transform(new_incoming_df)
```

Data Engine also implements `get_params()` and `set_params()` for seamless integration into scikit-learn `Pipeline` and hyperparameter search tools:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ("cleaner", DataEngine(target_column="label")),
    ("model", RandomForestClassifier())
])
```

---

## Configuration Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `interactive` | `bool` | `False` | Display pre-cleaning diagnostic summary and ask user confirmation before cleaning. |
| `steps` | `list[str]` | `["duplicates", "datetime", "missing", "outliers", "encoding"]` | List and order of steps to execute. |
| `drop_duplicates` | `bool` | `True` | Whether to remove duplicate rows. |
| `duplicates_subset` | `list[str]` | `None` | Columns to consider for duplicates (or all if None). |
| `duplicates_keep` | `{"first", "last", False}` | `"first"` | Which duplicate occurrence to keep. |
| `missing_numeric_strategy` | `{"median", "mean", "mode", "constant", "drop_rows", "none"}` | `"median"` | Numeric imputation strategy. |
| `missing_numeric_fill_value` | `float` | `0.0` | Fill value when numeric strategy is `"constant"`. |
| `missing_categorical_strategy`| `{"constant", "mode", "drop_rows", "none"}` | `"constant"` | Categorical imputation strategy. |
| `missing_categorical_fill_value`| `str` | `"missing"` | Fill value when categorical strategy is `"constant"`. |
| `missing_column_threshold` | `float` | `None` | Drop columns with missing fraction > threshold (e.g. 0.8). |
| `missing_custom_strategies` | `dict` | `None` | Per-column overrides, e.g. `{"age": "mean", "city": "NYC"}`. |
| `outliers_method` | `{"iqr", "zscore", "none"}` | `"iqr"` | Outlier boundary calculation method. |
| `outliers_action` | `{"clip", "drop", "nan", "none"}` | `"clip"` | Action on outliers (`"clip"` preserves row count). |
| `outliers_iqr_factor` | `float` | `1.5` | IQR multiplier (1.5 for mild, 3.0 for extreme). |
| `outliers_zscore_threshold` | `float` | `3.0` | Z-score threshold for outliers. |
| `outliers_columns` | `list[str]` | `None` | Numeric columns to inspect (auto-detects if None). |
| `datetime_columns` | `list[str]` | `None` | Datetime columns to process (auto-detects if None). |
| `datetime_features` | `list[str]` | `["year", "month", "day", "day_of_week", "is_weekend"]` | Calendar features to extract. |
| `datetime_drop_original` | `bool` | `True` | Whether to drop raw datetime columns after extraction. |
| `encoding_strategy` | `{"auto", "onehot", "label", "frequency", "none"}` | `"auto"` | Categorical encoding strategy. |
| `max_one_hot_cardinality` | `int` | `10` | Max unique values for one-hot encoding under `"auto"`. |
| `encoding_drop_first` | `bool` | `False` | Whether to drop first dummy level in one-hot encoding. |
| `target_column` | `str` | `None` | Column name to exclude from encoding or aggressive transformations. |
| `verbose` | `bool` | `True` | Whether to display step-by-step cleaning logs. |

---

## Running Examples & Tests

Run the interactive notebook workflow script:

```bash
python examples/data_engine_notebook.py
```

Run the interactive demo script:

```bash
python examples/demo.py
```

Run the automated test suite:

```bash
python -m pytest tests -v
```

---

## License

MIT License. Data Engine is open-source and free for academic and commercial use.
