"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Data Engine — Complete Usage Notebook  (pseudo-notebook)           ║
║  Run this file end-to-end:  python examples/data_engine_notebook.py         ║
╚══════════════════════════════════════════════════════════════════════════════╝

SECTIONS
────────
 0.  Setup & imports
 1.  Build a realistic messy dataset
 2.  Inspect before you touch  ->  profile() & diagnose_dataframe()
 3.  One-liner auto clean       ->  clean_dataframe()
 4.  Automatic mode with target ->  DataEngine(target_column=...)
 5.  Interactive mode           ->  clean_interactive()  [prompt simulation]
 6.  Manual mode                ->  choose only the steps you want
 7.  Granular chaining          ->  df.pipe() method-chain API
 8.  Custom strategies          ->  per-column imputation, zscore, frequency enc
 9.  Train / Test split         ->  fit() + transform()  (no data leakage)
10.  Model persistence          ->  save() / load() for production
11.  Cleaning report            ->  cleaner.report  &  .summary()
12.  column_types()             ->  quick semantic type map
13.  CLI cheatsheet             ->  what you would type in a terminal
14.  Real-world workflow        ->  end-to-end mini ML prep pipeline
"""

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 0 · Setup & imports
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

# Top-level convenience functions
import data_engine as de
from data_engine import (
    DataEngine,             # Primary class
    DataFrameCleaner,       # Class alias
    clean_dataframe,        # One-liner auto clean
    clean_interactive,      # Prompt-based interactive clean
    diagnose_dataframe,     # Inspect without touching data
    profile_dataframe,      # Per-column statistics table
    load_cleaner,           # Restore a saved pipeline from disk
)

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 120)

DIVIDER = "\n" + "=" * 72 + "\n"

print("Data Engine version:", de.__version__)
print("All imports OK.")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 · Build a realistic messy dataset
# ──────────────────────────────────────────────────────────────────────────────
# Imagine this is a raw CSV you just loaded — full of the usual mess:
#   * duplicate rows               (customer 104 appears twice)
#   * missing numeric values       (age has NaNs)
#   * extreme numeric outliers     (age 145, income 1.2 M)
#   * missing categorical values   (loyalty_tier, region)
#   * date columns stored as text  (signup_date)

print(DIVIDER)
print("SECTION 1 — Raw dirty dataset")
print(DIVIDER)

np.random.seed(42)

raw_df = pd.DataFrame({
    "customer_id":  [101, 102, 103, 104, 104, 105, 106, 107, 108, 109, 110, 111],
    "signup_date":  [
        "2023-01-15 08:30", "2023-02-14 12:45", "2023-03-22 17:10",
        "2023-04-05 09:15", "2023-04-05 09:15",        # duplicate row
        "2023-05-18 20:00", "2023-06-30 11:20", "2023-07-04 15:50",
        "2023-08-19 14:10", "2023-09-02 18:30", "2023-10-10 10:05",
        "2023-11-25 19:40",
    ],
    "age":          [24., 31., np.nan, 45., 45., 29., 52., 36., np.nan, 28., 145., 33.],
    "annual_income":[45000, 62000, 58000, 75000, 75000, 51000, 89000, 68000, 72000, 64000, 95000, 1_200_000],
    "loyalty_tier": ["Silver","Gold","Bronze","Gold","Gold", None,"Silver","Platinum","Bronze", None,"Gold","Silver"],
    "region":       ["North","South", None,"East","East","West","North","South","East","West", None,"North"],
    "department":   ["sales","SALES","Marketing","sales","sales","HR","marketing","HR","Sales","marketing","SALES","HR"],
    "churned":      [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],   # target column
})

print(raw_df.to_string())
print(f"\nShape       : {raw_df.shape}")
print(f"Duplicates  : {raw_df.duplicated().sum()}")
print(f"Missing vals: {raw_df.isna().sum().sum()}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 · Inspect before you touch
# ──────────────────────────────────────────────────────────────────────────────
# Always look at the data BEFORE cleaning.
# CleanFrame gives two read-only inspection tools.

print(DIVIDER)
print("SECTION 2A — Per-column profile  (never modifies data)")
print(DIVIDER)

# profile_dataframe() returns a DataFrame — one row per column.
profile = profile_dataframe(raw_df)
print(profile[[
    "dtype", "missing_count", "missing_pct", "unique_count",
    "mean", "min", "max", "is_numeric", "is_categorical", "is_datetime"
]].to_string())

print(DIVIDER)
print("SECTION 2B — Pre-cleaning diagnostic report  (with proposed blueprint)")
print(DIVIDER)

# diagnose_dataframe() audits missing cells, duplicates, outliers, datetimes,
# categoricals, and proposes a step-by-step cleaning blueprint.
diag = diagnose_dataframe(raw_df, display=True, target_column="churned")

print("\nProposed cleaning blueprint as a table:")
print(diag.to_dataframe().to_string(index=False))

print(f"\nHas issues?          {diag.has_issues}")
print(f"Total issues found:  {diag.total_issue_count}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 · One-liner auto clean  ->  clean_dataframe()
# ──────────────────────────────────────────────────────────────────────────────
# Fastest way to get a clean DataFrame. Zero config.
# Runs:  duplicates -> datetime -> missing -> outliers -> encoding

print(DIVIDER)
print("SECTION 3 — One-liner:  clean_dataframe(df)")
print(DIVIDER)

clean = clean_dataframe(raw_df, verbose=False)

print(f"Before : {raw_df.shape}  ->  After : {clean.shape}")
print(f"Any NaN remaining?  {clean.isna().any().any()}")
print(f"\nColumns after cleaning:\n  {list(clean.columns)}")
print(f"\nFirst 3 rows:")
print(clean.head(3).to_string())


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 · Auto mode with target column  ->  DataFrameCleaner(target_column=)
# ──────────────────────────────────────────────────────────────────────────────
# Tell CleanFrame which column is your ML label so it:
#   * skips outlier treatment on it
#   * skips encoding on it
#   * preserves it exactly as-is

print(DIVIDER)
print("SECTION 4 — Auto mode, protecting the target column 'churned'")
print(DIVIDER)

cleaner = DataFrameCleaner(
    target_column="churned",
    verbose=True,
)
auto_df = cleaner.clean(raw_df)

print(f"\nShape  : {raw_df.shape}  ->  {auto_df.shape}")
print(f"'churned' preserved?  {'churned' in auto_df.columns}")
print(f"Unique churned values: {sorted(auto_df['churned'].unique().tolist())}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 · Interactive mode  ->  prompt simulation
# ──────────────────────────────────────────────────────────────────────────────
# In a real notebook this prints the diagnosis and prompts:
#   >>> Should CleanFrame proceed? [y / n / c]
#
# We simulate all three flows below.

print(DIVIDER)
print("SECTION 5A — Interactive: user types 'y'  (confirm all steps)")
print(DIVIDER)

c_yes = DataFrameCleaner(target_column="churned", verbose=False)
result_yes = c_yes.clean(raw_df, interactive=True, prompt_fn=lambda _: "y")
print(f"Cleaned shape: {result_yes.shape}   |  Aborted: {c_yes.aborted}")

print(DIVIDER)
print("SECTION 5B — Interactive: user types 'n'  (abort, data returned unchanged)")
print(DIVIDER)

c_no = DataFrameCleaner(target_column="churned", verbose=False)
result_no = c_no.clean(raw_df, interactive=True, prompt_fn=lambda _: "n")
print(f"Returned shape: {result_no.shape}   |  Aborted: {c_no.aborted}")
print(f"Data unchanged? {result_no.shape == raw_df.shape}")

print(DIVIDER)
print("SECTION 5C — Interactive: user types 'c'  then '1,3'  (customize steps)")
print(DIVIDER)

# Simulate choosing only steps 1 (duplicates) and 3 (missing).
responses = iter(["c", "1,3"])
c_custom = DataFrameCleaner(target_column="churned", verbose=True)
result_custom = c_custom.clean(
    raw_df,
    interactive=True,
    prompt_fn=lambda _: next(responses),
)
print(f"\nOnly 'duplicates' + 'missing' ran  ->  shape: {result_custom.shape}")
print(f"signup_date still raw?   {'signup_date' in result_custom.columns}")
print(f"loyalty_tier still text? {result_custom['loyalty_tier'].dtype}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 · Manual mode  ->  choose only the steps you want
# ──────────────────────────────────────────────────────────────────────────────

print(DIVIDER)
print("SECTION 6 — Manual mode: steps=['duplicates', 'missing', 'outliers']")
print(DIVIDER)

manual_cleaner = DataFrameCleaner(
    steps=["duplicates", "missing", "outliers"],   # skip datetime & encoding
    missing_numeric_strategy="mean",
    missing_categorical_strategy="mode",
    outliers_method="zscore",
    outliers_action="clip",
    outliers_zscore_threshold=2.5,                 # stricter than default 3.0
    verbose=True,
)
manual_df = manual_cleaner.clean(raw_df)

print(f"\nShape : {raw_df.shape}  ->  {manual_df.shape}")
print("signup_date still present (datetime step skipped)?", "signup_date" in manual_df.columns)
print("loyalty_tier still text  (encoding step skipped)?", manual_df["loyalty_tier"].dtype)
print(f"\nAge — before max: {raw_df['age'].max():.1f}  |  after max: {manual_df['age'].max():.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 · Granular chaining  ->  df.pipe() API
# ──────────────────────────────────────────────────────────────────────────────
# Every step is independently callable. Build a fully custom pipeline using
# pandas pipe() for explicit control.

print(DIVIDER)
print("SECTION 7 — Granular df.pipe() chaining")
print(DIVIDER)

c = DataFrameCleaner(verbose=False)

piped_df = (
    raw_df
    .pipe(c.remove_duplicates, keep="first")
    .pipe(c.impute_missing,
          numeric_strategy="median",
          categorical_strategy="constant",
          categorical_fill_value="Unknown")
    .pipe(c.treat_outliers,
          method="iqr",
          action="clip",
          iqr_factor=1.5)
    .pipe(c.extract_datetime_features,
          features=["year", "month", "day", "is_weekend"],
          drop_original=True)
    .pipe(c.encode_categoricals,
          strategy="auto",
          max_one_hot_cardinality=8,
          target_column="churned")
)

print(f"Shape after chaining: {raw_df.shape} -> {piped_df.shape}")
print(f"Columns: {list(piped_df.columns)}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 · Custom strategies  ->  per-column overrides & frequency encoding
# ──────────────────────────────────────────────────────────────────────────────

print(DIVIDER)
print("SECTION 8 — Custom per-column strategies + frequency encoding")
print(DIVIDER)

custom_cleaner = DataFrameCleaner(
    target_column="churned",
    steps=["duplicates", "missing", "outliers", "encoding"],
    # Override imputation per column:
    missing_custom_strategies={
        "age":    "median",    # use median specifically for age
        "region": "North",     # fill region with the literal value "North"
    },
    missing_categorical_strategy="mode",     # fallback for all other cats
    # Frequency encoding: replace each category with its relative frequency.
    # Useful for high-cardinality features or tree-based models.
    encoding_strategy="frequency",
    verbose=True,
)
custom_df = custom_cleaner.clean(raw_df)

print(f"\nShape: {raw_df.shape} -> {custom_df.shape}")
print(f"'loyalty_tier' dtype after freq encoding: {custom_df['loyalty_tier'].dtype}")
print(f"'loyalty_tier' sample values: {custom_df['loyalty_tier'].tolist()}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 · Train / Test split  ->  fit() + transform()  (no data leakage)
# ──────────────────────────────────────────────────────────────────────────────
# CleanFrame follows scikit-learn conventions:
#   fit()       — learns all statistics from training data ONLY
#   transform() — applies learned stats to any DataFrame (train or test)

print(DIVIDER)
print("SECTION 9 — Train / Test split with fit() + transform()")
print(DIVIDER)

train_df = raw_df.iloc[:8].copy().reset_index(drop=True)
test_df  = raw_df.iloc[8:].copy().reset_index(drop=True)

ml_cleaner = DataFrameCleaner(target_column="churned", verbose=False)

# Learn everything from training data ONLY
ml_cleaner.fit(train_df)

# Apply the same learned parameters to both splits
X_train = ml_cleaner.transform(train_df)
X_test  = ml_cleaner.transform(test_df)

print(f"Train : {train_df.shape}  ->  {X_train.shape}")
print(f"Test  : {test_df.shape}   ->  {X_test.shape}")
print(f"\nColumn schemas identical? {list(X_train.columns) == list(X_test.columns)}")

X_train_feats = X_train.drop(columns=["churned"])
y_train       = X_train["churned"]
X_test_feats  = X_test.drop(columns=["churned"])
y_test        = X_test["churned"]
print(f"\nReady for model:")
print(f"  X_train {X_train_feats.shape}  y_train {y_train.shape}")
print(f"  X_test  {X_test_feats.shape}   y_test  {y_test.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 · Model persistence  ->  save() / load()
# ──────────────────────────────────────────────────────────────────────────────

print(DIVIDER)
print("SECTION 10 — Save & load fitted pipeline for production inference")
print(DIVIDER)

import tempfile, os
from pathlib import Path

save_path = Path(tempfile.gettempdir()) / "cleanframe_pipeline.pkl"
ml_cleaner.save(save_path)
print(f"Saved  : {save_path}  ({save_path.stat().st_size:,} bytes)")

prod_cleaner = load_cleaner(save_path)
print(f"Loaded : {prod_cleaner}")

# Transform a brand new incoming batch using ONLY learned training parameters
new_batch = pd.DataFrame({
    "customer_id":  [999, 1000],
    "signup_date":  ["2024-01-20 10:00", "2024-03-15 14:30"],
    "age":          [27., np.nan],
    "annual_income":[58000, 71000],
    "loyalty_tier": ["Gold", None],
    "region":       ["East", None],
    "department":   ["HR", "sales"],
    "churned":      [0, 1],
})

production_output = prod_cleaner.transform(new_batch)
print(f"\nNew batch: {new_batch.shape} -> {production_output.shape}")
print(f"Columns match training? {list(production_output.columns) == list(X_train.columns)}")
os.remove(save_path)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11 · Cleaning report  ->  cleaner.report & .summary()
# ──────────────────────────────────────────────────────────────────────────────

print(DIVIDER)
print("SECTION 11 — Post-cleaning report")
print(DIVIDER)

reporter = DataFrameCleaner(target_column="churned", verbose=False)
reporter.clean(raw_df)
report = reporter.report

# A. Table form
print("Report as a DataFrame:")
print(report.to_dataframe().to_string(index=False))

# B. Key metrics
print(f"\nDuplicates removed  : {report.duplicates_removed}")
print(f"Shape change        : {report.initial_shape} -> {report.final_shape}")
print(f"Missing before/after: {report.initial_missing} -> {report.final_missing}")

# C. Full text summary
print("\nText summary:")
print(reporter.summary())

# D. In Jupyter notebooks just type:  reporter   or   reporter.report
html = reporter._repr_html_()
print(f"HTML Jupyter card: {len(html):,} chars   (renders automatically in notebooks)")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12 · column_types()  ->  quick semantic type map
# ──────────────────────────────────────────────────────────────────────────────

print(DIVIDER)
print("SECTION 12 — column_types(): semantic type map")
print(DIVIDER)

c12 = DataFrameCleaner(verbose=False)
types = c12.column_types(raw_df)
for col, t in types.items():
    print(f"  {col:<18} ->  {t}")

numeric_cols     = [col for col, t in types.items() if t == "numeric"]
categorical_cols = [col for col, t in types.items() if t == "categorical"]
print(f"\nNumeric cols    : {numeric_cols}")
print(f"Categorical cols: {categorical_cols}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 13 · CLI cheatsheet  ->  what you would type in a terminal
# ──────────────────────────────────────────────────────────────────────────────

print(DIVIDER)
print("SECTION 13 — CLI cheatsheet  (run these in your terminal, not in Python)")
print(DIVIDER)

print("""
  # Preview data quality issues — NO changes made:
  python -m data_engine my_data.csv --preview

  # Interactive mode (shows diagnosis then asks y / n / c):
  python -m data_engine my_data.csv

  # Fully automatic — no prompts, save result:
  python -m data_engine my_data.csv --yes --output cleaned.csv

  # Protect a target column from encoding:
  python -m data_engine my_data.csv --yes --target churned --output out.csv

  # Run only specific steps:
  python -m data_engine my_data.csv --steps duplicates,missing --yes

  # Tune strategies from the command line:
  python -m data_engine my_data.csv --missing-strategy mean
      --outliers-action drop --encoding label --yes --output out.csv
""")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 14 · Real-world workflow  ->  end-to-end mini ML prep pipeline
# ──────────────────────────────────────────────────────────────────────────────

print(DIVIDER)
print("SECTION 14 — Real-world end-to-end ML prep workflow")
print(DIVIDER)

# 14.1  Peek at raw data
print("14.1  Peek at raw data")
print(f"  Shape: {raw_df.shape}  |  Missing: {raw_df.isna().sum().sum()}  |  Dups: {raw_df.duplicated().sum()}")

# 14.2  Non-destructive profile to understand the data
print("\n14.2  Profile to understand the data")
p = profile_dataframe(raw_df)
high_missing = p[p["missing_pct"] > 10]
if not high_missing.empty:
    print("  Columns with > 10% missing:")
    print(high_missing[["dtype", "missing_count", "missing_pct"]].to_string())
else:
    print("  No columns with > 10% missing.")

# 14.3  Diagnose without cleaning
print("\n14.3  Diagnose without cleaning")
plan = diagnose_dataframe(raw_df, display=False, target_column="churned")
print(f"  Issues detected : {plan.total_issue_count}")
print(f"  Planned actions : {len(plan.action_plan)}")

# 14.4  Fit cleaner on training data ONLY
print("\n14.4  Fit pipeline on training split")
train_split = raw_df.sample(frac=0.75, random_state=99).reset_index(drop=True)
test_split  = raw_df.drop(train_split.index).reset_index(drop=True)

final_cleaner = DataFrameCleaner(
    target_column="churned",
    steps=["duplicates", "datetime", "missing", "outliers", "encoding"],
    missing_numeric_strategy="median",
    missing_custom_strategies={"region": "North"},   # domain knowledge override
    outliers_method="iqr",
    outliers_action="clip",
    encoding_strategy="auto",
    max_one_hot_cardinality=6,
    verbose=False,
)
final_cleaner.fit(train_split)
print(f"  Fitted on {len(train_split)} rows.")

# 14.5  Transform both splits
X_tr = final_cleaner.transform(train_split)
X_te = final_cleaner.transform(test_split)
print(f"\n14.5  Transform results")
print(f"  Train: {train_split.shape} -> {X_tr.shape}")
print(f"  Test : {test_split.shape}  -> {X_te.shape}")
print(f"  Column alignment: {list(X_tr.columns) == list(X_te.columns)}")

# 14.6  Save for reproducibility
save_p = Path(tempfile.gettempdir()) / "final_cleaner.pkl"
final_cleaner.save(save_p)
print(f"\n14.6  Pipeline saved ({save_p.stat().st_size:,} bytes)")
os.remove(save_p)

# 14.7  Report snapshot
r = final_cleaner.report
print(f"\n14.7  Cleaning report snapshot")
print(f"  Duplicates removed  : {r.duplicates_removed}")
print(f"  Missing cells fixed : {r.initial_missing - r.final_missing}")
print(f"  Outliers handled    : {sum(v.get('count',0) for v in r.outlier_actions.values())}")
print(f"  Columns encoded     : {list(r.encoded_columns.keys())}")

print(DIVIDER)
print("  Notebook complete — all 14 sections ran successfully!")
print(DIVIDER)
