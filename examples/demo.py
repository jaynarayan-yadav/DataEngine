"""
Data Engine Interactive Demonstration
Shows Pre-Cleaning Diagnosis, User Confirmation ("Should I clean or not?"),
Automatic Mode, Manual Customization, and .pipe() Method Chaining.
"""

import sys
import numpy as np
import pandas as pd
from data_engine import DataFrameCleaner, DataEngine, clean_dataframe, clean_interactive, diagnose_dataframe


def create_sample_dataset() -> pd.DataFrame:
    """Generate a realistic messy dataset with duplicates, NaNs, outliers, dates, and categories."""
    np.random.seed(42)
    
    data = {
        "customer_id": [101, 102, 103, 104, 104, 105, 106, 107, 108, 109, 110, 111],
        "signup_date": [
            "2023-01-15 08:30:00", "2023-02-14 12:45:00", "2023-03-22 17:10:00",
            "2023-04-05 09:15:00", "2023-04-05 09:15:00",  # Duplicate row
            "2023-05-18 20:00:00", "2023-06-30 11:20:00", "2023-07-04 15:50:00",
            "2023-08-19 14:10:00", "2023-09-02 18:30:00", "2023-10-10 10:05:00",
            "2023-11-25 19:40:00"
        ],
        "age": [24.0, 31.0, np.nan, 45.0, 45.0, 29.0, 52.0, 36.0, np.nan, 28.0, 145.0, 33.0],  # NaN + Outlier 145
        "annual_income": [45000, 62000, 58000, 75000, 75000, 51000, 89000, 68000, 72000, 64000, 95000, 1200000],  # Outlier 1.2M
        "loyalty_tier": ["Silver", "Gold", "Bronze", "Gold", "Gold", None, "Silver", "Platinum", "Bronze", None, "Gold", "Silver"],
        "city": ["New York", "Chicago", "Boston", "Austin", "Austin", "Chicago", "New York", "Seattle", "Denver", "Boston", "Austin", "New York"],
        "churned": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


def main():
    print("===================================================================")
    print("              DATA ENGINE: DEMO & WORKFLOW SHOWCASE                ")
    print("===================================================================\n")

    raw_df = create_sample_dataset()
    print("1. RAW DIRTY DATAFRAME:")
    print("-------------------------------------------------------------------")
    print(raw_df)
    print(f"\nInitial Shape: {raw_df.shape}")
    print(f"Missing Values: {raw_df.isna().sum().sum()}")
    print(f"Duplicate Rows: {raw_df.duplicated().sum()}")
    print("\n" + "="*67 + "\n")

    # -------------------------------------------------------------------------
    # FEATURE 1: Pre-Cleaning Diagnosis & Issue Inspection
    # -------------------------------------------------------------------------
    print("2. PRE-CLEANING DIAGNOSIS & BLUEPRINT:")
    print("-------------------------------------------------------------------")
    diag_report = diagnose_dataframe(raw_df, display=True, target_column="churned")

    print("\nPROPOSED ACTION TABLE:")
    print(diag_report.to_dataframe().to_string(index=False))
    print("\n" + "="*67 + "\n")

    # -------------------------------------------------------------------------
    # FEATURE 2: Interactive Confirmation Mode ("Should I clean or not?")
    # -------------------------------------------------------------------------
    print("3. INTERACTIVE CONFIRMATION DEMO:")
    print("-------------------------------------------------------------------")
    print("Testing interactive confirmation with simulated user approval ('y'):\n")
    
    cleaner = DataFrameCleaner(target_column="churned", verbose=True)
    # Simulate user confirmation via prompt_fn
    interactive_df = cleaner.clean_interactive(raw_df, prompt_fn=lambda prompt: "y")
    print(f"\nInteractive cleanup result shape: {interactive_df.shape}")
    print(f"Cleaner aborted: {cleaner.aborted}")

    print("\nTesting interactive abort when user declines ('n'):\n")
    aborted_cleaner = DataFrameCleaner(target_column="churned", verbose=True)
    aborted_df = aborted_cleaner.clean_interactive(raw_df, prompt_fn=lambda prompt: "n")
    print(f"Cleaner aborted: {aborted_cleaner.aborted}")
    print(f"Original shape preserved: {aborted_df.shape == raw_df.shape}")

    print("\n" + "="*67 + "\n")

    # -------------------------------------------------------------------------
    # FEATURE 3: Manual Mode (Selected Steps & Custom Strategies)
    # -------------------------------------------------------------------------
    print("4. RUNNING MANUAL MODE (Selected steps & custom parameters):")
    print("-------------------------------------------------------------------")
    manual_cleaner = DataFrameCleaner(
        steps=["duplicates", "missing", "outliers"],  # Skip datetime and encoding
        missing_numeric_strategy="mean",
        missing_categorical_strategy="mode",
        outliers_method="zscore",
        outliers_action="clip",
        outliers_zscore_threshold=2.5,
        verbose=True
    )
    manual_df = manual_cleaner.clean(raw_df)
    print("\nResult of Manual Selected Steps:")
    print(manual_df[["customer_id", "age", "annual_income", "loyalty_tier"]].head())

    print("\n" + "="*67 + "\n")

    # -------------------------------------------------------------------------
    # FEATURE 4: Granular Step Execution & DataFrame Chaining (.pipe)
    # -------------------------------------------------------------------------
    print("5. GRANULAR FUNCTIONAL / CHAINING API (df.pipe):")
    print("-------------------------------------------------------------------")
    custom_cleaner = DataFrameCleaner(verbose=False)
    
    chained_df = (
        raw_df
        .pipe(custom_cleaner.remove_duplicates)
        .pipe(custom_cleaner.impute_missing, numeric_strategy="median", categorical_fill_value="Unknown")
        .pipe(custom_cleaner.treat_outliers, method="iqr", action="clip")
        .pipe(custom_cleaner.extract_datetime_features, features=["year", "month", "day", "is_weekend"])
    )
    print("Cleaned via chained .pipe() methods:")
    print(chained_df.columns.tolist())
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
