"""
Command-line interface for data_engine.
Provides interactive pre-cleaning diagnosis, confirmation, and automatic data preprocessing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd

from .cleaner import DataFrameCleaner
from .config import CleaningConfig
from .diagnostics import DatasetDiagnostics


BANNER = r"""
   ____ _                  _____                      
  / ___| | ___  __ _ _ __ |  ___| __ __ _ _ __ ___   ___
 | |   | |/ _ \/ _` | '_ \| |_ | '__/ _` | '_ ` _ \ / _ \
 | |___| |  __/ (_| | | | |  _|| | | (_| | | | | | |  __/
  \____|_|\___|\__,_|_| |_|_|  |_|  \__,_|_| |_| |_|\___|
        Data Cleaning & Preprocessing Engine
"""


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load dataset from disk based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(file_path)
    elif suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    elif suffix == ".parquet":
        return pd.read_parquet(file_path)
    elif suffix == ".json":
        return pd.read_json(file_path)
    else:
        # Fallback to csv reader
        return pd.read_csv(file_path)


def save_dataset(df: pd.DataFrame, file_path: Path) -> None:
    """Save cleaned dataset to disk."""
    suffix = file_path.suffix.lower()
    if suffix in (".csv", ".txt"):
        df.to_csv(file_path, index=False)
    elif suffix == ".tsv":
        df.to_csv(file_path, sep="\t", index=False)
    elif suffix in (".xlsx", ".xls"):
        df.to_excel(file_path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(file_path, index=False)
    elif suffix == ".json":
        df.to_json(file_path, orient="records", indent=2)
    else:
        df.to_csv(file_path, index=False)


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="cleanframe",
        description="CleanFrame: Pre-cleaning diagnosis, inspection, and automated pandas data preprocessing.",
    )
    parser.add_argument("input_file", help="Path to input data file (.csv, .xlsx, .parquet, .tsv, .json)")
    parser.add_argument("-o", "--output", help="Path to save the cleaned data file. (e.g. -o clean_data.csv)")
    parser.add_argument(
        "-p", "--preview",
        action="store_true",
        help="Preview-only mode: Run diagnostic inspection and show proposed cleaning plan without modifying data.",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Non-interactive mode: Automatically approve and run all cleaning steps without asking.",
    )
    parser.add_argument(
        "-t", "--target",
        help="Target column name (will be protected from encoding and aggressive transformations).",
    )
    parser.add_argument(
        "--steps",
        help="Comma-separated cleaning steps to run (e.g. 'duplicates,missing,outliers').",
    )
    parser.add_argument(
        "--missing-strategy",
        choices=["median", "mean", "mode", "constant", "drop_rows", "none"],
        default="median",
        help="Numeric missing value strategy (default: median).",
    )
    parser.add_argument(
        "--outliers-action",
        choices=["clip", "drop", "nan", "none"],
        default="clip",
        help="Action for detected numeric outliers (default: clip).",
    )
    parser.add_argument(
        "--encoding",
        choices=["auto", "onehot", "label", "frequency", "none"],
        default="auto",
        help="Categorical encoding strategy (default: auto).",
    )

    args = parser.parse_args(argv)

    input_path = Path(args.input_file)
    if not input_path.exists():
        sys.stderr.write(f"Error: File not found: {args.input_file}\n")
        return 1

    print(BANNER)
    print(f"Loading data from: {input_path} ...")
    try:
        df = load_dataset(input_path)
    except Exception as exc:
        sys.stderr.write(f"Error reading file '{input_path}': {exc}\n")
        return 1

    steps = [s.strip() for s in args.steps.split(",")] if args.steps else None

    cleaner = DataFrameCleaner(
        steps=steps,
        target_column=args.target,
        missing_numeric_strategy=args.missing_strategy,
        outliers_action=args.outliers_action,
        encoding_strategy=args.encoding,
        verbose=True,
    )

    # 1. Run Pre-Cleaning Diagnosis
    report = cleaner.diagnose(df, display=True)

    if args.preview:
        print("\n[Preview mode] Pre-cleaning diagnosis complete. No changes were made.")
        return 0

    # 2. Interactive confirmation
    if not args.yes:
        clean_df = cleaner.clean(df, interactive=True)
    else:
        print("\n[--yes specified] Auto-confirming cleaning pipeline...")
        clean_df = cleaner.clean(df, interactive=False)

    if cleaner.aborted:
        print("\nCleaning was aborted. No output file written.")
        return 0

    # 3. Display post-cleaning summary
    print("\n" + cleaner.report.summary())

    # 4. Save cleaned output if specified
    out_file = args.output
    if not out_file and not args.yes:
        # Prompt user if they'd like to save
        try:
            save_prompt = input("\n>>> Enter output filepath to save cleaned data (or press Enter to skip): ").strip()
            if save_prompt:
                out_file = save_prompt
        except (EOFError, KeyboardInterrupt):
            out_file = None

    if out_file:
        out_path = Path(out_file)
        save_dataset(clean_df, out_path)
        print(f"\n[cleanframe] Successfully saved cleaned DataFrame ({clean_df.shape[0]} rows, {clean_df.shape[1]} cols) to: {out_path.resolve()}")
    else:
        print("\n[cleanframe] Cleaning finished successfully (no output file saved).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
