"""
Tests for cleanframe command line interface.
"""

from pathlib import Path
import pandas as pd
import pytest

from data_engine.cli import main


@pytest.fixture
def temp_csv(tmp_path):
    csv_file = tmp_path / "test_dirty.csv"
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "age": [20.0, np_nan := None, None, 40.0],
        "category": ["A", "B", "B", "C"]
    })
    df.to_csv(csv_file, index=False)
    return csv_file


def test_cli_preview(temp_csv, capsys):
    code = main([str(temp_csv), "--preview"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "CLEANFRAME PRE-CLEANING DIAGNOSIS" in captured
    assert "PROPOSED CLEANING BLUEPRINT" in captured
    assert "[Preview mode]" in captured


def test_cli_auto_confirm_and_output(temp_csv, tmp_path):
    out_file = tmp_path / "cleaned_result.csv"
    code = main([str(temp_csv), "--yes", "-o", str(out_file)])
    assert code == 0
    assert out_file.exists()

    cleaned = pd.read_csv(out_file)
    # Duplicates removed (4 -> 3)
    assert len(cleaned) == 3
    # Age missing values imputed
    assert cleaned["age"].isna().sum() == 0


def test_cli_file_not_found(capsys):
    code = main(["non_existent_file_12345.csv"])
    assert code == 1
    err = capsys.readouterr().err
    assert "File not found" in err


def test_cli_interactive_abort(temp_csv, monkeypatch, capsys):
    # Simulate user entering 'n' to abort
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    code = main([str(temp_csv)])
    assert code == 0
    captured = capsys.readouterr().out
    assert "Cleaning was aborted" in captured
