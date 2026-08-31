"""tests/test_loader.py — Unit tests for src/preprocessing/loader.py
Extended in Sprint 1 final quality review with edge-case CSV tests.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.loader import load_raw_unswnb15


def _make_config(raw_dir, train_file="train.csv", test_file="test.csv"):
    return {
        "data": {
            "raw_dir": str(raw_dir),
            "train_file": train_file,
            "test_file": test_file,
            "header": True,
            "encoding": "utf-8",
        }
    }


@pytest.fixture
def csv_dir(tmp_path):
    """Create minimal synthetic CSVs that mimic UNSW-NB15 layout."""
    train_content = "label,attack_cat,feat1\n0,Normal,1.0\n1,Backdoor,2.0\n"
    test_content = "label,attack_cat,feat1\n0,Normal,3.0\n1,DoS,4.0\n"
    (tmp_path / "train.csv").write_text(train_content, encoding="utf-8")
    (tmp_path / "test.csv").write_text(test_content, encoding="utf-8")
    return tmp_path


class TestLoadRawUnswnb15:
    def test_returns_two_splits(self, csv_dir):
        config = _make_config(csv_dir)
        splits = load_raw_unswnb15(config)
        assert "train" in splits
        assert "test" in splits

    def test_returns_dataframes(self, csv_dir):
        config = _make_config(csv_dir)
        splits = load_raw_unswnb15(config)
        assert isinstance(splits["train"], pd.DataFrame)
        assert isinstance(splits["test"], pd.DataFrame)

    def test_preserves_column_names(self, csv_dir):
        config = _make_config(csv_dir)
        splits = load_raw_unswnb15(config)
        assert list(splits["train"].columns) == ["label", "attack_cat", "feat1"]

    def test_correct_row_count(self, csv_dir):
        config = _make_config(csv_dir)
        splits = load_raw_unswnb15(config)
        assert len(splits["train"]) == 2
        assert len(splits["test"]) == 2

    def test_missing_train_file_raises(self, tmp_path):
        """Only test file exists — train missing → FileNotFoundError."""
        test_content = "label,attack_cat\n0,Normal\n"
        (tmp_path / "test.csv").write_text(test_content)
        config = _make_config(tmp_path)
        with pytest.raises(FileNotFoundError, match="train.csv"):
            load_raw_unswnb15(config)

    def test_missing_test_file_raises(self, tmp_path):
        """Only train file exists — test missing → FileNotFoundError."""
        train_content = "label,attack_cat\n0,Normal\n"
        (tmp_path / "train.csv").write_text(train_content)
        config = _make_config(tmp_path)
        with pytest.raises(FileNotFoundError, match="test.csv"):
            load_raw_unswnb15(config)

    def test_no_files_raises(self, tmp_path):
        config = _make_config(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_raw_unswnb15(config)

    def test_raw_values_not_modified(self, csv_dir):
        """attack_cat raw strings are returned exactly as in the CSV."""
        config = _make_config(csv_dir)
        splits = load_raw_unswnb15(config)
        assert "Normal" in splits["train"]["attack_cat"].values
        assert "Backdoor" in splits["train"]["attack_cat"].values

    def test_empty_csv_raises_or_returns_empty(self, tmp_path):
        """
        A CSV with only a header (zero data rows) must not crash the loader.
        It may return an empty DataFrame — the caller must handle that.
        """
        (tmp_path / "train.csv").write_text("label,attack_cat,feat1\n", encoding="utf-8")
        (tmp_path / "test.csv").write_text("label,attack_cat,feat1\n", encoding="utf-8")
        config = _make_config(tmp_path)
        splits = load_raw_unswnb15(config)
        assert len(splits["train"]) == 0  # header-only → 0 rows
        assert len(splits["test"]) == 0

    def test_does_not_write_to_raw_dir(self, csv_dir):
        """Loading must not create any new files in the raw directory."""
        config = _make_config(csv_dir)
        files_before = set(p.name for p in csv_dir.iterdir())
        load_raw_unswnb15(config)
        files_after = set(p.name for p in csv_dir.iterdir())
        assert files_before == files_after, (
            f"Loader created unexpected files: {files_after - files_before}"
        )
