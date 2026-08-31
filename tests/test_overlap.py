"""tests/test_overlap.py — Unit tests for check_overlap in schema_audit.py"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.schema_audit import check_overlap


class TestCheckOverlap:
    def test_no_overlap(self):
        a = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        b = pd.DataFrame({"x": [5, 6], "y": [7, 8]})
        r = check_overlap(a, b)
        assert r["overlap_count"] == 0
        assert r["overlap_percentage"] == 0.0

    def test_one_shared_row(self):
        a = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        b = pd.DataFrame({"x": [3, 7], "y": [6, 8]})
        r = check_overlap(a, b)
        assert r["overlap_count"] == 1

    def test_complete_overlap(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        r = check_overlap(df, df.copy())
        assert r["overlap_count"] == 3

    def test_no_shared_columns_returns_zero(self):
        a = pd.DataFrame({"a": [1, 2]})
        b = pd.DataFrame({"b": [3, 4]})
        r = check_overlap(a, b)
        assert r["overlap_count"] == 0

    def test_reports_row_counts(self):
        a = pd.DataFrame({"x": [1, 2, 3]})
        b = pd.DataFrame({"x": [4, 5]})
        r = check_overlap(a, b)
        assert r["train_row_count"] == 3
        assert r["test_row_count"] == 2

    def test_limitation_documented(self):
        """Result always includes limitation note."""
        a = pd.DataFrame({"x": [1]})
        b = pd.DataFrame({"x": [2]})
        r = check_overlap(a, b)
        assert "limitation" in r or "note" in r
