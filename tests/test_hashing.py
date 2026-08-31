"""tests/test_hashing.py — Unit tests for src/utils/hashing.py"""

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.hashing import sha256_file, sha256_dataframe


class TestSha256File:
    def test_deterministic(self, tmp_path):
        """Same file hashed twice produces identical digest."""
        f = tmp_path / "test.csv"
        f.write_bytes(b"col1,col2\n1,2\n3,4\n")
        h1 = sha256_file(f)
        h2 = sha256_file(f)
        assert h1 == h2

    def test_different_files_differ(self, tmp_path):
        """Two files with different content produce different digests."""
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_bytes(b"hello")
        f2.write_bytes(b"world")
        assert sha256_file(f1) != sha256_file(f2)

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            sha256_file(tmp_path / "nonexistent.csv")

    def test_returns_64_char_hex(self, tmp_path):
        """SHA-256 hex digest is always 64 lowercase hex characters."""
        f = tmp_path / "x.bin"
        f.write_bytes(b"data")
        digest = sha256_file(f)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_empty_file(self, tmp_path):
        """Empty file produces a valid (known) SHA-256 digest."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        digest = sha256_file(f)
        expected = hashlib.sha256(b"").hexdigest()
        assert digest == expected


class TestSha256DataFrame:
    def _make_df(self):
        return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    def test_deterministic(self):
        """Same DataFrame produces same hash across two calls."""
        df = self._make_df()
        assert sha256_dataframe(df) == sha256_dataframe(df)

    def test_different_content_differs(self):
        """DataFrames with different values produce different hashes."""
        df1 = self._make_df()
        df2 = pd.DataFrame({"a": [1, 2, 99], "b": ["x", "y", "z"]})
        assert sha256_dataframe(df1) != sha256_dataframe(df2)

    def test_column_order_matters(self):
        """Changing column order changes the hash."""
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"b": [2], "a": [1]})
        # Column order should produce different CSV byte streams
        assert sha256_dataframe(df1) != sha256_dataframe(df2)
