"""tests/test_canonicalization.py — Unit tests for attack_cat_canonicalization.py"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.attack_cat_canonicalization import (
    CANONICAL_MAP,
    canonicalize_attack_cat,
    get_canonicalization_audit,
)


class TestCanonicalizeAttackCat:
    def test_known_mapped_correctly(self):
        s = pd.Series(["Backdoor"])
        result = canonicalize_attack_cat(s, CANONICAL_MAP)
        assert result.iloc[0] == "Backdoor"

    def test_lowercase_input_mapped(self):
        """'backdoor' (lowercase) should map to 'Backdoor'."""
        s = pd.Series(["backdoor"])
        result = canonicalize_attack_cat(s, CANONICAL_MAP)
        assert result.iloc[0] == "Backdoor"

    def test_leading_trailing_whitespace_stripped_and_mapped(self):
        """' Backdoor ' with spaces should map to 'Backdoor'."""
        s = pd.Series([" Backdoor "])
        result = canonicalize_attack_cat(s, CANONICAL_MAP)
        assert result.iloc[0] == "Backdoor"

    def test_unknown_value_preserved(self):
        """Values not in the map are returned unchanged (after stripping)."""
        s = pd.Series(["SomeUnknownCategory"])
        result = canonicalize_attack_cat(s, CANONICAL_MAP)
        assert result.iloc[0] == "SomeUnknownCategory"

    def test_nan_preserved(self):
        """NaN values are preserved as NaN, not coerced to string."""
        s = pd.Series([None, "Backdoor"])
        result = canonicalize_attack_cat(s, CANONICAL_MAP)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == "Backdoor"

    def test_original_series_not_modified(self):
        """The input Series must not be modified in-place."""
        original_vals = ["Backdoor", " dos "]
        s = pd.Series(original_vals.copy())
        _ = canonicalize_attack_cat(s, CANONICAL_MAP)
        # Original values must be unchanged
        assert list(s) == original_vals

    def test_all_documented_categories_mapped(self):
        """All keys in CANONICAL_MAP should produce a canonical value."""
        for raw_key, expected_canonical in CANONICAL_MAP.items():
            s = pd.Series([raw_key])
            result = canonicalize_attack_cat(s, CANONICAL_MAP)
            assert result.iloc[0] == expected_canonical, (
                f"Key '{raw_key}' did not map to '{expected_canonical}' "
                f"(got '{result.iloc[0]}')"
            )

    def test_custom_map_overrides_default(self):
        """A custom map takes precedence over CANONICAL_MAP."""
        custom_map = {"test": "TestCategory"}
        s = pd.Series(["test"])
        result = canonicalize_attack_cat(s, custom_map)
        assert result.iloc[0] == "TestCategory"


class TestGetCanonicalizationAudit:
    def test_mapped_status(self):
        s = pd.Series(["backdoor"])
        records = get_canonicalization_audit(s, CANONICAL_MAP)
        record = next(r for r in records if r["raw_value"] == "backdoor")
        assert record["status"] == "MAPPED"
        assert record["canonical_value"] == "Backdoor"

    def test_unknown_status(self):
        s = pd.Series(["WeirdCategory"])
        records = get_canonicalization_audit(s, CANONICAL_MAP)
        record = records[0]
        assert record["status"] == "PRESERVED_UNKNOWN"
        assert record["canonical_value"] == "WeirdCategory"

    def test_nan_status(self):
        s = pd.Series([None])
        records = get_canonicalization_audit(s, CANONICAL_MAP)
        nan_record = next(r for r in records if r["status"] == "NAN")
        assert nan_record["canonical_value"] is None
