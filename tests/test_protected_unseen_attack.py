"""tests/test_protected_unseen_attack.py — Unit tests for protected_unseen_attack.py"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.attack_cat_canonicalization import canonicalize_attack_cat
from src.preprocessing.protected_unseen_attack import (
    ReservationError,
    reserve_protected_unseen_attack,
    build_split_metadata,
)
from src.utils.hashing import sha256_dataframe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_test_df(n_backdoor=100, n_dos=200, n_normal=300, n_generic=50):
    """Create a synthetic TEST DataFrame with known category counts."""
    rows = []
    for _ in range(n_backdoor):
        rows.append({"label": 1, "attack_cat": "Backdoor", "feat1": 1.0})
    for _ in range(n_dos):
        rows.append({"label": 1, "attack_cat": "DoS", "feat1": 2.0})
    for _ in range(n_normal):
        rows.append({"label": 0, "attack_cat": "Normal", "feat1": 3.0})
    for _ in range(n_generic):
        rows.append({"label": 1, "attack_cat": "Generic", "feat1": 4.0})
    return pd.DataFrame(rows)


@pytest.fixture
def df_test():
    return _make_test_df()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReserveProtectedUnseenAttack:
    def test_protected_contains_only_backdoor(self, df_test):
        protected, _ = reserve_protected_unseen_attack(df_test, cat_col="attack_cat")
        canon = canonicalize_attack_cat(protected["attack_cat"])
        assert (canon == "Backdoor").all()

    def test_development_contains_zero_backdoor(self, df_test):
        _, development = reserve_protected_unseen_attack(df_test, cat_col="attack_cat")
        canon = canonicalize_attack_cat(development["attack_cat"])
        assert (canon == "Backdoor").sum() == 0

    def test_row_conservation(self, df_test):
        protected, development = reserve_protected_unseen_attack(df_test, cat_col="attack_cat")
        assert len(protected) + len(development) == len(df_test)

    def test_protected_count_correct(self, df_test):
        protected, _ = reserve_protected_unseen_attack(df_test, cat_col="attack_cat")
        assert len(protected) == 100  # matches n_backdoor fixture

    def test_column_order_preserved(self, df_test):
        protected, development = reserve_protected_unseen_attack(df_test, cat_col="attack_cat")
        assert list(protected.columns) == list(df_test.columns)
        assert list(development.columns) == list(df_test.columns)

    def test_source_dataframe_not_modified(self, df_test):
        """Verify the input DataFrame is not changed by the reservation."""
        original_hash = sha256_dataframe(df_test.reset_index(drop=True))
        _, _ = reserve_protected_unseen_attack(df_test, cat_col="attack_cat")
        post_hash = sha256_dataframe(df_test.reset_index(drop=True))
        assert original_hash == post_hash

    def test_exact_reconstruction(self, df_test):
        """Protected + development (sorted by original index) == original."""
        protected, development = reserve_protected_unseen_attack(df_test, cat_col="attack_cat")
        reconstructed = pd.concat([protected, development]).sort_index()
        original_sorted = df_test.sort_index()
        hash_orig = sha256_dataframe(original_sorted.reset_index(drop=True))
        hash_rec = sha256_dataframe(reconstructed.reset_index(drop=True))
        assert hash_orig == hash_rec

    def test_deterministic_rerun(self, df_test):
        """Running twice produces identical protected and development sets."""
        p1, d1 = reserve_protected_unseen_attack(df_test, cat_col="attack_cat")
        p2, d2 = reserve_protected_unseen_attack(df_test, cat_col="attack_cat")
        assert sha256_dataframe(p1.reset_index(drop=True)) == sha256_dataframe(p2.reset_index(drop=True))
        assert sha256_dataframe(d1.reset_index(drop=True)) == sha256_dataframe(d2.reset_index(drop=True))

    def test_below_threshold_raises(self):
        """Backdoor count below 50 must raise ReservationError."""
        df = _make_test_df(n_backdoor=30)  # only 30, below 50 threshold
        with pytest.raises(ReservationError, match="eligibility"):
            reserve_protected_unseen_attack(df, cat_col="attack_cat", min_count=50)

    def test_backdoor_absent_raises(self):
        """If no Backdoor rows exist, raise ReservationError."""
        df = pd.DataFrame({
            "label": [0, 1, 1],
            "attack_cat": ["Normal", "DoS", "Generic"],
            "feat1": [1.0, 2.0, 3.0],
        })
        with pytest.raises(ReservationError):
            reserve_protected_unseen_attack(df, cat_col="attack_cat")

    def test_missing_cat_col_raises(self, df_test):
        """If cat_col is absent from DataFrame, raise ReservationError."""
        df = df_test.drop(columns=["attack_cat"])
        with pytest.raises(ReservationError, match="attack_cat"):
            reserve_protected_unseen_attack(df, cat_col="attack_cat")

    def test_whitespace_variant_backdoor_handled(self):
        """' Backdoor ' (with spaces) must be canonicalized and captured."""
        rows = [
            {"label": 1, "attack_cat": " Backdoor ", "feat1": 1.0}
            for _ in range(60)
        ] + [
            {"label": 0, "attack_cat": "Normal", "feat1": 2.0}
            for _ in range(40)
        ]
        df = pd.DataFrame(rows)
        protected, development = reserve_protected_unseen_attack(df, cat_col="attack_cat")
        assert len(protected) == 60
        assert len(development) == 40


class TestBuildSplitMetadata:
    def test_structure(self, tmp_path, df_test):
        protected, development = reserve_protected_unseen_attack(df_test, cat_col="attack_cat")
        fpath = tmp_path / "test.csv"
        df_test.to_csv(fpath, index=False)
        from src.utils.hashing import sha256_file
        src_hash = sha256_file(fpath)
        meta = build_split_metadata(
            source_test_path=fpath,
            df_protected=protected,
            df_development=development,
            original_row_count=len(df_test),
            source_sha256=src_hash,
        )
        assert meta["row_conservation"] == "PASS"
        assert meta["reconstruction_verified"] == "PASS"
        assert meta["seed"] is None
        assert "No random sampling" in meta["notes"]
        assert meta["protected_row_count"] == 100
        assert meta["development_test_row_count"] == len(df_test) - 100
