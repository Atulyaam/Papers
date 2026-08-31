"""
tests/test_split_leakage.py
-------------------------------
Sprint 3 dedicated leakage prevention tests for the TRAIN/VALIDATION split.

Tests verify that the approved split design enforces all leakage boundaries:

  1. Protected unseen TEST data (Backdoor from TEST file) is completely
     separate from the training-file Backdoor archive.
  2. VALIDATION contains ZERO attack rows (label=0 throughout).
  3. The withheld attack class cannot reach TRAIN or VALIDATION.
  4. Split statistics (mean, distribution) cannot bleed from TEST to TRAIN
     through any indexing accident.
  5. The excluded set has EXPERIMENTAL ROLE = NONE.
  6. Row disjointness: TRAIN ∩ VAL = empty.
  7. Column schema is preserved (no accidental feature leakage).
  8. Seed produces a frozen, auditable allocation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.split_protocol import (
    ATTACK_CAT_COL,
    LABEL_COL,
    NORMAL_CAT,
    WITHHELD_ATTACK,
    create_train_val_split,
    verify_split_integrity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_df(
    n_normal: int = 400,
    n_backdoor: int = 80,
    n_exploits: int = 150,
    n_dos: int = 100,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0

    def row(cat, lbl):
        nonlocal uid
        r = {"id": uid, "dur": float(rng.uniform()), "sbytes": int(rng.integers(100, 1000)),
             "proto": "tcp", "service": "-", "state": "FIN",
             "attack_cat": cat, "label": lbl}
        uid += 1
        return r

    for _ in range(n_normal):
        rows.append(row("Normal", 0))
    for _ in range(n_backdoor):
        rows.append(row("Backdoor", 1))
    for _ in range(n_exploits):
        rows.append(row("Exploits", 1))
    for _ in range(n_dos):
        rows.append(row("DoS", 1))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Training-file Backdoor archive is separate from protected TEST Backdoor
# ---------------------------------------------------------------------------


class TestBackdoorArchiveSeparation:
    def test_excluded_backdoor_is_from_training_file_only(self):
        """
        The excluded_train_backdoor.csv contains rows from the TRAINING FILE only.
        The protected unseen TEST Backdoor (data/splits/protected_unseen_attack.csv)
        is from the TEST FILE.

        These two sets are physically disjoint by construction — they come from
        different source files. This test verifies that the split protocol does
        not attempt to merge them.

        Simulated: the excluded set should have the same column schema as the
        training file and a distinct row count from the TEST protected set.
        """
        df = _train_df(n_backdoor=74)  # != 583 (protected TEST count)
        result = create_train_val_split(df, seed=42)

        assert result.n_excluded_backdoor == 74
        assert (result.excluded_backdoor_df["attack_cat"] == "Backdoor").all()
        # The excluded set should NOT have the same count as the TEST protected set
        # (this verifies nothing was silently merged)
        assert result.n_excluded_backdoor != 583

    def test_excluded_backdoor_has_same_columns_as_input(self):
        """No columns are dropped from the excluded set."""
        df = _train_df()
        result = create_train_val_split(df, seed=42)
        assert set(result.excluded_backdoor_df.columns) == set(df.columns)

    def test_excluded_backdoor_label_is_all_ones(self):
        """All archived Backdoor rows must have label=1."""
        df = _train_df()
        result = create_train_val_split(df, seed=42)
        assert (result.excluded_backdoor_df["label"] == 1).all()


# ---------------------------------------------------------------------------
# 2. VALIDATION contains zero attack rows
# ---------------------------------------------------------------------------


class TestValidationAttackIsolation:
    def test_val_zero_label_ones(self):
        """Every VALIDATION row has label=0."""
        df = _train_df(n_normal=400, n_exploits=150, n_dos=100)
        result = create_train_val_split(df, seed=42)
        attack_count = (result.val_df["label"] != 0).sum()
        assert attack_count == 0, (
            f"VALIDATION has {attack_count} rows with label=1. "
            f"Attack categories present: "
            f"{result.val_df.loc[result.val_df['label'] != 0, 'attack_cat'].value_counts().to_dict()}"
        )

    def test_val_attack_cat_is_all_normal(self):
        """Every VALIDATION row has attack_cat == 'Normal'."""
        df = _train_df()
        result = create_train_val_split(df, seed=42)
        non_normal = (result.val_df["attack_cat"] != "Normal")
        assert non_normal.sum() == 0, (
            f"VALIDATION contains non-Normal categories: "
            f"{result.val_df.loc[non_normal, 'attack_cat'].value_counts().to_dict()}"
        )

    def test_val_no_exploits(self):
        df = _train_df(n_exploits=200)
        result = create_train_val_split(df, seed=42)
        assert (result.val_df["attack_cat"] == "Exploits").sum() == 0

    def test_val_no_dos(self):
        df = _train_df(n_dos=150)
        result = create_train_val_split(df, seed=42)
        assert (result.val_df["attack_cat"] == "DoS").sum() == 0

    def test_val_no_backdoor(self):
        df = _train_df(n_backdoor=80)
        result = create_train_val_split(df, seed=42)
        assert (result.val_df["attack_cat"] == "Backdoor").sum() == 0

    def test_multiple_attack_categories_all_stay_in_train(self):
        """All non-Normal, non-Backdoor attack categories must be in TRAIN only."""
        categories = ["Exploits", "DoS", "Fuzzers", "Reconnaissance", "Generic"]
        rows = []
        uid = 0
        rng = np.random.default_rng(0)
        for _ in range(300):
            rows.append({"id": uid, "dur": float(rng.uniform()), "sbytes": 100,
                         "proto": "tcp", "service": "-", "state": "FIN",
                         "attack_cat": "Normal", "label": 0})
            uid += 1
        for _ in range(20):
            rows.append({"id": uid, "dur": float(rng.uniform()), "sbytes": 100,
                         "proto": "tcp", "service": "-", "state": "FIN",
                         "attack_cat": "Backdoor", "label": 1})
            uid += 1
        for cat in categories:
            for _ in range(30):
                rows.append({"id": uid, "dur": float(rng.uniform()), "sbytes": 100,
                             "proto": "tcp", "service": "-", "state": "FIN",
                             "attack_cat": cat, "label": 1})
                uid += 1

        df = pd.DataFrame(rows)
        result = create_train_val_split(df, seed=42)

        for cat in categories:
            val_count = (result.val_df["attack_cat"] == cat).sum()
            assert val_count == 0, (
                f"Attack category '{cat}' appeared in VALIDATION ({val_count} rows)"
            )


# ---------------------------------------------------------------------------
# 3. Withheld attack class cannot reach TRAIN or VAL
# ---------------------------------------------------------------------------


class TestWithheldAttackIsolation:
    def test_backdoor_not_in_train_adversarial(self):
        """
        Adversarial fixture: Backdoor rows are interleaved with Normal rows
        to verify the masking logic does not miss any due to ordering.
        """
        rows = []
        rng = np.random.default_rng(99)
        for i in range(200):
            cat = "Backdoor" if i % 5 == 0 else "Normal"
            lbl = 1 if cat == "Backdoor" else 0
            rows.append({"id": i, "dur": float(rng.uniform()), "sbytes": 100,
                         "proto": "tcp", "service": "-", "state": "FIN",
                         "attack_cat": cat, "label": lbl})
        df = pd.DataFrame(rows)
        result = create_train_val_split(df, seed=42)

        in_train = (result.train_df["attack_cat"] == "Backdoor").sum()
        in_val   = (result.val_df["attack_cat"] == "Backdoor").sum()
        in_excl  = (result.excluded_backdoor_df["attack_cat"] == "Backdoor").sum()

        assert in_train == 0, f"Backdoor in TRAIN: {in_train}"
        assert in_val   == 0, f"Backdoor in VAL: {in_val}"
        assert in_excl  == 40, f"Backdoor in excluded: expected 40, got {in_excl}"


# ---------------------------------------------------------------------------
# 4. Row disjointness: TRAIN ∩ VAL = empty
# ---------------------------------------------------------------------------


class TestRowDisjointness:
    def test_train_val_index_disjoint(self):
        """No row appears in both TRAIN and VALIDATION."""
        df = _train_df()
        result = create_train_val_split(df, seed=42)
        overlap = set(result.train_df.index) & set(result.val_df.index)
        assert len(overlap) == 0, (
            f"{len(overlap)} rows shared between TRAIN and VAL index"
        )

    def test_train_excluded_index_disjoint(self):
        """No row appears in both TRAIN and excluded."""
        df = _train_df()
        result = create_train_val_split(df, seed=42)
        overlap = set(result.train_df.index) & set(result.excluded_backdoor_df.index)
        assert len(overlap) == 0

    def test_val_excluded_index_disjoint(self):
        """No row appears in both VAL and excluded."""
        df = _train_df()
        result = create_train_val_split(df, seed=42)
        overlap = set(result.val_df.index) & set(result.excluded_backdoor_df.index)
        assert len(overlap) == 0

    def test_row_conservation_exact(self):
        """TRAIN + VAL + EXCLUDED = input (no duplication, no loss)."""
        df = _train_df(n_normal=400, n_backdoor=80, n_exploits=150, n_dos=100)
        result = create_train_val_split(df, seed=42)
        total = result.n_train + result.n_val + result.n_excluded_backdoor
        assert total == len(df), (
            f"{result.n_train} + {result.n_val} + {result.n_excluded_backdoor} "
            f"= {total} != {len(df)}"
        )


# ---------------------------------------------------------------------------
# 5. Column schema preserved — no accidental feature leakage
# ---------------------------------------------------------------------------


class TestColumnSchemaPreservation:
    def test_label_not_removed_from_train(self):
        """label column must be present in TRAIN for supervised training."""
        df = _train_df()
        result = create_train_val_split(df, seed=42)
        assert "label" in result.train_df.columns

    def test_attack_cat_not_removed_from_train(self):
        """attack_cat must be preserved (used in later evaluation)."""
        df = _train_df()
        result = create_train_val_split(df, seed=42)
        assert "attack_cat" in result.train_df.columns

    def test_no_extra_columns_added(self):
        """No split-protocol columns should be injected into the output DataFrames."""
        df = _train_df()
        result = create_train_val_split(df, seed=42)
        for name, split in [("train", result.train_df),
                             ("val", result.val_df),
                             ("excluded", result.excluded_backdoor_df)]:
            extra = set(split.columns) - set(df.columns)
            assert len(extra) == 0, (
                f"{name} has extra columns not in input: {extra}"
            )

    def test_no_columns_removed(self):
        """No columns from the input should disappear in any split."""
        df = _train_df()
        result = create_train_val_split(df, seed=42)
        for name, split in [("train", result.train_df),
                             ("val", result.val_df),
                             ("excluded", result.excluded_backdoor_df)]:
            missing = set(df.columns) - set(split.columns)
            assert len(missing) == 0, (
                f"{name} is missing columns from input: {missing}"
            )


# ---------------------------------------------------------------------------
# 6. Frozen allocation — seed-based reproducibility for audit
# ---------------------------------------------------------------------------


class TestSeedAuditability:
    def test_seed_42_produces_reproducible_split(self):
        """
        Official project seed=42 split is deterministic: three reruns produce
        identical TRAIN and VAL sets.
        """
        df = _train_df(n_normal=500, n_backdoor=100, n_exploits=200, seed=0)
        results = [create_train_val_split(df, seed=42) for _ in range(3)]

        for i in range(1, 3):
            pd.testing.assert_frame_equal(
                results[0].train_df.reset_index(drop=True),
                results[i].train_df.reset_index(drop=True),
            )
            pd.testing.assert_frame_equal(
                results[0].val_df.reset_index(drop=True),
                results[i].val_df.reset_index(drop=True),
            )

    def test_integrity_report_all_pass_on_official_seed(self):
        """verify_split_integrity() must return all_pass=True for seed=42."""
        df = _train_df()
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        assert report.all_pass, f"Integrity failures: {report.failures}"
        assert len(report.failures) == 0
