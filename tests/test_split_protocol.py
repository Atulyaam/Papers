"""
tests/test_split_protocol.py
-------------------------------
Unit tests for Sprint 3: TRAIN / VALIDATION split protocol.

All tests use synthetic DataFrames that mirror the UNSW-NB15 schema.
No dependency on real data files.

Coverage:
  TestCreateTrainValSplit      — core split logic and row counts
  TestSplitIntegrity           — verify_split_integrity() checks
  TestSplitDeterminism         — same seed reproduces identical split
  TestSplitEdgeCases           — boundary conditions and error paths
  TestBuildSplitProvenance     — provenance dict structure and serialisation
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.split_protocol import (
    NORMAL_CAT,
    NORMAL_TRAIN_FRAC,
    NORMAL_VAL_FRAC,
    WITHHELD_ATTACK,
    TrainValSplitResult,
    build_split_provenance,
    create_train_val_split,
    verify_split_integrity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_training_df(
    n_normal: int = 500,
    n_backdoor: int = 100,
    n_exploits: int = 200,
    n_dos: int = 150,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a synthetic UNSW-NB15-style training DataFrame.

    Normal rows have label=0. All attack rows have label=1.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_normal):
        rows.append({"id": i, "dur": rng.uniform(), "sbytes": rng.integers(100, 1000),
                     "proto": "tcp", "service": "-", "state": "FIN",
                     "attack_cat": "Normal", "label": 0})

    for i in range(n_backdoor):
        rows.append({"id": n_normal + i, "dur": rng.uniform(), "sbytes": rng.integers(100, 1000),
                     "proto": "tcp", "service": "-", "state": "FIN",
                     "attack_cat": "Backdoor", "label": 1})

    for i in range(n_exploits):
        rows.append({"id": n_normal + n_backdoor + i, "dur": rng.uniform(),
                     "sbytes": rng.integers(100, 1000),
                     "proto": "tcp", "service": "-", "state": "FIN",
                     "attack_cat": "Exploits", "label": 1})

    for i in range(n_dos):
        rows.append({"id": n_normal + n_backdoor + n_exploits + i, "dur": rng.uniform(),
                     "sbytes": rng.integers(100, 1000),
                     "proto": "udp", "service": "-", "state": "CON",
                     "attack_cat": "DoS", "label": 1})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestCreateTrainValSplit
# ---------------------------------------------------------------------------


class TestCreateTrainValSplit:
    def test_returns_train_val_split_result(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        assert isinstance(result, TrainValSplitResult)

    def test_row_conservation(self):
        """TRAIN + VAL + EXCLUDED must equal total input rows."""
        df = _make_training_df(n_normal=500, n_backdoor=100, n_exploits=200, n_dos=150)
        result = create_train_val_split(df, seed=42)
        total = result.n_train + result.n_val + result.n_excluded_backdoor
        assert total == result.n_input == len(df), (
            f"Row conservation failed: {result.n_train} + {result.n_val} + "
            f"{result.n_excluded_backdoor} = {total} != {len(df)}"
        )

    def test_excluded_contains_all_backdoor_rows(self):
        n_backdoor = 87
        df = _make_training_df(n_backdoor=n_backdoor)
        result = create_train_val_split(df, seed=42)
        assert result.n_excluded_backdoor == n_backdoor
        assert (result.excluded_backdoor_df["attack_cat"] == "Backdoor").all()

    def test_backdoor_not_in_train(self):
        df = _make_training_df(n_backdoor=50)
        result = create_train_val_split(df, seed=42)
        backdoor_in_train = (result.train_df["attack_cat"] == "Backdoor").sum()
        assert backdoor_in_train == 0, (
            f"TRAIN contains {backdoor_in_train} Backdoor rows (expected 0)"
        )

    def test_backdoor_not_in_val(self):
        df = _make_training_df(n_backdoor=50)
        result = create_train_val_split(df, seed=42)
        backdoor_in_val = (result.val_df["attack_cat"] == "Backdoor").sum()
        assert backdoor_in_val == 0

    def test_validation_is_normal_only(self):
        """VALIDATION must contain ZERO attack rows."""
        df = _make_training_df(n_normal=500, n_exploits=200, n_dos=150)
        result = create_train_val_split(df, seed=42)
        non_normal = (result.val_df["attack_cat"] != "Normal").sum()
        assert non_normal == 0, (
            f"VALIDATION contains {non_normal} non-Normal rows. "
            f"Categories: {result.val_df['attack_cat'].value_counts().to_dict()}"
        )

    def test_validation_label_all_zero(self):
        """VALIDATION rows must all have label=0."""
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        attack_in_val = (result.val_df["label"] != 0).sum()
        assert attack_in_val == 0, (
            f"VALIDATION has {attack_in_val} rows with label != 0"
        )

    def test_train_contains_attack_rows(self):
        """TRAIN must contain the non-Backdoor attack rows."""
        df = _make_training_df(n_exploits=200, n_dos=150)
        result = create_train_val_split(df, seed=42)
        attack_in_train = (result.train_df["label"] == 1).sum()
        assert attack_in_train > 0

    def test_all_attack_rows_go_to_train(self):
        """100% of non-Backdoor attack rows must be in TRAIN."""
        n_exploits = 200
        n_dos = 150
        df = _make_training_df(n_exploits=n_exploits, n_dos=n_dos)
        result = create_train_val_split(df, seed=42)
        train_exploits = (result.train_df["attack_cat"] == "Exploits").sum()
        train_dos      = (result.train_df["attack_cat"] == "DoS").sum()
        assert train_exploits == n_exploits, (
            f"Expected {n_exploits} Exploits in TRAIN, got {train_exploits}"
        )
        assert train_dos == n_dos, (
            f"Expected {n_dos} DoS in TRAIN, got {train_dos}"
        )

    def test_normal_split_fraction_approximately_correct(self):
        """80% Normal in TRAIN, 20% in VALIDATION (within rounding tolerance)."""
        n_normal = 1000
        df = _make_training_df(n_normal=n_normal, n_backdoor=50, n_exploits=100)
        result = create_train_val_split(df, seed=42)

        n_normal_train = (result.train_df["attack_cat"] == "Normal").sum()
        n_normal_val   = (result.val_df["attack_cat"] == "Normal").sum()

        assert abs(n_normal_train - 800) <= 5, (
            f"Normal in TRAIN: expected ~800, got {n_normal_train}"
        )
        assert abs(n_normal_val - 200) <= 5, (
            f"Normal in VAL: expected ~200, got {n_normal_val}"
        )

    def test_train_df_contains_correct_columns(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        assert set(result.train_df.columns) == set(df.columns)
        assert set(result.val_df.columns) == set(df.columns)
        assert set(result.excluded_backdoor_df.columns) == set(df.columns)

    def test_seed_recorded_in_result(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=99)
        assert result.seed == 99

    def test_n_input_recorded_correctly(self):
        df = _make_training_df(n_normal=300, n_backdoor=30, n_exploits=100)
        result = create_train_val_split(df, seed=42)
        assert result.n_input == len(df)

    def test_no_index_overlap_between_train_and_val(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        overlap = set(result.train_df.index) & set(result.val_df.index)
        assert len(overlap) == 0, (
            f"{len(overlap)} rows appear in both TRAIN and VALIDATION"
        )

    def test_missing_attack_cat_col_raises(self):
        df = _make_training_df()
        df = df.drop(columns=["attack_cat"])
        with pytest.raises(ValueError, match="attack_cat"):
            create_train_val_split(df, seed=42)

    def test_missing_label_col_raises(self):
        df = _make_training_df()
        df = df.drop(columns=["label"])
        with pytest.raises(ValueError, match="label"):
            create_train_val_split(df, seed=42)

    def test_no_backdoor_rows_raises(self):
        df = _make_training_df(n_backdoor=0)
        with pytest.raises(ValueError, match="Backdoor"):
            create_train_val_split(df, seed=42)

    def test_no_normal_rows_raises(self):
        """
        When the non-Backdoor pool contains no Normal rows, the split is
        undefined and must raise ValueError.
        """
        rng = np.random.default_rng(5)
        rows = []
        for i in range(50):
            rows.append({"id": i, "dur": float(rng.uniform()), "sbytes": 100,
                         "proto": "tcp", "service": "-", "state": "FIN",
                         "attack_cat": "Backdoor", "label": 1})
        for i in range(100):
            rows.append({"id": 50 + i, "dur": float(rng.uniform()), "sbytes": 100,
                         "proto": "tcp", "service": "-", "state": "FIN",
                         "attack_cat": "Exploits", "label": 1})
        df = pd.DataFrame(rows)
        # No Normal rows at all in non-Backdoor pool → must raise
        with pytest.raises(ValueError, match="Normal"):
            create_train_val_split(df, seed=42)


# ---------------------------------------------------------------------------
# TestSplitDeterminism
# ---------------------------------------------------------------------------


class TestSplitDeterminism:
    def test_same_seed_same_split(self):
        """Two runs with same seed must produce identical TRAIN and VAL."""
        df = _make_training_df(seed=0)
        r1 = create_train_val_split(df, seed=42)
        r2 = create_train_val_split(df, seed=42)

        pd.testing.assert_frame_equal(
            r1.train_df.reset_index(drop=True),
            r2.train_df.reset_index(drop=True),
            check_like=False,
        )
        pd.testing.assert_frame_equal(
            r1.val_df.reset_index(drop=True),
            r2.val_df.reset_index(drop=True),
            check_like=False,
        )

    def test_different_seed_different_split(self):
        """Different seeds should produce different Normal allocations."""
        df = _make_training_df(n_normal=500, seed=0)
        r42  = create_train_val_split(df, seed=42)
        r123 = create_train_val_split(df, seed=123)

        val_idx_42  = set(r42.val_df.index)
        val_idx_123 = set(r123.val_df.index)
        assert val_idx_42 != val_idx_123, (
            "Different seeds produced identical VAL splits — seeding may be broken"
        )

    def test_all_three_primary_seeds_produce_valid_splits(self):
        """Seeds 42, 123, and 2024 (the project's three-seed protocol) all work."""
        df = _make_training_df()
        for seed in (42, 123, 2024):
            result = create_train_val_split(df, seed=seed)
            report = verify_split_integrity(result)
            assert report.all_pass, (
                f"Seed {seed} produced a split that failed integrity: {report.failures}"
            )


# ---------------------------------------------------------------------------
# TestSplitIntegrity
# ---------------------------------------------------------------------------


class TestSplitIntegrity:
    def test_clean_split_passes_all_checks(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        assert report.all_pass, f"Integrity failures: {report.failures}"

    def test_row_conservation_check(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        assert report.row_conservation

    def test_val_zero_attack_check(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        assert report.val_zero_attack
        assert report.val_zero_attack_count == 0

    def test_backdoor_not_in_train_check(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        assert report.backdoor_not_in_train
        assert report.backdoor_in_train_count == 0

    def test_backdoor_not_in_val_check(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        assert report.backdoor_not_in_val
        assert report.backdoor_in_val_count == 0

    def test_normal_only_in_val_check(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        assert report.normal_only_in_val
        assert report.non_normal_in_val_count == 0

    def test_excluded_all_backdoor_check(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        assert report.excluded_all_backdoor

    def test_label_consistency_check(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        assert report.label_consistency

    def test_no_index_overlap_check(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        assert report.no_index_overlap_train_val

    def test_train_has_attacks_check(self):
        df = _make_training_df(n_exploits=200)
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        assert report.train_has_attacks

    def test_integrity_detects_injected_backdoor_in_val(self):
        """
        Adversarial: manually inject a Backdoor row into the VAL set and
        verify that integrity detection catches it.
        """
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)

        backdoor_row = result.excluded_backdoor_df.iloc[[0]].copy()
        backdoor_row["attack_cat"] = "Backdoor"
        backdoor_row["label"] = 1
        result.val_df = pd.concat([result.val_df, backdoor_row], axis=0)
        result.n_val = len(result.val_df)

        report = verify_split_integrity(result)
        assert not report.val_zero_attack, "Should detect Backdoor in VAL"
        assert report.val_zero_attack_count >= 1
        assert not report.normal_only_in_val
        assert not report.all_pass

    def test_integrity_detects_attack_in_val(self):
        """Adversarial: inject a non-Backdoor attack row into VAL."""
        df = _make_training_df(n_exploits=100)
        result = create_train_val_split(df, seed=42)

        exploit_row = result.train_df.loc[
            result.train_df["attack_cat"] == "Exploits"
        ].iloc[[0]].copy()
        result.val_df = pd.concat([result.val_df, exploit_row], axis=0)
        result.n_val = len(result.val_df)

        report = verify_split_integrity(result)
        assert not report.val_zero_attack
        assert not report.normal_only_in_val
        assert not report.all_pass


# ---------------------------------------------------------------------------
# TestSplitEdgeCases
# ---------------------------------------------------------------------------


class TestSplitEdgeCases:
    def test_minimum_viable_normal_pool(self):
        """Smallest sensible Normal pool that allows 80/20 split."""
        df = _make_training_df(n_normal=10, n_backdoor=5, n_exploits=5)
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        assert result.n_train + result.n_val + result.n_excluded_backdoor == len(df)
        assert report.val_zero_attack

    def test_single_attack_category(self):
        """Only one non-Backdoor attack category in pool."""
        df = _make_training_df(n_normal=200, n_backdoor=20, n_exploits=80, n_dos=0)
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        assert report.all_pass

    def test_large_backdoor_proportion_archived(self):
        """Even if Backdoor outnumbers all other attacks, it is fully archived."""
        df = _make_training_df(n_normal=200, n_backdoor=1000, n_exploits=50)
        result = create_train_val_split(df, seed=42)
        assert result.n_excluded_backdoor == 1000
        assert (result.excluded_backdoor_df["attack_cat"] == "Backdoor").all()
        report = verify_split_integrity(result)
        assert report.backdoor_not_in_train
        assert report.backdoor_not_in_val

    def test_custom_seed_parameter_used(self):
        """The seed parameter is passed to train_test_split (not hardcoded)."""
        df = _make_training_df(n_normal=300)
        r42   = create_train_val_split(df, seed=42)
        r9999 = create_train_val_split(df, seed=9999)
        assert set(r42.val_df.index) != set(r9999.val_df.index)

    def test_no_normal_rows_raises(self):
        """
        When the non-Backdoor pool has no Normal rows, the split is undefined.
        create_train_val_split must raise ValueError mentioning 'Normal'.
        """
        rng = np.random.default_rng(5)
        rows = []
        for i in range(50):
            rows.append({"id": i, "dur": float(rng.uniform()), "sbytes": 100,
                         "proto": "tcp", "service": "-", "state": "FIN",
                         "attack_cat": "Backdoor", "label": 1})
        for i in range(100):
            rows.append({"id": 50 + i, "dur": float(rng.uniform()), "sbytes": 100,
                         "proto": "tcp", "service": "-", "state": "FIN",
                         "attack_cat": "Exploits", "label": 1})
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="Normal"):
            create_train_val_split(df, seed=42)


# ---------------------------------------------------------------------------
# TestBuildSplitProvenance
# ---------------------------------------------------------------------------


class TestBuildSplitProvenance:
    def _make_result_and_report(self):
        df = _make_training_df()
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        return result, report

    def test_provenance_is_dict(self):
        result, report = self._make_result_and_report()
        prov = build_split_provenance(result, report, "abc123", git_commit="deadbeef")
        assert isinstance(prov, dict)

    def test_provenance_contains_required_keys(self):
        """All spec-mandated top-level keys must be present."""
        result, report = self._make_result_and_report()
        prov = build_split_provenance(result, report, "abc123", git_commit="deadbeef")
        required = (
            "protocol_version", "experiment_id", "dataset",
            "source_training_filename", "source_training_sha256",
            "split_seed", "normal_train_fraction", "normal_validation_fraction",
            "split_strategy",
            "train_count", "validation_count", "excluded_backdoor_count",
            "normal_train_count", "normal_validation_count",
            "non_backdoor_attack_train_count",
            "validation_attack_count", "backdoor_train_count", "backdoor_validation_count",
            "train_distribution", "val_distribution", "excluded_distribution",
            "row_conservation", "exact_reconstruction",
            "integrity_checks",
            "validation_percentile_adequacy",
        )
        for key in required:
            assert key in prov, f"Missing required key: '{key}'"

    def test_provenance_seed_matches(self):
        result, report = self._make_result_and_report()
        prov = build_split_provenance(result, report, "abc123")
        assert prov["split_seed"] == 42

    def test_provenance_counts_match_result(self):
        result, report = self._make_result_and_report()
        prov = build_split_provenance(result, report, "abc123")
        assert prov["train_count"]             == result.n_train
        assert prov["validation_count"]        == result.n_val
        assert prov["excluded_backdoor_count"] == result.n_excluded_backdoor

    def test_provenance_normal_counts_are_correct(self):
        """normal_train_count + normal_validation_count == all non-Backdoor Normal rows."""
        df = _make_training_df(n_normal=500, n_backdoor=100, n_exploits=200)
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        prov = build_split_provenance(result, report, "abc123")
        total_normal = prov["normal_train_count"] + prov["normal_validation_count"]
        expected_normal = int((df["attack_cat"] == "Normal").sum())
        assert total_normal == expected_normal, (
            f"normal_train_count + normal_validation_count = {total_normal} "
            f"!= {expected_normal} (all Normal rows)"
        )

    def test_provenance_validation_attack_count_is_zero(self):
        result, report = self._make_result_and_report()
        prov = build_split_provenance(result, report, "abc123")
        assert prov["validation_attack_count"] == 0

    def test_provenance_backdoor_train_count_is_zero(self):
        result, report = self._make_result_and_report()
        prov = build_split_provenance(result, report, "abc123")
        assert prov["backdoor_train_count"] == 0

    def test_provenance_integrity_reflects_report(self):
        result, report = self._make_result_and_report()
        prov = build_split_provenance(result, report, "abc123")
        assert prov["integrity_checks"]["all_pass"] == report.all_pass
        assert prov["integrity_checks"]["val_zero_attack"] == report.val_zero_attack

    def test_provenance_is_json_serialisable(self):
        """
        json.dumps() must succeed without a custom encoder.
        All values must be native Python types: str, int, float, bool, list, dict, None.
        numpy.bool_, numpy.int64, pandas scalars are forbidden.
        """
        result, report = self._make_result_and_report()
        prov = build_split_provenance(result, report, "abc123", git_commit="deadbeef")
        # Must not raise TypeError
        serialised = json.dumps(prov)
        recovered = json.loads(serialised)
        assert recovered["split_seed"] == 42

    def test_provenance_experiment_id_is_correct(self):
        result, report = self._make_result_and_report()
        prov = build_split_provenance(result, report, "sha256abc")
        assert prov["experiment_id"] == "EXP_TRAIN_VAL_SPLIT_V1"

    def test_provenance_val_composition_is_documented(self):
        """val_distribution must show only Normal rows (no attack categories)."""
        result, report = self._make_result_and_report()
        prov = build_split_provenance(result, report, "sha256abc")
        assert "Normal" in prov["val_distribution"]["attack_cat"]
        val_cats = prov["val_distribution"]["attack_cat"]
        for cat in val_cats:
            assert cat == "Normal", f"Unexpected category in VAL distribution: {cat}"

    def test_provenance_percentile_adequacy_present(self):
        """validation_percentile_adequacy must be present with required structure."""
        result, report = self._make_result_and_report()
        prov = build_split_provenance(result, report, "sha256abc")
        pa = prov["validation_percentile_adequacy"]
        assert "n_benign_validation" in pa
        assert "percentiles" in pa
        assert "assessment" in pa
        for p in ("90", "95", "97.5", "99"):
            assert p in pa["percentiles"], f"Missing percentile: {p}"
            assert "approx_tail_count" in pa["percentiles"][p]

    def test_provenance_percentile_counts_are_positive(self):
        """All upper-tail counts must be positive integers."""
        df = _make_training_df(n_normal=500)
        result = create_train_val_split(df, seed=42)
        report = verify_split_integrity(result)
        prov = build_split_provenance(result, report, "sha256abc")
        pa = prov["validation_percentile_adequacy"]
        for p, entry in pa["percentiles"].items():
            assert entry["approx_tail_count"] > 0, (
                f"approx_tail_count for {p}th percentile is {entry['approx_tail_count']}"
            )

    def test_provenance_with_optional_hashes(self):
        """output_hashes are stored when provided."""
        result, report = self._make_result_and_report()
        hashes = {"train_csv": "abc", "validation_csv": "def"}
        prov = build_split_provenance(result, report, "sha256abc", output_hashes=hashes)
        assert prov["output_hashes"] == hashes

    def test_provenance_with_library_versions(self):
        """library_versions are stored when provided."""
        result, report = self._make_result_and_report()
        versions = {"python": "3.11.9", "numpy": "1.26.0", "pandas": "2.2.0"}
        prov = build_split_provenance(result, report, "sha256abc", library_versions=versions)
        assert prov["library_versions"] == versions

    def test_provenance_none_optional_fields_without_args(self):
        """output_hashes and library_versions are None when not provided."""
        result, report = self._make_result_and_report()
        prov = build_split_provenance(result, report, "sha256abc")
        assert prov["output_hashes"] is None
        assert prov["library_versions"] is None
        assert prov["exact_reconstruction"] is None
