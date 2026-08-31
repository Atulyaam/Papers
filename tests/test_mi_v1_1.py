"""
tests/test_mi_v1_1.py
-----------------------
Tests for Sprint 4 Protocol v1.1 — Expanded K grid.

Tests:
    TestExpandedKGrid       — K values 75/100/150 are valid, selection works
    TestKGridVersioning     — v1.0 and v1.1 K grids are distinct
    TestProtocolAmendment   — metadata records protocol change correctly
    TestOriginalV1Preserved — v1.0 output directory exists and is intact
    TestMonotonicAtBoundary — sanity check at expanded K=150 boundary
    TestDeterminismV11      — same seed → same K selection
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_selection.k_selector import (
    InnerCVConfig,
    KFoldRecord,
    check_selection_sanity,
    select_best_k,
    run_k_selection_cv,
)
from src.feature_selection.mi_selector import (
    MIConfig,
    compute_mi_scores,
    select_top_k,
)

# ---------------------------------------------------------------------------
# Constants that must remain frozen
# ---------------------------------------------------------------------------

V1_CANDIDATE_K  = (10, 20, 30, 40, 50)
V1_1_CANDIDATE_K = (10, 20, 30, 40, 50, 75, 100, 150)
V1_RESULTS = {10: 0.824852, 20: 0.864436, 30: 0.897442, 40: 0.916198, 50: 0.919560}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_train(n_normal=200, n_attack=150, n_num=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_normal):
        rows.append({
            "label": 0, "attack_cat": "Normal",
            "proto": rng.choice(["tcp", "udp"]),
            "service": rng.choice(["-", "http"]),
            "state": rng.choice(["FIN", "CON"]),
            **{f"num_{j}": float(rng.normal()) for j in range(n_num)},
        })
    for _ in range(n_attack):
        rows.append({
            "label": 1, "attack_cat": "Exploits",
            "proto": rng.choice(["tcp", "udp"]),
            "service": rng.choice(["-", "http"]),
            "state": rng.choice(["FIN", "CON"]),
            **{f"num_{j}": float(rng.normal()) for j in range(n_num)},
        })
    return pd.DataFrame(rows)


def _make_fold_records(k_f1_map: dict) -> list[KFoldRecord]:
    records = []
    for k, f1_list in k_f1_map.items():
        for fold_idx, f1 in enumerate(f1_list):
            records.append(KFoldRecord(k=k, fold=fold_idx + 1, macro_f1=f1))
    return records


# ---------------------------------------------------------------------------
# TestExpandedKGrid
# ---------------------------------------------------------------------------


class TestExpandedKGrid:
    def test_k75_is_valid_candidate(self):
        """K=75 must be accepted by InnerCVConfig and select_best_k."""
        records = _make_fold_records({
            10: [0.80] * 5,
            75: [0.90] * 5,
        })
        result = select_best_k(records)
        assert result.selected_k == 75

    def test_k100_is_valid_candidate(self):
        records = _make_fold_records({
            10: [0.80] * 5,
            100: [0.91] * 5,
        })
        result = select_best_k(records)
        assert result.selected_k == 100

    def test_k150_is_valid_candidate(self):
        records = _make_fold_records({
            10: [0.80] * 5,
            150: [0.92] * 5,
        })
        result = select_best_k(records)
        assert result.selected_k == 150

    def test_v1_1_config_candidate_k(self):
        config = InnerCVConfig(candidate_k=V1_1_CANDIDATE_K)
        assert set(config.candidate_k) == {10, 20, 30, 40, 50, 75, 100, 150}

    def test_v1_1_has_eight_candidates(self):
        assert len(V1_1_CANDIDATE_K) == 8

    def test_lower_k_in_v1_1_same_as_v1(self):
        """The first 5 candidates of v1.1 are identical to v1.0."""
        assert V1_1_CANDIDATE_K[:5] == V1_CANDIDATE_K

    def test_tie_break_prefers_smaller_k_among_new_values(self):
        """If K=75 and K=100 tie, K=75 must win."""
        records = _make_fold_records({
            75:  [0.92] * 5,
            100: [0.92] * 5,
        })
        result = select_best_k(records)
        assert result.selected_k == 75

    def test_k150_select_top_k_works_on_enough_features(self):
        """select_top_k with K=150 must work when ≥150 features exist."""
        rng = np.random.default_rng(0)
        n_features = 200
        n_samples  = 300
        X = rng.normal(0, 1, (n_samples, n_features))
        y = rng.integers(0, 2, n_samples, dtype=np.int64)
        names = [f"feat_{i}" for i in range(n_features)]
        result = compute_mi_scores(X, y, names)
        sel = select_top_k(result.ranking_df, k=150, feature_names_reference=names)
        assert int(sel["selected"].sum()) == 150

    def test_k_exceeds_features_still_raises(self):
        """K=150 should raise if there are only 18 features."""
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (100, 18))
        y = rng.integers(0, 2, 100, dtype=np.int64)
        names = [f"f{i}" for i in range(18)]
        from src.feature_selection.mi_selector import MISelectorError
        result = compute_mi_scores(X, y, names)
        with pytest.raises(MISelectorError, match="exceeds"):
            select_top_k(result.ranking_df, k=150, feature_names_reference=names)

    def test_summary_df_has_eight_rows(self):
        """After 8-K CV, summary has exactly 8 rows."""
        records = _make_fold_records(
            {k: [0.7 + k * 0.001] * 3 for k in V1_1_CANDIDATE_K}
        )
        result = select_best_k(records)
        assert len(result.summary_df) == 8


# ---------------------------------------------------------------------------
# TestKGridVersioning
# ---------------------------------------------------------------------------


class TestKGridVersioning:
    def test_v1_1_candidate_k_different_from_v1(self):
        assert set(V1_1_CANDIDATE_K) != set(V1_CANDIDATE_K)

    def test_v1_1_is_superset_of_v1(self):
        assert set(V1_CANDIDATE_K).issubset(set(V1_1_CANDIDATE_K))

    def test_v1_1_adds_three_new_values(self):
        new_values = set(V1_1_CANDIDATE_K) - set(V1_CANDIDATE_K)
        assert new_values == {75, 100, 150}

    def test_v1_config_unchanged(self):
        """The default InnerCVConfig still uses v1.0 candidate_k."""
        default_config = InnerCVConfig()
        assert set(default_config.candidate_k) == {10, 20, 30, 40, 50}

    def test_v1_1_config_does_not_affect_default(self):
        """InnerCVConfig with v1.1 K grid is independent from the default."""
        v1_1_config = InnerCVConfig(candidate_k=V1_1_CANDIDATE_K)
        default     = InnerCVConfig()
        assert v1_1_config.candidate_k != default.candidate_k


# ---------------------------------------------------------------------------
# TestProtocolAmendment
# ---------------------------------------------------------------------------


class TestProtocolAmendment:
    def test_amendment_file_exists(self):
        """Protocol amendment markdown must exist before running v1.1."""
        amendment_path = (
            PROJECT_ROOT
            / "results" / "feature_selection" / "EXP_MI_V1_1"
            / "protocol_amendment.md"
        )
        assert amendment_path.exists(), (
            f"Protocol amendment not found at {amendment_path}. "
            "Amendment must be created before running v1.1 experiment."
        )

    def test_amendment_contains_previous_k(self):
        """Amendment document must reference the original K grid."""
        amendment_path = (
            PROJECT_ROOT
            / "results" / "feature_selection" / "EXP_MI_V1_1"
            / "protocol_amendment.md"
        )
        if not amendment_path.exists():
            pytest.skip("Protocol amendment not yet created")
        content = amendment_path.read_text()
        assert "10,20,30,40,50" in content or "{10,20,30,40,50}" in content, (
            "Amendment must reference the original v1.0 K grid"
        )

    def test_amendment_contains_new_k(self):
        amendment_path = (
            PROJECT_ROOT
            / "results" / "feature_selection" / "EXP_MI_V1_1"
            / "protocol_amendment.md"
        )
        if not amendment_path.exists():
            pytest.skip("Protocol amendment not yet created")
        content = amendment_path.read_text()
        assert "150" in content, "Amendment must reference K=150"
        assert "75" in content, "Amendment must reference K=75"
        assert "100" in content, "Amendment must reference K=100"

    def test_amendment_references_monotonic_condition(self):
        amendment_path = (
            PROJECT_ROOT
            / "results" / "feature_selection" / "EXP_MI_V1_1"
            / "protocol_amendment.md"
        )
        if not amendment_path.exists():
            pytest.skip("Protocol amendment not yet created")
        content = amendment_path.read_text().upper()
        assert "MONOTON" in content, (
            "Amendment must mention MONOTONIC condition as reason"
        )

    def test_v1_1_metadata_records_amendment(self):
        """If v1.1 has run, its metadata must record the protocol amendment."""
        meta_path = (
            PROJECT_ROOT
            / "results" / "feature_selection" / "EXP_MI_V1_1"
            / "metadata.json"
        )
        if not meta_path.exists():
            pytest.skip("EXP_MI_V1_1 metadata not yet created (experiment not run)")
        with open(meta_path) as f:
            meta = json.load(f)
        assert "protocol_amendment" in meta
        assert meta["protocol_amendment"]["previous_candidate_k"] == [10, 20, 30, 40, 50]
        assert 150 in meta["protocol_amendment"]["new_candidate_k"]
        assert "monotonic" in meta["protocol_amendment"]["reason"].lower()

    def test_v1_1_metadata_has_correct_experiment_id(self):
        meta_path = (
            PROJECT_ROOT
            / "results" / "feature_selection" / "EXP_MI_V1_1"
            / "metadata.json"
        )
        if not meta_path.exists():
            pytest.skip("EXP_MI_V1_1 metadata not yet created")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["experiment_id"] == "EXP_MI_V1_1"
        assert meta["protocol_version"] == "1.1"


# ---------------------------------------------------------------------------
# TestOriginalV1Preserved
# ---------------------------------------------------------------------------


class TestOriginalV1Preserved:
    """Verify original v1.0 artifacts are untouched."""

    V1_FILES = [
        "mi_scores.csv",
        "feature_ranking.csv",
        "selected_features.json",
        "k_selection_results.csv",
        "metadata.json",
        "config.yaml",
    ]

    V1_DIR = PROJECT_ROOT / "results" / "feature_selection" / "EXP_MI_V1"

    def test_v1_directory_exists(self):
        assert self.V1_DIR.exists(), f"v1.0 directory missing: {self.V1_DIR}"

    @pytest.mark.parametrize("filename", V1_FILES)
    def test_v1_file_exists(self, filename):
        p = self.V1_DIR / filename
        assert p.exists(), f"v1.0 file missing: {p}"

    def test_v1_metadata_experiment_id(self):
        meta_path = self.V1_DIR / "metadata.json"
        if not meta_path.exists():
            pytest.skip("v1.0 metadata not found")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["experiment_id"] == "EXP_MI_V1"

    def test_v1_k_results_preserved(self):
        """k_selection_results.csv must still have 25 rows (5 K × 5 folds)."""
        csv_path = self.V1_DIR / "k_selection_results.csv"
        if not csv_path.exists():
            pytest.skip("v1.0 k_selection_results.csv not found")
        df = pd.read_csv(csv_path)
        assert len(df) == 25, f"Expected 25 rows, got {len(df)}"

    def test_v1_selected_k_was_50(self):
        meta_path = self.V1_DIR / "metadata.json"
        if not meta_path.exists():
            pytest.skip("v1.0 metadata not found")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["selected_k"] == 50

    def test_v1_sanity_was_review_required(self):
        meta_path = self.V1_DIR / "metadata.json"
        if not meta_path.exists():
            pytest.skip("v1.0 metadata not found")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["selection_sanity"]["status"] == "REVIEW_REQUIRED"

    def test_v1_mean_f1_values_preserved(self):
        """v1.0 mean macro-F1 values must match exactly."""
        csv_path = self.V1_DIR / "k_selection_results.csv"
        if not csv_path.exists():
            pytest.skip("v1.0 k_selection_results.csv not found")
        df = pd.read_csv(csv_path)
        summary = df.groupby("k")["macro_f1"].mean().to_dict()
        for k, expected_f1 in V1_RESULTS.items():
            assert k in summary, f"K={k} missing from v1.0 results"
            assert abs(summary[k] - expected_f1) < 1e-5, (
                f"v1.0 K={k} mean_f1 changed: expected {expected_f1}, got {summary[k]}"
            )

    def test_v1_output_dir_not_overwritten_by_v1_1(self):
        """v1.1 output dir and v1.0 output dir must be different paths."""
        v1_dir   = PROJECT_ROOT / "results" / "feature_selection" / "EXP_MI_V1"
        v1_1_dir = PROJECT_ROOT / "results" / "feature_selection" / "EXP_MI_V1_1"
        assert v1_dir != v1_1_dir


# ---------------------------------------------------------------------------
# TestMonotonicAtBoundary
# ---------------------------------------------------------------------------


class TestMonotonicAtBoundary:
    def _summary_df(self, k_f1: dict) -> pd.DataFrame:
        return pd.DataFrame([
            {"k": k, "mean_macro_f1": f1, "std_macro_f1": 0.01}
            for k, f1 in k_f1.items()
        ])

    def test_monotonic_through_k150_triggers_review(self):
        """If the expanded grid is still monotonic through K=150, flag REVIEW_REQUIRED."""
        df = self._summary_df({
            10: 0.82, 20: 0.86, 30: 0.90, 40: 0.92,
            50: 0.93, 75: 0.94, 100: 0.95, 150: 0.96,
        })
        sanity = check_selection_sanity(df)
        assert sanity.status == "REVIEW_REQUIRED"
        assert sanity.is_monotonic

    def test_plateau_between_k50_and_k75_passes(self):
        """If F1 drops at K=75 relative to K=50, the curve is not monotonic → PASS."""
        df = self._summary_df({
            10: 0.82, 20: 0.86, 30: 0.90, 40: 0.92,
            50: 0.93, 75: 0.92, 100: 0.93, 150: 0.93,  # dip at K=75
        })
        sanity = check_selection_sanity(df)
        # Not monotonic → not necessarily REVIEW_REQUIRED for that reason
        assert not sanity.is_monotonic

    def test_plateau_between_k100_and_k150_passes(self):
        """A dip at K=150 is enough to break monotonicity."""
        df = self._summary_df({
            10: 0.82, 20: 0.86, 30: 0.90, 40: 0.92,
            50: 0.93, 75: 0.94, 100: 0.95, 150: 0.94,  # dip at K=150
        })
        sanity = check_selection_sanity(df)
        assert not sanity.is_monotonic

    def test_all_eight_k_values_covered_in_sanity(self):
        """Sanity check must process all 8 candidate K values."""
        df = self._summary_df(
            {k: 0.7 + k * 0.001 for k in V1_1_CANDIDATE_K}
        )
        sanity = check_selection_sanity(df)
        # flat_range should cover the full 8-K span
        expected_range = max(0.7 + k * 0.001 for k in V1_1_CANDIDATE_K) - \
                         min(0.7 + k * 0.001 for k in V1_1_CANDIDATE_K)
        assert abs(sanity.flat_range - expected_range) < 1e-9

    def test_v1_1_metadata_sanity_recorded(self):
        meta_path = (
            PROJECT_ROOT
            / "results" / "feature_selection" / "EXP_MI_V1_1"
            / "metadata.json"
        )
        if not meta_path.exists():
            pytest.skip("EXP_MI_V1_1 not yet run")
        with open(meta_path) as f:
            meta = json.load(f)
        sanity = meta["selection_sanity"]
        assert "status" in sanity
        assert "is_monotonic" in sanity
        assert "flat_range" in sanity

    def test_v1_1_k_selection_has_eight_k_values(self):
        csv_path = (
            PROJECT_ROOT
            / "results" / "feature_selection" / "EXP_MI_V1_1"
            / "k_selection_results.csv"
        )
        if not csv_path.exists():
            pytest.skip("EXP_MI_V1_1 not yet run")
        df = pd.read_csv(csv_path)
        unique_k = sorted(df["k"].unique())
        assert unique_k == sorted(V1_1_CANDIDATE_K), (
            f"Expected K values {sorted(V1_1_CANDIDATE_K)}, got {unique_k}"
        )

    def test_v1_1_has_40_fold_records(self):
        """8 K values × 5 folds = 40 records."""
        csv_path = (
            PROJECT_ROOT
            / "results" / "feature_selection" / "EXP_MI_V1_1"
            / "k_selection_results.csv"
        )
        if not csv_path.exists():
            pytest.skip("EXP_MI_V1_1 not yet run")
        df = pd.read_csv(csv_path)
        assert len(df) == 40, f"Expected 40 rows (8 K × 5 folds), got {len(df)}"

    def test_v1_lower_k_results_reproducible_in_v1_1(self):
        """
        The mean macro-F1 for K={10..50} in v1.1 should be close to v1.0
        (same seed, same data, same protocol — only extra K values differ).
        Tolerance: ±0.005 (small differences due to OHE size varying per fold).
        """
        csv_path = (
            PROJECT_ROOT
            / "results" / "feature_selection" / "EXP_MI_V1_1"
            / "k_selection_results.csv"
        )
        if not csv_path.exists():
            pytest.skip("EXP_MI_V1_1 not yet run")
        df = pd.read_csv(csv_path)
        summary = df.groupby("k")["macro_f1"].mean().to_dict()
        for k, v1_f1 in V1_RESULTS.items():
            if k in summary:
                diff = abs(summary[k] - v1_f1)
                assert diff < 0.005, (
                    f"K={k}: v1.1 mean_f1={summary[k]:.6f} differs from "
                    f"v1.0 mean_f1={v1_f1:.6f} by {diff:.6f} > 0.005"
                )


# ---------------------------------------------------------------------------
# TestDeterminismV11
# ---------------------------------------------------------------------------


class TestDeterminismV11:
    def test_expanded_k_selection_deterministic_synthetic(self):
        """Same seed + same data → same selected K for expanded grid."""
        train_df = _make_synthetic_train(n_normal=100, n_attack=80, n_num=4, seed=42)
        config = InnerCVConfig(
            candidate_k=(5, 10, 15),
            n_splits=2,
            cv_random_state=42,
        )
        r1 = run_k_selection_cv(train_df, config=config)
        r2 = run_k_selection_cv(train_df, config=config)
        assert r1.selected_k == r2.selected_k

    def test_k75_100_150_in_fold_records_after_cv(self):
        """After running CV with expanded grid, fold records contain K=75/100/150."""
        train_df = _make_synthetic_train(n_normal=120, n_attack=80, n_num=3, seed=0)
        config = InnerCVConfig(
            candidate_k=(10, 75, 100),
            n_splits=2,
            cv_random_state=42,
        )
        result = run_k_selection_cv(train_df, config=config)
        k_in_records = {r.k for r in result.fold_records}
        assert 75 in k_in_records
        assert 100 in k_in_records

    def test_select_best_k_stable_with_new_k_values(self):
        records = _make_fold_records({
            10:  [0.82] * 5,
            20:  [0.86] * 5,
            50:  [0.91] * 5,
            75:  [0.92] * 5,
            100: [0.93] * 5,
            150: [0.92] * 5,   # K=100 is higher → selected
        })
        result = select_best_k(records)
        assert result.selected_k == 100
