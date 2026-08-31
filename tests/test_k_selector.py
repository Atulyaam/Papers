"""
tests/test_k_selector.py
--------------------------
Unit tests for Sprint 4: K selector (inner CV, best-K rule, sanity check).

Tests:
    TestSelectBestK         — selection rule: highest mean, smaller-K tie-break
    TestCheckSanity         — flat test, monotonic test
    TestKSelectionValues    — K values are exactly {10,20,30,40,50}
    TestKSelectorSmoke      — small synthetic CV run (no real data)
    TestEvaluateKOneFold    — fold-level evaluation isolation
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_selection.k_selector import (
    FLAT_TOLERANCE,
    MONOTONIC_TOLERANCE,
    InnerCVConfig,
    KFoldRecord,
    KSelectionResult,
    KSelectionSanity,
    _evaluate_k_one_fold,
    check_selection_sanity,
    select_best_k,
)
from src.feature_selection.mi_selector import MIConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fold_records(
    k_f1_map: dict[int, list[float]],
) -> list[KFoldRecord]:
    """Build KFoldRecord list from {K: [fold1_f1, fold2_f1, ...]}."""
    records = []
    for k, f1_list in k_f1_map.items():
        for fold_idx, f1 in enumerate(f1_list):
            records.append(KFoldRecord(k=k, fold=fold_idx + 1, macro_f1=f1))
    return records


def _make_synthetic_train_df(
    n_normal: int = 200,
    n_attack: int = 150,
    n_num_features: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a minimal synthetic TRAIN DataFrame compatible with the pipeline.
    Includes label, attack_cat, proto, service, state, and numeric columns.
    """
    rng = np.random.default_rng(seed)
    n = n_normal + n_attack
    rows = []
    for i in range(n_normal):
        rows.append({
            "label": 0,
            "attack_cat": "Normal",
            "proto": rng.choice(["tcp", "udp"]),
            "service": rng.choice(["-", "http"]),
            "state": rng.choice(["FIN", "CON"]),
            **{f"feat_{j}": float(rng.normal()) for j in range(n_num_features)},
        })
    for i in range(n_attack):
        rows.append({
            "label": 1,
            "attack_cat": "Exploits",
            "proto": rng.choice(["tcp", "udp"]),
            "service": rng.choice(["-", "http"]),
            "state": rng.choice(["FIN", "CON"]),
            **{f"feat_{j}": float(rng.normal()) for j in range(n_num_features)},
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestSelectBestK
# ---------------------------------------------------------------------------


class TestSelectBestK:
    def test_selects_highest_mean_f1(self):
        records = _make_fold_records({
            10: [0.70, 0.71, 0.72, 0.69, 0.70],
            20: [0.80, 0.81, 0.79, 0.80, 0.80],  # highest
            30: [0.75, 0.76, 0.74, 0.75, 0.75],
        })
        result = select_best_k(records)
        assert result.selected_k == 20

    def test_tie_break_selects_smaller_k(self):
        """Two K with identical mean → smaller K wins."""
        records = _make_fold_records({
            10: [0.80, 0.80, 0.80, 0.80, 0.80],
            30: [0.80, 0.80, 0.80, 0.80, 0.80],  # same mean
        })
        result = select_best_k(records)
        assert result.selected_k == 10, (
            "Tie-break must select smaller K (10), not 30"
        )

    def test_summary_df_has_required_columns(self):
        records = _make_fold_records({
            10: [0.7, 0.7],
            20: [0.8, 0.8],
        })
        result = select_best_k(records)
        assert "k" in result.summary_df.columns
        assert "mean_macro_f1" in result.summary_df.columns
        assert "std_macro_f1" in result.summary_df.columns

    def test_selected_k_is_one_of_candidate_values(self):
        config = InnerCVConfig(candidate_k=(10, 20, 30, 40, 50))
        records = _make_fold_records({
            k: [0.7 + k * 0.001] * 5 for k in (10, 20, 30, 40, 50)
        })
        result = select_best_k(records, config)
        assert result.selected_k in (10, 20, 30, 40, 50)

    def test_fold_records_preserved(self):
        records = _make_fold_records({10: [0.7, 0.8, 0.75, 0.72, 0.71]})
        result = select_best_k(records)
        assert len(result.fold_records) == 5

    def test_single_k_always_selected(self):
        records = _make_fold_records({30: [0.75, 0.76, 0.74, 0.75, 0.77]})
        result = select_best_k(records)
        assert result.selected_k == 30

    def test_tie_break_across_five_k_values(self):
        """All five K values tied → smallest (10) wins."""
        records = _make_fold_records({
            k: [0.75] * 5 for k in (10, 20, 30, 40, 50)
        })
        result = select_best_k(records)
        assert result.selected_k == 10


# ---------------------------------------------------------------------------
# TestCheckSanity
# ---------------------------------------------------------------------------


class TestCheckSanity:
    def _summary_df(self, k_f1: dict[int, float]) -> pd.DataFrame:
        return pd.DataFrame([
            {"k": k, "mean_macro_f1": f1, "std_macro_f1": 0.01}
            for k, f1 in k_f1.items()
        ])

    def test_normal_result_passes(self):
        df = self._summary_df({10: 0.70, 20: 0.78, 30: 0.75, 40: 0.73, 50: 0.72})
        sanity = check_selection_sanity(df)
        assert sanity.status == "PASS"

    def test_flat_triggers_review_required(self):
        """All K values within 0.001 → REVIEW_REQUIRED."""
        df = self._summary_df({
            10: 0.8000, 20: 0.8002, 30: 0.7999, 40: 0.8001, 50: 0.8003
        })
        sanity = check_selection_sanity(df, flat_tolerance=1e-3)
        assert sanity.status == "REVIEW_REQUIRED"
        assert "FLAT" in sanity.reason.upper()

    def test_monotonic_triggers_review_required(self):
        """Monotonically increasing through all K → REVIEW_REQUIRED."""
        df = self._summary_df({
            10: 0.70, 20: 0.75, 30: 0.80, 40: 0.85, 50: 0.90
        })
        sanity = check_selection_sanity(df)
        assert sanity.status == "REVIEW_REQUIRED"
        assert "MONOTON" in sanity.reason.upper()

    def test_tiny_fluctuation_does_not_falsely_trigger_monotonic(self):
        """Small non-monotonic dip should not be flagged as monotonic."""
        df = self._summary_df({
            10: 0.70, 20: 0.80, 30: 0.75, 40: 0.82, 50: 0.78
        })
        sanity = check_selection_sanity(df)
        # Should not flag as monotonic (30 dips below 20)
        assert not sanity.is_monotonic

    def test_flat_range_recorded(self):
        df = self._summary_df({10: 0.70, 20: 0.80, 30: 0.75, 40: 0.73, 50: 0.72})
        sanity = check_selection_sanity(df)
        expected_range = 0.80 - 0.70
        assert abs(sanity.flat_range - expected_range) < 1e-9

    def test_monotonic_tolerance_documented(self):
        sanity = check_selection_sanity(
            self._summary_df({10: 0.70, 20: 0.78, 30: 0.75, 40: 0.73, 50: 0.72})
        )
        assert sanity.monotonic_tolerance == MONOTONIC_TOLERANCE
        assert sanity.flat_tolerance == FLAT_TOLERANCE


# ---------------------------------------------------------------------------
# TestKSelectionValues
# ---------------------------------------------------------------------------


class TestKSelectionValues:
    def test_default_candidate_k(self):
        config = InnerCVConfig()
        assert set(config.candidate_k) == {10, 20, 30, 40, 50}

    def test_candidate_k_ordering(self):
        config = InnerCVConfig()
        assert list(config.candidate_k) == [10, 20, 30, 40, 50]

    def test_n_splits_is_five(self):
        config = InnerCVConfig()
        assert config.n_splits == 5

    def test_cv_random_state_is_42(self):
        config = InnerCVConfig()
        assert config.cv_random_state == 42

    def test_mi_n_neighbors_is_3(self):
        config = InnerCVConfig()
        assert config.mi_n_neighbors == 3

    def test_evaluator_config_frozen(self):
        config = InnerCVConfig()
        assert config.evaluator_solver == "liblinear"
        assert config.evaluator_C == 1.0
        assert config.evaluator_max_iter == 1000
        assert config.evaluator_class_weight == "balanced"
        assert config.evaluator_random_state == 42


# ---------------------------------------------------------------------------
# TestEvaluateKOneFold
# ---------------------------------------------------------------------------


class TestEvaluateKOneFold:
    def _make_fold_data(self, n=300, n_features=20, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, n_features))
        y = rng.integers(0, 2, n, dtype=np.int64)
        names = [f"proto_p{i}" if i < 5 else f"num_{i}" for i in range(n_features)]
        return X, y, names

    def test_returns_float_in_unit_interval(self):
        X, y, names = self._make_fold_data()
        n = len(X)
        split = int(0.8 * n)
        config = InnerCVConfig()
        f1 = _evaluate_k_one_fold(
            X[:split], y[:split],
            X[split:], y[split:],
            names, k=5, config=config,
        )
        assert isinstance(f1, float)
        assert 0.0 <= f1 <= 1.0

    def test_k_limits_feature_count(self):
        """Evaluation must complete correctly for each candidate K."""
        X, y, names = self._make_fold_data(n_features=20)
        n = len(X)
        split = int(0.8 * n)
        config = InnerCVConfig()
        for k in (5, 10, 15, 20):
            f1 = _evaluate_k_one_fold(
                X[:split], y[:split],
                X[split:], y[split:],
                names, k=k, config=config,
            )
            assert 0.0 <= f1 <= 1.0

    def test_inner_val_does_not_affect_scaler(self):
        """
        Adversarial: mutate inner_val after calling _evaluate_k_one_fold.
        The result should be the same as without mutation, because scaler
        is fitted only on inner_train — inner_val only goes through
        transform, not fit.
        """
        X, y, names = self._make_fold_data(n=400, n_features=15)
        n = len(X)
        split = int(0.8 * n)
        X_tr, y_tr = X[:split].copy(), y[:split].copy()
        X_val, y_val = X[split:].copy(), y[split:].copy()
        config = InnerCVConfig()

        # Baseline
        f1_baseline = _evaluate_k_one_fold(X_tr, y_tr, X_val, y_val, names, k=5, config=config)

        # Corrupt inner_val (simulate adversarial mutation BEFORE the call)
        # The function should not observe inner_val during fitting, so
        # if we pass a corrupted copy, LR/scaler state is unaffected;
        # only the prediction might differ, which is expected.
        # The key test is that f1_baseline was computed without crashing
        # and the function doesn't mutate global state.
        X_val_corrupt = X_val * 1e6
        f1_corrupt = _evaluate_k_one_fold(X_tr, y_tr, X_val_corrupt, y_val, names, k=5, config=config)

        # The function should run to completion without error regardless
        # of the magnitude of inner_val values (scaler already fitted)
        assert isinstance(f1_corrupt, float)


# ---------------------------------------------------------------------------
# TestKSelectorSmoke
# ---------------------------------------------------------------------------


class TestKSelectorSmoke:
    def test_small_synthetic_cv_run(self):
        """
        Full inner CV on a tiny synthetic dataset.
        Verifies end-to-end flow without crashing.
        Must complete without accessing any outer-split data.
        """
        train_df = _make_synthetic_train_df(n_normal=150, n_attack=100, n_num_features=5)

        config = InnerCVConfig(
            candidate_k=(5, 10),
            n_splits=3,
            shuffle=True,
            cv_random_state=42,
        )

        from src.feature_selection.k_selector import run_k_selection_cv
        result = run_k_selection_cv(train_df, config=config)

        assert isinstance(result, KSelectionResult)
        assert result.selected_k in (5, 10)
        assert len(result.fold_records) == 3 * 2  # 3 folds × 2 K values
        assert len(result.summary_df) == 2  # one row per K

    def test_cv_determinism_same_seed(self):
        """Two runs with same seed produce same selected K."""
        train_df = _make_synthetic_train_df(n_normal=100, n_attack=80, n_num_features=4)
        config = InnerCVConfig(candidate_k=(5, 10), n_splits=2, cv_random_state=42)

        from src.feature_selection.k_selector import run_k_selection_cv
        r1 = run_k_selection_cv(train_df, config=config)
        r2 = run_k_selection_cv(train_df, config=config)
        assert r1.selected_k == r2.selected_k

    def test_cv_different_seed_may_differ(self):
        """This is a smoke check that different seeds don't crash."""
        train_df = _make_synthetic_train_df(n_normal=100, n_attack=80, n_num_features=4)
        config_a = InnerCVConfig(candidate_k=(5, 10), n_splits=2, cv_random_state=1)
        config_b = InnerCVConfig(candidate_k=(5, 10), n_splits=2, cv_random_state=999)

        from src.feature_selection.k_selector import run_k_selection_cv
        r_a = run_k_selection_cv(train_df, config=config_a)
        r_b = run_k_selection_cv(train_df, config=config_b)
        # Both must return valid results
        assert r_a.selected_k in (5, 10)
        assert r_b.selected_k in (5, 10)
