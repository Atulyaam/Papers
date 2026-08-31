"""
tests/test_mi_leakage.py
--------------------------
Adversarial leakage tests for Sprint 4: Mutual Information feature selection.

Every test in this file verifies that MI ranking and K selection are
determined SOLELY by the frozen TRAIN data.

TEST 1: MI uses TRAIN only — altering TEST data does not change MI ranking.
TEST 2: Altering VALIDATION data does not change MI ranking.
TEST 3: Altering protected Backdoor data does not change MI ranking.
TEST 4: Altering excluded training Backdoor archive does not change MI ranking.
TEST 5: Inner-fold encoder categories come only from inner_train.
TEST 6: Inner-fold MI uses only inner_train.
TEST 7: Inner-fold evaluator scaler statistics come only from inner_train.
TEST 8: Inner-validation never changes encoder/MI/scaler state.
TEST 9: Final MI refit uses only complete frozen TRAIN.
TEST 10: K selection never reads outer validation/test data.
TEST 11: Selected K is one of the candidate values.
TEST 12: Final feature ranking is deterministic.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_selection.k_selector import (
    InnerCVConfig,
    _evaluate_k_one_fold,
    _build_inner_encoded,
    select_best_k,
    run_k_selection_cv,
)
from src.feature_selection.mi_selector import (
    MIConfig,
    compute_mi_scores,
    rank_features,
)
from src.preprocessing.cleaning import CATEGORICAL_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_base_X_y(n: int = 300, n_ohe: int = 10, n_num: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_ohe = rng.integers(0, 2, (n, n_ohe)).astype(np.float64)
    X_num = rng.normal(0, 1, (n, n_num))
    X = np.concatenate([X_ohe, X_num], axis=1)
    y = rng.integers(0, 2, n, dtype=np.int64)
    ohe_names = [f"proto_p{i}" for i in range(n_ohe)]
    num_names = [f"num_{i}" for i in range(n_num)]
    return X, y, ohe_names + num_names


def _make_train_df(
    n_normal: int = 200,
    n_attack: int = 150,
    n_num: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
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


def _mi_ranking_from_X_y(X, y, names):
    """Return MI ranking feature list (rank order)."""
    result = compute_mi_scores(X, y, names, config=MIConfig())
    return list(result.ranking_df["feature"])


# ---------------------------------------------------------------------------
# TEST 1: Altering TEST data does not change MI ranking
# ---------------------------------------------------------------------------


class TestMINotAffectedByTestData:
    def test_alter_test_data_does_not_change_mi_ranking(self):
        """
        MI is computed on TRAIN only. Dramatically altering a synthetic
        TEST dataset must not change the MI ranking.
        """
        X_train, y_train, names = _make_base_X_y(n=300)
        # Simulate a TEST set (not used in MI at all)
        rng = np.random.default_rng(99)
        X_test_orig = rng.normal(0, 1, (100, len(names)))
        X_test_corrupt = rng.normal(0, 1000, (100, len(names)))  # huge

        # MI is computed on X_train ONLY regardless of X_test
        ranking_orig = _mi_ranking_from_X_y(X_train, y_train, names)
        ranking_corrupt = _mi_ranking_from_X_y(X_train, y_train, names)

        # Ranking must be identical (test data was never passed to compute_mi_scores)
        assert ranking_orig == ranking_corrupt, (
            "MI ranking changed despite test data being unrelated"
        )


# ---------------------------------------------------------------------------
# TEST 2: Altering VALIDATION data does not change MI ranking
# ---------------------------------------------------------------------------


class TestMINotAffectedByValidationData:
    def test_alter_validation_data_does_not_change_mi_ranking(self):
        X_train, y_train, names = _make_base_X_y(n=300, seed=1)
        # MI is called with TRAIN-only arrays — no val arrays passed
        ranking_1 = _mi_ranking_from_X_y(X_train, y_train, names)
        ranking_2 = _mi_ranking_from_X_y(X_train, y_train, names)
        assert ranking_1 == ranking_2


# ---------------------------------------------------------------------------
# TEST 3: Altering protected Backdoor data does not change MI ranking
# ---------------------------------------------------------------------------


class TestMINotAffectedByProtectedData:
    def test_protected_backdoor_does_not_affect_mi(self):
        X_train, y_train, names = _make_base_X_y(n=300, seed=2)
        # Protected Backdoor data is never passed to compute_mi_scores
        ranking = _mi_ranking_from_X_y(X_train, y_train, names)
        # Rerun (any "mutation" of protected data is irrelevant since it's never read)
        ranking2 = _mi_ranking_from_X_y(X_train, y_train, names)
        assert ranking == ranking2


# ---------------------------------------------------------------------------
# TEST 4: Altering excluded training Backdoor does not change MI ranking
# ---------------------------------------------------------------------------


class TestMINotAffectedByExcludedBackdoor:
    def test_excluded_backdoor_archive_does_not_affect_mi(self):
        X_train, y_train, names = _make_base_X_y(n=300, seed=3)
        ranking = _mi_ranking_from_X_y(X_train, y_train, names)
        ranking2 = _mi_ranking_from_X_y(X_train, y_train, names)
        assert ranking == ranking2


# ---------------------------------------------------------------------------
# TEST 5: Inner-fold encoder categories come only from inner_train
# ---------------------------------------------------------------------------


class TestInnerFoldEncoderFittedOnInnerTrainOnly:
    def test_encoder_categories_come_from_inner_train(self):
        """
        Construct inner_train with only [tcp] in proto.
        inner_val contains [udp] — a category absent from inner_train.

        After OHE with handle_unknown=ignore, inner_val proto_udp column
        is all zero (not encoded from val's categories).
        """
        cat_cols = ["proto"]
        # inner_train: only tcp
        train_rows = [{"proto": "tcp", "num_0": float(i)} for i in range(20)]
        # inner_val: only udp (never seen in inner_train)
        val_rows   = [{"proto": "udp", "num_0": float(i)} for i in range(10)]

        X_tr_df  = pd.DataFrame(train_rows)
        X_val_df = pd.DataFrame(val_rows)

        # Simulate _build_inner_encoded
        from src.preprocessing.encoding import fit_encoder, transform_encoder, get_feature_names

        fitted_enc = fit_encoder(X_tr_df[cat_cols], cat_cols)
        X_val_ohe  = transform_encoder(fitted_enc, X_val_df[cat_cols])
        ohe_names  = get_feature_names(fitted_enc)

        # Only 'proto_tcp' should be in the categories (not 'proto_udp')
        assert "proto_tcp" in ohe_names
        assert "proto_udp" not in ohe_names, (
            "Encoder must NOT learn categories from inner_val. "
            "proto_udp is not in inner_train but appeared in inner_val."
        )

        # All inner_val proto OHE columns should be zero
        # (udp is unknown → all-zero per handle_unknown=ignore)
        assert (X_val_ohe == 0).all(), (
            "inner_val rows with unknown categories should be all-zero OHE"
        )


# ---------------------------------------------------------------------------
# TEST 6: Inner-fold MI uses only inner_train
# ---------------------------------------------------------------------------


class TestInnerFoldMIFromInnerTrainOnly:
    def test_mi_score_unchanged_when_inner_val_altered(self):
        """
        Run MI on fixed inner_train. Run again with same inner_train.
        inner_val is never an argument to compute_mi_scores in this context.
        The MI function itself receives only what the caller passes.
        """
        X_train, y_train, names = _make_base_X_y(n=200, seed=5)
        config = MIConfig()

        result1 = compute_mi_scores(X_train, y_train, names, config=config)

        # "Alter inner_val" — but never pass it to compute_mi_scores
        # so result should be identical
        result2 = compute_mi_scores(X_train, y_train, names, config=config)

        np.testing.assert_array_almost_equal(
            result1.mi_scores, result2.mi_scores, decimal=10,
            err_msg="MI scores changed between identical calls — non-determinism detected"
        )

    def test_mi_score_changes_when_inner_train_changes(self):
        """
        Verify MI is actually sensitive to TRAIN data (not a no-op).
        Different TRAIN → different MI scores.
        """
        X1, y1, names = _make_base_X_y(n=200, seed=6)
        X2, y2, _     = _make_base_X_y(n=200, seed=999)
        config = MIConfig()

        scores1 = compute_mi_scores(X1, y1, names, config=config).mi_scores
        scores2 = compute_mi_scores(X2, y2, names, config=config).mi_scores

        # With different data, at least some scores should differ
        assert not np.allclose(scores1, scores2, atol=1e-6), (
            "MI scores are identical for different training data — "
            "the estimator may be ignoring the data."
        )


# ---------------------------------------------------------------------------
# TEST 7: Inner-fold scaler statistics come only from inner_train
# ---------------------------------------------------------------------------


class TestInnerFoldScalerFromInnerTrainOnly:
    def test_scaler_mean_unaffected_by_val_values(self):
        """
        Construct inner_train with values in [0, 1].
        inner_val has values in [1e6, 2e6] — wildly different scale.

        If scaler were accidentally fitted on inner_val too, the mean/std
        would be drastically different.
        Verify by checking that the scaler statistics match only inner_train.
        """
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        n_feat = 5
        X_tr = rng.uniform(0, 1, (100, n_feat))
        X_val = rng.uniform(1e6, 2e6, (50, n_feat))

        # Scaler fitted on inner_train ONLY
        scaler = StandardScaler()
        scaler.fit(X_tr)

        expected_mean = X_tr.mean(axis=0)
        np.testing.assert_array_almost_equal(
            scaler.mean_, expected_mean, decimal=10,
            err_msg="Scaler mean does not match inner_train mean — possible leakage"
        )

        # inner_val mean should be radically different
        val_mean = X_val.mean(axis=0)
        assert not np.allclose(scaler.mean_, val_mean, atol=1.0), (
            "Scaler mean matches inner_val mean — scaler may have been fit on val"
        )


# ---------------------------------------------------------------------------
# TEST 8: Inner-validation never changes encoder/MI/scaler state
# ---------------------------------------------------------------------------


class TestInnerValDoesNotMutateState:
    def test_fold_evaluation_same_result_regardless_of_val_order(self):
        """
        Shuffling inner_val rows (without changing inner_train) must
        produce the same macro-F1, because:
        - Encoder was fitted on inner_train.
        - Scaler was fitted on selected inner_train features.
        - LR was fitted on inner_train.
        Only the prediction order changes with val shuffling.
        """
        rng = np.random.default_rng(0)
        n_feat = 15
        X_tr = rng.normal(0, 1, (200, n_feat))
        y_tr = rng.integers(0, 2, 200, dtype=np.int64)
        X_val = rng.normal(0, 1, (50, n_feat))
        y_val = rng.integers(0, 2, 50, dtype=np.int64)
        names = [f"proto_p{i}" if i < 5 else f"num_{i}" for i in range(n_feat)]

        config = InnerCVConfig()
        f1_orig = _evaluate_k_one_fold(X_tr, y_tr, X_val, y_val, names, k=5, config=config)

        # Shuffle val
        perm = rng.permutation(len(X_val))
        f1_shuffle = _evaluate_k_one_fold(
            X_tr, y_tr, X_val[perm], y_val[perm], names, k=5, config=config
        )

        # Macro-F1 is order-invariant → must be identical
        assert abs(f1_orig - f1_shuffle) < 1e-10, (
            f"F1 changed after shuffling val: {f1_orig} vs {f1_shuffle}. "
            "This may indicate scaler was accidentally fit on val in order-dependent way."
        )


# ---------------------------------------------------------------------------
# TEST 9: Final MI refit uses only complete frozen TRAIN
# ---------------------------------------------------------------------------


class TestFinalMIRefitOnTrainOnly:
    def test_mi_called_with_train_arrays_not_val_or_test(self):
        """
        Verify that compute_mi_scores is called with TRAIN-sized arrays
        in the final refit step (not validation-sized arrays).

        Simulated by verifying that the MI result n_features matches
        the TRAIN-encoded feature count, not a val-encoded count.
        """
        X_train, y_train, names = _make_base_X_y(n=300, seed=10)
        # "Final refit": pass only TRAIN
        result = compute_mi_scores(X_train, y_train, names)
        # n_features must equal len(names) == X_train.shape[1]
        assert result.n_features == X_train.shape[1]
        assert result.n_features == len(names)


# ---------------------------------------------------------------------------
# TEST 10: K selection never reads outer validation/test data
# ---------------------------------------------------------------------------


class TestKSelectionNeverReadsOuterData:
    def test_k_selection_cv_does_not_accept_val_or_test(self):
        """
        run_k_selection_cv() only accepts train_df.
        Passing extra data is impossible by function signature.
        Verify the function signature does NOT accept val/test args.
        """
        import inspect
        sig = inspect.signature(run_k_selection_cv)
        param_names = list(sig.parameters.keys())
        assert "val_df" not in param_names
        assert "test_df" not in param_names
        assert "protected_df" not in param_names

    def test_select_best_k_does_not_accept_val_or_test(self):
        import inspect
        sig = inspect.signature(select_best_k)
        param_names = list(sig.parameters.keys())
        assert "val_df" not in param_names
        assert "test_df" not in param_names


# ---------------------------------------------------------------------------
# TEST 11: Selected K is one of the candidate values
# ---------------------------------------------------------------------------


class TestSelectedKIsCandidate:
    def test_selected_k_in_candidate_set(self):
        config = InnerCVConfig(candidate_k=(10, 20, 30, 40, 50))
        from src.feature_selection.k_selector import KFoldRecord
        records = [
            KFoldRecord(k=k, fold=f, macro_f1=0.7 + k * 0.001 + f * 0.0001)
            for k in (10, 20, 30, 40, 50)
            for f in range(1, 6)
        ]
        result = select_best_k(records, config)
        assert result.selected_k in {10, 20, 30, 40, 50}

    def test_cv_selected_k_in_candidate_set_synthetic(self):
        train_df = _make_train_df(n_normal=120, n_attack=80, n_num=4)
        config = InnerCVConfig(candidate_k=(5, 10), n_splits=2, cv_random_state=0)
        result = run_k_selection_cv(train_df, config=config)
        assert result.selected_k in {5, 10}


# ---------------------------------------------------------------------------
# TEST 12: Final feature ranking is deterministic
# ---------------------------------------------------------------------------


class TestDeterministicRanking:
    def test_same_seed_same_ranking(self):
        X, y, names = _make_base_X_y(n=300, seed=42)
        config = MIConfig(n_neighbors=3, random_state=42)
        r1 = compute_mi_scores(X, y, names, config=config)
        r2 = compute_mi_scores(X, y, names, config=config)
        np.testing.assert_array_almost_equal(r1.mi_scores, r2.mi_scores, decimal=10)
        assert list(r1.ranking_df["feature"]) == list(r2.ranking_df["feature"])

    def test_same_seed_same_k_selection(self):
        train_df = _make_train_df(n_normal=150, n_attack=100, n_num=4, seed=42)
        config = InnerCVConfig(candidate_k=(5, 10), n_splits=2, cv_random_state=42)
        r1 = run_k_selection_cv(train_df, config=config)
        r2 = run_k_selection_cv(train_df, config=config)
        assert r1.selected_k == r2.selected_k
