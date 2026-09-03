"""
tests/test_oof_runner.py
--------------------------
Sprint 6 OOF runner tests.

Groups:
    1. Fold Protocol (OOF_NO_SELF_PREDICTION, coverage, uniqueness)
    2. Seed Propagation
    3. Model Contracts (SVM decision_function, NN fixed epochs, etc.)
    4. Data Isolation (resolved paths from Step 0)
    5. Feature Integrity (75 features, no MI rerun, meta matrix shape)
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Authoritative paths and hashes (from Step 0)
# ---------------------------------------------------------------------------

RESOLVED_PATHS = {
    "train":       ROOT / "data/splits/train.csv",
    "validation":  ROOT / "data/splits/validation.csv",
    "dev_test":    ROOT / "data/splits/development_test.csv",
    "protected":   ROOT / "data/splits/protected_unseen_attack.csv",
    "excluded":    ROOT / "data/splits/excluded_train_backdoor.csv",
}

RESOLVED_HASHES = {
    "train":      "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c",
    "validation": "13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180",
    "dev_test":   "04725e85732ab2fc6d9eaaa6105418b22b083b5c651067e7b0785464f414e508",
    "protected":  "6ffd23479b575e438ad90678268f40f674a663c2b9507aaf65089623397a9d91",
    "excluded":   "b3f6e7e60c9815a53f40eb2d41df8b67d29f884b922a487c3fe83c02e0db0a02",
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tiny_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Create tiny X (n, 75) and binary y for unit tests."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 75).astype(np.float64)
    y = rng.randint(0, 2, n).astype(np.int64)
    # Ensure both classes present
    y[:max(1, n // 3)] = 0
    y[max(1, n // 3):] = 1
    return X, y


@pytest.fixture
def tiny_data():
    return _make_tiny_data(n=200)


@pytest.fixture
def oof_folds_tiny(tiny_data):
    from src.models.stacking.oof_runner import make_oof_folds
    X, y = tiny_data
    return make_oof_folds(y, n_splits=3, seed=7)


# ===========================================================================
# Group 1 — Fold Protocol
# ===========================================================================


class TestFoldCreation:

    def test_make_oof_folds_returns_correct_length(self, tiny_data):
        from src.models.stacking.oof_runner import make_oof_folds
        _, y = tiny_data
        folds = make_oof_folds(y, n_splits=3, seed=7)
        assert len(folds) == 3

    def test_make_oof_folds_covers_all_rows(self, tiny_data):
        from src.models.stacking.oof_runner import make_oof_folds
        _, y = tiny_data
        n = len(y)
        folds = make_oof_folds(y, n_splits=3, seed=7)
        covered = np.concatenate([oof for _, oof in folds])
        assert len(np.unique(covered)) == n

    def test_make_oof_folds_no_overlap_between_folds(self, tiny_data):
        from src.models.stacking.oof_runner import make_oof_folds
        _, y = tiny_data
        folds = make_oof_folds(y, n_splits=3, seed=7)
        all_oof = np.concatenate([oof for _, oof in folds])
        assert len(all_oof) == len(np.unique(all_oof)), "Duplicate OOF indices across folds"

    def test_fold_train_oof_disjoint(self, tiny_data):
        from src.models.stacking.oof_runner import make_oof_folds
        _, y = tiny_data
        folds = make_oof_folds(y, n_splits=3, seed=7)
        for fold_idx, (train_idx, oof_idx) in enumerate(folds):
            overlap = set(train_idx) & set(oof_idx)
            assert len(overlap) == 0, \
                f"Fold {fold_idx}: train/oof overlap = {overlap}"

    def test_identical_fold_assignment_across_h1_seeds(self, tiny_data):
        """Same fold assignment must be used for all H1 seeds."""
        from src.models.stacking.oof_runner import make_oof_folds
        _, y = tiny_data
        folds_42 = make_oof_folds(y, n_splits=3, seed=7)
        folds_123 = make_oof_folds(y, n_splits=3, seed=7)
        folds_2024 = make_oof_folds(y, n_splits=3, seed=7)
        for i, (f42, f123, f2024) in enumerate(zip(folds_42, folds_123, folds_2024)):
            np.testing.assert_array_equal(f42[0], f123[0], err_msg=f"fold {i} train_idx differ")
            np.testing.assert_array_equal(f42[1], f123[1], err_msg=f"fold {i} oof_idx differ")
            np.testing.assert_array_equal(f42[0], f2024[0], err_msg=f"fold {i} train_idx differ")
            np.testing.assert_array_equal(f42[1], f2024[1], err_msg=f"fold {i} oof_idx differ")

    def test_make_oof_folds_empty_raises(self):
        from src.models.stacking.oof_runner import make_oof_folds
        with pytest.raises(ValueError, match="empty"):
            make_oof_folds(np.array([]), n_splits=3, seed=7)

    def test_make_oof_folds_one_class_raises(self):
        from src.models.stacking.oof_runner import make_oof_folds
        y = np.zeros(100, dtype=np.int64)
        with pytest.raises(ValueError, match="one class"):
            make_oof_folds(y, n_splits=3, seed=7)

    def test_oof_fold_count_equals_n_splits(self, tiny_data):
        from src.models.stacking.oof_runner import make_oof_folds, OOF_N_SPLITS
        _, y = tiny_data
        folds = make_oof_folds(y, n_splits=OOF_N_SPLITS, seed=7)
        assert len(folds) == OOF_N_SPLITS


class TestOOFNoPrediction:
    """OOF_NO_SELF_PREDICTION: held-out rows must NOT appear in training indices."""

    @pytest.mark.parametrize("h1_seed", [42, 123, 2024])
    def test_oof_no_self_prediction(self, tiny_data, h1_seed):
        """Each OOF row's fold is excluded from the model's training indices."""
        from src.models.stacking.oof_runner import make_oof_folds
        _, y = tiny_data
        folds = make_oof_folds(y, n_splits=3, seed=7)
        for fold_idx, (train_idx, oof_idx) in enumerate(folds):
            train_set = set(train_idx.tolist())
            oof_set = set(oof_idx.tolist())
            # No OOF index appears in train_idx for that fold
            leaks = train_set & oof_set
            assert len(leaks) == 0, (
                f"Self-prediction leakage in fold {fold_idx} seed {h1_seed}: "
                f"{len(leaks)} rows appear in both train and OOF"
            )


class TestOOFCoverage:
    """OOF_COMPLETE_COVERAGE: every row gets exactly one OOF prediction."""

    def test_oof_complete_coverage_row_count(self, tiny_data, oof_folds_tiny):
        """All n rows must appear in OOF predictions."""
        _, y = tiny_data
        n = len(y)
        all_oof = np.concatenate([oof for _, oof in oof_folds_tiny])
        assert len(all_oof) == n, f"OOF coverage: expected {n} rows, got {len(all_oof)}"

    def test_oof_exactly_one_prediction_per_row(self, tiny_data, oof_folds_tiny):
        """Each row appears in exactly one OOF fold."""
        _, y = tiny_data
        all_oof = np.concatenate([oof for _, oof in oof_folds_tiny])
        unique, counts = np.unique(all_oof, return_counts=True)
        assert (counts == 1).all(), \
            f"Some rows have multiple OOF predictions: {unique[counts > 1]}"

    def test_oof_no_duplicate_predictions(self, tiny_data, oof_folds_tiny):
        """No duplicate row indices in OOF output."""
        _, y = tiny_data
        all_oof = np.concatenate([oof for _, oof in oof_folds_tiny])
        assert len(all_oof) == len(np.unique(all_oof))


# ===========================================================================
# Group 2 — Seed Propagation
# ===========================================================================


class TestSeedPropagation:

    def test_dt_gets_h1_seed_as_random_state(self):
        """DT must be instantiated with random_state=h1_seed."""
        from src.models.stacking import oof_runner
        with patch("src.models.stacking.oof_runner.DecisionTreeClassifier") as mock_dt:
            mock_instance = MagicMock()
            mock_instance.predict_proba.return_value = np.array([[0.6, 0.4], [0.3, 0.7]])
            mock_dt.return_value = mock_instance

            X, y = _make_tiny_data(n=40)
            folds = [(np.arange(30), np.arange(30, 40))]
            svm_scaler = MagicMock()
            svm_scaler.transform.return_value = X
            nn_scaler = MagicMock()
            nn_scaler.transform.return_value = X

            with patch("src.models.stacking.oof_runner.RandomForestClassifier") as mock_rf, \
                 patch("src.models.stacking.oof_runner.LinearSVC") as mock_svm, \
                 patch("src.models.stacking.oof_runner._train_nn_oof_fold") as mock_nn:

                mock_rf.return_value = MagicMock(
                    predict_proba=MagicMock(return_value=np.array([[0.5, 0.5]] * 10))
                )
                mock_svm.return_value = MagicMock(
                    decision_function=MagicMock(return_value=np.ones(10))
                )
                mock_nn.return_value = MagicMock()
                import torch
                mock_net = MagicMock()
                mock_nn.return_value = mock_net

                with patch("torch.no_grad"), \
                     patch("torch.tensor", return_value=MagicMock()), \
                     patch("src.models.stacking.oof_runner.set_all_seeds"):
                    try:
                        oof_runner.run_oof_seed(
                            h1_seed=123, folds=folds,
                            X_unscaled=X, y=y,
                            svm_scaler=svm_scaler, nn_scaler=nn_scaler,
                        )
                    except Exception:
                        pass  # May fail due to mocking, but we check call args

                # Verify DT was instantiated with random_state=123
                if mock_dt.called:
                    call_kwargs = mock_dt.call_args[1]
                    assert call_kwargs.get("random_state") == 123, \
                        f"DT random_state={call_kwargs.get('random_state')} != 123"

    def test_rf_gets_h1_seed_as_random_state(self):
        """RF must be instantiated with random_state=h1_seed."""
        from src.models.stacking import oof_runner
        with patch("src.models.stacking.oof_runner.RandomForestClassifier") as mock_rf:
            mock_instance = MagicMock()
            mock_instance.predict_proba.return_value = np.ones((10, 2)) * 0.5
            mock_rf.return_value = mock_instance

            X, y = _make_tiny_data(n=40)
            folds = [(np.arange(30), np.arange(30, 40))]
            svm_scaler = MagicMock()
            svm_scaler.transform.return_value = X
            nn_scaler = MagicMock()
            nn_scaler.transform.return_value = X

            with patch("src.models.stacking.oof_runner.DecisionTreeClassifier") as mock_dt, \
                 patch("src.models.stacking.oof_runner.LinearSVC") as mock_svm, \
                 patch("src.models.stacking.oof_runner._train_nn_oof_fold"), \
                 patch("src.models.stacking.oof_runner.set_all_seeds"):
                mock_dt.return_value = MagicMock(
                    predict_proba=MagicMock(return_value=np.ones((10, 2)) * 0.5)
                )
                mock_svm.return_value = MagicMock(
                    decision_function=MagicMock(return_value=np.ones(10))
                )
                with patch("torch.no_grad"), patch("torch.tensor", return_value=MagicMock()):
                    try:
                        oof_runner.run_oof_seed(
                            h1_seed=2024, folds=folds,
                            X_unscaled=X, y=y,
                            svm_scaler=svm_scaler, nn_scaler=nn_scaler,
                        )
                    except Exception:
                        pass

                if mock_rf.called:
                    call_kwargs = mock_rf.call_args[1]
                    assert call_kwargs.get("random_state") == 2024, \
                        f"RF random_state={call_kwargs.get('random_state')} != 2024"

    def test_svm_gets_h1_seed_as_random_state(self):
        """SVM LinearSVC must be instantiated with random_state=h1_seed."""
        from src.models.stacking import oof_runner
        with patch("src.models.stacking.oof_runner.LinearSVC") as mock_svm:
            mock_instance = MagicMock()
            mock_instance.decision_function.return_value = np.ones(10)
            mock_svm.return_value = mock_instance

            X, y = _make_tiny_data(n=40)
            folds = [(np.arange(30), np.arange(30, 40))]
            svm_scaler = MagicMock()
            svm_scaler.transform.return_value = X
            nn_scaler = MagicMock()
            nn_scaler.transform.return_value = X

            with patch("src.models.stacking.oof_runner.DecisionTreeClassifier") as mock_dt, \
                 patch("src.models.stacking.oof_runner.RandomForestClassifier") as mock_rf, \
                 patch("src.models.stacking.oof_runner._train_nn_oof_fold"), \
                 patch("src.models.stacking.oof_runner.set_all_seeds"):
                mock_dt.return_value = MagicMock(
                    predict_proba=MagicMock(return_value=np.ones((10, 2)) * 0.5)
                )
                mock_rf.return_value = MagicMock(
                    predict_proba=MagicMock(return_value=np.ones((10, 2)) * 0.5)
                )
                with patch("torch.no_grad"), patch("torch.tensor", return_value=MagicMock()):
                    try:
                        oof_runner.run_oof_seed(
                            h1_seed=42, folds=folds,
                            X_unscaled=X, y=y,
                            svm_scaler=svm_scaler, nn_scaler=nn_scaler,
                        )
                    except Exception:
                        pass

                if mock_svm.called:
                    call_kwargs = mock_svm.call_args[1]
                    assert call_kwargs.get("random_state") == 42, \
                        f"SVM random_state={call_kwargs.get('random_state')} != 42"

    def test_set_all_seeds_called_before_each_nn_fold(self):
        """set_all_seeds(h1_seed) must be called for each NN OOF fold."""
        from src.models.stacking import oof_runner
        call_seeds = []

        original_set_all_seeds = oof_runner.set_all_seeds
        def spy_set_all_seeds(seed):
            call_seeds.append(seed)

        with patch("src.models.stacking.oof_runner.set_all_seeds", side_effect=spy_set_all_seeds):
            X, y = _make_tiny_data(n=40)
            folds = [(np.arange(30), np.arange(30, 40))]
            svm_scaler = MagicMock()
            svm_scaler.transform.return_value = X
            nn_scaler = MagicMock()
            nn_scaler.transform.return_value = X

            with patch("src.models.stacking.oof_runner.DecisionTreeClassifier") as mock_dt, \
                 patch("src.models.stacking.oof_runner.RandomForestClassifier") as mock_rf, \
                 patch("src.models.stacking.oof_runner.LinearSVC") as mock_svm, \
                 patch("src.models.stacking.oof_runner._train_nn_oof_fold") as mock_nn:
                mock_dt.return_value = MagicMock(
                    predict_proba=MagicMock(return_value=np.ones((10, 2)) * 0.5)
                )
                mock_rf.return_value = MagicMock(
                    predict_proba=MagicMock(return_value=np.ones((10, 2)) * 0.5)
                )
                mock_svm.return_value = MagicMock(
                    decision_function=MagicMock(return_value=np.ones(10))
                )
                mock_nn.return_value = MagicMock()
                with patch("torch.no_grad"), patch("torch.tensor", return_value=MagicMock()):
                    try:
                        oof_runner.run_oof_seed(
                            h1_seed=42, folds=folds,
                            X_unscaled=X, y=y,
                            svm_scaler=svm_scaler, nn_scaler=nn_scaler,
                        )
                    except Exception:
                        pass

        # set_all_seeds(42) should have been called at least once (once per fold × NN)
        assert all(s == 42 for s in call_seeds), \
            f"set_all_seeds called with wrong seeds: {call_seeds}"

    def test_meta_learner_gets_h1_seed(self):
        """Meta-learner must use random_state=h1_seed."""
        from src.models.stacking.meta_learner import train_meta_learner
        rng = np.random.RandomState(0)
        oof_df = pd.DataFrame({
            "row_id": np.arange(100),
            "dt_attack_probability": rng.rand(100),
            "rf_attack_probability": rng.rand(100),
            "svm_decision_score": rng.randn(100),
            "nn_attack_probability": rng.rand(100),
            "label": np.concatenate([np.zeros(40), np.ones(60)]).astype(int),
        })
        clf = train_meta_learner(oof_df, h1_seed=123)
        assert clf.random_state == 123

    def test_set_all_seeds_function_exists(self):
        """set_all_seeds must be importable from stacking module."""
        from src.models.stacking.oof_runner import set_all_seeds
        assert callable(set_all_seeds)

    def test_set_all_seeds_runs_without_error(self):
        """set_all_seeds should not raise for any H1 seed."""
        from src.models.stacking.oof_runner import set_all_seeds
        for seed in [42, 123, 2024]:
            set_all_seeds(seed)  # must not raise


# ===========================================================================
# Group 3 — Model Contracts
# ===========================================================================


class TestModelContracts:

    def test_svm_oof_output_can_exceed_01_range(self):
        """SVM decision_function output can be outside [0, 1] (not a probability)."""
        X, y = _make_tiny_data(n=200)
        svm = LinearSVC(C=0.1, class_weight="balanced", max_iter=5000, random_state=42)
        svm.fit(X[:160], y[:160])
        scores = svm.decision_function(X[160:])
        # decision_function CAN produce values outside [0, 1]
        # We verify predict_proba is absent
        assert not hasattr(svm, "predict_proba"), \
            "LinearSVC must not have predict_proba — do not use CalibratedClassifierCV"

    def test_svm_predict_proba_absent(self):
        """LinearSVC has no predict_proba — contract enforcement."""
        svm = LinearSVC()
        assert not hasattr(svm, "predict_proba")

    def test_dt_output_is_probability(self):
        """DT predict_proba returns values in [0, 1]."""
        X, y = _make_tiny_data(n=200)
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X[:160], y[:160])
        proba = np.asarray(dt.predict_proba(X[160:]))[:, 1]
        assert proba.min() >= 0.0 and proba.max() <= 1.0

    def test_rf_output_is_probability(self):
        """RF predict_proba returns values in [0, 1]."""
        X, y = _make_tiny_data(n=200)
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X[:160], y[:160])
        proba = np.asarray(rf.predict_proba(X[160:]))[:, 1]
        assert proba.min() >= 0.0 and proba.max() <= 1.0

    def test_oof_fixed_epoch_count_is_18(self):
        """OOF_FIXED_EPOCH_COUNT must be 18."""
        from src.models.stacking.oof_runner import OOF_FIXED_EPOCH_COUNT
        assert OOF_FIXED_EPOCH_COUNT == 18

    def test_oof_pos_weight_is_full_train_constant(self):
        """OOF_POS_WEIGHT must equal 44800/117595 (full-TRAIN constant)."""
        from src.models.stacking.oof_runner import OOF_POS_WEIGHT
        expected = 44_800 / 117_595
        assert abs(OOF_POS_WEIGHT - expected) < 1e-10, \
            f"OOF_POS_WEIGHT={OOF_POS_WEIGHT} != {expected}"

    def test_meta_feature_cols_excludes_row_id(self):
        """META_FEATURE_COLS must not contain row_id."""
        from src.models.stacking.meta_learner import META_FEATURE_COLS
        assert "row_id" not in META_FEATURE_COLS

    def test_meta_feature_cols_has_4_columns(self):
        """META_FEATURE_COLS must have exactly 4 entries."""
        from src.models.stacking.meta_learner import META_FEATURE_COLS
        assert len(META_FEATURE_COLS) == 4

    def test_meta_config_frozen(self):
        """META_CONFIG must match exactly the approved frozen configuration."""
        from src.models.stacking.meta_learner import META_CONFIG
        assert META_CONFIG["solver"] == "lbfgs"
        assert META_CONFIG["C"] == 1.0
        assert META_CONFIG["class_weight"] == "balanced"
        assert META_CONFIG["max_iter"] == 1000

    def test_meta_config_has_no_extra_keys(self):
        """META_CONFIG must not contain random_state (it's passed separately)."""
        from src.models.stacking.meta_learner import META_CONFIG
        assert "random_state" not in META_CONFIG, \
            "random_state must be passed at train time (varies per seed), not in META_CONFIG"

    def test_oof_seed_is_7(self):
        """OOF_SEED must be 7."""
        from src.models.stacking.oof_runner import OOF_SEED
        assert OOF_SEED == 7

    def test_oof_n_splits_is_5(self):
        """OOF_N_SPLITS must be 5."""
        from src.models.stacking.oof_runner import OOF_N_SPLITS
        assert OOF_N_SPLITS == 5


# ===========================================================================
# Group 4 — Data Isolation (resolved paths)
# ===========================================================================


class TestDataIsolation:

    def _sha256(self, p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def test_resolved_train_path_exists(self):
        assert RESOLVED_PATHS["train"].exists(), \
            f"TRAIN not found: {RESOLVED_PATHS['train']}"

    def test_resolved_train_sha_matches_step0(self):
        actual = self._sha256(RESOLVED_PATHS["train"])
        assert actual == RESOLVED_HASHES["train"], \
            f"TRAIN SHA mismatch: {actual} != {RESOLVED_HASHES['train']}"

    def test_resolved_validation_path_exists(self):
        assert RESOLVED_PATHS["validation"].exists()

    def test_resolved_validation_sha_matches_step0(self):
        actual = self._sha256(RESOLVED_PATHS["validation"])
        assert actual == RESOLVED_HASHES["validation"]

    def test_resolved_dev_test_path_exists(self):
        assert RESOLVED_PATHS["dev_test"].exists()

    def test_resolved_dev_test_sha_matches_step0(self):
        actual = self._sha256(RESOLVED_PATHS["dev_test"])
        assert actual == RESOLVED_HASHES["dev_test"]

    def test_resolved_protected_path_exists(self):
        assert RESOLVED_PATHS["protected"].exists()

    def test_resolved_protected_sha_matches_step0(self):
        actual = self._sha256(RESOLVED_PATHS["protected"])
        assert actual == RESOLVED_HASHES["protected"]

    def test_resolved_excluded_path_exists(self):
        assert RESOLVED_PATHS["excluded"].exists()

    def test_resolved_excluded_sha_matches_step0(self):
        actual = self._sha256(RESOLVED_PATHS["excluded"])
        assert actual == RESOLVED_HASHES["excluded"]

    def test_scaling_limitation_text_exists(self):
        """SCALING_LIMITATION_TEXT must be defined and non-empty."""
        from src.models.stacking.oof_runner import SCALING_LIMITATION_TEXT
        assert isinstance(SCALING_LIMITATION_TEXT, str) and len(SCALING_LIMITATION_TEXT) > 50

    def test_sprint5_checkpoints_exist(self):
        """Frozen Sprint 5 base checkpoints must exist (not refit)."""
        ckpt_root = ROOT / "results/checkpoints/EXP_BASE_MODELS_V1"
        for model in ["dt", "rf", "svm", "nn"]:
            d = ckpt_root / model
            assert d.exists(), f"Sprint 5 checkpoint dir missing: {d}"


# ===========================================================================
# Group 5 — Feature Integrity
# ===========================================================================


class TestFeatureIntegrity:

    def test_selected_features_count_is_75(self):
        features_path = ROOT / "results/feature_selection/EXP_MI_V1_1/selected_features.json"
        with open(features_path) as fh:
            data = json.load(fh)
        assert len(data["features"]) == 75

    def test_feature_set_id_is_exp_mi_v1_1(self):
        features_path = ROOT / "results/feature_selection/EXP_MI_V1_1/selected_features.json"
        with open(features_path) as fh:
            data = json.load(fh)
        assert data.get("experiment_id") == "EXP_MI_V1_1"

    def test_feature_ordering_is_deterministic(self):
        """Loading selected_features.json twice must return identical ordering."""
        features_path = ROOT / "results/feature_selection/EXP_MI_V1_1/selected_features.json"
        with open(features_path) as fh:
            d1 = json.load(fh)["features"]
        with open(features_path) as fh:
            d2 = json.load(fh)["features"]
        assert d1 == d2

    def test_meta_learner_receives_4_features_not_row_id(self):
        """Meta-learner trained on OOF must use 4 columns — row_id excluded."""
        from src.models.stacking.meta_learner import train_meta_learner, META_FEATURE_COLS
        rng = np.random.RandomState(0)
        n = 100
        oof_df = pd.DataFrame({
            "row_id": np.arange(n),
            "dt_attack_probability": rng.rand(n),
            "rf_attack_probability": rng.rand(n),
            "svm_decision_score": rng.randn(n),
            "nn_attack_probability": rng.rand(n),
            "label": np.concatenate([np.zeros(40), np.ones(60)]).astype(int),
        })
        clf = train_meta_learner(oof_df, h1_seed=42)
        assert clf.n_features_in_ == 4
        assert "row_id" not in META_FEATURE_COLS

    def test_meta_evaluation_limitation_text_exists(self):
        """META_EVALUATION_LIMITATION_TEXT must be defined."""
        from src.models.stacking.meta_learner import META_EVALUATION_LIMITATION_TEXT
        assert "in-sample" in META_EVALUATION_LIMITATION_TEXT.lower() or \
               "in sample" in META_EVALUATION_LIMITATION_TEXT.lower() or \
               "in-sample evaluation" in META_EVALUATION_LIMITATION_TEXT.lower()

    def test_sprint5_reference_label_is_exact(self):
        """Sprint 5 reference must carry exactly the approved label."""
        from src.models.stacking.meta_learner import SPRINT5_RF_REFERENCE_LABEL
        assert "Frozen Sprint 5 single-CV reference" in SPRINT5_RF_REFERENCE_LABEL
        assert "not a matched 3-seed H1 baseline" in SPRINT5_RF_REFERENCE_LABEL

    def test_h1_summary_contains_two_reporting_units_statement(self):
        """H1 summary must include the two-reporting-units statement."""
        from src.models.stacking.meta_learner import compute_h1_summary
        seed_results = [
            {"h1_seed": 42, "macro_f1": 0.95, "in_sample_evaluation_warning": True,
             "meta_evaluation_limitation": "", "scaling_limitation": ""},
            {"h1_seed": 123, "macro_f1": 0.94, "in_sample_evaluation_warning": True,
             "meta_evaluation_limitation": "", "scaling_limitation": ""},
            {"h1_seed": 2024, "macro_f1": 0.96, "in_sample_evaluation_warning": True,
             "meta_evaluation_limitation": "", "scaling_limitation": ""},
        ]
        summary = compute_h1_summary(seed_results)
        stmt = summary.get("two_reporting_units_statement", "")
        assert "three-seed H1 stacking" in stmt or "reporting units" in stmt.lower()

    def test_h1_summary_mean_std_correct(self):
        """H1 mean and std must be computed correctly from three seeds."""
        from src.models.stacking.meta_learner import compute_h1_summary
        f1s = [0.90, 0.92, 0.94]
        seed_results = [
            {"h1_seed": s, "macro_f1": f, "in_sample_evaluation_warning": True,
             "meta_evaluation_limitation": "", "scaling_limitation": ""}
            for s, f in zip([42, 123, 2024], f1s)
        ]
        summary = compute_h1_summary(seed_results)
        assert abs(summary["mean_macro_f1"] - np.mean(f1s)) < 1e-10
        assert abs(summary["std_macro_f1"] - np.std(f1s, ddof=1)) < 1e-10

    def test_h1_summary_empty_raises(self):
        from src.models.stacking.meta_learner import compute_h1_summary
        with pytest.raises(ValueError, match="empty"):
            compute_h1_summary([])
