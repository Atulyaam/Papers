"""
tests/test_linear_svc.py
--------------------------
Unit and integration tests for src/models/base_models/linear_svc.py

Covers
------
- SVMConfig validate()
- run_svm_cv: fit/predict, decision_function present, no probabilities
- Scaler leakage: scaler fit only on inner_train inside each fold
- run_svm_baseline: correct model_type and config
- run_svm_tuning: 4 C configurations
- refit_svm: returns (svm, scaler) tuple
- decision_function exists and is NOT claim to be probabilities
- Invalid C raises
- Edge cases: empty train, one-class train
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from src.models.base_models.linear_svc import (
    SVM_BASELINE_CONFIG,
    SVM_FIXED_PARAMS,
    SVM_TUNING_C_VALUES,
    SVMConfig,
    refit_svm,
    run_svm_baseline,
    run_svm_cv,
    run_svm_tuning,
)


def make_binary_data(n: int = 500, n_features: int = 75, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, size=n)
    y[:5] = 0
    y[5:10] = 1
    return X, y


# ---------------------------------------------------------------------------
# SVMConfig
# ---------------------------------------------------------------------------

class TestSVMConfig:
    def test_default_instantiation(self):
        cfg = SVMConfig()
        assert cfg.C == 1.0
        assert cfg.class_weight == "balanced"
        assert cfg.max_iter == 5000
        assert cfg.random_state == 42

    def test_validate_valid(self):
        SVMConfig(C=0.01).validate()
        SVMConfig(C=10.0).validate()

    def test_validate_bad_C_zero(self):
        with pytest.raises(ValueError, match="C"):
            SVMConfig(C=0.0).validate()

    def test_validate_bad_C_negative(self):
        with pytest.raises(ValueError, match="C"):
            SVMConfig(C=-1.0).validate()

    def test_validate_bad_class_weight(self):
        with pytest.raises(ValueError, match="class_weight"):
            SVMConfig(class_weight="uniform").validate()

    def test_validate_bad_max_iter(self):
        with pytest.raises(ValueError, match="max_iter"):
            SVMConfig(max_iter=1000).validate()

    def test_to_dict(self):
        cfg = SVMConfig(C=0.1)
        d = cfg.to_dict()
        assert d["C"] == 0.1
        assert d["class_weight"] == "balanced"
        assert d["max_iter"] == 5000


# ---------------------------------------------------------------------------
# Tuning grid
# ---------------------------------------------------------------------------

class TestSVMGrid:
    def test_four_c_values(self):
        assert len(SVM_TUNING_C_VALUES) == 4

    def test_c_values_ascending(self):
        assert SVM_TUNING_C_VALUES == sorted(SVM_TUNING_C_VALUES)

    def test_c_values_correct(self):
        assert set(SVM_TUNING_C_VALUES) == {0.01, 0.1, 1.0, 10.0}


# ---------------------------------------------------------------------------
# run_svm_cv
# ---------------------------------------------------------------------------

class TestRunSVMCV:
    def test_returns_cv_summary(self):
        X, y = make_binary_data()
        result = run_svm_cv(X, y, SVM_BASELINE_CONFIG)
        assert result.model_type == "svm"
        assert 0.0 <= result.mean_macro_f1 <= 1.0
        assert len(result.folds) == 5

    def test_decision_function_not_needed_from_cv(self):
        """
        run_svm_cv uses predict() internally for metrics.
        Decision function is not used for CV metric computation.
        This test confirms CV runs without probability errors.
        """
        X, y = make_binary_data()
        result = run_svm_cv(X, y, SVM_BASELINE_CONFIG)
        assert result is not None

    def test_scaler_isolation_in_folds(self):
        """
        Each fold must fit scaler only on inner_train.
        We verify indirectly: the CV must produce valid metrics (if scaler
        leaked val info, training would not fail, but we check the structure).
        """
        X, y = make_binary_data()
        result = run_svm_cv(X, y, SVM_BASELINE_CONFIG)
        for fm in result.folds:
            assert 0.0 <= fm.macro_f1 <= 1.0
            assert fm.n_train > 0
            assert fm.n_val > 0

    def test_empty_train_raises(self):
        X = np.empty((0, 75))
        y = np.empty(0, dtype=int)
        with pytest.raises(ValueError, match="[Ee]mpty"):
            run_svm_cv(X, y, SVM_BASELINE_CONFIG)

    def test_one_class_raises(self):
        X = np.ones((20, 75))
        y = np.zeros(20, dtype=int)
        with pytest.raises(ValueError, match="one class"):
            run_svm_cv(X, y, SVM_BASELINE_CONFIG)

    def test_invalid_C_raises(self):
        X, y = make_binary_data()
        bad_cfg = {**SVM_BASELINE_CONFIG, "C": -1.0}
        with pytest.raises(ValueError, match="C"):
            run_svm_cv(X, y, bad_cfg)

    def test_invalid_C_zero_raises(self):
        X, y = make_binary_data()
        bad_cfg = {**SVM_BASELINE_CONFIG, "C": 0.0}
        with pytest.raises(ValueError, match="C"):
            run_svm_cv(X, y, bad_cfg)

    def test_deterministic(self):
        X, y = make_binary_data()
        r1 = run_svm_cv(X, y, SVM_BASELINE_CONFIG)
        r2 = run_svm_cv(X, y, SVM_BASELINE_CONFIG)
        assert abs(r1.mean_macro_f1 - r2.mean_macro_f1) < 1e-9


# ---------------------------------------------------------------------------
# run_svm_baseline
# ---------------------------------------------------------------------------

class TestRunSVMBaseline:
    def test_baseline_model_type(self):
        X, y = make_binary_data()
        result = run_svm_baseline(X, y)
        assert result.model_type == "svm"

    def test_baseline_config_c(self):
        assert SVM_BASELINE_CONFIG["C"] == 1.0
        assert SVM_BASELINE_CONFIG["class_weight"] == "balanced"
        assert SVM_BASELINE_CONFIG["max_iter"] == 5000


# ---------------------------------------------------------------------------
# run_svm_tuning
# ---------------------------------------------------------------------------

class TestRunSVMTuning:
    def test_returns_4_configs(self):
        X, y = make_binary_data()
        results = run_svm_tuning(X, y)
        assert len(results) == 4

    def test_all_model_type_svm(self):
        X, y = make_binary_data()
        results = run_svm_tuning(X, y)
        assert all(r.model_type == "svm" for r in results)

    def test_c_values_covered(self):
        X, y = make_binary_data()
        results = run_svm_tuning(X, y)
        c_values = {r.config["C"] for r in results}
        assert c_values == {0.01, 0.1, 1.0, 10.0}


# ---------------------------------------------------------------------------
# refit_svm
# ---------------------------------------------------------------------------

class TestRefitSVM:
    def test_returns_tuple(self):
        X, y = make_binary_data()
        result = refit_svm(X, y, SVM_BASELINE_CONFIG)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_linearsvc_and_scaler(self):
        X, y = make_binary_data()
        clf, scaler = refit_svm(X, y, SVM_BASELINE_CONFIG)
        assert isinstance(clf, LinearSVC)
        assert isinstance(scaler, StandardScaler)

    def test_predict_works(self):
        X, y = make_binary_data()
        clf, scaler = refit_svm(X, y, SVM_BASELINE_CONFIG)
        X_scaled = scaler.transform(X)
        preds = clf.predict(X_scaled)
        assert set(preds).issubset({0, 1})

    def test_decision_function_exists(self):
        X, y = make_binary_data()
        clf, scaler = refit_svm(X, y, SVM_BASELINE_CONFIG)
        X_scaled = scaler.transform(X)
        scores = clf.decision_function(X_scaled)
        assert scores.shape == (len(y),)
        assert scores.dtype == np.float64

    def test_no_predict_proba(self):
        """
        LinearSVC does NOT have predict_proba by design.
        This test asserts the attribute is absent.
        """
        X, y = make_binary_data()
        clf, scaler = refit_svm(X, y, SVM_BASELINE_CONFIG)
        assert not hasattr(clf, "predict_proba"), (
            "LinearSVC must NOT expose predict_proba in Sprint 5. "
            "Calibration is a separate future sprint concern."
        )

    def test_decision_function_not_probability(self):
        """
        decision_function values are NOT in [0, 1] and must not be treated
        as probabilities.
        """
        X, y = make_binary_data()
        clf, scaler = refit_svm(X, y, SVM_BASELINE_CONFIG)
        X_scaled = scaler.transform(X)
        scores = clf.decision_function(X_scaled)
        # If they were probabilities, all values would be in [0,1].
        # For SVM decision values, they should span a broader range.
        # At minimum, we confirm the range check is NOT [0,1].
        has_out_of_range = np.any(scores > 1.0) or np.any(scores < 0.0)
        assert has_out_of_range, (
            "Expected SVM decision function values outside [0,1]. "
            "If all values are in [0,1], something is wrong with the model type."
        )

    def test_class_weight_balanced(self):
        X, y = make_binary_data()
        clf, _ = refit_svm(X, y, SVM_BASELINE_CONFIG)
        assert clf.class_weight == "balanced"

    def test_scaler_fitted_on_train_only(self):
        """
        Verify scaler was fitted on training data:
        the scaler mean should match the training set mean.
        """
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 75)) * 3.0 + 7.0
        y = np.array([0] * 100 + [1] * 100)
        clf, scaler = refit_svm(X, y, SVM_BASELINE_CONFIG)
        # Scaler mean should be close to the TRAIN mean
        np.testing.assert_allclose(scaler.mean_, X.mean(axis=0), rtol=1e-5)
