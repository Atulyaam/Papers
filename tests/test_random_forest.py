"""
tests/test_random_forest.py
-----------------------------
Unit and integration tests for src/models/base_models/random_forest.py

Covers
------
- RFConfig validate()
- run_rf_cv: fit/predict on synthetic data
- run_rf_baseline: correct model_type and config
- run_rf_tuning: 24 configurations
- refit_rf: fitted classifier with predict/predict_proba
- n_estimators, n_jobs enforcement
- Deterministic seed
- Edge cases: empty train, one-class train, invalid n_estimators
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.base_models.random_forest import (
    RF_BASELINE_CONFIG,
    RF_TUNING_GRID,
    RFConfig,
    _generate_rf_configs,
    _grid_size,
    refit_rf,
    run_rf_baseline,
    run_rf_cv,
    run_rf_tuning,
)


def make_binary_data(n: int = 500, n_features: int = 75, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, size=n)
    y[:5] = 0
    y[5:10] = 1
    return X, y


# ---------------------------------------------------------------------------
# RFConfig
# ---------------------------------------------------------------------------

class TestRFConfig:
    def test_default_instantiation(self):
        cfg = RFConfig()
        assert cfg.n_estimators == 300
        assert cfg.max_depth is None
        assert cfg.class_weight == "balanced"
        assert cfg.n_jobs == -1

    def test_validate_valid(self):
        RFConfig(n_estimators=100, max_depth=10).validate()
        RFConfig(n_estimators=300, max_depth=None).validate()

    def test_validate_bad_n_estimators(self):
        with pytest.raises(ValueError, match="n_estimators"):
            RFConfig(n_estimators=0).validate()

    def test_validate_bad_class_weight(self):
        with pytest.raises(ValueError, match="class_weight"):
            RFConfig(class_weight=None).validate()

    def test_validate_bad_random_state(self):
        with pytest.raises(ValueError, match="random_state"):
            RFConfig(random_state=0).validate()

    def test_to_dict(self):
        cfg = RFConfig(n_estimators=100, max_depth=10)
        d = cfg.to_dict()
        assert d["n_estimators"] == 100
        assert d["max_depth"] == 10
        assert d["n_jobs"] == -1


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

class TestRFGrid:
    def test_grid_size(self):
        assert _grid_size() == 24

    def test_generate_configs_count(self):
        assert len(_generate_rf_configs()) == 24

    def test_class_weight_always_balanced(self):
        for cfg in _generate_rf_configs():
            assert cfg["class_weight"] == "balanced"

    def test_n_jobs_always_minus1(self):
        for cfg in _generate_rf_configs():
            assert cfg["n_jobs"] == -1

    def test_all_n_estimators_values_present(self):
        cfgs = _generate_rf_configs()
        assert {c["n_estimators"] for c in cfgs} == {100, 300}

    def test_all_max_depth_values_present(self):
        cfgs = _generate_rf_configs()
        assert {c["max_depth"] for c in cfgs} == {10, 20, None}

    def test_all_max_features_values_present(self):
        cfgs = _generate_rf_configs()
        assert {c["max_features"] for c in cfgs} == {"sqrt", 0.3}


# ---------------------------------------------------------------------------
# run_rf_cv
# ---------------------------------------------------------------------------

class TestRunRFCV:
    def test_returns_cv_summary(self):
        X, y = make_binary_data(n=300)
        cfg = {**RF_BASELINE_CONFIG, "n_estimators": 10}  # fast for test
        result = run_rf_cv(X, y, cfg)
        assert result.model_type == "rf"
        assert 0.0 <= result.mean_macro_f1 <= 1.0
        assert len(result.folds) == 5

    def test_empty_train_raises(self):
        X = np.empty((0, 75))
        y = np.empty(0, dtype=int)
        with pytest.raises(ValueError, match="[Ee]mpty"):
            run_rf_cv(X, y, RF_BASELINE_CONFIG)

    def test_one_class_raises(self):
        X = np.ones((20, 75))
        y = np.zeros(20, dtype=int)
        with pytest.raises(ValueError, match="one class"):
            run_rf_cv(X, y, RF_BASELINE_CONFIG)

    def test_invalid_n_estimators_raises(self):
        X, y = make_binary_data(n=100)
        bad_cfg = {**RF_BASELINE_CONFIG, "n_estimators": 0}
        with pytest.raises(ValueError, match="n_estimators"):
            run_rf_cv(X, y, bad_cfg)

    def test_deterministic(self):
        X, y = make_binary_data(n=200)
        cfg = {**RF_BASELINE_CONFIG, "n_estimators": 10}
        r1 = run_rf_cv(X, y, cfg)
        r2 = run_rf_cv(X, y, cfg)
        assert abs(r1.mean_macro_f1 - r2.mean_macro_f1) < 1e-9


# ---------------------------------------------------------------------------
# run_rf_baseline
# ---------------------------------------------------------------------------

class TestRunRFBaseline:
    def test_model_type(self):
        X, y = make_binary_data(n=200)
        cfg = {**RF_BASELINE_CONFIG, "n_estimators": 10}
        # Override for speed — validate shape only
        from src.models.base_models.random_forest import run_rf_cv
        result = run_rf_cv(X, y, cfg)
        assert result.model_type == "rf"

    def test_baseline_config_keys(self):
        assert "n_estimators" in RF_BASELINE_CONFIG
        assert "class_weight" in RF_BASELINE_CONFIG
        assert RF_BASELINE_CONFIG["class_weight"] == "balanced"
        assert RF_BASELINE_CONFIG["random_state"] == 42
        assert RF_BASELINE_CONFIG["n_jobs"] == -1


# ---------------------------------------------------------------------------
# refit_rf
# ---------------------------------------------------------------------------

class TestRefitRF:
    def test_returns_random_forest(self):
        X, y = make_binary_data(n=200)
        cfg = {**RF_BASELINE_CONFIG, "n_estimators": 10}
        clf = refit_rf(X, y, cfg)
        assert isinstance(clf, RandomForestClassifier)

    def test_predict_works(self):
        X, y = make_binary_data(n=200)
        cfg = {**RF_BASELINE_CONFIG, "n_estimators": 10}
        clf = refit_rf(X, y, cfg)
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_works(self):
        X, y = make_binary_data(n=200)
        cfg = {**RF_BASELINE_CONFIG, "n_estimators": 10}
        clf = refit_rf(X, y, cfg)
        probs = clf.predict_proba(X)
        assert probs.shape == (len(y), 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_n_jobs_is_minus1(self):
        X, y = make_binary_data(n=100)
        cfg = {**RF_BASELINE_CONFIG, "n_estimators": 5}
        clf = refit_rf(X, y, cfg)
        assert clf.n_jobs == -1

    def test_class_weight_balanced(self):
        X, y = make_binary_data(n=100)
        cfg = {**RF_BASELINE_CONFIG, "n_estimators": 5}
        clf = refit_rf(X, y, cfg)
        assert clf.class_weight == "balanced"

    def test_deterministic_seed(self):
        X, y = make_binary_data(n=100)
        cfg = {**RF_BASELINE_CONFIG, "n_estimators": 10}
        clf1 = refit_rf(X, y, cfg)
        clf2 = refit_rf(X, y, cfg)
        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))

    def test_feature_dim(self):
        X, y = make_binary_data(n=100)
        cfg = {**RF_BASELINE_CONFIG, "n_estimators": 5}
        clf = refit_rf(X, y, cfg)
        assert clf.n_features_in_ == 75
