"""
tests/test_decision_tree.py
-----------------------------
Unit and integration tests for src/models/base_models/decision_tree.py

Covers
------
- DTConfig validate()
- run_dt_cv: fit/predict on synthetic data
- run_dt_baseline: returns CVSummary with correct model_type
- run_dt_tuning: 24 configurations produced
- refit_dt: produces fitted classifier with predict/predict_proba
- Deterministic: same seed gives same result
- class_weight=balanced enforced
- Feature dimension check
- Edge cases: empty train, one-class train, invalid config
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.models.base_models.decision_tree import (
    DT_BASELINE_CONFIG,
    DT_TUNING_GRID,
    DTConfig,
    _generate_dt_configs,
    _grid_size,
    refit_dt,
    run_dt_baseline,
    run_dt_cv,
    run_dt_tuning,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_binary_data(n: int = 500, n_features: int = 75, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, size=n)
    # Ensure both classes present
    y[:5] = 0
    y[5:10] = 1
    return X, y


# ---------------------------------------------------------------------------
# DTConfig
# ---------------------------------------------------------------------------

class TestDTConfig:
    def test_default_instantiation(self):
        cfg = DTConfig()
        assert cfg.criterion == "gini"
        assert cfg.max_depth is None
        assert cfg.class_weight == "balanced"
        assert cfg.random_state == 42

    def test_validate_valid(self):
        DTConfig(criterion="gini", max_depth=10).validate()
        DTConfig(criterion="entropy", max_depth=None).validate()

    def test_validate_bad_criterion(self):
        with pytest.raises(ValueError, match="criterion"):
            DTConfig(criterion="log_loss").validate()

    def test_validate_bad_max_depth(self):
        with pytest.raises(ValueError, match="max_depth"):
            DTConfig(max_depth=0).validate()
        with pytest.raises(ValueError, match="max_depth"):
            DTConfig(max_depth=-1).validate()

    def test_validate_bad_min_samples_leaf(self):
        with pytest.raises(ValueError, match="min_samples_leaf"):
            DTConfig(min_samples_leaf=0).validate()

    def test_validate_bad_class_weight(self):
        with pytest.raises(ValueError, match="class_weight"):
            DTConfig(class_weight=None).validate()

    def test_validate_bad_random_state(self):
        with pytest.raises(ValueError, match="random_state"):
            DTConfig(random_state=0).validate()

    def test_to_dict(self):
        cfg = DTConfig(criterion="entropy", max_depth=5)
        d = cfg.to_dict()
        assert d["criterion"] == "entropy"
        assert d["max_depth"] == 5
        assert d["class_weight"] == "balanced"


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

class TestDTGrid:
    def test_grid_size_matches_declared(self):
        assert _grid_size() == 24

    def test_generate_configs_count(self):
        configs = _generate_dt_configs()
        assert len(configs) == 24

    def test_generate_configs_all_class_weight_balanced(self):
        configs = _generate_dt_configs()
        for cfg in configs:
            assert cfg["class_weight"] == "balanced"

    def test_generate_configs_all_random_state_42(self):
        configs = _generate_dt_configs()
        for cfg in configs:
            assert cfg["random_state"] == 42

    def test_generate_configs_covers_all_depths(self):
        configs = _generate_dt_configs()
        depths = {cfg["max_depth"] for cfg in configs}
        assert depths == {5, 10, 20, None}

    def test_generate_configs_covers_all_criteria(self):
        configs = _generate_dt_configs()
        criteria = {cfg["criterion"] for cfg in configs}
        assert criteria == {"gini", "entropy"}


# ---------------------------------------------------------------------------
# run_dt_cv
# ---------------------------------------------------------------------------

class TestRunDTCV:
    def test_returns_cv_summary(self):
        X, y = make_binary_data()
        result = run_dt_cv(X, y, DT_BASELINE_CONFIG)
        assert result.model_type == "dt"
        assert 0.0 <= result.mean_macro_f1 <= 1.0
        assert result.std_macro_f1 >= 0.0
        assert len(result.folds) == 5

    def test_all_fold_metrics_valid(self):
        X, y = make_binary_data()
        result = run_dt_cv(X, y, DT_BASELINE_CONFIG)
        for fm in result.folds:
            assert 0.0 <= fm.macro_f1 <= 1.0
            assert 0.0 <= fm.fpr <= 1.0
            assert 0.0 <= fm.specificity <= 1.0
            assert fm.n_train > 0
            assert fm.n_val > 0
            assert fm.runtime_seconds >= 0.0

    def test_feature_dimension_enforced(self):
        """DT does not enforce 75 itself (that's preprocessing's job), but
        it must accept whatever shape is passed."""
        X, y = make_binary_data(n=200, n_features=75)
        result = run_dt_cv(X, y, DT_BASELINE_CONFIG)
        assert result is not None

    def test_empty_train_raises(self):
        X = np.empty((0, 75))
        y = np.empty(0, dtype=int)
        with pytest.raises(ValueError, match="[Ee]mpty"):
            run_dt_cv(X, y, DT_BASELINE_CONFIG)

    def test_one_class_train_raises(self):
        X = np.ones((20, 75))
        y = np.zeros(20, dtype=int)
        with pytest.raises(ValueError, match="one class"):
            run_dt_cv(X, y, DT_BASELINE_CONFIG)

    def test_deterministic(self):
        X, y = make_binary_data(n=300)
        r1 = run_dt_cv(X, y, DT_BASELINE_CONFIG)
        r2 = run_dt_cv(X, y, DT_BASELINE_CONFIG)
        assert abs(r1.mean_macro_f1 - r2.mean_macro_f1) < 1e-9


# ---------------------------------------------------------------------------
# run_dt_baseline
# ---------------------------------------------------------------------------

class TestRunDTBaseline:
    def test_model_type(self):
        X, y = make_binary_data()
        result = run_dt_baseline(X, y)
        assert result.model_type == "dt"

    def test_config_matches_baseline(self):
        X, y = make_binary_data()
        result = run_dt_baseline(X, y)
        assert result.config["criterion"] == "gini"
        assert result.config["max_depth"] is None
        assert result.config["min_samples_leaf"] == 1
        assert result.config["class_weight"] == "balanced"


# ---------------------------------------------------------------------------
# run_dt_tuning
# ---------------------------------------------------------------------------

class TestRunDTTuning:
    def test_returns_24_configs(self):
        X, y = make_binary_data()
        results = run_dt_tuning(X, y)
        assert len(results) == 24

    def test_all_model_type_dt(self):
        X, y = make_binary_data()
        results = run_dt_tuning(X, y)
        assert all(r.model_type == "dt" for r in results)

    def test_all_valid_f1(self):
        X, y = make_binary_data()
        results = run_dt_tuning(X, y)
        for r in results:
            assert 0.0 <= r.mean_macro_f1 <= 1.0


# ---------------------------------------------------------------------------
# refit_dt
# ---------------------------------------------------------------------------

class TestRefitDT:
    def test_returns_decision_tree(self):
        X, y = make_binary_data()
        clf = refit_dt(X, y, DT_BASELINE_CONFIG)
        assert isinstance(clf, DecisionTreeClassifier)

    def test_predict_works(self):
        X, y = make_binary_data()
        clf = refit_dt(X, y, DT_BASELINE_CONFIG)
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})
        assert preds.shape == (len(y),)

    def test_predict_proba_works(self):
        X, y = make_binary_data()
        clf = refit_dt(X, y, DT_BASELINE_CONFIG)
        probs = clf.predict_proba(X)
        assert probs.shape == (len(y), 2)
        assert np.all(probs >= 0) and np.all(probs <= 1)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_class_weight_balanced(self):
        X, y = make_binary_data()
        clf = refit_dt(X, y, DT_BASELINE_CONFIG)
        assert clf.class_weight == "balanced"

    def test_deterministic_seed(self):
        X, y = make_binary_data()
        clf1 = refit_dt(X, y, DT_BASELINE_CONFIG)
        clf2 = refit_dt(X, y, DT_BASELINE_CONFIG)
        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))

    def test_invalid_config_zero_leaf(self):
        X, y = make_binary_data()
        bad_cfg = {**DT_BASELINE_CONFIG, "min_samples_leaf": 0}
        # sklearn will raise an error
        with pytest.raises(Exception):
            refit_dt(X, y, bad_cfg)

    def test_feature_dimension_stored(self):
        X, y = make_binary_data()
        clf = refit_dt(X, y, DT_BASELINE_CONFIG)
        assert clf.n_features_in_ == 75
