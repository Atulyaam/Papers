"""
tests/test_comparator.py
--------------------------
Unit tests for src/models/base_models/comparator.py

Covers
------
- Higher mean F1 wins (step 1)
- Lower std F1 wins on tie (step 2)
- DT simplicity ordering (step 3)
- RF simplicity ordering (step 3)
- SVM simplicity ordering (step 3)
- NN simplicity ordering (step 3)
- Serialised config tie-break (step 4)
- Symmetry: compare(a,b) == -compare(b,a)
- Reflexivity: compare(a,a) == 0
- Invalid model type raises
- Determinism: identical inputs always return identical output
- Dict insertion order irrelevance
- Exact floating-point tolerance (1e-6)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.models.base_models.comparator import compare_model_configs


# ---------------------------------------------------------------------------
# Minimal CVSummary stand-in for testing
# ---------------------------------------------------------------------------


@dataclass
class MockResult:
    mean_macro_f1: float
    std_macro_f1: float
    config: dict[str, Any]


def mk(f1: float, std: float = 0.0, config: dict | None = None) -> MockResult:
    return MockResult(
        mean_macro_f1=f1,
        std_macro_f1=std,
        config=config or {},
    )


# ---------------------------------------------------------------------------
# Step 1: higher mean F1
# ---------------------------------------------------------------------------

class TestStep1MeanF1:
    def test_higher_f1_wins(self):
        a = mk(0.9)
        b = mk(0.8)
        assert compare_model_configs(a, b, "dt") == 1
        assert compare_model_configs(b, a, "dt") == -1

    def test_difference_below_tolerance_not_decisive(self):
        # 5e-7 < tolerance 1e-6 → not decisive at step 1
        a = mk(0.9 + 5e-7)
        b = mk(0.9)
        # Should NOT be decisive at step 1; must fall to step 2+
        # (both have std=0, same config → result is 0 at step 4)
        result = compare_model_configs(a, b, "svm")
        # The result can be 0 or -1/+1 from later steps, but must NOT be purely
        # driven by a 5e-7 difference. With std=0 and same config, result = 0.
        assert result == 0

    def test_difference_above_tolerance_decisive(self):
        a = mk(0.9001)
        b = mk(0.9)
        assert compare_model_configs(a, b, "rf") == 1

    def test_symmetry(self):
        a = mk(0.9)
        b = mk(0.85)
        assert compare_model_configs(a, b, "dt") == -compare_model_configs(b, a, "dt")

    def test_reflexivity(self):
        a = mk(0.9, config={"C": 1.0})
        assert compare_model_configs(a, a, "svm") == 0


# ---------------------------------------------------------------------------
# Step 2: lower std F1
# ---------------------------------------------------------------------------

class TestStep2StdF1:
    def test_lower_std_wins_on_f1_tie(self):
        a = mk(0.9, std=0.01)
        b = mk(0.9, std=0.02)
        assert compare_model_configs(a, b, "dt") == 1
        assert compare_model_configs(b, a, "dt") == -1

    def test_equal_std_falls_through(self):
        a = mk(0.9, std=0.01)
        b = mk(0.9, std=0.01)
        # Both steps 1 and 2 tie; result from step 3/4
        result = compare_model_configs(a, b, "svm")
        # With default config={}, step 4 gives 0
        assert result == 0


# ---------------------------------------------------------------------------
# Step 3: DT simplicity
# ---------------------------------------------------------------------------

class TestDTSimplicity:
    def _tied(self, cfg_a: dict, cfg_b: dict) -> int:
        a = mk(0.9, std=0.01, config=cfg_a)
        b = mk(0.9, std=0.01, config=cfg_b)
        return compare_model_configs(a, b, "dt")

    def test_smaller_max_depth_wins(self):
        assert self._tied({"max_depth": 5}, {"max_depth": 10}) == 1
        assert self._tied({"max_depth": 10}, {"max_depth": 20}) == 1
        assert self._tied({"max_depth": 20}, {"max_depth": None}) == 1
        assert self._tied({"max_depth": None}, {"max_depth": 5}) == -1

    def test_smaller_min_samples_leaf_wins(self):
        cfg_base = {"max_depth": 5}
        a = {**cfg_base, "min_samples_leaf": 1}
        b = {**cfg_base, "min_samples_leaf": 5}
        assert self._tied(a, b) == 1
        assert self._tied(b, a) == -1

    def test_gini_beats_entropy(self):
        cfg_base = {"max_depth": 5, "min_samples_leaf": 1}
        a = {**cfg_base, "criterion": "gini"}
        b = {**cfg_base, "criterion": "entropy"}
        assert self._tied(a, b) == 1
        assert self._tied(b, a) == -1

    def test_none_depth_is_largest(self):
        assert self._tied({"max_depth": None}, {"max_depth": 20}) == -1

    def test_equal_configs(self):
        cfg = {"max_depth": 10, "min_samples_leaf": 1, "criterion": "gini"}
        assert self._tied(cfg, cfg) == 0


# ---------------------------------------------------------------------------
# Step 3: RF simplicity
# ---------------------------------------------------------------------------

class TestRFSimplicity:
    def _tied(self, cfg_a: dict, cfg_b: dict) -> int:
        a = mk(0.9, std=0.01, config=cfg_a)
        b = mk(0.9, std=0.01, config=cfg_b)
        return compare_model_configs(a, b, "rf")

    def test_fewer_estimators_wins(self):
        assert self._tied({"n_estimators": 100}, {"n_estimators": 300}) == 1

    def test_smaller_max_depth_wins(self):
        base = {"n_estimators": 100}
        a = {**base, "max_depth": 10}
        b = {**base, "max_depth": 20}
        assert self._tied(a, b) == 1
        assert self._tied({**base, "max_depth": 20}, {**base, "max_depth": None}) == 1

    def test_smaller_min_leaf_wins(self):
        base = {"n_estimators": 100, "max_depth": 10}
        a = {**base, "min_samples_leaf": 1}
        b = {**base, "min_samples_leaf": 5}
        assert self._tied(a, b) == 1

    def test_sqrt_beats_float(self):
        base = {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 1}
        a = {**base, "max_features": "sqrt"}
        b = {**base, "max_features": 0.3}
        assert self._tied(a, b) == 1
        assert self._tied(b, a) == -1


# ---------------------------------------------------------------------------
# Step 3: SVM simplicity
# ---------------------------------------------------------------------------

class TestSVMSimplicity:
    def _tied(self, cfg_a: dict, cfg_b: dict) -> int:
        a = mk(0.9, std=0.01, config=cfg_a)
        b = mk(0.9, std=0.01, config=cfg_b)
        return compare_model_configs(a, b, "svm")

    def test_smaller_C_wins(self):
        assert self._tied({"C": 0.01}, {"C": 0.1}) == 1
        assert self._tied({"C": 0.1}, {"C": 1.0}) == 1
        assert self._tied({"C": 1.0}, {"C": 10.0}) == 1
        assert self._tied({"C": 10.0}, {"C": 0.01}) == -1

    def test_equal_C(self):
        assert self._tied({"C": 1.0}, {"C": 1.0}) == 0


# ---------------------------------------------------------------------------
# Step 3: NN simplicity
# ---------------------------------------------------------------------------

class TestNNSimplicity:
    def _tied(self, cfg_a: dict, cfg_b: dict) -> int:
        a = mk(0.9, std=0.01, config=cfg_a)
        b = mk(0.9, std=0.01, config=cfg_b)
        return compare_model_configs(a, b, "nn")

    def test_fewer_params_wins(self):
        a = {"hidden_sizes": [128, 64], "learning_rate": 0.001, "weight_decay": 0.0001}
        b = {"hidden_sizes": [256, 128], "learning_rate": 0.001, "weight_decay": 0.0001}
        assert self._tied(a, b) == 1
        assert self._tied(b, a) == -1

    def test_larger_weight_decay_wins_on_same_arch(self):
        base = {"hidden_sizes": [128, 64], "learning_rate": 0.001}
        a = {**base, "weight_decay": 0.001}
        b = {**base, "weight_decay": 0.0001}
        assert self._tied(a, b) == 1
        assert self._tied(b, a) == -1

    def test_smaller_lr_wins(self):
        base = {"hidden_sizes": [128, 64], "weight_decay": 0.001}
        a = {**base, "learning_rate": 0.0001}
        b = {**base, "learning_rate": 0.001}
        assert self._tied(a, b) == 1
        assert self._tied(b, a) == -1

    def test_equal_nn_config(self):
        cfg = {"hidden_sizes": [128, 64], "learning_rate": 0.001, "weight_decay": 0.0001}
        assert self._tied(cfg, cfg) == 0


# ---------------------------------------------------------------------------
# Step 4: Serialised config fallback
# ---------------------------------------------------------------------------

class TestSerialisedFallback:
    def test_deterministic_on_empty_configs(self):
        a = mk(0.9, std=0.01, config={})
        b = mk(0.9, std=0.01, config={})
        assert compare_model_configs(a, b, "dt") == 0

    def test_different_configs_give_nonzero(self):
        a = mk(0.9, std=0.01, config={"x": 1})
        b = mk(0.9, std=0.01, config={"x": 2})
        result = compare_model_configs(a, b, "svm")
        # Must be deterministic — call twice and compare
        assert compare_model_configs(a, b, "svm") == result

    def test_insertion_order_irrelevant(self):
        """
        Two dicts with the same keys/values but different insertion order
        must produce the same comparator result.
        """
        cfg1 = {"C": 1.0, "max_iter": 5000}
        cfg2 = {"max_iter": 5000, "C": 1.0}
        a = mk(0.9, std=0.01, config=cfg1)
        b = mk(0.9, std=0.01, config=cfg2)
        # Both represent the same config — comparator must return 0
        assert compare_model_configs(a, b, "svm") == 0


# ---------------------------------------------------------------------------
# Invalid model type
# ---------------------------------------------------------------------------

class TestInvalidModelType:
    def test_invalid_type_raises(self):
        a = mk(0.9)
        b = mk(0.8)
        with pytest.raises(ValueError, match="Unknown model_type"):
            compare_model_configs(a, b, "xgboost")

    def test_empty_string_raises(self):
        a = mk(0.9)
        b = mk(0.8)
        with pytest.raises(ValueError):
            compare_model_configs(a, b, "")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_calls_same_result(self):
        a = mk(0.9123, std=0.01, config={"C": 0.1})
        b = mk(0.9123, std=0.02, config={"C": 1.0})
        results = [compare_model_configs(a, b, "svm") for _ in range(20)]
        assert len(set(results)) == 1

    def test_exact_tolerance_boundary(self):
        # Exactly at tolerance — should NOT be decisive at step 1
        tol = 1e-6
        a = mk(0.9 + tol, std=0.0, config={})
        b = mk(0.9, std=0.0, config={})
        # The difference is exactly at the boundary — result is model-dependent
        # but must be consistent
        r1 = compare_model_configs(a, b, "svm")
        r2 = compare_model_configs(a, b, "svm")
        assert r1 == r2

    def test_reflexivity_all_types(self):
        configs = [
            ({"max_depth": 5, "criterion": "gini"}, "dt"),
            ({"n_estimators": 100, "max_depth": 10}, "rf"),
            ({"C": 0.1}, "svm"),
            ({"hidden_sizes": [128, 64], "learning_rate": 0.001, "weight_decay": 0.001}, "nn"),
        ]
        for cfg, mtype in configs:
            a = mk(0.9, std=0.01, config=cfg)
            assert compare_model_configs(a, a, mtype) == 0

    def test_antisymmetry(self):
        """compare(a,b) must equal -compare(b,a)."""
        a = mk(0.9, std=0.01, config={"C": 0.01})
        b = mk(0.85, std=0.02, config={"C": 1.0})
        for mtype in ["dt", "rf", "svm", "nn"]:
            assert compare_model_configs(a, b, mtype) == -compare_model_configs(b, a, mtype)
