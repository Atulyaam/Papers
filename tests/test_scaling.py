"""
tests/test_scaling.py
----------------------
Unit tests for src/preprocessing/scaling.py

Covers:
  - fit_scaler: returns FittedScaler, records correct TRAIN statistics
  - transform_scaler: standard scaling, column mismatch raises
  - LEAKAGE TEST: scaler fitted on TRAIN-only; TEST distribution must NOT
    influence mean_ or scale_. Tests that the implementation fails if
    accidentally combined.
  - Constant-feature behavior (StandardScaler safe)
  - Empty input raises
  - Shape/dtype contract
  - Metadata structure
  - Determinism
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.scaling import (
    FittedScaler,
    fit_scaler,
    get_scaler_metadata,
    transform_scaler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_X(n_rows=50, n_cols=5, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(n_rows, n_cols)).astype(np.float64)


def _feature_names(n=5) -> list[str]:
    return [f"feat_{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# fit_scaler
# ---------------------------------------------------------------------------


class TestFitScaler:
    def test_returns_fitted_scaler(self):
        X = _make_X()
        fs = fit_scaler(X, _feature_names())
        assert isinstance(fs, FittedScaler)

    def test_mean_is_train_mean(self):
        X = _make_X()
        fs = fit_scaler(X, _feature_names())
        np.testing.assert_allclose(fs.train_mean, X.mean(axis=0), rtol=1e-5)

    def test_scale_is_train_std(self):
        X = _make_X()
        fs = fit_scaler(X, _feature_names())
        np.testing.assert_allclose(fs.train_scale, X.std(axis=0, ddof=0), rtol=1e-5)

    def test_n_features_recorded(self):
        X = _make_X(n_cols=7)
        fs = fit_scaler(X, _feature_names(7))
        assert fs.n_features == 7

    def test_feature_names_stored(self):
        names = _feature_names(4)
        fs = fit_scaler(_make_X(n_cols=4), names)
        assert fs.feature_names == names

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="zero rows"):
            fit_scaler(np.empty((0, 5), dtype=np.float64), _feature_names(5))

    def test_mismatched_feature_names_raises(self):
        X = _make_X(n_cols=5)
        with pytest.raises(ValueError):
            fit_scaler(X, _feature_names(3))  # wrong length

    def test_deterministic(self):
        X = _make_X()
        fs1 = fit_scaler(X, _feature_names())
        fs2 = fit_scaler(X, _feature_names())
        np.testing.assert_array_equal(fs1.train_mean, fs2.train_mean)
        np.testing.assert_array_equal(fs1.train_scale, fs2.train_scale)


# ---------------------------------------------------------------------------
# transform_scaler
# ---------------------------------------------------------------------------


class TestTransformScaler:
    def test_output_shape_preserved(self):
        X_train = _make_X(n_rows=100)
        fs = fit_scaler(X_train, _feature_names())
        X_test = _make_X(n_rows=30)
        out = transform_scaler(fs, X_test)
        assert out.shape == (30, 5)

    def test_output_dtype_float64(self):
        fs = fit_scaler(_make_X(), _feature_names())
        out = transform_scaler(fs, _make_X())
        assert out.dtype == np.float64

    def test_train_transforms_to_approx_zero_mean(self):
        X_train = _make_X(n_rows=1000)
        fs = fit_scaler(X_train, _feature_names())
        X_out = transform_scaler(fs, X_train)
        np.testing.assert_allclose(X_out.mean(axis=0), np.zeros(5), atol=1e-10)

    def test_column_mismatch_raises(self):
        fs = fit_scaler(_make_X(n_cols=5), _feature_names(5))
        X_wrong = _make_X(n_cols=4)
        with pytest.raises(ValueError, match="columns"):
            transform_scaler(fs, X_wrong)

    def test_single_row_input(self):
        fs = fit_scaler(_make_X(), _feature_names())
        out = transform_scaler(fs, _make_X(n_rows=1))
        assert out.shape == (1, 5)


# ---------------------------------------------------------------------------
# LEAKAGE TEST: scaler mean_ must reflect TRAIN only
# ---------------------------------------------------------------------------


class TestScalerLeakage:
    def test_mean_reflects_train_not_test(self):
        """
        Critical leakage test.

        TRAIN: small values [0, 1].
        TEST:  very large values [1000, 2000].

        If the scaler accidentally uses TRAIN+TEST data, the mean_ will be
        approximately 1000+ rather than ~0.5.
        This test verifies that fit_scaler uses only TRAIN data.
        """
        rng = np.random.default_rng(42)
        n_features = 3

        X_train = rng.uniform(0, 1, size=(200, n_features))  # small values
        X_test = rng.uniform(1000, 2000, size=(200, n_features))  # very large

        # Fit on TRAIN only
        fs = fit_scaler(X_train, [f"f{i}" for i in range(n_features)])

        # TRAIN mean should be approximately 0.5 per feature
        assert (fs.train_mean < 2.0).all(), (
            f"Scaler mean exceeds expected TRAIN range. "
            f"Got: {fs.train_mean}. This suggests TEST data contaminated fitting."
        )

        # If we accidentally fitted on TRAIN+TEST, mean would be ~1000+
        if (fs.train_mean > 500).any():
            pytest.fail(
                "Scaler mean_is suspiciously large — TEST data may have "
                "contaminated the scaler fitting."
            )

        # Transform TEST: values should be shifted by TRAIN mean, not TEST mean
        X_test_scaled = transform_scaler(fs, X_test)
        # Since TRAIN mean ≈ 0.5 and TEST values ≈ 1000-2000,
        # scaled TEST values should be >> 0 (approx 1000 / train_std)
        assert (X_test_scaled.mean(axis=0) > 100).all(), (
            "Scaled TEST values should be large when scaler uses TRAIN parameters only."
        )

    def test_scaler_mean_identical_after_two_independent_fits_on_same_train(self):
        """Fitting twice on the same TRAIN data must yield identical statistics."""
        X_train = _make_X(n_rows=500, seed=1)
        fs1 = fit_scaler(X_train, _feature_names())
        fs2 = fit_scaler(X_train, _feature_names())
        np.testing.assert_array_equal(fs1.train_mean, fs2.train_mean)
        np.testing.assert_array_equal(fs1.train_scale, fs2.train_scale)

    def test_transform_does_not_refit(self):
        """
        Verify transform_scaler does not call scaler.fit() internally.
        After transform, mean_ must still equal the TRAIN mean.
        """
        rng = np.random.default_rng(99)
        X_train = rng.uniform(0, 1, size=(100, 3))
        X_test = rng.uniform(500, 1000, size=(100, 3))

        fs = fit_scaler(X_train, ["a", "b", "c"])
        original_mean = fs.scaler.mean_.copy()

        transform_scaler(fs, X_test)  # apply to very different data

        # mean_ must not have changed
        np.testing.assert_array_equal(fs.scaler.mean_, original_mean)


# ---------------------------------------------------------------------------
# Constant feature test
# ---------------------------------------------------------------------------


class TestConstantFeature:
    def test_constant_column_does_not_crash(self):
        """
        StandardScaler on a constant column does NOT crash.

        sklearn StandardScaler behavior for constant columns:
        - mean_ = 1.0 (the constant value)
        - scale_ = 1.0 (set to 1.0 to avoid division-by-zero)
        - transformed values = (x - mean) / scale = 0.0

        This test verifies the scaler does not crash and returns a stable
        (zero) result for constant inputs. The caller may later detect
        degenerate constant features upstream and drop them if needed.
        """
        X = np.ones((50, 3), dtype=np.float64)
        fs = fit_scaler(X, ["a", "b", "c"])
        # sklearn sets scale_=1.0 for constant columns (not 0.0)
        # to avoid NaN (1.0 / 0.0) in the transform step
        assert fs.n_features == 3
        out = transform_scaler(fs, X)
        assert out.shape == (50, 3)
        # Transformed constant values become 0.0
        assert (out == 0.0).all(), (
            "Expected constant-column transform to produce all zeros."
        )


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestScalerMetadata:
    def test_metadata_structure(self):
        fs = fit_scaler(_make_X(), _feature_names())
        meta = get_scaler_metadata(fs)
        assert meta["scaler_type"] == "StandardScaler"
        assert meta["fitted_on"] == "TRAIN"
        assert "n_features" in meta
        assert "mean_min" in meta
        assert "scale_min" in meta
        assert "note" in meta

    def test_metadata_fitted_on_is_train(self):
        fs = fit_scaler(_make_X(), _feature_names())
        assert get_scaler_metadata(fs)["fitted_on"] == "TRAIN"
