"""
tests/test_base_models_preprocessing.py
-----------------------------------------
Unit tests for src/models/base_models/preprocessing.py

Covers
------
- load_selected_features: happy path, missing file, wrong count, missing key
- build_feature_matrix: happy path, missing feature, duplicate names,
  wrong shape, non-finite values, extra columns ignored
- fit_scaler: happy path, empty X, non-finite X
- Edge cases: NaN, +inf, -inf, constant features, one-row X
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.base_models.preprocessing import (
    EXPECTED_FEATURE_COUNT,
    FEATURE_SET_ID,
    build_feature_matrix,
    fit_scaler,
    load_selected_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dummy_features_json(tmp_path: Path) -> Path:
    """Write a valid selected_features.json with 75 dummy feature names."""
    features = [f"feat_{i}" for i in range(75)]
    data = {
        "experiment_id": "EXP_MI_V1_1",
        "features": features,
        "feature_count": 75,
        "selected_k": 75,
    }
    p = tmp_path / "selected_features.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture()
def dummy_df(dummy_features_json: Path) -> pd.DataFrame:
    """DataFrame with exactly the 75 dummy features plus extra cols."""
    features = [f"feat_{i}" for i in range(75)]
    rng = np.random.default_rng(0)
    data = rng.standard_normal((50, 75))
    df = pd.DataFrame(data, columns=features)
    df["label"] = rng.integers(0, 2, size=50)   # extra col, should be ignored
    return df


# ---------------------------------------------------------------------------
# load_selected_features
# ---------------------------------------------------------------------------

class TestLoadSelectedFeatures:
    def test_happy_path(self, dummy_features_json: Path):
        features = load_selected_features(dummy_features_json)
        assert len(features) == 75
        assert features[0] == "feat_0"
        assert features[74] == "feat_74"

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_selected_features(tmp_path / "nonexistent.json")

    def test_wrong_count(self, tmp_path: Path):
        bad = {"experiment_id": "EXP_MI_V1_1", "features": ["a", "b"]}
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="75"):
            load_selected_features(p)

    def test_missing_features_key(self, tmp_path: Path):
        bad = {"experiment_id": "EXP_MI_V1_1"}
        p = tmp_path / "nofeatures.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="'features' key missing"):
            load_selected_features(p)

    def test_returns_list(self, dummy_features_json: Path):
        features = load_selected_features(dummy_features_json)
        assert isinstance(features, list)

    def test_all_strings(self, dummy_features_json: Path):
        features = load_selected_features(dummy_features_json)
        assert all(isinstance(f, str) for f in features)

    def test_path_str_accepted(self, dummy_features_json: Path):
        """str path should also work."""
        features = load_selected_features(str(dummy_features_json))
        assert len(features) == 75


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_happy_path(self, dummy_features_json: Path, dummy_df: pd.DataFrame):
        features = load_selected_features(dummy_features_json)
        X = build_feature_matrix(dummy_df, features)
        assert X.shape == (50, 75)
        assert X.dtype == np.float64

    def test_correct_column_order(self, dummy_features_json: Path):
        features = load_selected_features(dummy_features_json)
        rng = np.random.default_rng(1)
        data = rng.standard_normal((10, 75))
        df = pd.DataFrame(data, columns=features)
        X = build_feature_matrix(df, features)
        np.testing.assert_array_equal(X, df[features].to_numpy())

    def test_missing_feature_raises(self, dummy_features_json: Path):
        features = load_selected_features(dummy_features_json)
        # Drop one feature from the DataFrame
        rng = np.random.default_rng(2)
        df = pd.DataFrame(rng.standard_normal((10, 74)), columns=features[:74])
        with pytest.raises(ValueError, match="missing"):
            build_feature_matrix(df, features)

    def test_duplicate_feature_names_raises(self):
        features = ["a"] * 75  # all duplicates
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            build_feature_matrix(df, features)

    def test_wrong_feature_count_raises(self, dummy_df: pd.DataFrame):
        features = ["feat_0", "feat_1"]  # only 2, not 75
        with pytest.raises(ValueError, match="75"):
            build_feature_matrix(dummy_df, features)

    def test_nan_raises(self, dummy_features_json: Path):
        features = load_selected_features(dummy_features_json)
        rng = np.random.default_rng(3)
        data = rng.standard_normal((10, 75))
        data[0, 0] = float("nan")
        df = pd.DataFrame(data, columns=features)
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            build_feature_matrix(df, features)

    def test_inf_raises(self, dummy_features_json: Path):
        features = load_selected_features(dummy_features_json)
        rng = np.random.default_rng(4)
        data = rng.standard_normal((10, 75))
        data[5, 10] = float("inf")
        df = pd.DataFrame(data, columns=features)
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            build_feature_matrix(df, features)

    def test_neg_inf_raises(self, dummy_features_json: Path):
        features = load_selected_features(dummy_features_json)
        rng = np.random.default_rng(5)
        data = rng.standard_normal((10, 75))
        data[3, 3] = float("-inf")
        df = pd.DataFrame(data, columns=features)
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            build_feature_matrix(df, features)

    def test_extra_columns_ignored(self, dummy_features_json: Path):
        features = load_selected_features(dummy_features_json)
        rng = np.random.default_rng(6)
        data = rng.standard_normal((10, 75))
        df = pd.DataFrame(data, columns=features)
        df["extra_col"] = 999.0
        X = build_feature_matrix(df, features)
        assert X.shape == (10, 75)

    def test_single_row(self, dummy_features_json: Path):
        features = load_selected_features(dummy_features_json)
        df = pd.DataFrame([[1.0] * 75], columns=features)
        X = build_feature_matrix(df, features)
        assert X.shape == (1, 75)

    def test_constant_feature(self, dummy_features_json: Path):
        """Constant features are valid (scaler handles them separately)."""
        features = load_selected_features(dummy_features_json)
        rng = np.random.default_rng(7)
        data = rng.standard_normal((10, 75))
        data[:, 0] = 0.0  # constant
        df = pd.DataFrame(data, columns=features)
        X = build_feature_matrix(df, features)
        assert X.shape == (10, 75)
        assert np.all(X[:, 0] == 0.0)


# ---------------------------------------------------------------------------
# fit_scaler
# ---------------------------------------------------------------------------

class TestFitScaler:
    def test_happy_path(self):
        rng = np.random.default_rng(8)
        X = rng.standard_normal((100, 75))
        scaler = fit_scaler(X)
        assert hasattr(scaler, "mean_")
        assert scaler.mean_.shape == (75,)

    def test_transform_works(self):
        rng = np.random.default_rng(9)
        X_tr = rng.standard_normal((100, 75)) * 5 + 10
        X_va = rng.standard_normal((20, 75)) * 5 + 10
        scaler = fit_scaler(X_tr)
        X_tr_scaled = scaler.transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        # Training data should be approximately zero-mean
        assert abs(X_tr_scaled.mean()) < 0.1
        assert X_va_scaled.shape == (20, 75)

    def test_empty_X_raises(self):
        X = np.empty((0, 75))
        with pytest.raises(ValueError, match="empty"):
            fit_scaler(X)

    def test_nan_raises(self):
        X = np.ones((10, 75))
        X[0, 0] = float("nan")
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            fit_scaler(X)

    def test_inf_raises(self):
        X = np.ones((10, 75))
        X[5, 5] = float("inf")
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            fit_scaler(X)

    def test_one_row(self):
        """fit_scaler on a single row should not raise (std will be 0)."""
        X = np.array([[1.0] * 75])
        scaler = fit_scaler(X)
        assert scaler.mean_.shape == (75,)

    def test_scaler_not_fit_on_val(self):
        """
        Leakage guard: if we fit on train only, val transformation must differ
        from a scaler that mistakenly used val stats.
        """
        rng = np.random.default_rng(10)
        X_tr = rng.standard_normal((100, 75))
        X_va = rng.standard_normal((20, 75)) * 10 + 50  # very different distribution

        scaler_train_only = fit_scaler(X_tr)
        X_va_transformed = scaler_train_only.transform(X_va)

        # The val-transformed data should NOT be zero-mean (because scaler
        # was fit on train, which has different stats).
        # If scaler had been fit on val, mean would be ~0.
        assert abs(X_va_transformed.mean()) > 1.0, (
            "Val transformation appears to use val statistics — "
            "this would indicate a scaler leakage bug."
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_expected_feature_count(self):
        assert EXPECTED_FEATURE_COUNT == 75

    def test_feature_set_id(self):
        assert FEATURE_SET_ID == "EXP_MI_V1_1"
