"""
tests/test_processed_dataset.py
---------------------------------
Unit tests for src/preprocessing/processed_dataset.py

Covers:
  - Construction with valid data
  - __post_init__ integrity checks (shape, alignment)
  - label, attack_cat, id NOT in feature_names
  - view_type identity
  - to_summary_dict structure
  - y/attack_cat row alignment preserved
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.processed_dataset import (
    ProcessedDataset,
    VIEW_SCALED,
    VIEW_UNSCALED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    n_rows: int = 20,
    n_features: int = 10,
    view_type: str = VIEW_SCALED,
    split_name: str = "train",
) -> ProcessedDataset:
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(n_rows, n_features)).astype(np.float64)
    y = pd.Series(rng.integers(0, 2, size=n_rows), name="label")
    attack_cat = pd.Series(["Normal"] * n_rows, name="attack_cat")
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return ProcessedDataset(
        X=X,
        y=y,
        attack_cat=attack_cat,
        feature_names=feature_names,
        view_type=view_type,
        split_name=split_name,
        n_rows=n_rows,
        n_features=n_features,
        encoder_metadata={"encoder_type": "OneHotEncoder"},
        scaler_metadata={"scaler_type": "StandardScaler"},
        categorical_cols=["proto"],
        numeric_cols=["dur", "sbytes"],
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestProcessedDatasetConstruction:
    def test_valid_construction(self):
        ds = _make_dataset()
        assert ds.n_rows == 20
        assert ds.n_features == 10

    def test_x_shape_correct(self):
        ds = _make_dataset()
        assert ds.X.shape == (20, 10)

    def test_y_length_matches_rows(self):
        ds = _make_dataset()
        assert len(ds.y) == 20

    def test_attack_cat_length_matches_rows(self):
        ds = _make_dataset()
        assert len(ds.attack_cat) == 20

    def test_feature_names_length_matches_features(self):
        ds = _make_dataset()
        assert len(ds.feature_names) == 10

    def test_view_type_stored_correctly(self):
        ds_scaled = _make_dataset(view_type=VIEW_SCALED)
        ds_unscaled = _make_dataset(view_type=VIEW_UNSCALED)
        assert ds_scaled.view_type == "scaled"
        assert ds_unscaled.view_type == "unscaled"

    def test_split_name_stored(self):
        ds = _make_dataset(split_name="development_test")
        assert ds.split_name == "development_test"


# ---------------------------------------------------------------------------
# Integrity checks (__post_init__)
# ---------------------------------------------------------------------------


class TestIntegrityChecks:
    def test_x_shape_mismatch_raises(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(20, 8))  # wrong n_features (8 != 10)
        y = pd.Series(np.zeros(20))
        attack_cat = pd.Series(["N"] * 20)
        with pytest.raises(ValueError, match="X.shape"):
            ProcessedDataset(
                X=X, y=y, attack_cat=attack_cat,
                feature_names=[f"f{i}" for i in range(10)],
                view_type=VIEW_SCALED, split_name="t",
                n_rows=20, n_features=10,
            )

    def test_y_length_mismatch_raises(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(20, 10))
        y = pd.Series(np.zeros(15))  # wrong length
        attack_cat = pd.Series(["N"] * 20)
        with pytest.raises(ValueError, match="y length"):
            ProcessedDataset(
                X=X, y=y, attack_cat=attack_cat,
                feature_names=[f"f{i}" for i in range(10)],
                view_type=VIEW_SCALED, split_name="t",
                n_rows=20, n_features=10,
            )

    def test_attack_cat_length_mismatch_raises(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(20, 10))
        y = pd.Series(np.zeros(20))
        attack_cat = pd.Series(["N"] * 5)  # wrong length
        with pytest.raises(ValueError, match="attack_cat length"):
            ProcessedDataset(
                X=X, y=y, attack_cat=attack_cat,
                feature_names=[f"f{i}" for i in range(10)],
                view_type=VIEW_SCALED, split_name="t",
                n_rows=20, n_features=10,
            )

    def test_feature_names_length_mismatch_raises(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(20, 10))
        y = pd.Series(np.zeros(20))
        ac = pd.Series(["N"] * 20)
        with pytest.raises(ValueError, match="feature_names length"):
            ProcessedDataset(
                X=X, y=y, attack_cat=ac,
                feature_names=["only_3_names"],  # wrong
                view_type=VIEW_SCALED, split_name="t",
                n_rows=20, n_features=10,
            )


# ---------------------------------------------------------------------------
# Metadata / target leakage prevention
# ---------------------------------------------------------------------------


class TestTargetLeakagePrevention:
    def test_label_not_in_feature_names(self):
        ds = _make_dataset()
        assert "label" not in ds.feature_names

    def test_attack_cat_not_in_feature_names(self):
        ds = _make_dataset()
        assert "attack_cat" not in ds.feature_names

    def test_id_not_in_feature_names(self):
        ds = _make_dataset()
        assert "id" not in ds.feature_names

    def test_y_alignment_preserved(self):
        """y rows must correspond 1-to-1 with X rows."""
        rng = np.random.default_rng(7)
        n = 10
        X = rng.uniform(size=(n, 5))
        y_vals = list(range(n))  # unique labels to verify ordering
        y = pd.Series(y_vals)
        ac = pd.Series(["N"] * n)
        ds = ProcessedDataset(
            X=X, y=y, attack_cat=ac,
            feature_names=[f"f{i}" for i in range(5)],
            view_type=VIEW_SCALED, split_name="t",
            n_rows=n, n_features=5,
        )
        # Row 3 of X should align with y[3]
        assert ds.y.iloc[3] == 3

    def test_attack_cat_alignment_preserved(self):
        rng = np.random.default_rng(7)
        n = 10
        X = rng.uniform(size=(n, 5))
        y = pd.Series([0] * n)
        ac_vals = [f"cat_{i}" for i in range(n)]
        ac = pd.Series(ac_vals)
        ds = ProcessedDataset(
            X=X, y=y, attack_cat=ac,
            feature_names=[f"f{i}" for i in range(5)],
            view_type=VIEW_SCALED, split_name="t",
            n_rows=n, n_features=5,
        )
        assert ds.attack_cat.iloc[5] == "cat_5"


# ---------------------------------------------------------------------------
# to_summary_dict
# ---------------------------------------------------------------------------


class TestSummaryDict:
    def test_summary_dict_structure(self):
        ds = _make_dataset()
        summary = ds.to_summary_dict()
        assert "split_name" in summary
        assert "view_type" in summary
        assert "n_rows" in summary
        assert "n_features" in summary
        assert "feature_names_head" in summary
        assert "y_value_counts" in summary
        assert "attack_cat_counts" in summary
        assert "encoder_metadata" in summary
        assert "scaler_metadata" in summary

    def test_summary_dict_does_not_contain_raw_x(self):
        ds = _make_dataset()
        summary = ds.to_summary_dict()
        # The dict should be small — no raw numpy arrays
        import json
        json_str = json.dumps(summary)  # must be JSON-serialisable
        assert len(json_str) < 50_000  # no huge raw array dumps


# ---------------------------------------------------------------------------
# Single-row / small input
# ---------------------------------------------------------------------------


class TestSmallInputs:
    def test_single_row_dataset(self):
        ds = _make_dataset(n_rows=1)
        assert ds.n_rows == 1
        assert ds.X.shape == (1, 10)

    def test_two_row_dataset(self):
        ds = _make_dataset(n_rows=2)
        assert ds.n_rows == 2
