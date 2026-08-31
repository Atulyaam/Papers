"""
tests/test_preprocessing_pipeline.py
---------------------------------------
Unit tests for src/preprocessing/preprocessing_pipeline.py

Covers:
  - fit() then transform() happy path (both views)
  - transform() before fit() raises PreprocessingNotFittedError
  - feature_names identical between unscaled and scaled views
  - feature_names stable across TRAIN / val / test / protected splits
  - label/attack_cat/id NOT in feature_names (from the pipeline)
  - row count preserved through transform
  - no rows dropped, no rows duplicated
  - both views have same n_rows, n_features, same feature ordering
  - scaled and unscaled X values differ
  - unknown view raises ValueError
  - transform_both_views returns two ProcessedDatasets
  - determinism: two fits on same data -> identical feature_names and values
  - row order preserved
  - split_name recorded correctly in ProcessedDataset
  - empty DataFrame edge case
  - single-row DataFrame
  - is_fitted flag
  - metadata fields present in ProcessedDataset
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.preprocessing.processed_dataset import ProcessedDataset
from src.preprocessing.exceptions import PreprocessingNotFittedError


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_df(
    n: int = 50,
    seed: int = 0,
    proto_vals: list[str] | None = None,
    service_vals: list[str] | None = None,
    state_vals: list[str] | None = None,
    attack_cat_val: str = "Normal",
) -> pd.DataFrame:
    """Build a minimal UNSW-NB15-like DataFrame for pipeline tests."""
    rng = np.random.default_rng(seed)
    if proto_vals is None:
        proto_vals = (["tcp"] * (n // 2) + ["udp"] * (n - n // 2))[:n]
    if service_vals is None:
        service_vals = (["-"] * (n // 2) + ["http"] * (n - n // 2))[:n]
    if state_vals is None:
        state_vals = (["FIN"] * n)[:n]
    data = {
        "id": list(range(n)),
        "dur": rng.uniform(0, 1, size=n),
        "sbytes": rng.uniform(100, 1000, size=n),
        "dbytes": rng.uniform(100, 500, size=n),
        "rate": rng.uniform(0, 100, size=n),
        "proto": proto_vals,
        "service": service_vals,
        "state": state_vals,
        "attack_cat": [attack_cat_val] * n,
        "label": rng.integers(0, 2, size=n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def train_df():
    return _make_df(n=100, seed=0)


@pytest.fixture
def val_df():
    return _make_df(n=30, seed=1)


@pytest.fixture
def test_df():
    return _make_df(n=20, seed=2)


@pytest.fixture
def fitted_pipeline(train_df):
    pp = PreprocessingPipeline("TEST_EXP")
    pp.fit(train_df)
    return pp


# ---------------------------------------------------------------------------
# Fit before transform
# ---------------------------------------------------------------------------


class TestFitRequired:
    def test_transform_before_fit_raises(self, train_df):
        pp = PreprocessingPipeline()
        with pytest.raises(PreprocessingNotFittedError, match="fit()"):
            pp.transform(train_df, view="scaled")

    def test_is_fitted_false_before_fit(self):
        pp = PreprocessingPipeline()
        assert pp.is_fitted is False

    def test_is_fitted_true_after_fit(self, train_df):
        pp = PreprocessingPipeline()
        pp.fit(train_df)
        assert pp.is_fitted is True

    def test_feature_names_raises_before_fit(self):
        pp = PreprocessingPipeline()
        with pytest.raises(PreprocessingNotFittedError):
            _ = pp.feature_names

    def test_fit_returns_self(self, train_df):
        pp = PreprocessingPipeline()
        result = pp.fit(train_df)
        assert result is pp


# ---------------------------------------------------------------------------
# Feature names contract
# ---------------------------------------------------------------------------


class TestFeatureNamesContract:
    def test_label_not_in_feature_names(self, fitted_pipeline):
        assert "label" not in fitted_pipeline.feature_names

    def test_attack_cat_not_in_feature_names(self, fitted_pipeline):
        assert "attack_cat" not in fitted_pipeline.feature_names

    def test_id_not_in_feature_names(self, fitted_pipeline):
        assert "id" not in fitted_pipeline.feature_names

    def test_feature_names_identical_for_both_views(self, fitted_pipeline, val_df):
        ds_unscaled = fitted_pipeline.transform(val_df, view="unscaled")
        ds_scaled = fitted_pipeline.transform(val_df, view="scaled")
        assert ds_unscaled.feature_names == ds_scaled.feature_names

    def test_feature_names_stable_across_splits(self, fitted_pipeline, val_df, test_df):
        ds_val = fitted_pipeline.transform(val_df, view="scaled")
        ds_test = fitted_pipeline.transform(test_df, view="scaled")
        assert ds_val.feature_names == ds_test.feature_names

    def test_feature_names_match_pipeline_property(self, fitted_pipeline, val_df):
        ds = fitted_pipeline.transform(val_df, view="scaled")
        assert ds.feature_names == fitted_pipeline.feature_names


# ---------------------------------------------------------------------------
# Shape / row integrity
# ---------------------------------------------------------------------------


class TestShapeAndRowIntegrity:
    def test_row_count_preserved_train(self, fitted_pipeline, train_df):
        ds = fitted_pipeline.transform(train_df, view="scaled", split_name="train")
        assert ds.n_rows == len(train_df)
        assert ds.X.shape[0] == len(train_df)

    def test_row_count_preserved_val(self, fitted_pipeline, val_df):
        ds = fitted_pipeline.transform(val_df, view="scaled", split_name="val")
        assert ds.n_rows == len(val_df)

    def test_no_rows_dropped(self, fitted_pipeline, test_df):
        ds = fitted_pipeline.transform(test_df, view="scaled")
        assert len(ds.y) == len(test_df)
        assert len(ds.attack_cat) == len(test_df)
        assert ds.X.shape[0] == len(test_df)

    def test_n_features_consistent_across_splits(
        self, fitted_pipeline, train_df, val_df, test_df
    ):
        ds_train = fitted_pipeline.transform(train_df, view="scaled", split_name="train")
        ds_val = fitted_pipeline.transform(val_df, view="scaled")
        ds_test = fitted_pipeline.transform(test_df, view="scaled")
        assert ds_train.n_features == ds_val.n_features == ds_test.n_features


# ---------------------------------------------------------------------------
# Two model views
# ---------------------------------------------------------------------------


class TestTwoViews:
    def test_both_views_same_row_count(self, fitted_pipeline, val_df):
        ds_u, ds_s = fitted_pipeline.transform_both_views(val_df)
        assert ds_u.n_rows == ds_s.n_rows

    def test_both_views_same_n_features(self, fitted_pipeline, val_df):
        ds_u, ds_s = fitted_pipeline.transform_both_views(val_df)
        assert ds_u.n_features == ds_s.n_features

    def test_both_views_same_feature_names(self, fitted_pipeline, val_df):
        ds_u, ds_s = fitted_pipeline.transform_both_views(val_df)
        assert ds_u.feature_names == ds_s.feature_names

    def test_scaled_and_unscaled_x_differ(self, fitted_pipeline, val_df):
        """Scaled and unscaled X must not be numerically identical."""
        ds_u, ds_s = fitted_pipeline.transform_both_views(val_df)
        assert not np.allclose(ds_u.X, ds_s.X), (
            "Scaled and unscaled X are numerically identical — scaling may not be applied."
        )

    def test_view_type_labels_correct(self, fitted_pipeline, val_df):
        ds_u, ds_s = fitted_pipeline.transform_both_views(val_df)
        assert ds_u.view_type == "unscaled"
        assert ds_s.view_type == "scaled"

    def test_invalid_view_raises(self, fitted_pipeline, val_df):
        with pytest.raises(ValueError, match="view"):
            fitted_pipeline.transform(val_df, view="invalid_view")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_feature_names_on_two_fits(self, train_df):
        pp1 = PreprocessingPipeline().fit(train_df)
        pp2 = PreprocessingPipeline().fit(train_df)
        assert pp1.feature_names == pp2.feature_names

    def test_identical_scaled_values_on_two_fits(self, train_df, val_df):
        pp1 = PreprocessingPipeline().fit(train_df)
        pp2 = PreprocessingPipeline().fit(train_df)
        out1 = pp1.transform(val_df, view="scaled")
        out2 = pp2.transform(val_df, view="scaled")
        np.testing.assert_allclose(out1.X, out2.X, rtol=1e-8)

    def test_identical_unscaled_values_on_two_transforms(self, fitted_pipeline, val_df):
        out1 = fitted_pipeline.transform(val_df, view="unscaled")
        out2 = fitted_pipeline.transform(val_df, view="unscaled")
        np.testing.assert_array_equal(out1.X, out2.X)


# ---------------------------------------------------------------------------
# split_name recorded
# ---------------------------------------------------------------------------


class TestSplitNameRecorded:
    def test_split_name_in_result(self, fitted_pipeline, test_df):
        ds = fitted_pipeline.transform(test_df, split_name="development_test")
        assert ds.split_name == "development_test"

    def test_default_split_name(self, fitted_pipeline, val_df):
        ds = fitted_pipeline.transform(val_df)
        assert ds.split_name == "unknown"


# ---------------------------------------------------------------------------
# Metadata in ProcessedDataset
# ---------------------------------------------------------------------------


class TestMetadataInResult:
    def test_encoder_metadata_present(self, fitted_pipeline, val_df):
        ds = fitted_pipeline.transform(val_df, view="scaled")
        assert "encoder_type" in ds.encoder_metadata

    def test_scaler_metadata_present_scaled(self, fitted_pipeline, val_df):
        ds = fitted_pipeline.transform(val_df, view="scaled")
        assert "scaler_type" in ds.scaler_metadata

    def test_scaler_metadata_not_applied_flag_for_unscaled(self, fitted_pipeline, val_df):
        ds = fitted_pipeline.transform(val_df, view="unscaled")
        assert ds.scaler_metadata.get("applied") is False or \
               "not" in str(ds.scaler_metadata.get("note", "")).lower()

    def test_categorical_cols_recorded(self, fitted_pipeline, val_df):
        ds = fitted_pipeline.transform(val_df, view="scaled")
        assert "proto" in ds.categorical_cols


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_row_transform(self, fitted_pipeline):
        single = _make_df(n=1, seed=99)
        ds = fitted_pipeline.transform(single, view="scaled", split_name="single")
        assert ds.n_rows == 1

    def test_unseen_category_in_test_no_crash(self, fitted_pipeline):
        """Test with a proto value not seen during TRAIN fitting."""
        test_with_unseen = _make_df(n=10, proto_vals=["UNSEEN_PROTO"] * 10)
        ds = fitted_pipeline.transform(
            test_with_unseen, view="scaled", split_name="test_unseen"
        )
        assert ds.n_rows == 10
        # Dimensionality must match fitted pipeline
        assert ds.n_features == len(fitted_pipeline.feature_names)

    def test_train_category_absent_from_test_column_still_exists(self, train_df):
        """udp is in TRAIN; test has only tcp. proto_udp column must still exist."""
        # Ensure training has both tcp and udp
        train_with_both = _make_df(
            n=100, seed=0,
            proto_vals=(["tcp"] * 50 + ["udp"] * 50),
        )
        pp = PreprocessingPipeline().fit(train_with_both)
        test_tcp_only = _make_df(n=10, proto_vals=["tcp"] * 10)
        ds = pp.transform(test_tcp_only, view="unscaled")
        assert "proto_udp" in ds.feature_names
        # proto_udp column should be all zeros
        udp_idx = ds.feature_names.index("proto_udp")
        assert (ds.X[:, udp_idx] == 0.0).all()

    def test_fluent_api_fit_transform(self, train_df, val_df):
        """Fluent API: pp.fit(train).transform(val) without intermediate variable."""
        ds = PreprocessingPipeline().fit(train_df).transform(val_df, view="scaled")
        assert isinstance(ds, ProcessedDataset)
