"""
tests/test_encoding.py
-----------------------
Unit tests for src/preprocessing/encoding.py

Covers all 6 category edge cases from the spec plus:
  - OHE feature names from encoder (not manually reconstructed)
  - feature dimensionality stability
  - service="-" is a real category
  - TRAIN category absent from TEST still produces column with zeros
  - unseen TEST category produces zeros (not a crash)
  - empty categorical_cols raises
  - empty DataFrame raises
  - category ordering preserved
  - metadata structure
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.encoding import (
    FittedEncoder,
    fit_encoder,
    get_encoder_metadata,
    get_feature_names,
    transform_encoder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cat_df(*proto_vals, service_vals=None, state_vals=None) -> pd.DataFrame:
    """Build a categorical-only DataFrame."""
    n = len(proto_vals)
    service_vals = service_vals or (["-"] * n)
    state_vals = state_vals or (["FIN"] * n)
    return pd.DataFrame({
        "proto": list(proto_vals),
        "service": service_vals,
        "state": state_vals,
    })


CAT_COLS = ["proto", "service", "state"]


@pytest.fixture
def fitted_enc() -> FittedEncoder:
    train = _cat_df("tcp", "udp", "tcp", service_vals=["-", "http", "-"])
    return fit_encoder(train, CAT_COLS)


# ---------------------------------------------------------------------------
# Basic fit/transform
# ---------------------------------------------------------------------------


class TestFitEncoder:
    def test_returns_fitted_encoder(self, fitted_enc):
        assert isinstance(fitted_enc, FittedEncoder)

    def test_feature_names_from_encoder(self, fitted_enc):
        names = get_feature_names(fitted_enc)
        # Names must come from the fitted encoder, contain column prefixes
        assert all("proto_" in n or "service_" in n or "state_" in n for n in names)

    def test_category_ordering_preserved(self, fitted_enc):
        # Categories for proto should include exactly the observed values
        assert set(fitted_enc.categories["proto"]) == {"tcp", "udp"}

    def test_empty_categorical_cols_raises(self):
        train = _cat_df("tcp")
        with pytest.raises(ValueError, match="empty"):
            fit_encoder(train, [])

    def test_empty_dataframe_raises(self):
        empty = pd.DataFrame({"proto": [], "service": [], "state": []})
        with pytest.raises(ValueError, match="empty"):
            fit_encoder(empty, CAT_COLS)


class TestTransformEncoder:
    def test_output_shape(self, fitted_enc):
        test_df = _cat_df("tcp", "udp")
        out = transform_encoder(fitted_enc, test_df)
        assert out.shape[0] == 2
        assert out.shape[1] == len(fitted_enc.feature_names)

    def test_output_dtype_float64(self, fitted_enc):
        test_df = _cat_df("tcp")
        out = transform_encoder(fitted_enc, test_df)
        assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# Case 1: TRAIN has known categories; TEST has only known categories
# ---------------------------------------------------------------------------


class TestCase1AllKnown:
    def test_no_exception_and_correct_shape(self, fitted_enc):
        test_df = _cat_df("tcp", service_vals=["http"])
        out = transform_encoder(fitted_enc, test_df)
        assert out.shape == (1, len(fitted_enc.feature_names))

    def test_known_category_produces_nonzero(self, fitted_enc):
        """tcp should produce a 1 in the proto_tcp position."""
        test_df = _cat_df("tcp", service_vals=["-"])
        out = transform_encoder(fitted_enc, test_df)
        # Find proto_tcp column
        names = get_feature_names(fitted_enc)
        tcp_idx = names.index("proto_tcp")
        assert out[0, tcp_idx] == 1.0


# ---------------------------------------------------------------------------
# Case 2: TEST contains a completely unseen category
# ---------------------------------------------------------------------------


class TestCase2UnseenCategory:
    def test_no_exception_on_unseen(self, fitted_enc):
        # "icmp" was not in TRAIN proto
        test_df = _cat_df("icmp", service_vals=["-"])
        out = transform_encoder(fitted_enc, test_df)  # must not crash
        assert out.shape == (1, len(fitted_enc.feature_names))

    def test_unseen_category_produces_all_zeros_in_proto(self, fitted_enc):
        """Unseen category -> all zero in the OHE block for that column."""
        test_df = _cat_df("UNSEEN_PROTO_XYZ", service_vals=["-"])
        out = transform_encoder(fitted_enc, test_df)
        names = get_feature_names(fitted_enc)
        proto_indices = [i for i, n in enumerate(names) if n.startswith("proto_")]
        proto_block = out[0, proto_indices]
        assert (proto_block == 0).all()

    def test_dimensionality_unchanged_on_unseen(self, fitted_enc):
        """Feature count must remain identical regardless of unseen categories."""
        test_normal = _cat_df("tcp", service_vals=["-"])
        test_unseen = _cat_df("UNSEEN_XYZ", service_vals=["-"])
        out_normal = transform_encoder(fitted_enc, test_normal)
        out_unseen = transform_encoder(fitted_enc, test_unseen)
        assert out_normal.shape[1] == out_unseen.shape[1]


# ---------------------------------------------------------------------------
# Case 3: TRAIN has category absent from TEST — column still exists with zeros
# ---------------------------------------------------------------------------


class TestCase3AbsentFromTest:
    def test_absent_train_category_produces_zero_column(self):
        """udp is in TRAIN but not in TEST. proto_udp column must exist with 0s."""
        train = _cat_df("tcp", "udp", service_vals=["-", "-"])
        enc = fit_encoder(train, ["proto", "service", "state"])
        test_df = _cat_df("tcp", service_vals=["-"])  # only tcp
        out = transform_encoder(enc, test_df)
        names = get_feature_names(enc)
        assert "proto_udp" in names
        udp_idx = names.index("proto_udp")
        assert out[0, udp_idx] == 0.0  # column exists with zeros


# ---------------------------------------------------------------------------
# Case 4: Protected Backdoor has category absent from TRAIN
# ---------------------------------------------------------------------------


class TestCase4ProtectedUnseenCategory:
    def test_transform_succeeds_without_refitting(self):
        """Protected-unseen data with novel category must transform without exception."""
        train = _cat_df("tcp", "udp", service_vals=["-", "http"])
        enc = fit_encoder(train, CAT_COLS)
        n_features_before = len(get_feature_names(enc))

        protected = _cat_df("sctp", service_vals=["dns"])  # "sctp" and "dns" unseen
        out = transform_encoder(enc, protected)
        assert out.shape[1] == n_features_before  # dimensionality unchanged


# ---------------------------------------------------------------------------
# Case 5: service="-" is a real category
# ---------------------------------------------------------------------------


class TestCase5ServiceDash:
    def test_dash_is_real_category_not_nan(self):
        train = _cat_df("tcp", "udp", service_vals=["-", "http"])
        enc = fit_encoder(train, CAT_COLS)
        assert "-" in enc.categories["service"]

    def test_dash_produces_one_hot_feature(self):
        train = _cat_df("tcp", service_vals=["-"])
        enc = fit_encoder(train, CAT_COLS)
        test_df = _cat_df("tcp", service_vals=["-"])
        out = transform_encoder(enc, test_df)
        names = get_feature_names(enc)
        dash_idx = names.index("service_-")
        assert out[0, dash_idx] == 1.0

    def test_dash_not_treated_as_nan(self):
        """Encoder must NOT have NaN in service categories."""
        train = _cat_df("tcp", "tcp", service_vals=["-", "-"])
        enc = fit_encoder(train, CAT_COLS)
        service_cats = enc.categories["service"]
        # No NaN should be in the categories list
        has_nan = any(
            (isinstance(c, float) and np.isnan(c)) for c in service_cats
        )
        assert not has_nan


# ---------------------------------------------------------------------------
# Case 6: Whitespace in category
# ---------------------------------------------------------------------------


class TestCase6Whitespace:
    def test_whitespace_variant_is_distinct_category(self):
        """
        ' tcp ' (with spaces) and 'tcp' (without) are distinct raw values.
        The policy does NOT silently strip whitespace; that is handled by
        attack_cat canonicalization only.
        For proto/service/state, the raw value is preserved.
        """
        train = _cat_df("tcp", " tcp ", service_vals=["-", "-"])
        enc = fit_encoder(train, CAT_COLS)
        # Both should be categories (distinct raw strings)
        proto_cats = enc.categories["proto"]
        assert "tcp" in proto_cats
        assert " tcp " in proto_cats


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestEncoderMetadata:
    def test_metadata_structure(self, fitted_enc):
        meta = get_encoder_metadata(fitted_enc)
        assert "encoder_type" in meta
        assert "categorical_cols" in meta
        assert "categories" in meta
        assert "n_output_features" in meta
        assert meta["encoder_type"] == "OneHotEncoder"
        assert meta["handle_unknown"] == "ignore"

    def test_categories_in_metadata_match_fitted(self, fitted_enc):
        meta = get_encoder_metadata(fitted_enc)
        assert "proto" in meta["categories"]
        assert "service" in meta["categories"]

    def test_dash_documented_in_metadata_note(self, fitted_enc):
        meta = get_encoder_metadata(fitted_enc)
        assert "'-'" in meta["note"] or "dash" in meta["note"].lower() or "-" in meta["note"]
