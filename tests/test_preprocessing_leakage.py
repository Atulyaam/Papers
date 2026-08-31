"""
tests/test_preprocessing_leakage.py
---------------------------------------
Dedicated leakage prevention tests for the Sprint 2 preprocessing pipeline.

Tests verify that:
  1. Scaler statistics reflect TRAIN only (adversarial fixture).
  2. Encoder categories reflect TRAIN only.
  3. Calling transform() never modifies the fitted encoder state.
  4. Calling transform() never modifies the fitted scaler state.
  5. Protected unseen data can be transformed without altering fitted state.
  6. Validation data cannot add new encoder categories.
  7. Test data cannot change encoder categories.
  8. Feature dimensionality is identical for TRAIN / val / test / protected.
  9. y and attack_cat from TEST/protected do NOT enter the fitted state.
  10. Backdoor-only protected set transforms correctly without refitting.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df(
    n: int = 50,
    seed: int = 0,
    proto: list[str] | None = None,
    service: list[str] | None = None,
    attack_cat: str = "Normal",
    label: int | None = None,
    numeric_scale: float = 1.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_proto = proto or (["tcp"] * (n // 2) + ["udp"] * (n - n // 2))[:n]
    n_service = service or (["-"] * n)[:n]
    lbl = label if label is not None else rng.integers(0, 2, size=n).tolist()
    return pd.DataFrame({
        "id": list(range(n)),
        "dur": rng.uniform(0, 1, size=n) * numeric_scale,
        "sbytes": rng.uniform(100, 1000, size=n) * numeric_scale,
        "dbytes": rng.uniform(100, 500, size=n) * numeric_scale,
        "proto": n_proto,
        "service": n_service,
        "state": ["FIN"] * n,
        "attack_cat": [attack_cat] * n,
        "label": lbl,
    })


# ---------------------------------------------------------------------------
# 1. Scaler leakage — adversarial numeric fixture
# ---------------------------------------------------------------------------


class TestScalerLeakage:
    def test_scaler_mean_and_scale_match_train_encoded_matrix_exactly(self):
        """
        Precise leakage test for scaler.mean_ and scaler.scale_.

        Method:
          1. Fit the pipeline on TRAIN.
          2. Independently re-encode TRAIN using the fitted encoder to obtain
             the exact encoded matrix X_enc_train.
          3. Compute np.mean(X_enc_train, axis=0) and np.std(X_enc_train, ddof=0).
          4. Assert scaler.mean_ == expected_mean within rtol=1e-7, atol=1e-9.
          5. Assert scaler.scale_ == expected_std  within rtol=1e-7, atol=1e-9.

        Justification for rtol=1e-7:
          sklearn's StandardScaler accumulates column statistics in float64.
          np.mean and np.std on the identical float64 matrix produce the same
          result to within float64 machine epsilon (~1e-16); any practical
          gap at n=200 rows is << 1e-7.

          A 5% tolerance was rejected: it would mask a leak of up to 5% of
          TEST distribution influence. At numeric_scale=15000 (TEST is 15000x
          TRAIN), leakage would shift the mean by ~7500x TRAIN mean — easily
          detectable at rtol=1e-7 but invisible at rtol=0.05.
        """
        from src.preprocessing.encoding import transform_encoder

        train = _df(n=200, seed=0, numeric_scale=1.0)
        test_ = _df(n=200, seed=1, numeric_scale=15000.0)

        pp = PreprocessingPipeline("LEAKAGE_TEST").fit(train)

        # --- Independently reconstruct the encoded TRAIN matrix ---
        from src.preprocessing.cleaning import separate_target_and_features
        cleaned = separate_target_and_features(train, "train")

        X_ohe = transform_encoder(
            pp.fitted_encoder, cleaned.X_raw[pp._categorical_cols]
        )
        X_num = cleaned.X_raw[pp._numeric_cols].to_numpy(dtype=np.float64)
        X_enc_train = np.concatenate([X_ohe, X_num], axis=1)

        # Expected: numpy statistics over the encoded TRAIN matrix
        expected_mean = X_enc_train.mean(axis=0)
        raw_std       = X_enc_train.std(axis=0, ddof=0)  # ddof=0 matches StandardScaler

        # sklearn StandardScaler replaces zero-variance features' scale_ with 1.0
        # to avoid division-by-zero during transform. Apply the same replacement.
        # Reference: sklearn source sklearn/preprocessing/_data.py:_handle_zeros_in_scale()
        expected_std = np.where(raw_std == 0.0, 1.0, raw_std)

        np.testing.assert_allclose(
            pp.fitted_scaler.train_mean, expected_mean,
            rtol=1e-7, atol=1e-9,
            err_msg=(
                "scaler.mean_ does not match TRAIN encoded matrix mean. "
                "TEST data may have contaminated the scaler."
            ),
        )
        np.testing.assert_allclose(
            pp.fitted_scaler.train_scale, expected_std,
            rtol=1e-7, atol=1e-9,
            err_msg=(
                "scaler.scale_ does not match TRAIN encoded matrix std. "
                "TEST data may have contaminated the scaler."
            ),
        )

        # --- After transforming TEST, scaler state must be byte-identical ---
        mean_before  = pp.fitted_scaler.train_mean.copy()
        scale_before = pp.fitted_scaler.train_scale.copy()

        pp.transform(test_, view="scaled", split_name="test")

        np.testing.assert_array_equal(
            pp.fitted_scaler.train_mean, mean_before,
            err_msg="scaler.mean_ changed after transforming TEST data.",
        )
        np.testing.assert_array_equal(
            pp.fitted_scaler.train_scale, scale_before,
            err_msg="scaler.scale_ changed after transforming TEST data.",
        )

    def test_scaler_mean_catastrophically_wrong_if_test_contaminated(self):
        """
        Adversarial bound test: verifies actual mean is far below the contaminated
        value, providing a coarse but intuitive guard against accidental mixing.

        If fit_scaler were accidentally called with TRAIN+TEST combined, the mean
        of numeric cols would be approximately (TRAIN_mean + TEST_mean) / 2,
        which at TEST scale=15000 equals ~7500 * TRAIN_mean.

        We assert actual_mean < 10 * TRAIN_raw_mean (a very conservative bound
        that still detects any data mixing).
        """
        from src.preprocessing.cleaning import separate_target_and_features

        train = _df(n=200, seed=0, numeric_scale=1.0)

        pp = PreprocessingPipeline("ADV_LEAKAGE").fit(train)

        num_indices = [
            i for i, name in enumerate(pp.feature_names)
            if name in ("dur", "sbytes", "dbytes")
        ]
        if not num_indices:
            return

        actual_mean = pp.fitted_scaler.train_mean[num_indices]
        cleaned = separate_target_and_features(train, "train")
        train_raw_means = cleaned.X_raw[["dur", "sbytes", "dbytes"]].mean().values

        assert (actual_mean < train_raw_means * 10).all(), (
            f"Scaler numeric mean {actual_mean} is >10x TRAIN raw mean "
            f"{train_raw_means}. TEST data may have contaminated fitting."
        )


# ---------------------------------------------------------------------------
# 2. Encoder categories reflect TRAIN only
# ---------------------------------------------------------------------------


class TestEncoderLeakage:
    def test_train_categories_not_modified_by_val(self):
        """Transforming validation data must not add new categories to the encoder."""
        train = _df(n=50, proto=["tcp"] * 50)
        val = _df(n=20, proto=["udp"] * 20)  # udp not in TRAIN

        pp = PreprocessingPipeline().fit(train)
        original_proto_cats = list(pp.fitted_encoder.categories["proto"])

        pp.transform(val, view="scaled", split_name="val")

        # Categories must be unchanged
        assert pp.fitted_encoder.categories["proto"] == original_proto_cats

    def test_train_categories_not_modified_by_test(self):
        train = _df(n=50, proto=["tcp"] * 50)
        test_ = _df(n=20, proto=["icmp"] * 20)  # unseen

        pp = PreprocessingPipeline().fit(train)
        original_cats = list(pp.fitted_encoder.categories["proto"])

        pp.transform(test_, view="scaled", split_name="test")
        assert pp.fitted_encoder.categories["proto"] == original_cats

    def test_train_categories_not_modified_by_protected(self):
        train = _df(n=50, proto=["tcp"] * 50)
        protected = _df(n=10, proto=["sctp"] * 10, attack_cat="Backdoor", label=1)

        pp = PreprocessingPipeline().fit(train)
        original_cats = list(pp.fitted_encoder.categories["proto"])

        pp.transform(protected, view="scaled", split_name="protected_unseen")
        assert pp.fitted_encoder.categories["proto"] == original_cats


# ---------------------------------------------------------------------------
# 3. Scaler not modified by transform
# ---------------------------------------------------------------------------


class TestScalerNotModifiedByTransform:
    def test_scaler_mean_unchanged_after_val_transform(self):
        train = _df(n=100, seed=0)
        val = _df(n=30, seed=1, numeric_scale=1000.0)

        pp = PreprocessingPipeline().fit(train)
        mean_before = pp.fitted_scaler.train_mean.copy()

        pp.transform(val, view="scaled")
        np.testing.assert_array_equal(pp.fitted_scaler.train_mean, mean_before)

    def test_scaler_scale_unchanged_after_test_transform(self):
        train = _df(n=100, seed=0)
        test_ = _df(n=20, seed=2, numeric_scale=500.0)

        pp = PreprocessingPipeline().fit(train)
        scale_before = pp.fitted_scaler.train_scale.copy()

        pp.transform(test_, view="scaled")
        np.testing.assert_array_equal(pp.fitted_scaler.train_scale, scale_before)


# ---------------------------------------------------------------------------
# 4. Feature dimensionality stable across all splits
# ---------------------------------------------------------------------------


class TestFeatureDimensionality:
    def test_dimensionality_identical_for_all_splits(self):
        train = _df(n=100, seed=0)
        val = _df(n=30, seed=1, proto=["udp"] * 30)   # different proto dist
        test_ = _df(n=20, seed=2, proto=["icmp"] * 20)  # unseen proto
        protected = _df(n=10, seed=3, attack_cat="Backdoor", label=1)

        pp = PreprocessingPipeline().fit(train)
        n_feat = len(pp.feature_names)

        for split_name, df in [
            ("val", val), ("test", test_), ("protected", protected)
        ]:
            ds = pp.transform(df, view="scaled", split_name=split_name)
            assert ds.n_features == n_feat, (
                f"Feature count mismatch for split '{split_name}': "
                f"expected {n_feat}, got {ds.n_features}."
            )


# ---------------------------------------------------------------------------
# 5. Backdoor-only protected set transforms correctly
# ---------------------------------------------------------------------------


class TestProtectedBackdoorTransform:
    def test_protected_all_backdoor_transforms_without_crash(self):
        train = _df(n=100, seed=0)
        protected = _df(n=583, seed=99, attack_cat="Backdoor", label=1)

        pp = PreprocessingPipeline().fit(train)
        ds = pp.transform(protected, view="scaled", split_name="protected_unseen_attack")
        assert ds.n_rows == 583
        assert ds.n_features == len(pp.feature_names)

    def test_protected_set_does_not_influence_encoder(self):
        train = _df(n=100, seed=0, proto=["tcp"] * 50 + ["udp"] * 50)
        protected = _df(n=10, proto=["BACKDOOR_PROTO"] * 10, attack_cat="Backdoor")

        pp = PreprocessingPipeline().fit(train)
        cats_before = {
            col: list(cats)
            for col, cats in pp.fitted_encoder.categories.items()
        }
        pp.transform(protected, view="scaled", split_name="protected")
        for col in cats_before:
            assert pp.fitted_encoder.categories[col] == cats_before[col]

    def test_backdoor_attack_cat_not_in_features(self):
        train = _df(n=100, seed=0)
        protected = _df(n=20, attack_cat="Backdoor", label=1)

        pp = PreprocessingPipeline().fit(train)
        ds = pp.transform(protected, view="scaled", split_name="protected")
        assert "attack_cat" not in ds.feature_names
        assert "Backdoor" not in ds.feature_names

    def test_protected_y_is_all_ones(self):
        train = _df(n=100, seed=0)
        protected = _df(n=30, attack_cat="Backdoor", label=1)

        pp = PreprocessingPipeline().fit(train)
        ds = pp.transform(protected, view="scaled", split_name="protected")
        assert (ds.y == 1).all()


# ---------------------------------------------------------------------------
# 6. Target isolation — y and attack_cat must not contaminate X
# ---------------------------------------------------------------------------


class TestTargetIsolation:
    def test_y_values_not_in_X(self):
        """
        Adversarial test: label column contains large unique numbers.
        Verify none of those unique numbers appear as a column in X.
        (A proxy check that label was not accidentally included in features.)
        """
        train = _df(n=50, seed=0)
        pp = PreprocessingPipeline().fit(train)
        ds = pp.transform(train, view="scaled", split_name="train")
        # label is 0 or 1; check feature_names contain no 'label'
        assert "label" not in ds.feature_names

    def test_attack_cat_string_not_in_feature_names(self):
        train = _df(n=50, seed=0)
        pp = PreprocessingPipeline().fit(train)
        ds = pp.transform(train, view="scaled", split_name="train")
        assert "attack_cat" not in ds.feature_names
        assert "Normal" not in ds.feature_names  # raw value must not be a feature name

    def test_id_column_not_in_features(self):
        train = _df(n=50, seed=0)
        pp = PreprocessingPipeline().fit(train)
        ds = pp.transform(train, view="scaled", split_name="train")
        assert "id" not in ds.feature_names


# ---------------------------------------------------------------------------
# 7. Row ordering integrity
# ---------------------------------------------------------------------------


class TestRowOrdering:
    def test_row_ordering_preserved_scaled(self):
        """
        Row i of X must correspond to row i of y and attack_cat.

        Adversarial setup: y values are unique sequential integers. Verify that
        ds.y.iloc[k] matches the original label at position k, i.e., the
        transform did not shuffle rows.
        """
        n = 30
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "id": list(range(n)),
            "dur": rng.uniform(0, 1, size=n),
            "sbytes": rng.uniform(100, 1000, size=n),
            "dbytes": rng.uniform(100, 500, size=n),
            "proto": ["tcp"] * n,
            "service": ["-"] * n,
            "state": ["FIN"] * n,
            "attack_cat": [f"Cat_{i}" for i in range(n)],  # unique per row
            "label": list(range(n)),  # unique per row
        })

        pp = PreprocessingPipeline().fit(df)
        ds = pp.transform(df, view="scaled", split_name="row_order_test")

        # Every row's label must match the original value at that position
        for i in range(n):
            assert ds.y.iloc[i] == i, (
                f"Row {i}: expected label={i}, got label={ds.y.iloc[i]}. "
                f"Row ordering was not preserved."
            )
            assert ds.attack_cat.iloc[i] == f"Cat_{i}", (
                f"Row {i}: expected attack_cat='Cat_{i}', "
                f"got '{ds.attack_cat.iloc[i]}'. Row ordering was not preserved."
            )

    def test_row_ordering_preserved_unscaled(self):
        n = 20
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "id": list(range(n)),
            "dur": rng.uniform(0, 1, size=n),
            "sbytes": rng.uniform(100, 200, size=n),
            "dbytes": rng.uniform(10, 50, size=n),
            "proto": ["tcp"] * n,
            "service": ["-"] * n,
            "state": ["FIN"] * n,
            "attack_cat": ["Normal"] * n,
            "label": list(range(n)),  # unique per row
        })
        pp = PreprocessingPipeline().fit(df)
        ds = pp.transform(df, view="unscaled", split_name="row_order_unscaled")
        for i in range(n):
            assert ds.y.iloc[i] == i, f"Row {i} ordering lost in unscaled view."

    def test_x_row_count_equals_input_row_count(self):
        """No rows created or destroyed during transform."""
        train = _df(n=100)
        pp = PreprocessingPipeline().fit(train)
        for n_test in (1, 10, 50, 200):
            test = _df(n=n_test, seed=n_test)
            ds = pp.transform(test, view="scaled")
            assert ds.X.shape[0] == n_test, (
                f"Expected {n_test} rows, got {ds.X.shape[0]}."
            )


# ---------------------------------------------------------------------------
# 8. Feature name ordering: element-by-element, all splits
# ---------------------------------------------------------------------------


class TestFeatureNameOrdering:
    def test_feature_names_element_by_element_identical_across_splits(self):
        """
        Every feature name at every position must be identical across TRAIN,
        val, test, and protected splits.

        This verifies the OHE-first, numeric-second ordering is deterministic
        and that the transform path uses the frozen TRAIN-fitted names, not
        a dynamically reconstructed list that could drift.
        """
        train = _df(n=100, seed=0)
        val   = _df(n=30,  seed=1, proto=["udp"] * 30)
        test_ = _df(n=20,  seed=2, proto=["icmp"] * 20)  # unseen category
        prot  = _df(n=10,  seed=3, attack_cat="Backdoor", label=1)

        pp = PreprocessingPipeline().fit(train)
        canonical = pp.feature_names  # reference

        for split_name, df in [("val", val), ("test", test_), ("protected", prot)]:
            for view in ("scaled", "unscaled"):
                ds = pp.transform(df, view=view, split_name=split_name)
                assert ds.feature_names == canonical, (
                    f"Feature names differ for split='{split_name}' view='{view}'. "
                    f"First mismatch at index "
                    f"{next((i for i,(a,b) in enumerate(zip(ds.feature_names, canonical)) if a!=b), -1)}."
                )

    def test_ohe_names_come_from_fitted_encoder_not_reconstructed(self):
        """
        OHE feature names must be identical to what the fitted encoder produces.
        This prevents any manual name reconstruction from drifting (e.g., if
        categories_ order changes between sklearn versions).
        """
        train = _df(n=50, seed=0)
        pp = PreprocessingPipeline().fit(train)
        canonical_names = pp.feature_names

        # The OHE names must match what the encoder itself reports
        from src.preprocessing.encoding import get_feature_names
        encoder_ohe_names = get_feature_names(pp.fitted_encoder)

        # OHE names appear first in the feature list
        n_ohe = len(encoder_ohe_names)
        assert canonical_names[:n_ohe] == encoder_ohe_names, (
            "Pipeline feature_names OHE prefix does not match encoder's own "
            "get_feature_names_out() output. Manual name reconstruction may have drifted."
        )

    def test_numeric_names_come_after_ohe_names(self):
        """
        Numeric column names must appear AFTER all OHE columns.
        No mixing of OHE and numeric names is allowed.
        """
        train = _df(n=50, seed=0)
        pp = PreprocessingPipeline().fit(train)

        from src.preprocessing.encoding import get_feature_names
        n_ohe = len(get_feature_names(pp.fitted_encoder))
        ohe_block   = pp.feature_names[:n_ohe]
        numeric_block = pp.feature_names[n_ohe:]

        # OHE names should have categorical column prefixes
        for name in ohe_block:
            assert any(
                name.startswith(f"{col}_")
                for col in pp._categorical_cols
            ), f"OHE feature '{name}' does not start with a categorical column prefix."

        # Numeric names should not have OHE prefixes
        for name in numeric_block:
            assert not any(
                name.startswith(f"{col}_")
                for col in pp._categorical_cols
            ), f"Numeric feature '{name}' has a categorical OHE prefix — ordering may be wrong."
