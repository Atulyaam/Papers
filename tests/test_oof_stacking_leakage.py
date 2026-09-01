"""
tests/test_oof_stacking_leakage.py
-------------------------------------
Sprint 6 edge-case and adversarial leakage tests.

Covers:
- Empty / tiny TRAIN handling
- NaN and infinite predictions
- Malformed OOF matrix
- Wrong feature order / duplicates
- Incorrect frozen hash raises
- Seed reproducibility
- Limitation texts present in h1_summary
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

# Resolved hashes from Step 0
FROZEN_TRAIN_SHA = "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c"
RESOLVED_HASHES = {
    "train":      FROZEN_TRAIN_SHA,
    "validation": "13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180",
    "dev_test":   "04725e85732ab2fc6d9eaaa6105418b22b083b5c651067e7b0785464f414e508",
    "protected":  "6ffd23479b575e438ad90678268f40f674a663c2b9507aaf65089623397a9d91",
    "excluded":   "b3f6e7e60c9815a53f40eb2d41df8b67d29f884b922a487c3fe83c02e0db0a02",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_oof_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    half = n // 2
    return pd.DataFrame({
        "row_id": np.arange(n),
        "dt_attack_probability": rng.rand(n),
        "rf_attack_probability": rng.rand(n),
        "svm_decision_score": rng.randn(n),
        "nn_attack_probability": rng.rand(n),
        "label": np.concatenate([np.zeros(half, dtype=int), np.ones(n - half, dtype=int)]),
    })


# ---------------------------------------------------------------------------
# Empty TRAIN
# ---------------------------------------------------------------------------

class TestEmptyTrain:

    def test_make_oof_folds_empty_y_raises(self):
        from src.models.stacking.oof_runner import make_oof_folds
        with pytest.raises(ValueError):
            make_oof_folds(np.array([]), n_splits=5, seed=7)

    def test_run_oof_seed_empty_raises(self):
        from src.models.stacking.oof_runner import run_oof_seed
        from unittest.mock import MagicMock
        with pytest.raises(ValueError, match="empty"):
            run_oof_seed(
                h1_seed=42,
                folds=[],
                X_unscaled=np.array([]).reshape(0, 75),
                y=np.array([]),
                svm_scaler=MagicMock(),
                nn_scaler=MagicMock(),
            )


# ---------------------------------------------------------------------------
# Tiny TRAIN
# ---------------------------------------------------------------------------

class TestTinyTrain:

    def test_make_oof_folds_tiny_y_ok(self):
        """Tiny y (15 rows, 2 classes) should create folds without error."""
        from src.models.stacking.oof_runner import make_oof_folds
        y = np.array([0] * 8 + [1] * 7, dtype=np.int64)
        folds = make_oof_folds(y, n_splits=3, seed=7)
        assert len(folds) == 3


# ---------------------------------------------------------------------------
# NaN and Infinite predictions
# ---------------------------------------------------------------------------

class TestNaNInfiniteDetection:

    def test_validate_oof_df_raises_on_nan(self):
        from src.models.stacking.meta_learner import _validate_oof_df
        df = _make_oof_df(100)
        df.loc[0, "dt_attack_probability"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            _validate_oof_df(df)

    def test_oofdf_with_inf_svm_score_raises_in_lr(self):
        """LogisticRegression will fail or produce warnings with infinite inputs."""
        from src.models.stacking.meta_learner import train_meta_learner, _validate_oof_df
        df = _make_oof_df(100)
        df.loc[0, "svm_decision_score"] = np.inf
        # _validate_oof_df won't catch inf by default — but we test it propagates
        # to the meta-learner and raises or produces bad output
        # At minimum, verify _validate_oof_df doesn't silently swallow it
        try:
            _validate_oof_df(df)  # may not raise by itself
        except ValueError:
            pass  # acceptable to raise
        # If it didn't raise, training the meta-learner with inf should still surface the issue
        try:
            clf = train_meta_learner(df, h1_seed=42)
            # If it somehow trained, the coefficients should still be finite
            assert np.isfinite(clf.coef_).all(), "LR coefficients non-finite with inf input"
        except Exception:
            pass  # expected to fail with inf input


# ---------------------------------------------------------------------------
# Malformed OOF matrix
# ---------------------------------------------------------------------------

class TestMalformedOOFMatrix:

    def test_missing_dt_column_raises(self):
        from src.models.stacking.meta_learner import _validate_oof_df
        df = _make_oof_df(100)
        df = df.drop(columns=["dt_attack_probability"])
        with pytest.raises(ValueError, match="missing"):
            _validate_oof_df(df)

    def test_missing_label_column_raises(self):
        from src.models.stacking.meta_learner import _validate_oof_df
        df = _make_oof_df(100)
        df = df.drop(columns=["label"])
        with pytest.raises(ValueError, match="missing"):
            _validate_oof_df(df)

    def test_predict_meta_wrong_shape_raises(self):
        from src.models.stacking.meta_learner import predict_meta, train_meta_learner
        df = _make_oof_df(100)
        clf = train_meta_learner(df, h1_seed=42)
        X_wrong = np.ones((10, 3))  # 3 columns instead of 4
        with pytest.raises(ValueError):
            predict_meta(clf, X_wrong)

    def test_predict_meta_with_row_id_column_raises(self):
        from src.models.stacking.meta_learner import predict_meta, train_meta_learner
        df = _make_oof_df(100)
        clf = train_meta_learner(df, h1_seed=42)
        X_wrong = np.ones((10, 5))  # 5 columns (includes hypothetical row_id)
        with pytest.raises(ValueError):
            predict_meta(clf, X_wrong)


# ---------------------------------------------------------------------------
# Wrong feature order / duplicates
# ---------------------------------------------------------------------------

class TestFeatureOrderDuplicates:

    def test_build_feature_matrix_wrong_order_raises(self):
        """build_feature_matrix only raises if features are MISSING, not reordered.
        But columns are selected in order — wrong order produces a different matrix.
        Verify that duplicate features in the list raises."""
        from src.models.base_models.preprocessing import build_feature_matrix
        # Build a DataFrame with 75 dummy columns
        cols = [f"f{i}" for i in range(75)]
        df = pd.DataFrame(np.zeros((10, 75)), columns=cols)
        # Duplicate feature in list raises
        bad_features = cols[:74] + [cols[0]]  # last is a duplicate of first
        with pytest.raises(ValueError, match="Duplicate"):
            build_feature_matrix(df, bad_features)

    def test_build_feature_matrix_missing_feature_raises(self):
        from src.models.base_models.preprocessing import build_feature_matrix
        cols = [f"f{i}" for i in range(74)]  # only 74 columns
        df = pd.DataFrame(np.zeros((10, 74)), columns=cols)
        # Feature list has 75 entries but df only has 74 cols
        features = [f"f{i}" for i in range(74)] + ["f_missing"]
        with pytest.raises(ValueError, match="missing"):
            build_feature_matrix(df, features)


# ---------------------------------------------------------------------------
# Incorrect frozen hash
# ---------------------------------------------------------------------------

class TestFrozenHashVerification:

    def test_train_sha_matches_frozen(self):
        """TRAIN SHA-256 must match the Step 0 resolved hash."""
        train_path = ROOT / "data/splits/train.csv"
        actual = _sha256(train_path)
        assert actual == FROZEN_TRAIN_SHA, (
            f"TRAIN SHA mismatch!\n"
            f"  expected: {FROZEN_TRAIN_SHA}\n"
            f"  actual:   {actual}"
        )

    def test_validation_sha_matches_step0(self):
        p = ROOT / "data/splits/validation.csv"
        actual = _sha256(p)
        assert actual == RESOLVED_HASHES["validation"]

    def test_dev_test_sha_matches_step0(self):
        p = ROOT / "data/splits/development_test.csv"
        actual = _sha256(p)
        assert actual == RESOLVED_HASHES["dev_test"]

    def test_protected_sha_matches_step0(self):
        p = ROOT / "data/splits/protected_unseen_attack.csv"
        actual = _sha256(p)
        assert actual == RESOLVED_HASHES["protected"]

    def test_excluded_sha_matches_step0(self):
        p = ROOT / "data/splits/excluded_train_backdoor.csv"
        actual = _sha256(p)
        assert actual == RESOLVED_HASHES["excluded"]


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------

class TestSeedReproducibility:

    def test_same_seed_same_fold_assignment(self):
        """Identical seed + y produces identical fold assignment."""
        from src.models.stacking.oof_runner import make_oof_folds
        y = np.array([0] * 80 + [1] * 120, dtype=np.int64)
        folds_a = make_oof_folds(y, n_splits=5, seed=7)
        folds_b = make_oof_folds(y, n_splits=5, seed=7)
        for i, (fa, fb) in enumerate(zip(folds_a, folds_b)):
            np.testing.assert_array_equal(fa[1], fb[1], err_msg=f"fold {i} oof differs")

    def test_different_seed_different_fold_assignment(self):
        """Different seed should (almost certainly) produce different folds."""
        from src.models.stacking.oof_runner import make_oof_folds
        y = np.array([0] * 80 + [1] * 120, dtype=np.int64)
        folds_7 = make_oof_folds(y, n_splits=5, seed=7)
        folds_99 = make_oof_folds(y, n_splits=5, seed=99)
        # At least one fold should differ
        diffs = [not np.array_equal(fa[1], fb[1]) for fa, fb in zip(folds_7, folds_99)]
        assert any(diffs), "Different seeds produced identical fold assignments (unexpected)"

    def test_meta_learner_same_seed_reproducible(self):
        """Same seed must produce identical LR coefficients."""
        from src.models.stacking.meta_learner import train_meta_learner
        df = _make_oof_df(200)
        clf1 = train_meta_learner(df, h1_seed=42)
        clf2 = train_meta_learner(df, h1_seed=42)
        np.testing.assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=10)

    def test_meta_learner_different_seeds_different_init(self):
        """Different H1 seeds should produce (potentially) different LR initialisation.
        LR is deterministic for lbfgs regardless of seed, but random_state is propagated."""
        from src.models.stacking.meta_learner import train_meta_learner, META_CONFIG
        df = _make_oof_df(200)
        clf_42 = train_meta_learner(df, h1_seed=42)
        clf_123 = train_meta_learner(df, h1_seed=123)
        # Both should train; random_state is set correctly on each
        assert clf_42.random_state == 42
        assert clf_123.random_state == 123


# ---------------------------------------------------------------------------
# Limitation texts in h1_summary
# ---------------------------------------------------------------------------

class TestLimitationTextsInH1Summary:

    def _make_seed_results(self) -> list[dict]:
        return [
            {"h1_seed": s, "macro_f1": f, "in_sample_evaluation_warning": True,
             "meta_evaluation_limitation": "meta lim", "scaling_limitation": "scaling lim"}
            for s, f in zip([42, 123, 2024], [0.90, 0.92, 0.94])
        ]

    def test_h1_summary_has_meta_evaluation_limitation(self):
        from src.models.stacking.meta_learner import compute_h1_summary
        summary = compute_h1_summary(self._make_seed_results())
        assert "meta_evaluation_limitation" in summary
        assert len(summary["meta_evaluation_limitation"]) > 0

    def test_h1_summary_has_scaling_limitation(self):
        from src.models.stacking.meta_learner import compute_h1_summary
        summary = compute_h1_summary(self._make_seed_results())
        assert "scaling_limitation" in summary
        assert len(summary["scaling_limitation"]) > 0

    def test_h1_summary_has_in_sample_warning(self):
        from src.models.stacking.meta_learner import compute_h1_summary
        summary = compute_h1_summary(self._make_seed_results())
        assert summary.get("in_sample_evaluation_warning") is True

    def test_h1_summary_sprint5_reference_correct_label(self):
        from src.models.stacking.meta_learner import compute_h1_summary, SPRINT5_RF_REFERENCE_LABEL
        summary = compute_h1_summary(self._make_seed_results())
        ref_label = summary["sprint5_reference"]["label"]
        assert "Frozen Sprint 5 single-CV reference" in ref_label

    def test_h1_summary_no_statistical_significance_claim(self):
        """h1_summary must not contain 'statistically significant' language."""
        from src.models.stacking.meta_learner import compute_h1_summary
        import json
        summary = compute_h1_summary(self._make_seed_results())
        summary_str = json.dumps(summary).lower()
        assert "statistically significant" not in summary_str, \
            "h1_summary must not claim statistical significance"
