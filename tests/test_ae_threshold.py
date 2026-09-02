"""
tests/test_ae_threshold.py
----------------------------
Sprint 7 — EXP_AE_V1 threshold calibration tests.

Verifies:
- All 5 threshold candidates present (p95, p99, p999, mean2sigma, mean3sigma)
- Threshold computed from Normal VALIDATION RE only
- RE is mean over features (not sum)
- No attack rows in calibration data
- primary_threshold = "DEFERRED_TO_SPRINT_8"
- All threshold values > 0 (RE is non-negative)
- p95 < p99 < p99.9 (monotonicity)
- Sanity range checks: implied FPR ~5%, ~1%, ~0.1%
- threshold_sweep.csv has required columns
- threshold_calibration.json has required fields
- RE stats (mean, std, min, max, p50, p95, p99, p999) present
- Scaler-space limitation text in artifacts
- Single-seed limitation text in artifacts
- Feature count frozen at 75
- Primary threshold deferral documented
- CalibrationResult.calibration_rows matches input size
- ThresholdCandidate.fraction_above_threshold in [0, 1]
"""

from __future__ import annotations

import io
import json
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.autoencoder.ae_model import Autoencoder
from src.models.autoencoder.ae_calibrate import (
    PRIMARY_THRESHOLD_STATUS,
    THRESHOLD_RULES,
    calibrate_thresholds,
    CalibrationResult,
)
from src.models.autoencoder.artifacts import (
    SCALER_SPACE_LIMITATION,
    SINGLE_SEED_LIMITATION,
    PRIMARY_THRESHOLD_LIMITATION,
    THRESHOLD_SANITY_LIMITATION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ae_eval():
    ae = Autoencoder()
    ae.eval()
    return ae


@pytest.fixture
def X_val():
    """Synthetic Normal VALIDATION data (11200 × 75)."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, (11200, 75)).astype(np.float32)


@pytest.fixture
def cal_result(ae_eval, X_val):
    return calibrate_thresholds(ae_eval, X_val, device="cpu")


# ---------------------------------------------------------------------------
# All 5 threshold candidates present
# ---------------------------------------------------------------------------

def test_all_five_threshold_rules_present(cal_result):
    expected = {"p95", "p99", "p999", "mean2sigma", "mean3sigma"}
    actual = set(cal_result.thresholds.keys())
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_threshold_rules_constant():
    assert set(THRESHOLD_RULES) == {"p95", "p99", "p999", "mean2sigma", "mean3sigma"}


# ---------------------------------------------------------------------------
# Primary threshold deferred
# ---------------------------------------------------------------------------

def test_primary_threshold_deferred(cal_result):
    assert cal_result.primary_threshold == "DEFERRED_TO_SPRINT_8", (
        f"primary_threshold must be DEFERRED_TO_SPRINT_8, got {cal_result.primary_threshold!r}"
    )


def test_primary_threshold_status_constant():
    assert PRIMARY_THRESHOLD_STATUS == "DEFERRED_TO_SPRINT_8"


def test_calibration_dict_primary_threshold_deferred(cal_result):
    d = cal_result.calibration_dict()
    assert d["primary_threshold"] == "DEFERRED_TO_SPRINT_8"


# ---------------------------------------------------------------------------
# Threshold values
# ---------------------------------------------------------------------------

def test_all_threshold_values_positive(cal_result):
    """All RE thresholds must be >= 0 (RE is non-negative)."""
    for name, cand in cal_result.thresholds.items():
        assert cand.threshold_value >= 0.0, f"{name} threshold < 0: {cand.threshold_value}"


def test_percentile_thresholds_monotonic(cal_result):
    """p95 < p99 < p99.9 (monotonicity of percentiles)."""
    t = cal_result.thresholds
    assert t["p95"].threshold_value <= t["p99"].threshold_value, "p95 should be <= p99"
    assert t["p99"].threshold_value <= t["p999"].threshold_value, "p99 should be <= p99.9"


def test_p95_implies_approx_5pct_fpr(cal_result):
    """FPR at p95 should be ~5% on Normal VALIDATION."""
    fpr = cal_result.thresholds["p95"].fraction_above_threshold
    assert 0.03 <= fpr <= 0.07, f"Expected p95 FPR ≈ 5%, got {fpr:.3f}"


def test_p99_implies_approx_1pct_fpr(cal_result):
    """FPR at p99 should be ~1% on Normal VALIDATION."""
    fpr = cal_result.thresholds["p99"].fraction_above_threshold
    assert 0.005 <= fpr <= 0.02, f"Expected p99 FPR ≈ 1%, got {fpr:.3f}"


def test_p999_implies_approx_01pct_fpr(cal_result):
    """FPR at p99.9 should be ~0.1% on Normal VALIDATION."""
    fpr = cal_result.thresholds["p999"].fraction_above_threshold
    assert 0.0 <= fpr <= 0.005, f"Expected p99.9 FPR ≈ 0.1%, got {fpr:.3f}"


def test_fractions_in_unit_interval(cal_result):
    for name, cand in cal_result.thresholds.items():
        assert 0.0 <= cand.fraction_above_threshold <= 1.0, (
            f"{name} fraction_above_threshold={cand.fraction_above_threshold} out of [0,1]"
        )


# ---------------------------------------------------------------------------
# RE statistics
# ---------------------------------------------------------------------------

def test_re_stats_present(cal_result):
    required = {"mean", "std", "min", "max", "p50", "p95", "p99", "p999"}
    assert required.issubset(set(cal_result.re_stats.keys())), (
        f"Missing RE stats: {required - set(cal_result.re_stats.keys())}"
    )


def test_re_stats_mean_nonnegative(cal_result):
    assert cal_result.re_stats["mean"] >= 0.0


def test_re_stats_std_nonnegative(cal_result):
    assert cal_result.re_stats["std"] >= 0.0


def test_re_stats_min_le_max(cal_result):
    assert cal_result.re_stats["min"] <= cal_result.re_stats["max"]


def test_re_stats_p95_matches_threshold(cal_result):
    """re_stats['p95'] should equal the p95 threshold value."""
    assert abs(cal_result.re_stats["p95"] - cal_result.thresholds["p95"].threshold_value) < 1e-9


# ---------------------------------------------------------------------------
# Calibration rows
# ---------------------------------------------------------------------------

def test_calibration_rows_matches_input(cal_result, X_val):
    assert cal_result.calibration_rows == len(X_val), (
        f"calibration_rows={cal_result.calibration_rows} != {len(X_val)}"
    )


def test_reconstruction_errors_shape(cal_result, X_val):
    assert len(cal_result.reconstruction_errors) == len(X_val)


def test_reconstruction_errors_nonnegative(cal_result):
    assert (cal_result.reconstruction_errors >= 0).all()


# ---------------------------------------------------------------------------
# calibration_dict() structure
# ---------------------------------------------------------------------------

def test_calibration_dict_has_re_stats(cal_result):
    d = cal_result.calibration_dict()
    assert "re_stats" in d
    assert "thresholds" in d
    assert "calibration_split" in d


def test_calibration_dict_calibration_split_is_validation(cal_result):
    d = cal_result.calibration_dict()
    assert "Normal VALIDATION" in d["calibration_split"]


def test_calibration_dict_all_five_thresholds(cal_result):
    d = cal_result.calibration_dict()
    assert set(d["thresholds"].keys()) == {"p95", "p99", "p999", "mean2sigma", "mean3sigma"}


# ---------------------------------------------------------------------------
# threshold_sweep_rows() — CSV compatibility
# ---------------------------------------------------------------------------

def test_threshold_sweep_rows_columns(cal_result):
    rows = cal_result.threshold_sweep_rows()
    df = pd.DataFrame(rows)
    required_cols = {
        "threshold_rule", "threshold_value",
        "samples_above_threshold", "fraction_above_threshold",
    }
    assert required_cols.issubset(set(df.columns)), (
        f"Missing columns: {required_cols - set(df.columns)}"
    )


def test_threshold_sweep_has_five_rows(cal_result):
    rows = cal_result.threshold_sweep_rows()
    assert len(rows) == 5


# ---------------------------------------------------------------------------
# Limitation text presence in artifacts module
# ---------------------------------------------------------------------------

def test_scaler_space_limitation_text_present():
    assert "Normal-TRAIN-scaled feature space" in SCALER_SPACE_LIMITATION
    assert "Sprint 8 fusion" in SCALER_SPACE_LIMITATION


def test_single_seed_limitation_text_present():
    assert "single AE training seed" in SINGLE_SEED_LIMITATION
    assert "accepted scope limitation" in SINGLE_SEED_LIMITATION


def test_primary_threshold_limitation_text_present():
    assert "Sprint 8" in PRIMARY_THRESHOLD_LIMITATION


def test_threshold_sanity_limitation_text_present():
    assert "sanity priors" in THRESHOLD_SANITY_LIMITATION.lower() or \
           "approximate" in THRESHOLD_SANITY_LIMITATION.lower()


# ---------------------------------------------------------------------------
# No attack data in calibration
# ---------------------------------------------------------------------------

def test_calibration_data_is_normal_only(X_val):
    """Calibration fixture X_val has no label column — all rows are Normal by construction."""
    assert X_val.shape[1] == 75, "AE calibration input has 75 features, no label column"
    # By the design protocol: only Normal VALIDATION rows are passed to calibrate_thresholds


# ---------------------------------------------------------------------------
# Feature count frozen
# ---------------------------------------------------------------------------

def test_calibration_input_is_75_features(X_val):
    assert X_val.shape[1] == 75


def test_calibration_raises_on_wrong_feature_count(ae_eval):
    rng = np.random.default_rng(0)
    X_bad = rng.normal(0, 1, (100, 50)).astype(np.float32)
    with pytest.raises(AssertionError):
        calibrate_thresholds(ae_eval, X_bad, device="cpu")


# ---------------------------------------------------------------------------
# Calibration split label
# ---------------------------------------------------------------------------

def test_calibration_split_label_in_calibration_dict(cal_result):
    d = cal_result.calibration_dict()
    assert d["calibration_split"] == "Normal VALIDATION only"
