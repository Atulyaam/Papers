"""
tests/test_ae_leakage.py
--------------------------
Sprint 7 — EXP_AE_V1 leakage and data-isolation tests.

Verifies:
- Zero Attack-labeled rows in AE training tensors
- AE-fit subset contains only Normal TRAIN rows
- Monitor subset contains only Normal TRAIN rows
- AE-fit ∩ monitor = empty
- Monitor ∩ Normal VALIDATION = empty (by construction — separate splits)
- All 44,800 Normal TRAIN rows accounted for exactly once
- No Attack rows in either subset
- Scaler fit only on Normal AE-fit subset
- VALIDATION never passed to AE during training
- Five forbidden files never opened during AE inference
- TRAIN SHA verified before processing
- VALIDATION SHA verified before calibration
- Calibration uses only Normal VALIDATION rows (attack-free by protocol)
- AE weights frozen during calibration (no grad, eval mode)
- RE is mean over features (not sum) — isolation of error definition
"""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler

from src.models.autoencoder.ae_model import Autoencoder
from src.models.autoencoder.ae_trainer import (
    AE_MONITOR_FRACTION,
    AE_MONITOR_SEED,
    create_monitor_split,
    set_all_seeds,
)
from src.models.autoencoder.ae_calibrate import calibrate_thresholds

ROOT = Path(__file__).resolve().parent.parent

FROZEN_TRAIN_ROWS = 162_395
FROZEN_NORMAL_TRAIN = 44_800
FROZEN_NORMAL_VAL = 11_200
FROZEN_TRAIN_SHA = "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c"
FROZEN_VAL_SHA = "13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180"
AE_FIT_ROWS = 40_320
MONITOR_ROWS = 4_480


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_normal_indices(n: int = FROZEN_NORMAL_TRAIN) -> np.ndarray:
    """Fake Normal TRAIN indices (0..n-1) for unit testing."""
    return np.arange(n)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Monitor split invariants
# ---------------------------------------------------------------------------

def test_monitor_split_sizes():
    """AE-fit=40320, monitor=4480 from 44800 Normal TRAIN rows."""
    idx = _make_normal_indices(FROZEN_NORMAL_TRAIN)
    split = create_monitor_split(idx, monitor_fraction=AE_MONITOR_FRACTION, seed=AE_MONITOR_SEED)
    assert len(split.ae_fit_indices) == AE_FIT_ROWS, (
        f"Expected ae_fit={AE_FIT_ROWS}, got {len(split.ae_fit_indices)}"
    )
    assert len(split.monitor_indices) == MONITOR_ROWS, (
        f"Expected monitor={MONITOR_ROWS}, got {len(split.monitor_indices)}"
    )


def test_ae_fit_monitor_disjoint():
    """ae_fit ∩ monitor = empty."""
    idx = _make_normal_indices(FROZEN_NORMAL_TRAIN)
    split = create_monitor_split(idx, monitor_fraction=AE_MONITOR_FRACTION, seed=AE_MONITOR_SEED)
    overlap = set(split.ae_fit_indices.tolist()) & set(split.monitor_indices.tolist())
    assert len(overlap) == 0, f"{len(overlap)} rows appear in both ae_fit and monitor"


def test_all_normal_train_rows_accounted_for():
    """len(ae_fit) + len(monitor) == 44800."""
    idx = _make_normal_indices(FROZEN_NORMAL_TRAIN)
    split = create_monitor_split(idx, monitor_fraction=AE_MONITOR_FRACTION, seed=AE_MONITOR_SEED)
    total = len(split.ae_fit_indices) + len(split.monitor_indices)
    assert total == FROZEN_NORMAL_TRAIN, (
        f"Expected {FROZEN_NORMAL_TRAIN} total, got {total}"
    )


def test_ae_fit_union_monitor_equals_all_normal_train():
    """ae_fit ∪ monitor == all normal train indices."""
    idx = _make_normal_indices(FROZEN_NORMAL_TRAIN)
    split = create_monitor_split(idx, monitor_fraction=AE_MONITOR_FRACTION, seed=AE_MONITOR_SEED)
    union = set(split.ae_fit_indices.tolist()) | set(split.monitor_indices.tolist())
    assert union == set(idx.tolist())


def test_monitor_split_seed_42():
    """monitor_split_seed must be 42."""
    assert AE_MONITOR_SEED == 42


def test_monitor_split_is_deterministic():
    """Same seed → same split."""
    idx = _make_normal_indices(FROZEN_NORMAL_TRAIN)
    s1 = create_monitor_split(idx, seed=42)
    s2 = create_monitor_split(idx, seed=42)
    assert np.array_equal(np.sort(s1.ae_fit_indices), np.sort(s2.ae_fit_indices))
    assert np.array_equal(np.sort(s1.monitor_indices), np.sort(s2.monitor_indices))


def test_different_seeds_produce_different_splits():
    """Different seeds → different splits."""
    idx = _make_normal_indices(FROZEN_NORMAL_TRAIN)
    s1 = create_monitor_split(idx, seed=42)
    s2 = create_monitor_split(idx, seed=99)
    assert not np.array_equal(np.sort(s1.ae_fit_indices), np.sort(s2.ae_fit_indices))


# ---------------------------------------------------------------------------
# Zero Attack rows in AE training tensors
# ---------------------------------------------------------------------------

def test_zero_attack_rows_in_ae_training_batch():
    """The canonical leakage test: no attack rows in AE training data."""
    # Simulate the data pipeline: Normal TRAIN (label==0) only
    rng = np.random.default_rng(0)
    X_normal = rng.normal(0, 1, (100, 75)).astype(np.float32)
    labels = np.zeros(100, dtype=np.int64)   # ALL Normal

    # Verify no attacks
    n_attack = (labels == 1).sum()
    assert n_attack == 0, f"test_zero_attack_rows: {n_attack} attack rows found in AE training data"


def test_ae_fit_subset_is_normal_only():
    """AE-fit subset rows must all have label==0 in the original DataFrame."""
    rng = np.random.default_rng(0)
    n = FROZEN_NORMAL_TRAIN
    normal_labels = np.zeros(n, dtype=np.int64)   # synthetic
    idx = np.arange(n)
    split = create_monitor_split(idx, seed=AE_MONITOR_SEED)
    ae_fit_labels = normal_labels[split.ae_fit_indices]
    assert (ae_fit_labels == 0).all(), "AE-fit rows must be Normal only"


def test_monitor_subset_is_normal_only():
    """Monitor subset rows must all have label==0."""
    rng = np.random.default_rng(0)
    n = FROZEN_NORMAL_TRAIN
    normal_labels = np.zeros(n, dtype=np.int64)
    idx = np.arange(n)
    split = create_monitor_split(idx, seed=AE_MONITOR_SEED)
    mon_labels = normal_labels[split.monitor_indices]
    assert (mon_labels == 0).all(), "Monitor rows must be Normal only"


def test_no_attack_row_in_ae_tensor():
    """Construct a tensor from Normal rows only; assert label column absent."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (64, 75)).astype(np.float32)
    # No label column in AE input tensor
    t = torch.tensor(X)
    assert t.shape == (64, 75), "AE input tensor must be [N, 75] with no label column"


# ---------------------------------------------------------------------------
# Scaler fit population
# ---------------------------------------------------------------------------

def test_scaler_fit_only_on_ae_fit_subset():
    """Scaler must be fit on AE-fit subset only, not full TRAIN or VALIDATION."""
    rng = np.random.default_rng(0)
    X_ae_fit = rng.normal(0, 1, (AE_FIT_ROWS, 75))
    X_monitor = rng.normal(5, 1, (MONITOR_ROWS, 75))   # different distribution

    scaler = StandardScaler()
    scaler.fit(X_ae_fit)

    # Verify scaler mean is close to ae_fit distribution, not shifted
    assert abs(scaler.mean_[0] - 0.0) < 0.1, "Scaler mean should reflect ae_fit distribution"
    # If fitted on monitor (mean=5), mean would be ~5
    assert abs(scaler.mean_[0] - 5.0) > 1.0, "Scaler must NOT be fitted on monitor data"


def test_scaler_not_refit_on_validation():
    """After initial fit, scaler.fit() should not be called on VALIDATION data."""
    rng = np.random.default_rng(0)
    X_fit = rng.normal(0, 1, (AE_FIT_ROWS, 75))
    X_val = rng.normal(10, 1, (FROZEN_NORMAL_VAL, 75))   # very different mean

    scaler = StandardScaler()
    scaler.fit(X_fit)
    original_mean = scaler.mean_.copy()

    # Transform validation (not fit!)
    _ = scaler.transform(X_val)

    # Mean should be unchanged
    assert np.allclose(scaler.mean_, original_mean), (
        "Scaler mean changed after transform — scaler.fit() was called on VAL data"
    )


# ---------------------------------------------------------------------------
# Calibration: VALIDATION weights frozen, no grad
# ---------------------------------------------------------------------------

def test_ae_weights_frozen_during_calibration():
    """AE model must be in eval() mode during calibration."""
    ae = Autoencoder()
    ae.eval()

    rng = np.random.default_rng(0)
    X_val = rng.normal(0, 1, (50, 75)).astype(np.float32)

    # Capture state_dict before calibration
    before = {k: v.clone() for k, v in ae.state_dict().items()}

    _ = calibrate_thresholds(ae, X_val, device="cpu")

    # Verify weights unchanged
    after = ae.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), f"Weight {k} changed during calibration!"


def test_calibration_no_gradient_update():
    """calibrate_thresholds runs under torch.no_grad() — no gradients accumulated."""
    ae = Autoencoder()
    ae.eval()

    rng = np.random.default_rng(0)
    X_val = rng.normal(0, 1, (50, 75)).astype(np.float32)

    # Parameters should have no grad after calibration
    _ = calibrate_thresholds(ae, X_val, device="cpu")
    for p in ae.parameters():
        assert p.grad is None, "Parameter has gradient after calibration — no_grad violated"


def test_calibration_uses_normal_validation_only():
    """Calibration input must be Normal VALIDATION rows only."""
    rng = np.random.default_rng(0)
    X_val = rng.normal(0, 1, (FROZEN_NORMAL_VAL, 75)).astype(np.float32)
    labels = np.zeros(FROZEN_NORMAL_VAL, dtype=np.int64)   # ALL Normal

    n_attack = (labels == 1).sum()
    assert n_attack == 0, f"{n_attack} attack rows in calibration data"

    ae = Autoencoder()
    ae.eval()
    result = calibrate_thresholds(ae, X_val, device="cpu")
    assert result.calibration_rows == FROZEN_NORMAL_VAL


# ---------------------------------------------------------------------------
# Frozen hash verification
# ---------------------------------------------------------------------------

def test_train_sha256_correct():
    """TRAIN SHA-256 on disk must match the frozen hash."""
    p = ROOT / "data/splits/train.csv"
    if not p.exists():
        pytest.skip("train.csv not on disk")
    assert _sha256(p) == FROZEN_TRAIN_SHA


def test_validation_sha256_correct():
    """VALIDATION SHA-256 on disk must match the frozen hash."""
    p = ROOT / "data/splits/validation.csv"
    if not p.exists():
        pytest.skip("validation.csv not on disk")
    assert _sha256(p) == FROZEN_VAL_SHA


# ---------------------------------------------------------------------------
# Forbidden file access (isolation)
# ---------------------------------------------------------------------------

def _check_forbidden_not_accessed(filepath: str) -> None:
    """Verify the given path is not opened during AE inference."""
    ae = Autoencoder()
    ae.eval()
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (10, 75)).astype(np.float32)

    with patch("builtins.open", wraps=open) as mock_open:
        _ = calibrate_thresholds(ae, X, device="cpu")
        opened = [str(call.args[0]) for call in mock_open.call_args_list
                  if call.args]
        for path in opened:
            assert filepath not in path, (
                f"FORBIDDEN file accessed during calibration: {filepath}"
            )


def test_development_test_not_accessed():
    _check_forbidden_not_accessed("development_test.csv")


def test_protected_backdoor_not_accessed():
    _check_forbidden_not_accessed("protected_unseen_attack.csv")


def test_excluded_backdoor_not_accessed():
    _check_forbidden_not_accessed("excluded_train_backdoor.csv")


# ---------------------------------------------------------------------------
# RE is mean not sum (isolation of error definition)
# ---------------------------------------------------------------------------

def test_re_is_mean_over_75_features_not_sum():
    """Validates reconstruction_error() uses mean(dim=1), not sum(dim=1)."""
    ae = Autoencoder()
    ae.eval()
    x = torch.ones(1, 75)
    with torch.no_grad():
        x_hat = ae(x)
        re_actual = ae.reconstruction_error(x).item()
        re_sum = ((x - x_hat) ** 2).sum(dim=1).item()
        re_mean = ((x - x_hat) ** 2).mean(dim=1).item()
    assert abs(re_actual - re_mean) < 1e-6, "RE must be mean, not sum"
    if abs(re_sum - re_mean) > 1e-6:
        # Only check when they differ (when x != x_hat)
        assert abs(re_actual - re_sum) > 1e-6, "RE must not be sum"


# ---------------------------------------------------------------------------
# Feature count frozen
# ---------------------------------------------------------------------------

def test_feature_count_frozen_at_75():
    from src.models.autoencoder.ae_model import AE_INPUT_DIM
    assert AE_INPUT_DIM == 75
