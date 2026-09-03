"""
src/models/autoencoder/ae_calibrate.py
----------------------------------------
Sprint 7 — EXP_AE_V1 Threshold Calibration.

Calibration protocol (FINAL):
    1. Frozen AE weights (model.eval(), torch.no_grad())
    2. Pass Normal VALIDATION (11,200 rows) through AE + frozen scaler
    3. Compute RE(x) = mean((x - x_hat)^2) for each row
    4. Compute 5 fixed threshold candidates from RE distribution
    5. Save raw RE values and threshold sweep

Threshold candidates (FINAL — fixed):
    p95       95th percentile of Normal VAL RE
    p99       99th percentile of Normal VAL RE
    p999      99.9th percentile of Normal VAL RE
    mean2sigma  mean(RE) + 2*std(RE)
    mean3sigma  mean(RE) + 3*std(RE)

Primary threshold: DEFERRED TO SPRINT 8.

Isolation invariant:
    - No AE weight updates during calibration
    - No gradient computation
    - No Attack data used
    - VALIDATION SHA verified before calibration

Mandatory limitation (must be preserved in all downstream uses):
    AE operates in a Normal-TRAIN-scaled feature space distinct from the
    full-TRAIN-scaled space used by DT/RF/SVM/NN. Sprint 8 fusion must
    account for this representation difference when combining the AE
    reconstruction error with the supervised branch outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.autoencoder.ae_model import Autoencoder

logger = logging.getLogger(__name__)

AE_BATCH_SIZE: int = 256

# ---------------------------------------------------------------------------
# Threshold candidate definitions
# ---------------------------------------------------------------------------

THRESHOLD_RULES: list[str] = ["p95", "p99", "p999", "mean2sigma", "mean3sigma"]

PRIMARY_THRESHOLD_STATUS: str = "DEFERRED_TO_SPRINT_8"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ThresholdCandidate:
    rule: str
    percentile: float | None          # e.g. 95.0 for p95; None for sigma rules
    threshold_value: float
    samples_above_threshold: int
    fraction_above_threshold: float

    def to_dict(self) -> dict:
        return {
            "threshold_rule": self.rule,
            "percentile": self.percentile,
            "threshold_value": self.threshold_value,
            "samples_above_threshold": self.samples_above_threshold,
            "fraction_above_threshold": self.fraction_above_threshold,
        }


@dataclass
class CalibrationResult:
    reconstruction_errors: np.ndarray   # shape [N_val] — one per VAL row
    re_stats: dict                       # mean, std, min, max, p50, p95, ...
    thresholds: dict[str, ThresholdCandidate]
    calibration_rows: int
    primary_threshold: str = PRIMARY_THRESHOLD_STATUS

    def threshold_sweep_rows(self) -> list[dict]:
        return [t.to_dict() for t in self.thresholds.values()]

    def calibration_dict(self) -> dict:
        return {
            "re_stats": self.re_stats,
            "thresholds": {k: v.to_dict() for k, v in self.thresholds.items()},
            "primary_threshold": self.primary_threshold,
            "calibration_rows": self.calibration_rows,
            "calibration_split": "Normal VALIDATION only",
        }


# ---------------------------------------------------------------------------
# Core calibration function
# ---------------------------------------------------------------------------

def calibrate_thresholds(
    model: Autoencoder,
    X_val_scaled: np.ndarray,
    device: str = "cpu",
) -> CalibrationResult:
    """
    Compute per-sample reconstruction errors and all 5 threshold candidates.

    Parameters
    ----------
    model : Autoencoder
        Must be in eval() mode with frozen weights (no gradient).
    X_val_scaled : np.ndarray  shape [11200, 75]
        Normal VALIDATION rows, transformed by the frozen AE scaler.
        MUST contain only label==0 rows (verified by leakage tests).
    device : str
        Device to run inference on.

    Returns
    -------
    CalibrationResult
    """
    assert X_val_scaled.shape[1] == 75, (
        f"Expected 75 features, got {X_val_scaled.shape[1]}"
    )
    assert X_val_scaled.shape[0] > 0, "Empty validation set"

    model = model.to(device)
    model.eval()

    # --- Compute reconstruction errors ---
    t = torch.tensor(X_val_scaled, dtype=torch.float32)
    ds = TensorDataset(t)
    loader = DataLoader(ds, batch_size=AE_BATCH_SIZE, shuffle=False)

    re_list: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            re_batch = model.reconstruction_error(batch)  # shape [B]
            re_list.append(re_batch.cpu().numpy())

    re = np.concatenate(re_list, axis=0)   # shape [N_val]
    n = len(re)

    logger.info(
        "Calibration | n=%d | mean_RE=%.6f | std_RE=%.6f | max_RE=%.6f",
        n, re.mean(), re.std(), re.max(),
    )

    # --- RE distribution statistics ---
    re_stats = {
        "mean":  float(re.mean()),
        "std":   float(re.std(ddof=0)),
        "min":   float(re.min()),
        "max":   float(re.max()),
        "p50":   float(np.percentile(re, 50)),
        "p95":   float(np.percentile(re, 95)),
        "p99":   float(np.percentile(re, 99)),
        "p999":  float(np.percentile(re, 99.9)),
    }

    # --- Compute 5 threshold candidates ---
    candidates: dict[str, ThresholdCandidate] = {}

    def _make(rule: str, pct: float | None, val: float) -> ThresholdCandidate:
        n_above = int((re > val).sum())
        return ThresholdCandidate(
            rule=rule,
            percentile=pct,
            threshold_value=val,
            samples_above_threshold=n_above,
            fraction_above_threshold=n_above / n,
        )

    candidates["p95"]        = _make("p95",        95.0,  re_stats["p95"])
    candidates["p99"]        = _make("p99",        99.0,  re_stats["p99"])
    candidates["p999"]       = _make("p999",       99.9,  re_stats["p999"])
    candidates["mean2sigma"] = _make("mean2sigma", None,
                                     re_stats["mean"] + 2.0 * re_stats["std"])
    candidates["mean3sigma"] = _make("mean3sigma", None,
                                     re_stats["mean"] + 3.0 * re_stats["std"])

    for name, cand in candidates.items():
        logger.info(
            "  Threshold %-12s = %.6f | n_above=%d | frac=%.4f",
            name, cand.threshold_value,
            cand.samples_above_threshold, cand.fraction_above_threshold,
        )

    return CalibrationResult(
        reconstruction_errors=re,
        re_stats=re_stats,
        thresholds=candidates,
        calibration_rows=n,
        primary_threshold=PRIMARY_THRESHOLD_STATUS,
    )
