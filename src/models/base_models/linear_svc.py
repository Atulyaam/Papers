"""
src/models/base_models/linear_svc.py
--------------------------------------
LinearSVC base model for Sprint 5.

Backend: sklearn.svm.LinearSVC

CRITICAL DESIGN DECISIONS
--------------------------
1. LinearSVC does NOT produce calibrated probabilities.
   Output contract:  predict()  +  decision_function()
   CalibratedClassifierCV is NOT introduced in Sprint 5.
   SVM calibration will be designed separately in the OOF stacking sprint.

2. Scaler isolation:
   A fresh StandardScaler is fit on inner_train ONLY inside each fold.
   The outer val/test data are NEVER seen by the scaler's fit() call.

3. The final refit returns BOTH the fitted SVM and the fitted scaler,
   which must be persisted together as the checkpoint pair.

Phases
------
Phase A — Baseline CV  : run_svm_baseline(X_unscaled, y)
Phase B — Tuning grid  : run_svm_tuning(X_unscaled, y)
Phase C — Best select  : uses comparator.compare_model_configs (caller)
Phase D — Final refit  : refit_svm(X_unscaled, y, config) -> (svm, scaler)

Representation
--------------
Input to run_svm_* is the UNSCALED 75-feature matrix.
Scaling is performed per-fold, fit on inner_train only.

Tuning grid (frozen)
--------------------
C:           0.01 | 0.1 | 1.0 | 10.0
class_weight: balanced  (fixed)
max_iter:     5000       (fixed)
random_state: 42         (fixed)

Total grid: 4 configurations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from src.models.base_models.cv_utils import (
    CVSummary,
    FoldMetrics,
    aggregate_cv_results,
    compute_fold_metrics,
    make_model_skf,
)
from src.models.base_models.preprocessing import fit_scaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline configuration (frozen)
# ---------------------------------------------------------------------------

SVM_BASELINE_CONFIG: dict[str, Any] = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 5000,
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Tuning grid (frozen)
# ---------------------------------------------------------------------------

SVM_TUNING_C_VALUES: list[float] = [0.01, 0.1, 1.0, 10.0]

# Fixed parameters (never tuned)
SVM_FIXED_PARAMS: dict[str, Any] = {
    "class_weight": "balanced",
    "max_iter": 5000,
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SVMConfig:
    """
    LinearSVC hyperparameter configuration.

    All values must be from the approved tuning grid or the baseline.
    """

    C: float = 1.0
    class_weight: str = "balanced"
    max_iter: int = 5000
    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "C": self.C,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }

    def validate(self) -> None:
        """Raise ValueError for out-of-protocol configurations."""
        if self.C <= 0:
            raise ValueError(f"SVM C must be > 0, got {self.C}")
        if self.class_weight != "balanced":
            raise ValueError(
                f"class_weight must be 'balanced' (frozen), got {self.class_weight!r}"
            )
        if self.max_iter != 5000:
            raise ValueError(
                f"max_iter must be 5000 (frozen), got {self.max_iter}"
            )
        if self.random_state != 42:
            raise ValueError(
                f"random_state must be 42 (frozen), got {self.random_state}"
            )


# ---------------------------------------------------------------------------
# Single CV run for one SVM configuration
# ---------------------------------------------------------------------------


def run_svm_cv(
    X_unscaled: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
) -> CVSummary:
    """
    Run 5-fold StratifiedKFold CV for one SVM configuration.

    SCALER ISOLATION: A fresh StandardScaler is fit on inner_train ONLY
    inside each fold.  inner_val is only transformed (never fit).

    Parameters
    ----------
    X_unscaled : np.ndarray
        Feature matrix (encoded, UNSCALED), shape (n, 75).
        Scaling is applied per-fold inside this function.
    y : np.ndarray
        Binary labels 0/1.
    config : dict
        SVM hyperparameter dict.

    Returns
    -------
    CVSummary
    """
    _validate_inputs(X_unscaled, y, config)

    skf = make_model_skf()
    fold_metrics: list[FoldMetrics] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_unscaled, y)):
        t0 = time.perf_counter()

        X_tr_raw, X_va_raw = X_unscaled[train_idx], X_unscaled[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # ----------------------------------------------------------------
        # SCALER: fit ONLY on inner_train — this is the leakage boundary.
        # ----------------------------------------------------------------
        scaler = fit_scaler(X_tr_raw)
        X_tr = scaler.transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)

        clf = _build_svm(config)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_va)

        elapsed = time.perf_counter() - t0

        fm = compute_fold_metrics(
            y_true=y_va,
            y_pred=y_pred,
            fold_idx=fold_idx,
            runtime_seconds=elapsed,
            n_train=len(train_idx),
            n_val=len(val_idx),
        )
        fold_metrics.append(fm)

        logger.info(
            "SVM CV fold=%d | macro_f1=%.6f | runtime=%.2fs | C=%.4f",
            fold_idx,
            fm.macro_f1,
            elapsed,
            config.get("C", 1.0),
        )

    summary = aggregate_cv_results(fold_metrics, config=config, model_type="svm")
    logger.info(
        "SVM CV done | mean_f1=%.6f ± %.6f | C=%.4f",
        summary.mean_macro_f1,
        summary.std_macro_f1,
        config.get("C", 1.0),
    )
    return summary


# ---------------------------------------------------------------------------
# Phase A — Baseline
# ---------------------------------------------------------------------------


def run_svm_baseline(X_unscaled: np.ndarray, y: np.ndarray) -> CVSummary:
    """
    Phase A: Run 5-fold CV with the frozen baseline configuration.

    Parameters
    ----------
    X_unscaled, y : np.ndarray
        Encoded UNSCALED TRAIN feature matrix and labels.

    Returns
    -------
    CVSummary
    """
    logger.info("=== SVM BASELINE START ===")
    t0 = time.perf_counter()
    result = run_svm_cv(X_unscaled, y, SVM_BASELINE_CONFIG)
    elapsed = time.perf_counter() - t0
    logger.info(
        "=== SVM BASELINE DONE | mean_f1=%.6f | total_runtime=%.2fs ===",
        result.mean_macro_f1,
        elapsed,
    )
    return result


# ---------------------------------------------------------------------------
# Phase B — Tuning grid
# ---------------------------------------------------------------------------


def run_svm_tuning(X_unscaled: np.ndarray, y: np.ndarray) -> list[CVSummary]:
    """
    Phase B: Run 5-fold CV for all 4 C-value configurations.

    Parameters
    ----------
    X_unscaled, y : np.ndarray
        Encoded UNSCALED TRAIN feature matrix and labels.

    Returns
    -------
    list[CVSummary]
        One entry per C value (4 total).
    """
    logger.info("=== SVM TUNING START | grid_size=%d ===", len(SVM_TUNING_C_VALUES))
    t0 = time.perf_counter()

    results: list[CVSummary] = []
    for i, C_val in enumerate(SVM_TUNING_C_VALUES):
        config = dict(SVM_FIXED_PARAMS)
        config["C"] = C_val
        logger.info("SVM tuning config %d/%d | C=%.4f", i + 1, len(SVM_TUNING_C_VALUES), C_val)
        result = run_svm_cv(X_unscaled, y, config)
        results.append(result)

    elapsed = time.perf_counter() - t0
    logger.info(
        "=== SVM TUNING DONE | n_configs=%d | total_runtime=%.2fs ===",
        len(results),
        elapsed,
    )
    return results


# ---------------------------------------------------------------------------
# Phase D — Final refit on full TRAIN
# ---------------------------------------------------------------------------


def refit_svm(
    X_unscaled: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
) -> tuple[LinearSVC, StandardScaler]:
    """
    Phase D: Fit the final SVM on the complete frozen TRAIN data.

    A fresh StandardScaler is fit on the complete TRAIN data.
    No cross-validation occurs here.

    Parameters
    ----------
    X_unscaled : np.ndarray
        Full TRAIN feature matrix (encoded, UNSCALED), shape (n, 75).
    y : np.ndarray
        Full TRAIN labels.
    config : dict
        Selected best SVM configuration.

    Returns
    -------
    (LinearSVC, StandardScaler)
        Fitted model and scaler — both must be persisted in the checkpoint.

    Notes
    -----
    The returned model exposes:
        - predict(X_scaled)            -> class labels
        - decision_function(X_scaled)  -> raw SVM scores
    Do NOT claim decision_function() values are probabilities.
    """
    _validate_inputs(X_unscaled, y, config)
    logger.info("SVM final refit | C=%.4f | n_train=%d", config.get("C", 1.0), len(y))
    t0 = time.perf_counter()

    # Fresh scaler fit on complete TRAIN
    scaler = fit_scaler(X_unscaled)
    X_scaled = scaler.transform(X_unscaled)

    clf = _build_svm(config)
    clf.fit(X_scaled, y)
    elapsed = time.perf_counter() - t0
    logger.info("SVM final refit done | runtime=%.2fs", elapsed)
    return clf, scaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_svm(config: dict[str, Any]) -> LinearSVC:
    """Instantiate a LinearSVC from a config dict."""
    return LinearSVC(
        C=config.get("C", 1.0),
        class_weight=config.get("class_weight", "balanced"),
        max_iter=config.get("max_iter", 5000),
        random_state=config.get("random_state", 42),
    )


def _validate_inputs(
    X: np.ndarray, y: np.ndarray, config: dict[str, Any]
) -> None:
    """Validate X/y shapes and basic config constraints."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X rows ({len(X)}) != y length ({len(y)})")
    if X.shape[0] == 0:
        raise ValueError("Empty training set (0 rows).")
    unique_cls = np.unique(y)
    if len(unique_cls) < 2:
        raise ValueError(
            f"Training set contains only one class: {unique_cls}. "
            "SVM requires at least two classes."
        )
    C_val = config.get("C", 1.0)
    if not isinstance(C_val, (int, float)) or C_val <= 0:
        raise ValueError(f"SVM C must be a positive number, got {C_val!r}")
