"""
src/models/base_models/decision_tree.py
-----------------------------------------
Decision Tree base model for Sprint 5.

Backend: sklearn.tree.DecisionTreeClassifier

Phases
------
Phase A — Baseline CV  : run_dt_baseline(X, y)
Phase B — Tuning grid  : run_dt_tuning(X, y)
Phase C — Best select  : uses comparator.compare_model_configs (caller)
Phase D — Final refit  : refit_dt(X, y, config)

Representation
--------------
DT uses encoded UNSCALED features (75 MI-selected columns, no StandardScaler).

Tuning grid (frozen)
--------------------
criterion:        gini | entropy
max_depth:        5 | 10 | 20 | None
min_samples_leaf: 1 | 5 | 10
class_weight:     balanced   (fixed)
random_state:     42         (fixed)

Total grid: 2 × 4 × 3 = 24 configurations.

Leakage guarantee
-----------------
No scaler is applied in DT — no scaler leakage possible.
The caller must never pass validation/test data into X, y during Phase A-B.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from original_split_benchmark.src.models.base_models.cv_utils import (
    CVSummary,
    FoldMetrics,
    aggregate_cv_results,
    compute_fold_metrics,
    make_model_skf,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline configuration (frozen)
# ---------------------------------------------------------------------------

DT_BASELINE_CONFIG: dict[str, Any] = {
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Tuning grid (frozen)
# ---------------------------------------------------------------------------

DT_TUNING_GRID: dict[str, list[Any]] = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 20, None],
    "min_samples_leaf": [1, 5, 10],
}

# Fixed parameters (never tuned)
DT_FIXED_PARAMS: dict[str, Any] = {
    "min_samples_split": 2,
    "class_weight": "balanced",
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DTConfig:
    """
    Decision Tree hyperparameter configuration.

    All values must be from the approved tuning grid or the baseline.
    """

    criterion: str = "gini"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    class_weight: str = "balanced"
    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
        }

    def validate(self) -> None:
        """Raise ValueError for out-of-protocol configurations."""
        if self.criterion not in ("gini", "entropy"):
            raise ValueError(f"Invalid DT criterion: {self.criterion!r}")
        if self.max_depth is not None and (
            not isinstance(self.max_depth, int) or self.max_depth < 1
        ):
            raise ValueError(f"Invalid max_depth: {self.max_depth!r}")
        if self.min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {self.min_samples_leaf}")
        if self.class_weight != "balanced":
            raise ValueError(
                f"class_weight must be 'balanced' (frozen), got {self.class_weight!r}"
            )
        if self.random_state != 42:
            raise ValueError(
                f"random_state must be 42 (frozen), got {self.random_state}"
            )


# ---------------------------------------------------------------------------
# Single CV run for one DT configuration
# ---------------------------------------------------------------------------


def run_dt_cv(
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
) -> CVSummary:
    """
    Run 5-fold StratifiedKFold CV for one DT configuration.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (encoded, UNSCALED), shape (n, 75).
    y : np.ndarray
        Binary labels 0/1.
    config : dict
        DT hyperparameter dict (may include fixed params).

    Returns
    -------
    CVSummary
    """
    _validate_inputs(X, y, config)

    skf = make_model_skf()
    fold_metrics: list[FoldMetrics] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        t0 = time.perf_counter()

        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        clf = _build_dt(config)
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
            "DT CV fold=%d | macro_f1=%.6f | runtime=%.2fs | config=%s",
            fold_idx,
            fm.macro_f1,
            elapsed,
            _config_str(config),
        )

    summary = aggregate_cv_results(fold_metrics, config=config, model_type="dt")
    logger.info(
        "DT CV done | mean_f1=%.6f ± %.6f | config=%s",
        summary.mean_macro_f1,
        summary.std_macro_f1,
        _config_str(config),
    )
    return summary


# ---------------------------------------------------------------------------
# Phase A — Baseline
# ---------------------------------------------------------------------------


def run_dt_baseline(X: np.ndarray, y: np.ndarray) -> CVSummary:
    """
    Phase A: Run 5-fold CV with the frozen baseline configuration.

    Must be called before any tuning so that the baseline is established
    independently.

    Parameters
    ----------
    X, y : np.ndarray
        Encoded UNSCALED TRAIN feature matrix and labels.

    Returns
    -------
    CVSummary
    """
    logger.info("=== DT BASELINE START ===")
    t0 = time.perf_counter()
    result = run_dt_cv(X, y, DT_BASELINE_CONFIG)
    elapsed = time.perf_counter() - t0
    logger.info(
        "=== DT BASELINE DONE | mean_f1=%.6f | total_runtime=%.2fs ===",
        result.mean_macro_f1,
        elapsed,
    )
    return result


# ---------------------------------------------------------------------------
# Phase B — Tuning grid
# ---------------------------------------------------------------------------


def run_dt_tuning(X: np.ndarray, y: np.ndarray) -> list[CVSummary]:
    """
    Phase B: Run 5-fold CV for all 24 grid configurations.

    Parameters
    ----------
    X, y : np.ndarray
        Encoded UNSCALED TRAIN feature matrix and labels.

    Returns
    -------
    list[CVSummary]
        One entry per grid configuration (24 total).
    """
    logger.info("=== DT TUNING START | grid_size=%d ===", _grid_size())
    t0 = time.perf_counter()

    results: list[CVSummary] = []
    configs = _generate_dt_configs()

    for i, config in enumerate(configs):
        logger.info("DT tuning config %d/%d | %s", i + 1, len(configs), _config_str(config))
        result = run_dt_cv(X, y, config)
        results.append(result)

    elapsed = time.perf_counter() - t0
    logger.info(
        "=== DT TUNING DONE | n_configs=%d | total_runtime=%.2fs ===",
        len(results),
        elapsed,
    )
    return results


# ---------------------------------------------------------------------------
# Phase D — Final refit on full TRAIN
# ---------------------------------------------------------------------------


def refit_dt(
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
) -> DecisionTreeClassifier:
    """
    Phase D: Fit the final DT on the complete frozen TRAIN data.

    No cross-validation occurs here.  The caller must ensure X, y are the
    full frozen TRAIN (not validation or test).

    Parameters
    ----------
    X : np.ndarray
        Full TRAIN feature matrix (encoded, UNSCALED), shape (n, 75).
    y : np.ndarray
        Full TRAIN labels.
    config : dict
        Selected best DT configuration.

    Returns
    -------
    DecisionTreeClassifier
        Fitted final model.
    """
    _validate_inputs(X, y, config)
    logger.info("DT final refit | config=%s | n_train=%d", _config_str(config), len(y))
    t0 = time.perf_counter()
    clf = _build_dt(config)
    clf.fit(X, y)
    elapsed = time.perf_counter() - t0
    logger.info("DT final refit done | runtime=%.2fs", elapsed)
    return clf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dt(config: dict[str, Any]) -> DecisionTreeClassifier:
    """Instantiate a DecisionTreeClassifier from a config dict."""
    return DecisionTreeClassifier(
        criterion=config.get("criterion", "gini"),
        max_depth=config.get("max_depth", None),
        min_samples_split=config.get("min_samples_split", 2),
        min_samples_leaf=config.get("min_samples_leaf", 1),
        class_weight=config.get("class_weight", "balanced"),
        random_state=config.get("random_state", 42),
    )


def _generate_dt_configs() -> list[dict[str, Any]]:
    """Generate all 24 DT grid configurations in deterministic order."""
    configs = []
    for criterion, max_depth, min_samples_leaf in itertools.product(
        DT_TUNING_GRID["criterion"],
        DT_TUNING_GRID["max_depth"],
        DT_TUNING_GRID["min_samples_leaf"],
    ):
        cfg = dict(DT_FIXED_PARAMS)
        cfg["criterion"] = criterion
        cfg["max_depth"] = max_depth
        cfg["min_samples_leaf"] = min_samples_leaf
        configs.append(cfg)
    return configs


def _grid_size() -> int:
    return (
        len(DT_TUNING_GRID["criterion"])
        * len(DT_TUNING_GRID["max_depth"])
        * len(DT_TUNING_GRID["min_samples_leaf"])
    )


def _config_str(config: dict[str, Any]) -> str:
    return (
        f"criterion={config.get('criterion')} "
        f"max_depth={config.get('max_depth')} "
        f"min_samples_leaf={config.get('min_samples_leaf')}"
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
            "DT requires at least two classes."
        )
