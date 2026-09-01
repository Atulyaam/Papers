"""
src/models/stacking/meta_learner.py
--------------------------------------
Logistic Regression meta-learner for Sprint 6 OOF stacking.

Fixed configuration (frozen — no tuning)
-----------------------------------------
    solver       = lbfgs
    C            = 1.0
    class_weight = balanced
    max_iter     = 1000
    random_state = H1 seed  (varies per run: 42, 123, 2024)

Meta-feature input
-------------------
    4 columns only: dt_attack_probability, rf_attack_probability,
                    svm_decision_score, nn_attack_probability
    row_id is provenance only and MUST NOT enter the meta-learner.
    label is the target.

In-sample evaluation limitation (APPROVED — human decision)
-------------------------------------------------------------
H1 Macro-F1 is computed by evaluating the meta-learner on the same OOF
matrix used to train it. No separate meta-learner holdout exists under
the current data-isolation rules. This is in-sample evaluation at the
meta-learner level and is NOT a fully held-out end-to-end generalisation
estimate.

H1 reporting
-------------
    Three seeds (42, 123, 2024) each produce a per-seed Macro-F1.
    mean_macro_f1 and std_macro_f1 are computed across the three seeds.

    Frozen Sprint 5 reference:
    RF CV Macro-F1 = 0.9508532447968256
    Label: "Frozen Sprint 5 single-CV reference; not a matched 3-seed H1 baseline."
    No statistical significance testing between these mismatched reporting units.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.stacking.oof_runner import SCALING_LIMITATION_TEXT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen meta-learner configuration
# ---------------------------------------------------------------------------

META_CONFIG: dict[str, Any] = {
    "solver": "lbfgs",
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 1000,
}

# Mandatory limitation text (verbatim — must appear in metadata + reports)
META_EVALUATION_LIMITATION_TEXT: str = (
    "H1 Macro-F1 is computed by evaluating the meta-learner on the same OOF "
    "matrix used to train it. No separate meta-learner holdout exists under "
    "the current data-isolation rules. This is in-sample evaluation at the "
    "meta-learner level and is NOT a fully held-out end-to-end generalisation "
    "estimate."
)

# Sprint 5 reference (exact label — never call this a matched H1 baseline)
SPRINT5_RF_REFERENCE: float = 0.9508532447968256
SPRINT5_RF_REFERENCE_LABEL: str = (
    "Frozen Sprint 5 single-CV reference; not a matched 3-seed H1 baseline."
)

# Column names for the meta-feature input (row_id and label are excluded)
META_FEATURE_COLS: list[str] = [
    "dt_attack_probability",
    "rf_attack_probability",
    "svm_decision_score",
    "nn_attack_probability",
]


# ---------------------------------------------------------------------------
# Meta-learner training
# ---------------------------------------------------------------------------


def train_meta_learner(
    oof_df: pd.DataFrame,
    h1_seed: int,
) -> LogisticRegression:
    """
    Train a Logistic Regression meta-learner on the OOF matrix.

    Parameters
    ----------
    oof_df : pd.DataFrame
        OOF prediction matrix. Must contain columns:
        dt_attack_probability, rf_attack_probability,
        svm_decision_score, nn_attack_probability, label.
        row_id may be present but is IGNORED.
    h1_seed : int
        H1 seed for meta-learner random_state.

    Returns
    -------
    LogisticRegression
        Fitted meta-learner. Configuration is frozen (META_CONFIG).
        Only random_state varies by seed.

    Raises
    ------
    ValueError
        If required meta-feature columns are missing or label is absent.
    """
    _validate_oof_df(oof_df)

    X_meta = oof_df[META_FEATURE_COLS].to_numpy(dtype=np.float64)
    y_meta = oof_df["label"].to_numpy(dtype=np.int64)

    # Validate no row_id in X_meta
    assert X_meta.shape[1] == 4, (
        f"Meta-learner input must have exactly 4 columns, got {X_meta.shape[1]}. "
        "row_id must be excluded."
    )

    clf = LogisticRegression(
        **META_CONFIG,
        random_state=h1_seed,
    )
    clf.fit(X_meta, y_meta)

    logger.info(
        "Meta-learner trained | seed=%d | n_samples=%d | n_features=%d | "
        "classes=%s | config=%s",
        h1_seed, len(y_meta), X_meta.shape[1],
        clf.classes_.tolist(), META_CONFIG,
    )
    return clf


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_meta(
    clf: LogisticRegression,
    X_meta: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate binary predictions and attack probabilities from the meta-learner.

    Parameters
    ----------
    clf : LogisticRegression
        Fitted meta-learner (from train_meta_learner).
    X_meta : np.ndarray, shape (n, 4)
        Meta-feature matrix (no row_id, no label).
    threshold : float
        Classification threshold (default 0.5).

    Returns
    -------
    (y_pred, attack_proba)
        y_pred      : np.ndarray, int, shape (n,), values in {0, 1}
        attack_proba: np.ndarray, float, shape (n,), values in [0, 1]
    """
    if X_meta.ndim != 2 or X_meta.shape[1] != 4:
        raise ValueError(
            f"X_meta must have shape (n, 4), got {X_meta.shape}. "
            "Ensure row_id is excluded."
        )
    attack_proba = clf.predict_proba(X_meta)[:, 1]
    y_pred = (attack_proba >= threshold).astype(int)
    return y_pred, attack_proba


# ---------------------------------------------------------------------------
# OOF self-evaluation
# ---------------------------------------------------------------------------


def compute_oof_metrics(
    clf: LogisticRegression,
    oof_df: pd.DataFrame,
    h1_seed: int,
) -> dict[str, Any]:
    """
    Evaluate the meta-learner on the same OOF matrix it was trained on.

    LIMITATION: This is in-sample evaluation (training set = evaluation set).
    The limitation is documented in the returned dict. See module docstring.

    Parameters
    ----------
    clf : LogisticRegression
        Fitted meta-learner.
    oof_df : pd.DataFrame
        Full OOF matrix (columns: META_FEATURE_COLS + label).
    h1_seed : int
        H1 seed (for provenance).

    Returns
    -------
    dict
        macro_f1, weighted_f1, balanced_accuracy, precision_macro,
        recall_macro, confusion, fpr, specificity, h1_seed,
        in_sample_evaluation_warning (bool, always True),
        meta_evaluation_limitation (verbatim text),
        scaling_limitation (verbatim text).
    """
    _validate_oof_df(oof_df)

    X_meta = oof_df[META_FEATURE_COLS].to_numpy(dtype=np.float64)
    y_true = oof_df["label"].to_numpy(dtype=np.int64)

    y_pred, _ = predict_meta(clf, X_meta)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    denom_fp = float(fp + tn)
    fpr = float(fp) / denom_fp if denom_fp > 0 else 0.0
    specificity = float(tn) / denom_fp if denom_fp > 0 else 0.0

    logger.info(
        "OOF meta-evaluation | seed=%d | macro_f1=%.6f | weighted_f1=%.6f | "
        "balanced_acc=%.6f | IN-SAMPLE (limitation applies)",
        h1_seed, macro_f1, weighted_f1, bal_acc,
    )

    return {
        "h1_seed": h1_seed,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "confusion": cm.tolist(),
        "fpr": fpr,
        "specificity": specificity,
        "n_samples": len(y_true),
        "in_sample_evaluation_warning": True,
        "meta_evaluation_limitation": META_EVALUATION_LIMITATION_TEXT,
        "scaling_limitation": SCALING_LIMITATION_TEXT,
        "meta_config": {**META_CONFIG, "random_state": h1_seed},
    }


# ---------------------------------------------------------------------------
# H1 summary
# ---------------------------------------------------------------------------


def compute_h1_summary(
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate per-seed OOF metrics into the H1 stacking summary.

    Parameters
    ----------
    seed_results : list[dict]
        One dict per H1 seed, each returned by compute_oof_metrics().
        Must contain 'h1_seed' and 'macro_f1' keys.

    Returns
    -------
    dict
        h1_seeds, per_seed_macro_f1, mean_macro_f1, std_macro_f1,
        Sprint 5 reference (with exact label),
        both limitation texts,
        two_reporting_units_statement.

    Raises
    ------
    ValueError
        If seed_results is empty or any seed is missing 'macro_f1'.
    """
    if not seed_results:
        raise ValueError("seed_results is empty — cannot compute H1 summary.")

    seeds = [r["h1_seed"] for r in seed_results]
    f1s = np.array([r["macro_f1"] for r in seed_results], dtype=np.float64)

    mean_f1 = float(np.mean(f1s))
    std_f1 = float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0)

    per_seed = {str(r["h1_seed"]): r["macro_f1"] for r in seed_results}

    logger.info(
        "H1 summary | seeds=%s | per_seed_f1=%s | mean=%.6f | std=%.6f",
        seeds, per_seed, mean_f1, std_f1,
    )

    return {
        "experiment_id": "EXP_OOF_STACK_V1",
        "h1_seeds": seeds,
        "per_seed_macro_f1": per_seed,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "sprint5_reference": {
            "model": "RF",
            "macro_f1": SPRINT5_RF_REFERENCE,
            "label": SPRINT5_RF_REFERENCE_LABEL,
        },
        "two_reporting_units_statement": (
            "Two reporting units are used: "
            "(a) three-seed H1 stacking mean\u00b1std; "
            "(b) frozen Sprint 5 single-CV base-model reference. "
            "These are not statistically matched quantities."
        ),
        "in_sample_evaluation_warning": True,
        "meta_evaluation_limitation": META_EVALUATION_LIMITATION_TEXT,
        "scaling_limitation": SCALING_LIMITATION_TEXT,
        "per_seed_details": seed_results,
    }


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------


def _validate_oof_df(oof_df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    required = set(META_FEATURE_COLS) | {"label"}
    missing = required - set(oof_df.columns)
    if missing:
        raise ValueError(
            f"OOF DataFrame missing required columns: {sorted(missing)}. "
            f"Available: {list(oof_df.columns)}"
        )
    if "row_id" in META_FEATURE_COLS:
        raise RuntimeError(
            "row_id must NOT appear in META_FEATURE_COLS. Design violation."
        )
    n_nan = oof_df[META_FEATURE_COLS].isna().sum().sum()
    if n_nan > 0:
        raise ValueError(
            f"OOF DataFrame contains {n_nan} NaN values in meta-feature columns."
        )
