"""
src/models/base_models/cv_utils.py
------------------------------------
Cross-validation infrastructure for Sprint 5 base models.

Provides
--------
- make_model_skf()        : canonical 5-fold StratifiedKFold (seed=0)
- FoldMetrics             : per-fold metric container
- CVSummary               : aggregate CV result container
- compute_fold_metrics()  : compute all metrics from y_true/y_pred
- aggregate_cv_results()  : reduce list[FoldMetrics] -> CVSummary

CV protocol (frozen)
--------------------
    StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    Stratification target: label (binary 0/1)

Primary tuning metric: Macro-F1.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CV configuration constants (FROZEN)
# ---------------------------------------------------------------------------

MODEL_CV_N_SPLITS = 5
MODEL_CV_SHUFFLE = True
MODEL_CV_RANDOM_STATE = 0  # Intentionally different from MI seed=42


# ---------------------------------------------------------------------------
# Canonical SKF factory
# ---------------------------------------------------------------------------


def make_model_skf() -> StratifiedKFold:
    """
    Return the canonical StratifiedKFold for Sprint 5 model CV.

    Frozen parameters:
        n_splits=5, shuffle=True, random_state=0

    Returns
    -------
    StratifiedKFold
    """
    return StratifiedKFold(
        n_splits=MODEL_CV_N_SPLITS,
        shuffle=MODEL_CV_SHUFFLE,
        random_state=MODEL_CV_RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Per-fold metric container
# ---------------------------------------------------------------------------


@dataclass
class FoldMetrics:
    """
    All evaluation metrics recorded for a single CV fold.

    Attributes
    ----------
    fold_idx : int
        Zero-based fold index (0..4).
    macro_f1 : float
        Macro-averaged F1 score (primary tuning metric).
    weighted_f1 : float
        Weighted F1 score.
    balanced_accuracy : float
        Balanced accuracy.
    precision_macro : float
        Macro-averaged precision.
    recall_macro : float
        Macro-averaged recall.
    confusion : list[list[int]]
        2×2 confusion matrix [[TN, FP], [FN, TP]].
    fpr : float
        False positive rate = FP / (FP + TN).
    specificity : float
        Specificity = TN / (TN + FP).
    runtime_seconds : float
        Wall-clock time for this fold (training + evaluation).
    n_train : int
        Number of training samples in this fold.
    n_val : int
        Number of validation samples in this fold.
    """

    fold_idx: int
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    confusion: list[list[int]]
    fpr: float
    specificity: float
    runtime_seconds: float
    n_train: int
    n_val: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON/CSV output."""
        return {
            "fold_idx": self.fold_idx,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "confusion": self.confusion,
            "fpr": self.fpr,
            "specificity": self.specificity,
            "runtime_seconds": self.runtime_seconds,
            "n_train": self.n_train,
            "n_val": self.n_val,
        }


# ---------------------------------------------------------------------------
# Aggregate CV result container
# ---------------------------------------------------------------------------


@dataclass
class CVSummary:
    """
    Aggregate result for a single model configuration across all CV folds.

    Attributes
    ----------
    config : dict
        Model hyperparameter dict (serialisable).
    model_type : str
        One of {"dt", "rf", "svm", "nn"}.
    folds : list[FoldMetrics]
        Per-fold metrics (length = MODEL_CV_N_SPLITS).
    mean_macro_f1 : float
        Mean Macro-F1 across folds (primary comparator key).
    std_macro_f1 : float
        Standard deviation of Macro-F1 across folds.
    mean_weighted_f1 : float
    std_weighted_f1 : float
    mean_balanced_accuracy : float
    std_balanced_accuracy : float
    mean_precision_macro : float
    mean_recall_macro : float
    mean_fpr : float
    mean_specificity : float
    total_runtime_seconds : float
        Sum of all fold runtimes.
    """

    config: dict[str, Any]
    model_type: str
    folds: list[FoldMetrics]
    mean_macro_f1: float
    std_macro_f1: float
    mean_weighted_f1: float
    std_weighted_f1: float
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    mean_precision_macro: float
    mean_recall_macro: float
    mean_fpr: float
    mean_specificity: float
    total_runtime_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON/CSV output."""
        return {
            "model_type": self.model_type,
            "config": self.config,
            "mean_macro_f1": self.mean_macro_f1,
            "std_macro_f1": self.std_macro_f1,
            "mean_weighted_f1": self.mean_weighted_f1,
            "std_weighted_f1": self.std_weighted_f1,
            "mean_balanced_accuracy": self.mean_balanced_accuracy,
            "std_balanced_accuracy": self.std_balanced_accuracy,
            "mean_precision_macro": self.mean_precision_macro,
            "mean_recall_macro": self.mean_recall_macro,
            "mean_fpr": self.mean_fpr,
            "mean_specificity": self.mean_specificity,
            "total_runtime_seconds": self.total_runtime_seconds,
            "folds": [f.to_dict() for f in self.folds],
        }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_idx: int,
    runtime_seconds: float,
    n_train: int,
    n_val: int,
) -> FoldMetrics:
    """
    Compute all evaluation metrics for a single CV fold.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (0=Normal, 1=Attack).
    y_pred : np.ndarray
        Predicted labels from the model.
    fold_idx : int
        Zero-based fold index.
    runtime_seconds : float
        Wall-clock time for this fold.
    n_train, n_val : int
        Sample counts.

    Returns
    -------
    FoldMetrics
    """
    # Primary metric
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec_macro = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # FPR and specificity
    denom_fp = float(fp + tn)
    fpr = float(fp) / denom_fp if denom_fp > 0 else 0.0
    specificity = float(tn) / denom_fp if denom_fp > 0 else 0.0

    return FoldMetrics(
        fold_idx=fold_idx,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        balanced_accuracy=bal_acc,
        precision_macro=prec_macro,
        recall_macro=rec_macro,
        confusion=cm.tolist(),
        fpr=fpr,
        specificity=specificity,
        runtime_seconds=runtime_seconds,
        n_train=n_train,
        n_val=n_val,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_cv_results(
    folds: list[FoldMetrics],
    config: dict[str, Any],
    model_type: str,
) -> CVSummary:
    """
    Reduce a list of per-fold metrics to a CVSummary.

    Parameters
    ----------
    folds : list[FoldMetrics]
        Must have exactly MODEL_CV_N_SPLITS entries.
    config : dict
        Model hyperparameter dict.
    model_type : str
        One of {"dt", "rf", "svm", "nn"}.

    Returns
    -------
    CVSummary
    """
    if len(folds) == 0:
        raise ValueError("aggregate_cv_results: folds list is empty.")

    macro_f1s = np.array([f.macro_f1 for f in folds])
    weighted_f1s = np.array([f.weighted_f1 for f in folds])
    bal_accs = np.array([f.balanced_accuracy for f in folds])
    precs = np.array([f.precision_macro for f in folds])
    recs = np.array([f.recall_macro for f in folds])
    fprs = np.array([f.fpr for f in folds])
    specs = np.array([f.specificity for f in folds])
    total_rt = sum(f.runtime_seconds for f in folds)

    return CVSummary(
        config=config,
        model_type=model_type,
        folds=folds,
        mean_macro_f1=float(np.mean(macro_f1s)),
        std_macro_f1=float(np.std(macro_f1s, ddof=1) if len(macro_f1s) > 1 else 0.0),
        mean_weighted_f1=float(np.mean(weighted_f1s)),
        std_weighted_f1=float(np.std(weighted_f1s, ddof=1) if len(weighted_f1s) > 1 else 0.0),
        mean_balanced_accuracy=float(np.mean(bal_accs)),
        std_balanced_accuracy=float(np.std(bal_accs, ddof=1) if len(bal_accs) > 1 else 0.0),
        mean_precision_macro=float(np.mean(precs)),
        mean_recall_macro=float(np.mean(recs)),
        mean_fpr=float(np.mean(fprs)),
        mean_specificity=float(np.mean(specs)),
        total_runtime_seconds=float(total_rt),
    )
