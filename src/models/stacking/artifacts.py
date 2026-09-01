"""
src/models/stacking/artifacts.py
----------------------------------
Artifact save/load helpers for Sprint 6 OOF stacking.

All functions are pure I/O — no training logic.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OOF predictions
# ---------------------------------------------------------------------------


def save_oof_predictions(
    oof_df: pd.DataFrame,
    seed_dir: pathlib.Path,
) -> pathlib.Path:
    """
    Save OOF prediction matrix to seed_dir/oof_predictions.csv.

    Columns: row_id, dt_attack_probability, rf_attack_probability,
             svm_decision_score, nn_attack_probability, label
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    out = seed_dir / "oof_predictions.csv"
    oof_df.to_csv(out, index=False)
    logger.info("OOF predictions saved | path=%s | shape=%s", out, oof_df.shape)
    return out


# ---------------------------------------------------------------------------
# Fold assignments
# ---------------------------------------------------------------------------


def save_fold_assignments(
    folds: list[tuple[np.ndarray, np.ndarray]],
    n: int,
    seed_dir: pathlib.Path,
) -> pathlib.Path:
    """
    Save fold assignment as a CSV: row_id, fold_idx.

    The same file is identical across H1 seeds (folds are created once).
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    fold_col = np.full(n, -1, dtype=np.int64)
    for fold_idx, (_, oof_idx) in enumerate(folds):
        fold_col[oof_idx] = fold_idx

    df = pd.DataFrame({
        "row_id": np.arange(n, dtype=np.int64),
        "fold_idx": fold_col,
    })
    out = seed_dir / "fold_assignments.csv"
    df.to_csv(out, index=False)
    logger.info("Fold assignments saved | path=%s | n=%d", out, n)
    return out


# ---------------------------------------------------------------------------
# Per-seed metrics
# ---------------------------------------------------------------------------


def save_seed_metrics(
    metrics: dict[str, Any],
    seed_dir: pathlib.Path,
) -> pathlib.Path:
    """Save per-seed metrics to seed_dir/metrics.json."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    out = seed_dir / "metrics.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(metrics), fh, indent=2)
    logger.info("Seed metrics saved | path=%s", out)
    return out


# ---------------------------------------------------------------------------
# Per-seed metadata
# ---------------------------------------------------------------------------


def save_seed_metadata(
    metadata: dict[str, Any],
    seed_dir: pathlib.Path,
) -> pathlib.Path:
    """Save per-seed metadata to seed_dir/metadata.json."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    out = seed_dir / "metadata.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(metadata), fh, indent=2)
    logger.info("Seed metadata saved | path=%s", out)
    return out


# ---------------------------------------------------------------------------
# Meta-learner checkpoint
# ---------------------------------------------------------------------------


def save_meta_learner_checkpoint(
    clf: LogisticRegression,
    ckpt_dir: pathlib.Path,
) -> pathlib.Path:
    """Persist meta-learner to ckpt_dir/meta_learner.joblib."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / "meta_learner.joblib"
    joblib.dump(clf, out)
    logger.info("Meta-learner checkpoint saved | path=%s", out)
    return out


def save_meta_learner_metadata(
    metadata: dict[str, Any],
    ckpt_dir: pathlib.Path,
) -> pathlib.Path:
    """Save meta-learner checkpoint metadata to ckpt_dir/metadata.json."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / "metadata.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(metadata), fh, indent=2)
    logger.info("Meta-learner checkpoint metadata saved | path=%s", out)
    return out


def load_meta_learner_checkpoint(ckpt_dir: pathlib.Path) -> LogisticRegression:
    """Load meta-learner from ckpt_dir/meta_learner.joblib."""
    p = ckpt_dir / "meta_learner.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Meta-learner checkpoint not found: {p}")
    clf = joblib.load(p)
    logger.info("Meta-learner checkpoint loaded | path=%s", p)
    return clf


# ---------------------------------------------------------------------------
# Experiment-level artifacts
# ---------------------------------------------------------------------------


def save_h1_summary(
    summary: dict[str, Any],
    results_dir: pathlib.Path,
) -> pathlib.Path:
    """Save H1 summary to results_dir/h1_summary.json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "h1_summary.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(summary), fh, indent=2)
    logger.info("H1 summary saved | path=%s", out)
    return out


def save_experiment_metadata(
    metadata: dict[str, Any],
    results_dir: pathlib.Path,
) -> pathlib.Path:
    """Save experiment-level metadata to results_dir/metadata.json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "metadata.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(metadata), fh, indent=2)
    logger.info("Experiment metadata saved | path=%s", out)
    return out


def save_runtime_report(
    report: dict[str, Any],
    results_dir: pathlib.Path,
) -> pathlib.Path:
    """Save runtime report to results_dir/runtime_report.json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "runtime_report.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=2)
    logger.info("Runtime report saved | path=%s", out)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return obj
