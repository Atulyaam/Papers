"""
src/models/autoencoder/artifacts.py
--------------------------------------
Sprint 7 — EXP_AE_V1 Artifact I/O helpers.

Handles saving and loading of all Sprint 7 artifacts:
    - monitor_split_indices.csv
    - training_history.csv
    - epoch_diagnostics.json
    - validation_reconstruction_errors.csv
    - threshold_sweep.csv
    - threshold_calibration.json
    - ae_final.pt
    - ae_scaler.joblib
    - ae_architecture.json
    - ae_metadata.json
    - threshold_config.json
    - config.yaml
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.models.autoencoder.ae_model import Autoencoder
from src.models.autoencoder.ae_calibrate import CalibrationResult, PRIMARY_THRESHOLD_STATUS
from src.models.autoencoder.ae_trainer import AETrainResult, MonitorSplit

logger = logging.getLogger(__name__)

SCALER_SPACE_LIMITATION = (
    "AE operates in a Normal-TRAIN-scaled feature space distinct from the "
    "full-TRAIN-scaled space used by DT/RF/SVM/NN. Sprint 8 fusion must "
    "account for this representation difference when combining the AE "
    "reconstruction error with the supervised branch outputs."
)

SINGLE_SEED_LIMITATION = (
    "Sprint 7 uses a single AE training seed (42). No multi-seed stability "
    "estimate exists for AE reconstruction error or threshold values. This "
    "is an accepted scope limitation and not a null result."
)

THRESHOLD_SANITY_LIMITATION = (
    "Threshold sanity counts are approximate priors only. Actual threshold "
    "values and counts are computed from the saved Normal VALIDATION "
    "reconstruction-error distribution."
)

PRIMARY_THRESHOLD_LIMITATION = "Primary threshold selection is deferred to Sprint 8."


# ---------------------------------------------------------------------------
# Monitor split
# ---------------------------------------------------------------------------

def save_monitor_split(
    split: MonitorSplit,
    normal_train_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Save monitor_split_indices.csv and monitor_metadata.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in split.ae_fit_indices:
        rows.append({"row_id": int(idx), "split": "ae_fit"})
    for idx in split.monitor_indices:
        rows.append({"row_id": int(idx), "split": "monitor"})
    df = pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)
    df.to_csv(out_dir / "monitor_split_indices.csv", index=False)

    meta = {
        **split.to_dict(),
        "monitor_split_seed": split.split_seed,
        "invariants": {
            "ae_fit_count": len(split.ae_fit_indices),
            "monitor_count": len(split.monitor_indices),
            "total": len(split.ae_fit_indices) + len(split.monitor_indices),
            "disjoint": len(set(split.ae_fit_indices.tolist()) &
                           set(split.monitor_indices.tolist())) == 0,
        },
    }
    with open(out_dir / "monitor_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Monitor split saved: %s", out_dir)


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def save_training_history(
    result: AETrainResult,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"epoch": r.epoch, "ae_fit_mse": r.ae_fit_mse, "monitor_mse": r.monitor_mse}
        for r in result.training_history
    ]
    pd.DataFrame(rows).to_csv(out_dir / "training_history.csv", index=False)

    diag = result.epoch_diagnostics()
    with open(out_dir / "epoch_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    logger.info("Training history saved: %s", out_dir)


# ---------------------------------------------------------------------------
# Threshold artifacts
# ---------------------------------------------------------------------------

def save_threshold_artifacts(
    cal: CalibrationResult,
    val_row_ids: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw reconstruction errors
    re_df = pd.DataFrame({"row_id": val_row_ids, "re_value": cal.reconstruction_errors})
    re_df.to_csv(out_dir / "validation_reconstruction_errors.csv", index=False)

    # Threshold sweep
    sweep_rows = cal.threshold_sweep_rows()
    pd.DataFrame(sweep_rows).to_csv(out_dir / "threshold_sweep.csv", index=False)

    # Calibration JSON
    with open(out_dir / "threshold_calibration.json", "w", encoding="utf-8") as f:
        json.dump(cal.calibration_dict(), f, indent=2)

    logger.info("Threshold artifacts saved: %s", out_dir)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoints(
    model: Autoencoder,
    scaler,
    result: AETrainResult,
    cal: CalibrationResult,
    out_dir: Path,
    metadata: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # AE weights
    torch.save(model.state_dict(), out_dir / "ae_final.pt")

    # Scaler
    joblib.dump(scaler, out_dir / "ae_scaler.joblib")

    # Architecture
    with open(out_dir / "ae_architecture.json", "w", encoding="utf-8") as f:
        json.dump(model.architecture_dict(), f, indent=2)

    # AE metadata
    ae_meta = {
        **metadata,
        "best_epoch": result.best_epoch,
        "final_epoch_count": result.final_epoch_count,
        "ae_fit_rows": result.ae_fit_rows,
        "monitor_rows": result.monitor_rows,
        "device": result.device,
        "runtime_seconds": round(result.runtime_seconds, 3),
        "scaler_space_limitation": SCALER_SPACE_LIMITATION,
        "single_seed_limitation": SINGLE_SEED_LIMITATION,
    }
    with open(out_dir / "ae_metadata.json", "w", encoding="utf-8") as f:
        json.dump(ae_meta, f, indent=2)

    # Threshold config
    threshold_cfg = {
        **cal.calibration_dict(),
        "primary_threshold": PRIMARY_THRESHOLD_STATUS,
        "scaler_space_limitation": SCALER_SPACE_LIMITATION,
        "single_seed_limitation": SINGLE_SEED_LIMITATION,
        "threshold_sanity_limitation": THRESHOLD_SANITY_LIMITATION,
        "primary_threshold_limitation": PRIMARY_THRESHOLD_LIMITATION,
    }
    with open(out_dir / "threshold_config.json", "w", encoding="utf-8") as f:
        json.dump(threshold_cfg, f, indent=2)

    logger.info("Checkpoints saved: %s", out_dir)


# ---------------------------------------------------------------------------
# Full metadata.json
# ---------------------------------------------------------------------------

def build_metadata(
    train_sha: str,
    val_sha: str,
    result: AETrainResult,
    cal: CalibrationResult,
    model: Autoencoder,
    scaler,
    git_commit: str,
    timestamp_utc: str,
    torch_version: str,
    cuda_version: str,
    gpu: str,
) -> dict:
    return {
        "experiment_id": "EXP_AE_V1",
        "sprint": 7,
        "dataset": "UNSW-NB15",
        "feature_set": "EXP_MI_V1_1",
        "feature_count": 75,
        "train_sha256": train_sha,
        "validation_sha256": val_sha,
        "training_rows": result.ae_fit_rows,
        "monitor_rows": result.monitor_rows,
        "normal_train_total": result.ae_fit_rows + result.monitor_rows,
        "calibration_rows": cal.calibration_rows,
        "monitor_split_seed": result.monitor_split.split_seed,
        "ae_seed": 42,
        "architecture": model.architecture_dict(),
        "loss": "MSELoss",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 256,
        "max_epochs": 100,
        "patience": 5,
        "best_epoch": result.best_epoch,
        "final_epoch_count": result.final_epoch_count,
        "scaler_source": "new StandardScaler fit on Normal AE-fit subset only",
        "scaler_fit_population": f"Normal AE-fit subset: {result.ae_fit_rows} rows",
        "ohe_mapping_source": "frozen upstream Sprint 2/4 preprocessing mapping",
        "threshold_candidates": list(cal.thresholds.keys()),
        "primary_threshold": PRIMARY_THRESHOLD_STATUS,
        "data_access_boundary": (
            "TRAIN (Normal AE-fit + monitor subsets only) + "
            "VALIDATION (Normal rows, threshold calibration only). "
            "FORBIDDEN: development_test.csv, protected_unseen_attack.csv, "
            "excluded_train_backdoor.csv."
        ),
        "scaler_space_limitation": SCALER_SPACE_LIMITATION,
        "single_seed_limitation": SINGLE_SEED_LIMITATION,
        "threshold_sanity_limitation": THRESHOLD_SANITY_LIMITATION,
        "primary_threshold_limitation": PRIMARY_THRESHOLD_LIMITATION,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu": gpu,
        "git_commit": git_commit,
        "timestamp_utc": timestamp_utc,
    }


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_ae(checkpoint_dir: Path, device: str = "cpu") -> Autoencoder:
    model = Autoencoder()
    state = torch.load(checkpoint_dir / "ae_final.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_scaler(checkpoint_dir: Path):
    return joblib.load(checkpoint_dir / "ae_scaler.joblib")
