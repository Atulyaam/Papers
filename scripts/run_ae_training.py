"""
scripts/run_ae_training.py
----------------------------
Sprint 7 — EXP_AE_V1 Full AE Training Pipeline.

Steps 5–14 of the 18-step execution order:
  5.  Create deterministic 90/10 Normal TRAIN monitor split
  6.  Fit scaler on AE-fit Normal TRAIN subset
  7.  Train AE (architecture 75→12→6→12→75, seed=42, patience=5)
  8.  Record best_epoch
  9.  Final refit on all 44,800 Normal TRAIN rows for best_epoch
  10. Freeze AE weights
  11. Transform Normal VALIDATION using frozen AE scaler
  12. Compute reconstruction errors for all 11,200 VAL rows
  13. Compute all 5 threshold candidates
  14. Save all artifacts

CRITICAL EXECUTION RULE: Run in Antigravity IDE Integrated Terminal.
"""

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FROZEN_TRAIN_SHA = "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c"
FROZEN_VAL_SHA   = "13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180"


# ---------------------------------------------------------------------------
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.perf_counter()
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    cuda_version = torch.version.cuda or "N/A"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    logger.info("Device: %s | GPU: %s | CUDA: %s | PyTorch: %s",
                device_str, gpu_name, cuda_version, torch.__version__)

    # ------------------------------------------------------------------ #
    # STEP 0 — Hash verification                                           #
    # ------------------------------------------------------------------ #
    logger.info("=== STEP 0: Verifying frozen upstream artifacts ===")
    actual_train = sha256(ROOT / "data/splits/train.csv")
    assert actual_train == FROZEN_TRAIN_SHA, f"TRAIN SHA mismatch: {actual_train}"
    actual_val = sha256(ROOT / "data/splits/validation.csv")
    assert actual_val == FROZEN_VAL_SHA, f"VALIDATION SHA mismatch: {actual_val}"
    logger.info("TRAIN SHA: PASS | VALIDATION SHA: PASS")

    # ------------------------------------------------------------------ #
    # Load data                                                            #
    # ------------------------------------------------------------------ #
    logger.info("Loading data ...")
    train_df = pd.read_csv(ROOT / "data/splits/train.csv")
    val_df   = pd.read_csv(ROOT / "data/splits/validation.csv")

    sf = json.load(open(ROOT / "results/feature_selection/EXP_MI_V1_1/selected_features.json"))
    features = sf["features"]
    assert len(features) == 75, f"Expected 75 features, got {len(features)}"

    from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
    from src.models.base_models.preprocessing import build_feature_matrix

    pipeline = PreprocessingPipeline()
    pipeline.fit(train_df)
    train_enc = pipeline.transform(train_df, view="unscaled", split_name="train")
    val_enc   = pipeline.transform(val_df,   view="unscaled", split_name="validation")

    train_feat_df = pd.DataFrame(train_enc.X, columns=train_enc.feature_names)
    val_feat_df   = pd.DataFrame(val_enc.X,   columns=val_enc.feature_names)
    X_train_full  = build_feature_matrix(train_feat_df, features)
    X_val_full    = build_feature_matrix(val_feat_df,   features)

    # Filter Normal rows — MANDATORY
    normal_mask_train = train_df["label"].values == 0
    normal_mask_val   = val_df["label"].values   == 0
    X_normal_train = X_train_full[normal_mask_train]
    X_normal_val   = X_val_full[normal_mask_val]
    y_train_labels = train_df["label"].values[normal_mask_train]
    y_val_labels   = val_df["label"].values[normal_mask_val]

    assert X_normal_train.shape == (44800, 75), f"Expected (44800,75) got {X_normal_train.shape}"
    assert X_normal_val.shape   == (11200, 75), f"Expected (11200,75) got {X_normal_val.shape}"
    assert (y_train_labels == 0).all(), "Attack rows in Normal TRAIN subset!"
    assert (y_val_labels   == 0).all(), "Attack rows in Normal VAL subset!"
    logger.info("Normal TRAIN: 44800 | Normal VAL: 11200 | All label==0: VERIFIED")

    # ------------------------------------------------------------------ #
    # STEP 5 — Monitor split                                               #
    # ------------------------------------------------------------------ #
    logger.info("=== STEP 5: Creating monitor split ===")
    from src.models.autoencoder.ae_trainer import create_monitor_split, AETrainer

    normal_pos_idx = np.arange(len(X_normal_train))
    split = create_monitor_split(normal_pos_idx, seed=42)

    assert len(split.ae_fit_indices)  == 40320, f"ae_fit={len(split.ae_fit_indices)}"
    assert len(split.monitor_indices) ==  4480, f"monitor={len(split.monitor_indices)}"
    overlap = set(split.ae_fit_indices.tolist()) & set(split.monitor_indices.tolist())
    assert len(overlap) == 0, "ae_fit ∩ monitor NOT empty!"
    logger.info("Monitor split: ae_fit=40320 | monitor=4480 | disjoint=VERIFIED")

    # Save monitor split indices
    out_mon_dir = ROOT / "results/autoencoder/EXP_AE_V1/monitor"
    out_mon_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in split.ae_fit_indices:
        rows.append({"row_id": int(idx), "split": "ae_fit"})
    for idx in split.monitor_indices:
        rows.append({"row_id": int(idx), "split": "monitor"})
    pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True).to_csv(
        out_mon_dir / "monitor_split_indices.csv", index=False
    )
    mon_meta = {
        **split.to_dict(),
        "monitor_split_seed": split.split_seed,
        "normal_train_total": 44800,
        "invariants": {
            "ae_fit_count": int(len(split.ae_fit_indices)),
            "monitor_count": int(len(split.monitor_indices)),
            "total": int(len(split.ae_fit_indices) + len(split.monitor_indices)),
            "disjoint": len(overlap) == 0,
        },
    }
    with open(out_mon_dir / "monitor_metadata.json", "w", encoding="utf-8") as f:
        json.dump(mon_meta, f, indent=2)
    logger.info("Monitor split saved.")

    # ------------------------------------------------------------------ #
    # STEP 6 — Fit scaler on AE-fit subset only                           #
    # ------------------------------------------------------------------ #
    logger.info("=== STEP 6: Fitting AE scaler on AE-fit subset ===")
    X_ae_fit_raw  = X_normal_train[split.ae_fit_indices]
    X_monitor_raw = X_normal_train[split.monitor_indices]

    scaler = StandardScaler()
    X_ae_fit  = scaler.fit_transform(X_ae_fit_raw).astype(np.float32)
    X_monitor = scaler.transform(X_monitor_raw).astype(np.float32)
    logger.info("Scaler fitted on %d Normal AE-fit rows (75 features).", len(X_ae_fit))

    # Pre-scale full Normal TRAIN and VAL for later steps
    X_normal_train_scaled = scaler.transform(X_normal_train).astype(np.float32)
    X_normal_val_scaled   = scaler.transform(X_normal_val).astype(np.float32)

    # Save scaler immediately to checkpoint
    ckpt_dir = ROOT / "results/checkpoints/EXP_AE_V1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, ckpt_dir / "ae_scaler.joblib")
    logger.info("Scaler saved: %s", ckpt_dir / "ae_scaler.joblib")

    # ------------------------------------------------------------------ #
    # STEP 7 — Train AE with monitor-based early stopping                  #
    # ------------------------------------------------------------------ #
    logger.info("=== STEP 7: Phase-1 AE training (monitor early stopping) ===")
    trainer = AETrainer(seed=42, device=device_str)
    phase1_result = trainer.fit(X_ae_fit, X_monitor, monitor_split=split)
    phase1_result.scaler = scaler

    # ------------------------------------------------------------------ #
    # STEP 8 — Record best_epoch                                           #
    # ------------------------------------------------------------------ #
    best_epoch = phase1_result.best_epoch
    logger.info("=== STEP 8: best_epoch = %d ===", best_epoch)

    # Save training history
    out_train_dir = ROOT / "results/autoencoder/EXP_AE_V1/training"
    out_train_dir.mkdir(parents=True, exist_ok=True)
    hist_rows = [
        {"epoch": r.epoch, "ae_fit_mse": r.ae_fit_mse, "monitor_mse": r.monitor_mse}
        for r in phase1_result.training_history
    ]
    pd.DataFrame(hist_rows).to_csv(out_train_dir / "training_history.csv", index=False)
    epoch_diag = phase1_result.epoch_diagnostics()
    with open(out_train_dir / "epoch_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(epoch_diag, f, indent=2)
    logger.info("Training history saved (epochs logged: %d)", len(hist_rows))

    # ------------------------------------------------------------------ #
    # STEP 9 — Final refit on all 44,800 Normal TRAIN rows                 #
    # ------------------------------------------------------------------ #
    logger.info("=== STEP 9: Phase-2 final refit on 44,800 Normal TRAIN rows ===")
    t_refit_start = time.perf_counter()
    final_model = trainer.final_refit(X_normal_train_scaled, best_epoch)
    t_refit = time.perf_counter() - t_refit_start
    logger.info("Final refit complete: epochs=%d | rows=44800 | %.1fs", best_epoch, t_refit)

    # ------------------------------------------------------------------ #
    # STEP 10 — Freeze weights                                             #
    # ------------------------------------------------------------------ #
    logger.info("=== STEP 10: Freezing AE weights ===")
    final_model.eval()
    # Save checkpoint
    torch.save(final_model.state_dict(), ckpt_dir / "ae_final.pt")
    with open(ckpt_dir / "ae_architecture.json", "w", encoding="utf-8") as f:
        json.dump(final_model.architecture_dict(), f, indent=2)
    logger.info("AE weights frozen and saved.")

    # ------------------------------------------------------------------ #
    # STEP 11 — Transform Normal VALIDATION                                #
    # ------------------------------------------------------------------ #
    logger.info("=== STEP 11: Transforming Normal VALIDATION ===")
    # Already done above as X_normal_val_scaled
    logger.info("Normal VALIDATION transformed: shape=%s", X_normal_val_scaled.shape)

    # ------------------------------------------------------------------ #
    # STEP 12 — Compute reconstruction errors                              #
    # ------------------------------------------------------------------ #
    logger.info("=== STEP 12: Computing reconstruction errors for 11,200 VAL rows ===")
    from src.models.autoencoder.ae_calibrate import calibrate_thresholds

    cal_result = calibrate_thresholds(final_model, X_normal_val_scaled, device=device_str)
    logger.info("RE stats: mean=%.6f | std=%.6f | max=%.6f",
                cal_result.re_stats["mean"],
                cal_result.re_stats["std"],
                cal_result.re_stats["max"])

    # ------------------------------------------------------------------ #
    # STEP 13 — Compute 5 threshold candidates                             #
    # ------------------------------------------------------------------ #
    logger.info("=== STEP 13: Threshold candidates ===")
    for name, cand in cal_result.thresholds.items():
        logger.info("  %-12s τ = %.6f | n_above=%d | frac=%.4f",
                    name, cand.threshold_value,
                    cand.samples_above_threshold, cand.fraction_above_threshold)

    # ------------------------------------------------------------------ #
    # STEP 14 — Save all artifacts                                         #
    # ------------------------------------------------------------------ #
    logger.info("=== STEP 14: Saving artifacts ===")

    # VAL row IDs (positional indices into Normal VAL subset)
    val_row_ids = np.arange(len(X_normal_val_scaled))

    # Threshold artifacts
    out_thresh_dir = ROOT / "results/autoencoder/EXP_AE_V1/threshold"
    out_thresh_dir.mkdir(parents=True, exist_ok=True)

    # Raw RE values
    re_df = pd.DataFrame({"row_id": val_row_ids, "re_value": cal_result.reconstruction_errors})
    re_df.to_csv(out_thresh_dir / "validation_reconstruction_errors.csv", index=False)

    # Threshold sweep
    sweep_rows = cal_result.threshold_sweep_rows()
    pd.DataFrame(sweep_rows).to_csv(out_thresh_dir / "threshold_sweep.csv", index=False)

    # Calibration JSON
    cal_dict = cal_result.calibration_dict()
    with open(out_thresh_dir / "threshold_calibration.json", "w", encoding="utf-8") as f:
        json.dump(cal_dict, f, indent=2)

    # Threshold config in checkpoint
    from src.models.autoencoder.artifacts import (
        SCALER_SPACE_LIMITATION, SINGLE_SEED_LIMITATION,
        THRESHOLD_SANITY_LIMITATION, PRIMARY_THRESHOLD_LIMITATION,
    )
    threshold_cfg = {
        **cal_dict,
        "scaler_space_limitation": SCALER_SPACE_LIMITATION,
        "single_seed_limitation": SINGLE_SEED_LIMITATION,
        "threshold_sanity_limitation": THRESHOLD_SANITY_LIMITATION,
        "primary_threshold_limitation": PRIMARY_THRESHOLD_LIMITATION,
    }
    with open(ckpt_dir / "threshold_config.json", "w", encoding="utf-8") as f:
        json.dump(threshold_cfg, f, indent=2)

    # AE metadata in checkpoint
    t_total = time.perf_counter() - t_start
    from src.models.autoencoder.ae_trainer import AE_MAX_EPOCHS
    ae_meta = {
        "experiment_id": "EXP_AE_V1",
        "sprint": 7,
        "dataset": "UNSW-NB15",
        "feature_set": "EXP_MI_V1_1",
        "feature_count": 75,
        "train_sha256": FROZEN_TRAIN_SHA,
        "validation_sha256": FROZEN_VAL_SHA,
        "normal_train_total": 44800,
        "training_rows": 40320,
        "monitor_rows": 4480,
        "calibration_rows": int(cal_result.calibration_rows),
        "monitor_split_seed": 42,
        "ae_seed": 42,
        "architecture": final_model.architecture_dict(),
        "loss": "MSELoss",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 256,
        "max_epochs": AE_MAX_EPOCHS,
        "patience": 5,
        "best_epoch": best_epoch,
        "final_epoch_count": best_epoch,
        "decision_4_revision": (
            "Decision 4 (Revised): max_epochs raised from 100 to 150, based on the "
            "EXP_AE_V1 diagnostic showing early stopping at epoch 138 with best_epoch=133 "
            "under the same seed, architecture, monitor split, scaler, and training "
            "configuration. Rationale: the 100-epoch run reached the maximum epoch while "
            "monitor MSE was still decreasing; the 300-epoch diagnostic reached its best "
            "at epoch 133 and triggered patience at epoch 138."
        ),
        "scaler_source": "new StandardScaler fit on Normal AE-fit subset only",
        "scaler_fit_population": "Normal AE-fit subset: 40320 rows",
        "ohe_mapping_source": "frozen upstream Sprint 2/4 preprocessing mapping",
        "threshold_candidates": list(cal_result.thresholds.keys()),
        "primary_threshold": "DEFERRED_TO_SPRINT_8",
        "threshold_caution": (
            "p95/p99/p99.9 are percentile-based thresholds from Normal VALIDATION. "
            "mean+2sigma and mean+3sigma are strongly affected by extreme high-RE rows "
            "(row_id 10737 RE~269, row_id 10731 RE~269 — short/aborted TCP sessions, RST/FIN state) "
            "and must be treated as outlier-inflated/cautious alternatives. "
            "Sanity tail counts are priors only; actual threshold values/counts come from "
            "the saved Normal VALIDATION reconstruction-error distribution."
        ),
        "outlier_note": (
            "Top-2 extreme Normal VALIDATION rows: row_id 10737 (RE~269.09) and "
            "row_id 10731 (RE~269.03). Both are short/aborted TCP sessions "
            "(RST/FIN state, low bytes/packets, services: '-'/ssh/pop3), labeled Normal. "
            "The AE's reconstruction error is elevated for short/aborted-connection Normal "
            "traffic, a structurally distinct sub-population. Sprint 8 fusion logic should "
            "be aware this may contribute to false positives for legitimately "
            "anomalous-but-benign connection patterns."
        ),
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
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "gpu": gpu_name,
        "git_commit": get_git_commit(),
        "timestamp_utc": timestamp_utc,
        "total_runtime_seconds": round(t_total, 2),
    }
    with open(ckpt_dir / "ae_metadata.json", "w", encoding="utf-8") as f:
        json.dump(ae_meta, f, indent=2)

    # Full metadata.json in results/autoencoder/EXP_AE_V1/
    out_dir = ROOT / "results/autoencoder/EXP_AE_V1"
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(ae_meta, f, indent=2)

    # config.yaml
    config = {
        "experiment_id": "EXP_AE_V1",
        "sprint": 7,
        "architecture": "75->12->6->12->75",
        "ae_seed": 42,
        "monitor_split_seed": 42,
        "loss": "MSELoss",
        "optimizer": "Adam",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 256,
        "max_epochs": AE_MAX_EPOCHS,
        "patience": 5,
        "scaler": "fresh_normal_ae_fit_only",
        "threshold_candidates": list(cal_result.thresholds.keys()),
        "primary_threshold": "DEFERRED_TO_SPRINT_8",
    }
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    # runtime_report.json
    runtime_report = {
        "total_runtime_seconds": round(t_total, 2),
        "best_epoch": best_epoch,
        "device": device_str,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "timestamp_utc": timestamp_utc,
    }
    with open(out_dir / "runtime_report.json", "w", encoding="utf-8") as f:
        json.dump(runtime_report, f, indent=2)

    logger.info("All artifacts saved.")

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("SPRINT 7 — EXP_AE_V1 TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Device:          {device_str} ({gpu_name})")
    print(f"  Architecture:    75->12->6->12->75 | params=2049")
    print(f"  Seed:            42")
    print(f"  AE-fit rows:     40,320")
    print(f"  Monitor rows:    4,480")
    print(f"  Best epoch:      {best_epoch}")
    print(f"  Final epoch:     {best_epoch}")
    print(f"  VAL rows:        11,200 (Normal only)")
    print()
    print("  RE Stats (Normal VAL):")
    for k, v in cal_result.re_stats.items():
        print(f"    {k:<8} = {v:.6f}")
    print()
    print("  Threshold Candidates:")
    for name, cand in cal_result.thresholds.items():
        print(f"    {name:<12} tau={cand.threshold_value:.6f}"
              f" | n_above={cand.samples_above_threshold}"
              f" | frac={cand.fraction_above_threshold:.4f}")
    print()
    print(f"  Primary threshold: DEFERRED TO SPRINT 8")
    print(f"  Total runtime:   {t_total:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
