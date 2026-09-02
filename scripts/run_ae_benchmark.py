"""
scripts/run_ae_benchmark.py
-----------------------------
Sprint 7 — EXP_AE_V1 Runtime Benchmark (Step 4).

Estimates runtime for each stage of the AE pipeline without
running the full training. Uses a small subset for timing extrapolation.
Does NOT save any artifacts or modify any frozen data.
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FROZEN_TRAIN_SHA = "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c"
FROZEN_VAL_SHA   = "13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180"

AE_FIT_ROWS   = 40_320
MONITOR_ROWS  =  4_480
NORMAL_TRAIN  = 44_800
NORMAL_VAL    = 11_200
MAX_EPOCHS    = 100
BATCH_SIZE    = 256


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("PyTorch: %s", torch.__version__)

    # ------------------------------------------------------------------ #
    # STAGE 0 — Verify hashes                                             #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    actual_train = sha256(ROOT / "data/splits/train.csv")
    actual_val   = sha256(ROOT / "data/splits/validation.csv")
    assert actual_train == FROZEN_TRAIN_SHA, "TRAIN SHA mismatch"
    assert actual_val   == FROZEN_VAL_SHA,   "VAL SHA mismatch"
    t_hash = time.perf_counter() - t0
    logger.info("Stage 0  Hash verify:             %.2f s", t_hash)

    # ------------------------------------------------------------------ #
    # STAGE 1 — Load + preprocessing estimate                             #
    # ------------------------------------------------------------------ #
    import pandas as pd
    t0 = time.perf_counter()
    train_df = pd.read_csv(ROOT / "data/splits/train.csv")
    val_df   = pd.read_csv(ROOT / "data/splits/validation.csv")
    t_load = time.perf_counter() - t0
    logger.info("Stage 1  CSV load (train+val):    %.2f s", t_load)

    # Load selected features
    sf = json.load(open(ROOT / "results/feature_selection/EXP_MI_V1_1/selected_features.json"))
    features = sf["features"]
    assert len(features) == 75

    from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
    t0 = time.perf_counter()
    pipeline = PreprocessingPipeline()
    pipeline.fit(train_df)
    train_enc = pipeline.transform(train_df, view="unscaled", split_name="train")
    val_enc   = pipeline.transform(val_df,   view="unscaled", split_name="validation")
    t_ohe = time.perf_counter() - t0
    logger.info("Stage 1  OHE fit+transform:       %.2f s", t_ohe)

    from src.models.base_models.preprocessing import build_feature_matrix
    t0 = time.perf_counter()
    train_feat_df = pd.DataFrame(train_enc.X, columns=train_enc.feature_names)
    val_feat_df   = pd.DataFrame(val_enc.X,   columns=val_enc.feature_names)
    X_train_full  = build_feature_matrix(train_feat_df, features)
    X_val_full    = build_feature_matrix(val_feat_df,   features)
    t_feat = time.perf_counter() - t0
    logger.info("Stage 1  build_feature_matrix:    %.2f s", t_feat)

    # Filter Normal only
    normal_mask_train = train_df["label"].values == 0
    normal_mask_val   = val_df["label"].values   == 0
    X_normal_train = X_train_full[normal_mask_train]
    X_normal_val   = X_val_full[normal_mask_val]
    logger.info("Normal TRAIN: %d | Normal VAL: %d", len(X_normal_train), len(X_normal_val))

    # ------------------------------------------------------------------ #
    # STAGE 2 — Monitor split + scaler                                    #
    # ------------------------------------------------------------------ #
    from src.models.autoencoder.ae_trainer import create_monitor_split, set_all_seeds
    t0 = time.perf_counter()
    # Pass positional 0..N-1 indices into X_normal_train for the split
    normal_pos_idx = np.arange(len(X_normal_train))
    split = create_monitor_split(normal_pos_idx, seed=42)
    t_split = time.perf_counter() - t0
    logger.info("Stage 2  Monitor split:           %.4f s", t_split)

    from sklearn.preprocessing import StandardScaler
    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_ae_fit   = scaler.fit_transform(X_normal_train[split.ae_fit_indices])
    X_monitor  = scaler.transform(X_normal_train[split.monitor_indices])
    X_all_norm = scaler.transform(X_normal_train)
    X_val_sc   = scaler.transform(X_normal_val)
    t_scaler = time.perf_counter() - t0
    logger.info("Stage 2  Scaler fit+transform:    %.4f s", t_scaler)

    # ------------------------------------------------------------------ #
    # STAGE 3 — AE single-epoch timing → extrapolate                      #
    # ------------------------------------------------------------------ #
    from src.models.autoencoder.ae_model import Autoencoder
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    set_all_seeds(42)
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.MSELoss()

    t_ae_fit = torch.tensor(X_ae_fit, dtype=torch.float32)
    t_mon    = torch.tensor(X_monitor, dtype=torch.float32)
    t_all    = torch.tensor(X_all_norm, dtype=torch.float32)

    fit_loader = DataLoader(TensorDataset(t_ae_fit), batch_size=BATCH_SIZE,
                            shuffle=True, generator=torch.Generator().manual_seed(42))
    mon_loader = DataLoader(TensorDataset(t_mon), batch_size=BATCH_SIZE, shuffle=False)
    all_loader = DataLoader(TensorDataset(t_all), batch_size=BATCH_SIZE, shuffle=True,
                            generator=torch.Generator().manual_seed(42))

    # Time 3 training epochs on ae_fit
    set_all_seeds(42)
    t0 = time.perf_counter()
    PROBE_EPOCHS = 3
    for ep in range(PROBE_EPOCHS):
        model.train()
        for (batch,) in fit_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            criterion(model(batch), batch).backward()
            optimizer.step()
    t_probe = time.perf_counter() - t0
    t_per_epoch_fit = t_probe / PROBE_EPOCHS

    # Time monitor eval
    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for (batch,) in mon_loader:
            batch = batch.to(device); model(batch)
    t_mon_eval = time.perf_counter() - t0

    # Time single final-refit epoch on all Normal TRAIN
    set_all_seeds(42)
    model2 = Autoencoder().to(device)
    opt2 = torch.optim.Adam(model2.parameters(), lr=0.001, weight_decay=0.0001)
    t0 = time.perf_counter()
    model2.train()
    for (batch,) in all_loader:
        batch = batch.to(device)
        opt2.zero_grad(); criterion(model2(batch), batch).backward(); opt2.step()
    t_per_epoch_refit = time.perf_counter() - t0

    # Estimate assuming early stopping at ~20 epochs (conservative)
    est_early_stop_epoch = 20
    t_phase1_est = est_early_stop_epoch * (t_per_epoch_fit + t_mon_eval)
    t_phase2_est = est_early_stop_epoch * t_per_epoch_refit

    logger.info("Stage 3  1 ae_fit epoch:          %.3f s", t_per_epoch_fit)
    logger.info("Stage 3  1 monitor eval:          %.3f s", t_mon_eval)
    logger.info("Stage 3  1 refit epoch (44800):   %.3f s", t_per_epoch_refit)
    logger.info("Stage 3  Phase-1 est (~20 ep):    %.1f s  (%.1f min)",
                t_phase1_est, t_phase1_est / 60)
    logger.info("Stage 3  Phase-2 est (~20 ep):    %.1f s  (%.1f min)",
                t_phase2_est, t_phase2_est / 60)

    # ------------------------------------------------------------------ #
    # STAGE 4 — Validation reconstruction + threshold sweep               #
    # ------------------------------------------------------------------ #
    t_val_sc = torch.tensor(X_val_sc, dtype=torch.float32)
    val_loader = DataLoader(TensorDataset(t_val_sc), batch_size=BATCH_SIZE, shuffle=False)

    t0 = time.perf_counter()
    model.eval()
    re_list = []
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(device)
            re_list.append(model.reconstruction_error(batch).cpu().numpy())
    re = np.concatenate(re_list)
    t_val_re = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = [np.percentile(re, p) for p in [95, 99, 99.9]]
    _ = re.mean() + 2 * re.std()
    _ = re.mean() + 3 * re.std()
    t_thresh = time.perf_counter() - t0

    logger.info("Stage 4  VAL reconstruction:      %.4f s", t_val_re)
    logger.info("Stage 4  Threshold sweep:         %.4f s", t_thresh)

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    t_total_est = (t_hash + t_load + t_ohe + t_feat + t_split + t_scaler
                   + t_phase1_est + t_phase2_est + t_val_re + t_thresh)

    print("\n" + "=" * 60)
    print("SPRINT 7 — AE RUNTIME BENCHMARK SUMMARY")
    print("=" * 60)
    rows = [
        ("Hash verification",         t_hash),
        ("CSV load (train+val)",       t_load),
        ("OHE fit+transform",          t_ohe),
        ("build_feature_matrix",       t_feat),
        ("Monitor split + scaler",     t_split + t_scaler),
        ("Phase-1 training (est ~20ep)", t_phase1_est),
        ("Phase-2 final refit (est ~20ep)", t_phase2_est),
        ("VAL reconstruction errors", t_val_re),
        ("Threshold sweep",            t_thresh),
    ]
    for name, secs in rows:
        mins = secs / 60
        print(f"  {name:<38} {secs:7.2f} s   ({mins:.2f} min)")
    print("-" * 60)
    print(f"  {'TOTAL (estimate)':38} {t_total_est:7.2f} s   ({t_total_est/60:.2f} min)")
    print("=" * 60)
    print(f"Device:       {device}")
    print(f"Max epochs:   {MAX_EPOCHS} (early stopping likely < 30)")
    print(f"Architecture: 75->12->6->12->75 | params=2049")
    print(f"AE-fit rows:  {AE_FIT_ROWS}")
    print(f"Monitor rows: {MONITOR_ROWS}")
    print(f"Val rows:     {NORMAL_VAL}")

    report = {
        "device": device,
        "torch_version": torch.__version__,
        "architecture": "75->12->6->12->75",
        "ae_fit_rows": AE_FIT_ROWS,
        "monitor_rows": MONITOR_ROWS,
        "val_rows": NORMAL_VAL,
        "t_hash_s": round(t_hash, 3),
        "t_load_s": round(t_load, 3),
        "t_ohe_s": round(t_ohe, 3),
        "t_feature_matrix_s": round(t_feat, 3),
        "t_monitor_split_scaler_s": round(t_split + t_scaler, 3),
        "t_per_epoch_fit_s": round(t_per_epoch_fit, 3),
        "t_per_epoch_monitor_eval_s": round(t_mon_eval, 3),
        "t_per_epoch_refit_s": round(t_per_epoch_refit, 3),
        "t_phase1_estimate_s": round(t_phase1_est, 2),
        "t_phase2_estimate_s": round(t_phase2_est, 2),
        "t_val_reconstruction_s": round(t_val_re, 4),
        "t_threshold_sweep_s": round(t_thresh, 6),
        "t_total_estimate_s": round(t_total_est, 2),
        "t_total_estimate_min": round(t_total_est / 60, 2),
        "early_stop_epoch_assumption": est_early_stop_epoch,
        "note": "Phase-1/2 times are estimates based on per-epoch timing x assumed 20 epochs",
    }

    out_dir = ROOT / "results/autoencoder/EXP_AE_V1"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runtime_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Runtime report saved: results/autoencoder/EXP_AE_V1/runtime_report.json")


if __name__ == "__main__":
    main()
