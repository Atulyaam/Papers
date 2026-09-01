"""
scripts/run_base_models_refit.py
----------------------------------
Phase D — Final refit on complete frozen TRAIN for all four base models.

This script reads selected_configs.json (produced by run_base_models_tuning.py)
and refits each model on the full TRAIN dataset.

Protocol
--------
- Reads selected_configs.json
- Fits fresh PreprocessingPipeline on TRAIN (NEW fit, not reusing benchmark/tuning scaler)
- Refits DT and RF using the selected configuration (no scaler)
- Refits SVM using fresh scaler fitted on full TRAIN → saves (svm, scaler)
- Refits NN using final_epoch_count = median(best_epoch from inner-CV)
  with fresh scaler fitted on full TRAIN → saves (net, scaler)
- Saves checkpoints to results/checkpoints/EXP_BASE_MODELS_V1/<model>/

Checkpoint contents per model
------------------------------
DT:  dt_final.joblib  + dt_metadata.json
RF:  rf_final.joblib  + rf_metadata.json
SVM: svm_final.joblib + svm_scaler.joblib + svm_metadata.json
NN:  nn_final.pt      + nn_scaler.joblib   + nn_metadata.json

Usage
-----
    python scripts/run_base_models_refit.py [--models dt,rf,svm,nn]

Outputs
-------
    results/checkpoints/EXP_BASE_MODELS_V1/<model>/<model>_final.*
    results/checkpoints/EXP_BASE_MODELS_V1/<model>/<model>_metadata.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logging_utils import get_experiment_logger
from src.utils.hashing import sha256_file
from src.utils.reproducibility import set_all_seeds
from src.models.base_models.preprocessing import load_selected_features, build_feature_matrix
from src.models.base_models.decision_tree import refit_dt
from src.models.base_models.random_forest import refit_rf
from src.models.base_models.linear_svc import refit_svm
from src.models.base_models.neural_network import (
    refit_nn, TRAIN_POS_WEIGHT, TRAIN_N_NORMAL, TRAIN_N_ATTACK
)
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

EXPERIMENT_ID = "EXP_BASE_MODELS_V1"
TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"
FEATURES_PATH = ROOT / "results" / "feature_selection" / "EXP_MI_V1_1" / "selected_features.json"
OUTPUT_DIR = ROOT / "results" / "base_models" / EXPERIMENT_ID
CHECKPOINT_ROOT = ROOT / "results" / "checkpoints" / EXPERIMENT_ID
SELECTED_CONFIGS_PATH = OUTPUT_DIR / "selected_configs.json"


def main(models_to_run: list[str]):
    set_all_seeds(42)
    logger = get_experiment_logger(
        f"{EXPERIMENT_ID}_REFIT",
        log_dir=str(ROOT / "results" / "logs"),
    )
    t_global = time.perf_counter()
    logger.info("=== REFIT START | experiment=%s | models=%s | timestamp=%s ===",
                EXPERIMENT_ID, models_to_run, datetime.now(timezone.utc).isoformat())

    # Load selected configs
    if not SELECTED_CONFIGS_PATH.exists():
        raise FileNotFoundError(
            f"selected_configs.json not found: {SELECTED_CONFIGS_PATH}. "
            "Run run_base_models_tuning.py first."
        )
    selected_configs = json.loads(SELECTED_CONFIGS_PATH.read_text())
    logger.info("Loaded selected configs for models: %s", list(selected_configs.keys()))

    # Load and preprocess TRAIN (fresh fit)
    logger.info("Loading TRAIN: %s", TRAIN_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    train_sha256 = sha256_file(TRAIN_PATH)
    logger.info("TRAIN | shape=%s | SHA256=%s", train_df.shape, train_sha256)

    logger.info("Fitting fresh preprocessing pipeline on TRAIN ...")
    pipe = PreprocessingPipeline(experiment_id=EXPERIMENT_ID)
    pipe.fit(train_df)
    ds_unscaled = pipe.transform(train_df, view="unscaled", split_name="train")

    features = load_selected_features(FEATURES_PATH)
    feature_df = pd.DataFrame(ds_unscaled.X, columns=ds_unscaled.feature_names)
    X = build_feature_matrix(feature_df, features)
    y = ds_unscaled.y.to_numpy(dtype=int)
    logger.info("Feature matrix | shape=%s", X.shape)
    assert X.shape == (162_395, 75), f"Unexpected X shape: {X.shape}"

    pos_weight = TRAIN_N_NORMAL / TRAIN_N_ATTACK

    # ------------------------------------------------------------------
    # DT refit
    # ------------------------------------------------------------------
    if "dt" in models_to_run and "dt" in selected_configs:
        logger.info("--- DT REFIT ---")
        cfg = selected_configs["dt"]["config"]
        t0 = time.perf_counter()
        dt_clf = refit_dt(X, y, cfg)
        dt_time = time.perf_counter() - t0

        ckpt_dir = CHECKPOINT_ROOT / "dt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(dt_clf, ckpt_dir / "dt_final.joblib")
        _save_metadata(ckpt_dir / "dt_metadata.json", "dt", cfg,
                       selected_configs["dt"], train_sha256, dt_time, {
                           "n_features_in_": dt_clf.n_features_in_,
                           "max_depth": dt_clf.max_depth,
                       })
        logger.info("DT checkpoint saved | runtime=%.2fs", dt_time)

    # ------------------------------------------------------------------
    # RF refit
    # ------------------------------------------------------------------
    if "rf" in models_to_run and "rf" in selected_configs:
        logger.info("--- RF REFIT ---")
        cfg = selected_configs["rf"]["config"]
        t0 = time.perf_counter()
        rf_clf = refit_rf(X, y, cfg)
        rf_time = time.perf_counter() - t0

        ckpt_dir = CHECKPOINT_ROOT / "rf"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf_clf, ckpt_dir / "rf_final.joblib")
        _save_metadata(ckpt_dir / "rf_metadata.json", "rf", cfg,
                       selected_configs["rf"], train_sha256, rf_time, {
                           "n_features_in_": rf_clf.n_features_in_,
                           "n_estimators": rf_clf.n_estimators,
                       })
        logger.info("RF checkpoint saved | runtime=%.2fs", rf_time)

    # ------------------------------------------------------------------
    # SVM refit
    # ------------------------------------------------------------------
    if "svm" in models_to_run and "svm" in selected_configs:
        logger.info("--- SVM REFIT ---")
        cfg = selected_configs["svm"]["config"]
        t0 = time.perf_counter()
        svm_clf, svm_scaler = refit_svm(X, y, cfg)
        svm_time = time.perf_counter() - t0

        ckpt_dir = CHECKPOINT_ROOT / "svm"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(svm_clf, ckpt_dir / "svm_final.joblib")
        joblib.dump(svm_scaler, ckpt_dir / "svm_scaler.joblib")
        _save_metadata(ckpt_dir / "svm_metadata.json", "svm", cfg,
                       selected_configs["svm"], train_sha256, svm_time, {
                           "output_contract": "predict + decision_function (NOT predict_proba)",
                           "scaler": "StandardScaler fitted on full frozen TRAIN",
                           "C": svm_clf.C,
                       })
        logger.info("SVM checkpoint saved | runtime=%.2fs", svm_time)

    # ------------------------------------------------------------------
    # NN refit
    # ------------------------------------------------------------------
    if "nn" in models_to_run and "nn" in selected_configs:
        logger.info("--- NN REFIT ---")
        nn_info = selected_configs["nn"]
        cfg = nn_info["config"]
        final_epoch_count = nn_info.get("final_epoch_count")
        if final_epoch_count is None:
            raise ValueError(
                "NN final_epoch_count is None in selected_configs.json. "
                "Re-run tuning to populate median(best_epoch)."
            )
        final_epoch_count = int(final_epoch_count)
        logger.info("NN refit | final_epoch_count=%d | config=%s", final_epoch_count, cfg)

        t0 = time.perf_counter()
        nn_net, nn_scaler = refit_nn(X, y, cfg, final_epoch_count, pos_weight_value=pos_weight)
        nn_time = time.perf_counter() - t0

        ckpt_dir = CHECKPOINT_ROOT / "nn"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(nn_net.state_dict(), ckpt_dir / "nn_final.pt")
        joblib.dump(nn_scaler, ckpt_dir / "nn_scaler.joblib")

        # Also save architecture config for re-loading
        arch_cfg = {
            "input_dim": 75,
            "hidden_sizes": cfg["hidden_sizes"],
        }
        (ckpt_dir / "nn_architecture.json").write_text(
            json.dumps(arch_cfg, indent=2), encoding="utf-8"
        )

        _save_metadata(ckpt_dir / "nn_metadata.json", "nn", cfg,
                       nn_info, train_sha256, nn_time, {
                           "final_epoch_count": final_epoch_count,
                           "pos_weight": pos_weight,
                           "pos_weight_source": f"N_normal({TRAIN_N_NORMAL}) / N_attack({TRAIN_N_ATTACK})",
                           "early_stopping": "NO (final refit trains fixed epochs)",
                           "scaler": "StandardScaler fitted on full frozen TRAIN",
                           "architecture": cfg["hidden_sizes"],
                           "output_contract": "predict + sigmoid probability in [0,1]",
                       })
        logger.info("NN checkpoint saved | runtime=%.2fs", nn_time)

    total_time = time.perf_counter() - t_global
    logger.info("=== REFIT COMPLETE | total_runtime=%.2fs ===", total_time)
    print(f"\n[REFIT COMPLETE] Total runtime: {total_time:.2f}s")
    print(f"Checkpoints saved to: {CHECKPOINT_ROOT}")


def _save_metadata(path, model_type, config, cv_info, train_sha256, runtime, extra):
    meta = {
        "experiment_id": EXPERIMENT_ID,
        "model_type": model_type,
        "sprint": 5,
        "config": config,
        "cv_mean_macro_f1": cv_info.get("mean_macro_f1"),
        "cv_std_macro_f1": cv_info.get("std_macro_f1"),
        "train_sha256": train_sha256,
        "refit_runtime_seconds": round(runtime, 3),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "REFIT_COMPLETE (not yet FROZEN)",
        **extra,
    }
    Path(path).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint 5 base model final refit.")
    parser.add_argument(
        "--models", type=str, default="dt,rf,svm,nn",
        help="Comma-separated list of models to refit (default: dt,rf,svm,nn)"
    )
    args = parser.parse_args()
    models_to_run = [m.strip().lower() for m in args.models.split(",")]
    valid = {"dt", "rf", "svm", "nn"}
    for m in models_to_run:
        if m not in valid:
            print(f"ERROR: Unknown model '{m}'. Valid: {sorted(valid)}")
            sys.exit(1)
    main(models_to_run)
