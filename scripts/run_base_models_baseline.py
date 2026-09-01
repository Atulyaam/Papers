"""
scripts/run_base_models_baseline.py
--------------------------------------
Phase A — Baseline CV for all four Sprint 5 base models.

This script MUST be run BEFORE tuning.
Baseline results are recorded independently of any tuning run.

Protocol
--------
- Loads frozen TRAIN (data/splits/train.csv)
- Fits fresh PreprocessingPipeline on TRAIN
- Selects 75 frozen MI features
- Runs 5-fold StratifiedKFold (seed=0) baseline CV for DT, RF, SVM, NN
- Records all fold metrics + mean/std
- Saves results to results/base_models/EXP_BASE_MODELS_V1/baseline_results.csv

Leakage guards
--------------
- NEVER reads validation.csv, development_test.csv, protected_unseen_attack.csv
- Scaler fitted on inner_train only (SVM/NN)
- NN early stopping uses inner_val only

Usage
-----
    python scripts/run_base_models_baseline.py

Outputs
-------
    results/base_models/EXP_BASE_MODELS_V1/baseline_results.csv
    results/base_models/EXP_BASE_MODELS_V1/fold_results.csv  (appended)
    results/logs/EXP_BASE_MODELS_V1_BASELINE/run.log
"""

import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logging_utils import get_experiment_logger
from src.utils.hashing import sha256_file
from src.utils.reproducibility import set_all_seeds
from src.models.base_models.preprocessing import load_selected_features, build_feature_matrix
from src.models.base_models.decision_tree import run_dt_baseline
from src.models.base_models.random_forest import run_rf_baseline
from src.models.base_models.linear_svc import run_svm_baseline
from src.models.base_models.neural_network import (
    run_nn_baseline, TRAIN_POS_WEIGHT, TRAIN_N_NORMAL, TRAIN_N_ATTACK
)
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

EXPERIMENT_ID = "EXP_BASE_MODELS_V1"
TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"
FEATURES_PATH = ROOT / "results" / "feature_selection" / "EXP_MI_V1_1" / "selected_features.json"
OUTPUT_DIR = ROOT / "results" / "base_models" / EXPERIMENT_ID


def main():
    set_all_seeds(42)
    logger = get_experiment_logger(
        f"{EXPERIMENT_ID}_BASELINE",
        log_dir=str(ROOT / "results" / "logs"),
    )
    t_global = time.perf_counter()
    logger.info("=== BASELINE START | experiment=%s | timestamp=%s ===",
                EXPERIMENT_ID, datetime.now(timezone.utc).isoformat())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load TRAIN
    # ------------------------------------------------------------------
    logger.info("Loading TRAIN: %s", TRAIN_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    train_sha256 = sha256_file(TRAIN_PATH)
    logger.info("TRAIN | shape=%s | SHA256=%s", train_df.shape, train_sha256)
    assert len(train_df) == 162_395, f"Expected 162395 rows, got {len(train_df)}"

    # ------------------------------------------------------------------
    # Fresh preprocessing fit on TRAIN
    # ------------------------------------------------------------------
    logger.info("Fitting preprocessing pipeline on TRAIN ...")
    pipe = PreprocessingPipeline(experiment_id=EXPERIMENT_ID)
    pipe.fit(train_df)
    ds_unscaled = pipe.transform(train_df, view="unscaled", split_name="train")
    logger.info("Preprocessing done | n_encoded_features=%d", ds_unscaled.n_features)

    # ------------------------------------------------------------------
    # Build feature matrix (75 frozen MI features)
    # ------------------------------------------------------------------
    features = load_selected_features(FEATURES_PATH)
    assert len(features) == 75
    feature_df = pd.DataFrame(ds_unscaled.X, columns=ds_unscaled.feature_names)
    X = build_feature_matrix(feature_df, features)
    y = ds_unscaled.y.to_numpy(dtype=int)
    logger.info("Feature matrix | shape=%s | classes=%s", X.shape, dict(zip(*np.unique(y, return_counts=True))))

    pos_weight = TRAIN_N_NORMAL / TRAIN_N_ATTACK
    logger.info("pos_weight (fixed from frozen TRAIN) = %.8f (N_normal=%d / N_attack=%d)",
                pos_weight, TRAIN_N_NORMAL, TRAIN_N_ATTACK)
    assert abs(pos_weight - TRAIN_POS_WEIGHT) < 1e-10

    # ------------------------------------------------------------------
    # Baseline CV — Decision Tree
    # ------------------------------------------------------------------
    logger.info("--- DT BASELINE ---")
    t0 = time.perf_counter()
    dt_result = run_dt_baseline(X, y)
    dt_time = time.perf_counter() - t0
    logger.info("DT BASELINE | mean_f1=%.6f | std_f1=%.6f | runtime=%.2fs",
                dt_result.mean_macro_f1, dt_result.std_macro_f1, dt_time)

    # ------------------------------------------------------------------
    # Baseline CV — Random Forest
    # ------------------------------------------------------------------
    logger.info("--- RF BASELINE ---")
    t0 = time.perf_counter()
    rf_result = run_rf_baseline(X, y)
    rf_time = time.perf_counter() - t0
    logger.info("RF BASELINE | mean_f1=%.6f | std_f1=%.6f | runtime=%.2fs",
                rf_result.mean_macro_f1, rf_result.std_macro_f1, rf_time)

    # ------------------------------------------------------------------
    # Baseline CV — LinearSVC
    # ------------------------------------------------------------------
    logger.info("--- SVM BASELINE ---")
    t0 = time.perf_counter()
    svm_result = run_svm_baseline(X, y)
    svm_time = time.perf_counter() - t0
    logger.info("SVM BASELINE | mean_f1=%.6f | std_f1=%.6f | runtime=%.2fs",
                svm_result.mean_macro_f1, svm_result.std_macro_f1, svm_time)

    # ------------------------------------------------------------------
    # Baseline CV — Neural Network
    # ------------------------------------------------------------------
    logger.info("--- NN BASELINE ---")
    t0 = time.perf_counter()
    nn_result, nn_diag = run_nn_baseline(X, y, pos_weight_value=pos_weight)
    nn_time = time.perf_counter() - t0
    logger.info("NN BASELINE | mean_f1=%.6f | std_f1=%.6f | median_best_epoch=%d | runtime=%.2fs",
                nn_result.mean_macro_f1, nn_result.std_macro_f1,
                nn_diag.final_epoch_count, nn_time)

    # ------------------------------------------------------------------
    # Save baseline_results.csv
    # ------------------------------------------------------------------
    rows = [
        _summary_to_row(dt_result, "DT", "baseline", dt_time),
        _summary_to_row(rf_result, "RF", "baseline", rf_time),
        _summary_to_row(svm_result, "SVM", "baseline", svm_time),
        _summary_to_row(nn_result, "NN", "baseline", nn_time),
    ]
    baseline_path = OUTPUT_DIR / "baseline_results.csv"
    _write_csv(rows, baseline_path)
    logger.info("Baseline results saved: %s", baseline_path)

    # ------------------------------------------------------------------
    # Save fold_results.csv
    # ------------------------------------------------------------------
    fold_rows = []
    for result in [dt_result, rf_result, svm_result, nn_result]:
        for fm in result.folds:
            fold_rows.append(_fold_to_row(result.model_type, "baseline", fm))
    fold_path = OUTPUT_DIR / "fold_results.csv"
    _write_csv(fold_rows, fold_path, mode="a")
    logger.info("Fold results saved: %s", fold_path)

    # NN epoch diagnostics for baseline
    nn_epoch_path = OUTPUT_DIR / "nn_epoch_diagnostics.json"
    diag_data = {}
    if nn_epoch_path.exists():
        diag_data = json.loads(nn_epoch_path.read_text())
    diag_data["baseline"] = nn_diag.to_dict()
    nn_epoch_path.write_text(json.dumps(diag_data, indent=2), encoding="utf-8")
    logger.info("NN epoch diagnostics saved: %s", nn_epoch_path)

    total_time = time.perf_counter() - t_global
    logger.info("=== BASELINE COMPLETE | total_runtime=%.2fs ===", total_time)

    # Summary print
    print("\n=== BASELINE RESULTS ===")
    print(f"{'Model':<8} {'Mean F1':>10} {'Std F1':>10} {'Bal Acc':>10} {'Runtime(s)':>12}")
    print("-" * 55)
    for model, result in [("DT", dt_result), ("RF", rf_result), ("SVM", svm_result), ("NN", nn_result)]:
        rt = {"DT": dt_time, "RF": rf_time, "SVM": svm_time, "NN": nn_time}[model]
        print(f"{model:<8} {result.mean_macro_f1:>10.6f} {result.std_macro_f1:>10.6f} "
              f"{result.mean_balanced_accuracy:>10.6f} {rt:>12.2f}")


def _summary_to_row(result, label, phase, runtime):
    return {
        "model": label,
        "phase": phase,
        "mean_macro_f1": result.mean_macro_f1,
        "std_macro_f1": result.std_macro_f1,
        "mean_weighted_f1": result.mean_weighted_f1,
        "std_weighted_f1": result.std_weighted_f1,
        "mean_balanced_accuracy": result.mean_balanced_accuracy,
        "mean_precision_macro": result.mean_precision_macro,
        "mean_recall_macro": result.mean_recall_macro,
        "mean_fpr": result.mean_fpr,
        "mean_specificity": result.mean_specificity,
        "total_runtime_seconds": runtime,
        "config": json.dumps(result.config),
    }


def _fold_to_row(model_type, phase, fm):
    return {
        "model": model_type,
        "phase": phase,
        "fold_idx": fm.fold_idx,
        "macro_f1": fm.macro_f1,
        "weighted_f1": fm.weighted_f1,
        "balanced_accuracy": fm.balanced_accuracy,
        "precision_macro": fm.precision_macro,
        "recall_macro": fm.recall_macro,
        "fpr": fm.fpr,
        "specificity": fm.specificity,
        "runtime_seconds": fm.runtime_seconds,
        "n_train": fm.n_train,
        "n_val": fm.n_val,
    }


def _write_csv(rows, path, mode="w"):
    if not rows:
        return
    path = Path(path)
    write_header = mode == "w" or not path.exists()
    with open(path, mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
