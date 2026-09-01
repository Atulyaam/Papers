"""
scripts/run_base_models_tuning.py
-----------------------------------
Phase B — Hyperparameter tuning CV for all four Sprint 5 base models.

Protocol
--------
- Loads frozen TRAIN (data/splits/train.csv)
- Fits fresh PreprocessingPipeline on TRAIN
- Runs 5-fold StratifiedKFold tuning grid for DT, RF, SVM, NN
- Selects best config per model using comparator.compare_model_configs
- Saves all grid results + selected best configs

DT  grid: 24 configurations  (2 × 4 × 3)
RF  grid: 24 configurations  (2 × 3 × 2 × 2)
SVM grid:  4 configurations  (4 C values)
NN  grid:  8 configurations  (2 × 2 × 2)

Leakage guards
--------------
- TRAIN only (no val/test seen)
- Scaler fitted on inner_train only (SVM/NN)
- NN early stopping uses inner_val only

Usage
-----
    python scripts/run_base_models_tuning.py [--models dt,rf,svm,nn]

Optional flag ``--models`` allows running a subset (e.g. for resuming).

Outputs
-------
    results/base_models/EXP_BASE_MODELS_V1/tuning_results.csv
    results/base_models/EXP_BASE_MODELS_V1/selected_configs.json
    results/base_models/EXP_BASE_MODELS_V1/nn_epoch_diagnostics.json
    results/base_models/EXP_BASE_MODELS_V1/fold_results.csv  (appended)
"""

import argparse
import csv
import json
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
from src.models.base_models.comparator import compare_model_configs
from src.models.base_models.decision_tree import run_dt_tuning
from src.models.base_models.random_forest import run_rf_tuning
from src.models.base_models.linear_svc import run_svm_tuning
from src.models.base_models.neural_network import (
    run_nn_tuning, TRAIN_POS_WEIGHT, TRAIN_N_NORMAL, TRAIN_N_ATTACK
)
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

EXPERIMENT_ID = "EXP_BASE_MODELS_V1"
TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"
FEATURES_PATH = ROOT / "results" / "feature_selection" / "EXP_MI_V1_1" / "selected_features.json"
OUTPUT_DIR = ROOT / "results" / "base_models" / EXPERIMENT_ID


def select_best(results, model_type):
    """Return the best CVSummary from a list using compare_model_configs."""
    if not results:
        raise ValueError(f"Empty results list for model_type={model_type}")
    best = results[0]
    for candidate in results[1:]:
        if compare_model_configs(candidate, best, model_type) == 1:
            best = candidate
    return best


def main(models_to_run: list[str]):
    set_all_seeds(42)
    logger = get_experiment_logger(
        f"{EXPERIMENT_ID}_TUNING",
        log_dir=str(ROOT / "results" / "logs"),
    )
    t_global = time.perf_counter()
    logger.info("=== TUNING START | experiment=%s | models=%s | timestamp=%s ===",
                EXPERIMENT_ID, models_to_run, datetime.now(timezone.utc).isoformat())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and preprocess TRAIN
    # ------------------------------------------------------------------
    train_df = pd.read_csv(TRAIN_PATH)
    train_sha256 = sha256_file(TRAIN_PATH)
    logger.info("TRAIN | shape=%s | SHA256=%s", train_df.shape, train_sha256)

    pipe = PreprocessingPipeline(experiment_id=EXPERIMENT_ID)
    pipe.fit(train_df)
    ds_unscaled = pipe.transform(train_df, view="unscaled", split_name="train")

    features = load_selected_features(FEATURES_PATH)
    feature_df = pd.DataFrame(ds_unscaled.X, columns=ds_unscaled.feature_names)
    X = build_feature_matrix(feature_df, features)
    y = ds_unscaled.y.to_numpy(dtype=int)
    logger.info("Feature matrix | shape=%s", X.shape)

    pos_weight = TRAIN_N_NORMAL / TRAIN_N_ATTACK

    all_results = []
    all_fold_rows = []
    selected_configs = {}
    nn_epoch_diagnostics = {}

    # ------------------------------------------------------------------
    # DT tuning
    # ------------------------------------------------------------------
    if "dt" in models_to_run:
        logger.info("=== DT TUNING ===")
        t0 = time.perf_counter()
        dt_results = run_dt_tuning(X, y)
        dt_time = time.perf_counter() - t0
        for r in dt_results:
            all_results.append(_summary_to_row(r, "dt", "tuning", None))
            for fm in r.folds:
                all_fold_rows.append(_fold_to_row("dt", "tuning", fm))
        best_dt = select_best(dt_results, "dt")
        selected_configs["dt"] = {"config": best_dt.config, "mean_macro_f1": best_dt.mean_macro_f1,
                                   "std_macro_f1": best_dt.std_macro_f1}
        logger.info("DT BEST | mean_f1=%.6f | config=%s | runtime=%.2fs",
                    best_dt.mean_macro_f1, best_dt.config, dt_time)

    # ------------------------------------------------------------------
    # RF tuning
    # ------------------------------------------------------------------
    if "rf" in models_to_run:
        logger.info("=== RF TUNING ===")
        t0 = time.perf_counter()
        rf_results = run_rf_tuning(X, y)
        rf_time = time.perf_counter() - t0
        for r in rf_results:
            all_results.append(_summary_to_row(r, "rf", "tuning", None))
            for fm in r.folds:
                all_fold_rows.append(_fold_to_row("rf", "tuning", fm))
        best_rf = select_best(rf_results, "rf")
        selected_configs["rf"] = {"config": best_rf.config, "mean_macro_f1": best_rf.mean_macro_f1,
                                   "std_macro_f1": best_rf.std_macro_f1}
        logger.info("RF BEST | mean_f1=%.6f | config=%s | runtime=%.2fs",
                    best_rf.mean_macro_f1, best_rf.config, rf_time)

    # ------------------------------------------------------------------
    # SVM tuning
    # ------------------------------------------------------------------
    if "svm" in models_to_run:
        logger.info("=== SVM TUNING ===")
        t0 = time.perf_counter()
        svm_results = run_svm_tuning(X, y)
        svm_time = time.perf_counter() - t0
        for r in svm_results:
            all_results.append(_summary_to_row(r, "svm", "tuning", None))
            for fm in r.folds:
                all_fold_rows.append(_fold_to_row("svm", "tuning", fm))
        best_svm = select_best(svm_results, "svm")
        selected_configs["svm"] = {"config": best_svm.config, "mean_macro_f1": best_svm.mean_macro_f1,
                                    "std_macro_f1": best_svm.std_macro_f1}
        logger.info("SVM BEST | mean_f1=%.6f | config=%s | runtime=%.2fs",
                    best_svm.mean_macro_f1, best_svm.config, svm_time)

    # ------------------------------------------------------------------
    # NN tuning
    # ------------------------------------------------------------------
    if "nn" in models_to_run:
        logger.info("=== NN TUNING ===")
        t0 = time.perf_counter()
        nn_results_and_diags = run_nn_tuning(X, y, pos_weight_value=pos_weight)
        nn_time = time.perf_counter() - t0
        nn_results = [r for r, _ in nn_results_and_diags]
        for (r, diag) in nn_results_and_diags:
            all_results.append(_summary_to_row(r, "nn", "tuning", None))
            for fm in r.folds:
                all_fold_rows.append(_fold_to_row("nn", "tuning", fm))
            cfg_key = json.dumps(r.config, sort_keys=True)
            nn_epoch_diagnostics[cfg_key] = diag.to_dict()

            if diag.diagnostic_flag:
                logger.warning(
                    "NN DIAGNOSTIC FLAG: config=%s | range/median=%.3f | best_epochs=%s",
                    r.config, diag.range_median_ratio, diag.best_epochs
                )
        best_nn_result = select_best(nn_results, "nn")
        best_nn_diag = None
        for r, diag in nn_results_and_diags:
            if r.config == best_nn_result.config:
                best_nn_diag = diag
                break
        selected_configs["nn"] = {
            "config": best_nn_result.config,
            "mean_macro_f1": best_nn_result.mean_macro_f1,
            "std_macro_f1": best_nn_result.std_macro_f1,
            "final_epoch_count": best_nn_diag.final_epoch_count if best_nn_diag else None,
            "best_epochs": best_nn_diag.best_epochs if best_nn_diag else None,
            "diagnostic_flag": best_nn_diag.diagnostic_flag if best_nn_diag else None,
            "range_median_ratio": best_nn_diag.range_median_ratio if best_nn_diag else None,
        }
        logger.info("NN BEST | mean_f1=%.6f | config=%s | final_epochs=%s | runtime=%.2fs",
                    best_nn_result.mean_macro_f1, best_nn_result.config,
                    selected_configs["nn"]["final_epoch_count"], nn_time)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    # tuning_results.csv
    tuning_path = OUTPUT_DIR / "tuning_results.csv"
    _write_csv(all_results, tuning_path, mode="w")
    logger.info("Tuning results saved: %s | n_rows=%d", tuning_path, len(all_results))

    # fold_results.csv (append)
    fold_path = OUTPUT_DIR / "fold_results.csv"
    _write_csv(all_fold_rows, fold_path, mode="a")
    logger.info("Fold results appended: %s | n_rows=%d", fold_path, len(all_fold_rows))

    # selected_configs.json
    sel_path = OUTPUT_DIR / "selected_configs.json"
    sel_path.write_text(json.dumps(selected_configs, indent=2, default=str), encoding="utf-8")
    logger.info("Selected configs saved: %s", sel_path)

    # NN epoch diagnostics
    if nn_epoch_diagnostics:
        diag_path = OUTPUT_DIR / "nn_epoch_diagnostics.json"
        existing = {}
        if diag_path.exists():
            existing = json.loads(diag_path.read_text())
        existing.update({"tuning": nn_epoch_diagnostics})
        diag_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
        logger.info("NN epoch diagnostics saved: %s", diag_path)

    total_time = time.perf_counter() - t_global
    logger.info("=== TUNING COMPLETE | total_runtime=%.2fs ===", total_time)

    # Summary
    print("\n=== TUNING RESULTS — SELECTED CONFIGS ===")
    for model, info in selected_configs.items():
        print(f"\n{model.upper()}: mean_f1={info['mean_macro_f1']:.6f} ± {info['std_macro_f1']:.6f}")
        print(f"  Config: {info['config']}")
        if "final_epoch_count" in info and info["final_epoch_count"] is not None:
            print(f"  NN final_epoch_count: {info['final_epoch_count']}")
            if info.get("diagnostic_flag"):
                print(f"  NN DIAGNOSTIC FLAG: range/median={info.get('range_median_ratio', '?'):.3f} "
                      "— REVIEW REQUIRED (no automatic action taken)")


def _summary_to_row(result, model, phase, runtime):
    return {
        "model": model,
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
        "config": json.dumps(result.config, default=str),
    }


def _fold_to_row(model, phase, fm):
    return {
        "model": model, "phase": phase, "fold_idx": fm.fold_idx,
        "macro_f1": fm.macro_f1, "weighted_f1": fm.weighted_f1,
        "balanced_accuracy": fm.balanced_accuracy, "precision_macro": fm.precision_macro,
        "recall_macro": fm.recall_macro, "fpr": fm.fpr, "specificity": fm.specificity,
        "runtime_seconds": fm.runtime_seconds, "n_train": fm.n_train, "n_val": fm.n_val,
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
    parser = argparse.ArgumentParser(description="Sprint 5 base model tuning.")
    parser.add_argument(
        "--models", type=str, default="dt,rf,svm,nn",
        help="Comma-separated list of models to tune (default: dt,rf,svm,nn)"
    )
    args = parser.parse_args()
    models_to_run = [m.strip().lower() for m in args.models.split(",")]
    valid = {"dt", "rf", "svm", "nn"}
    for m in models_to_run:
        if m not in valid:
            print(f"ERROR: Unknown model '{m}'. Valid: {sorted(valid)}")
            sys.exit(1)
    main(models_to_run)
