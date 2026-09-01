"""
scripts/run_base_models_benchmark.py
--------------------------------------
Runtime benchmark for Sprint 5 base models.

Purpose
-------
Estimate the total runtime of the hyperparameter tuning grids BEFORE
running the full search.

Benchmark approach
------------------
For each model family, run ONE representative configuration on a subsample
of TRAIN (up to BENCHMARK_N_ROWS rows, stratified) with BENCHMARK_N_FOLDS folds.
Extrapolate to the full grid runtime.

Benchmark does NOT affect:
  - selected hyperparameters
  - model architecture
  - feature selection
  - evaluation results

If estimated total runtime exceeds RUNTIME_WARNING_MINUTES, the script STOPS.

Usage
-----
    python scripts/run_base_models_benchmark.py

Outputs
-------
    results/base_models/EXP_BASE_MODELS_V1/runtime_report.json (benchmark section)
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logging_utils import get_experiment_logger
from src.utils.hashing import sha256_file
from src.utils.reproducibility import set_all_seeds
from src.models.base_models.preprocessing import load_selected_features, build_feature_matrix, fit_scaler
from src.models.base_models.decision_tree import DT_BASELINE_CONFIG, _build_dt
from src.models.base_models.random_forest import RF_BASELINE_CONFIG, _build_rf
from src.models.base_models.linear_svc import SVM_BASELINE_CONFIG, _build_svm
from src.models.base_models.cv_utils import make_model_skf
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

EXPERIMENT_ID = "EXP_BASE_MODELS_V1"
TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"
FEATURES_PATH = ROOT / "results" / "feature_selection" / "EXP_MI_V1_1" / "selected_features.json"
OUTPUT_DIR = ROOT / "results" / "base_models" / EXPERIMENT_ID
RUNTIME_REPORT = OUTPUT_DIR / "runtime_report.json"

BENCHMARK_N_ROWS = 10_000
BENCHMARK_N_FOLDS = 2
RUNTIME_WARNING_MINUTES = 120.0


def _stratified_subsample(X, y, n, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        n_cls = min(int(n * cls_idx.shape[0] / len(y)), len(cls_idx))
        idx.extend(rng.choice(cls_idx, size=n_cls, replace=False).tolist())
    return sorted(idx)


def main():
    set_all_seeds(42)
    logger = get_experiment_logger(
        f"{EXPERIMENT_ID}_BENCHMARK",
        log_dir=str(ROOT / "results" / "logs"),
    )
    logger.info("=== BENCHMARK START | experiment=%s ===", EXPERIMENT_ID)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and preprocess TRAIN
    logger.info("Loading TRAIN: %s", TRAIN_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    train_sha256 = sha256_file(TRAIN_PATH)
    logger.info("TRAIN shape=%s | SHA256=%s", train_df.shape, train_sha256)

    logger.info("Fitting preprocessing pipeline ...")
    pipe = PreprocessingPipeline(experiment_id=EXPERIMENT_ID)
    pipe.fit(train_df)
    ds_unscaled = pipe.transform(train_df, view="unscaled", split_name="train")

    features = load_selected_features(FEATURES_PATH)
    feature_df = pd.DataFrame(ds_unscaled.X, columns=ds_unscaled.feature_names)
    X_full = build_feature_matrix(feature_df, features)
    y_full = ds_unscaled.y.to_numpy(dtype=int)
    logger.info("Feature matrix | shape=%s", X_full.shape)

    # Subsample for benchmark
    sample_idx = _stratified_subsample(X_full, y_full, BENCHMARK_N_ROWS)
    X_bench = X_full[sample_idx]
    y_bench = y_full[sample_idx]
    logger.info("Benchmark subsample | n=%d", len(y_bench))

    # Get benchmark folds
    skf = make_model_skf()
    bench_folds = list(skf.split(X_bench, y_bench))[:BENCHMARK_N_FOLDS]
    scale_to_full = len(y_full) / len(y_bench)
    scale_to_all_folds = 5 / BENCHMARK_N_FOLDS

    benchmark_results = {}

    # ---- Decision Tree ----
    logger.info("Benchmarking DT ...")
    t0 = time.perf_counter()
    for tr_idx, va_idx in bench_folds:
        clf = _build_dt({**DT_BASELINE_CONFIG, "max_depth": 10})
        clf.fit(X_bench[tr_idx], y_bench[tr_idx])
        clf.predict(X_bench[va_idx])
    dt_bench = time.perf_counter() - t0
    dt_est = dt_bench * scale_to_full * scale_to_all_folds * 24
    benchmark_results["dt"] = {
        "benchmark_n_rows": len(y_bench), "benchmark_n_folds": BENCHMARK_N_FOLDS,
        "benchmark_runtime_seconds": round(dt_bench, 3),
        "estimated_full_tuning_seconds": round(dt_est, 1),
        "estimated_full_tuning_minutes": round(dt_est / 60, 2), "grid_size": 24,
    }
    logger.info("DT | bench=%.2fs | est=%.1fmin", dt_bench, dt_est / 60)

    # ---- Random Forest ----
    logger.info("Benchmarking RF ...")
    t0 = time.perf_counter()
    for tr_idx, va_idx in bench_folds:
        clf = _build_rf({**RF_BASELINE_CONFIG, "n_estimators": 100, "max_depth": 10})
        clf.fit(X_bench[tr_idx], y_bench[tr_idx])
        clf.predict(X_bench[va_idx])
    rf_bench = time.perf_counter() - t0
    rf_est = rf_bench * scale_to_full * scale_to_all_folds * 24
    benchmark_results["rf"] = {
        "benchmark_n_rows": len(y_bench), "benchmark_n_folds": BENCHMARK_N_FOLDS,
        "benchmark_runtime_seconds": round(rf_bench, 3),
        "estimated_full_tuning_seconds": round(rf_est, 1),
        "estimated_full_tuning_minutes": round(rf_est / 60, 2), "grid_size": 24,
    }
    logger.info("RF | bench=%.2fs | est=%.1fmin", rf_bench, rf_est / 60)

    # ---- LinearSVC ----
    logger.info("Benchmarking SVM ...")
    t0 = time.perf_counter()
    for tr_idx, va_idx in bench_folds:
        sc = fit_scaler(X_bench[tr_idx])
        X_tr_s = sc.transform(X_bench[tr_idx])
        X_va_s = sc.transform(X_bench[va_idx])
        clf = _build_svm(SVM_BASELINE_CONFIG)
        clf.fit(X_tr_s, y_bench[tr_idx])
        clf.predict(X_va_s)
    svm_bench = time.perf_counter() - t0
    svm_est = svm_bench * scale_to_full * scale_to_all_folds * 4
    benchmark_results["svm"] = {
        "benchmark_n_rows": len(y_bench), "benchmark_n_folds": BENCHMARK_N_FOLDS,
        "benchmark_runtime_seconds": round(svm_bench, 3),
        "estimated_full_tuning_seconds": round(svm_est, 1),
        "estimated_full_tuning_minutes": round(svm_est / 60, 2), "grid_size": 4,
    }
    logger.info("SVM | bench=%.2fs | est=%.1fmin", svm_bench, svm_est / 60)

    # ---- Neural Network ----
    logger.info("Benchmarking NN ...")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from src.models.base_models.neural_network import IDSNet, TRAIN_POS_WEIGHT, NN_BATCH_SIZE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("NN device: %s", device)

        t0 = time.perf_counter()
        for tr_idx, va_idx in bench_folds:
            sc = fit_scaler(X_bench[tr_idx])
            X_tr_s = sc.transform(X_bench[tr_idx])
            net = IDSNet(75, [128, 64]).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
            pw = torch.tensor([TRAIN_POS_WEIGHT], dtype=torch.float32, device=device)
            crit = nn.BCEWithLogitsLoss(pos_weight=pw)
            X_t = torch.tensor(X_tr_s, dtype=torch.float32)
            y_t = torch.tensor(y_bench[tr_idx], dtype=torch.float32)
            dl = DataLoader(TensorDataset(X_t, y_t), batch_size=NN_BATCH_SIZE, shuffle=True)
            for _ in range(3):
                net.train()
                for Xb, yb in dl:
                    Xb, yb = Xb.to(device), yb.to(device)
                    opt.zero_grad()
                    crit(net(Xb), yb).backward()
                    opt.step()
        nn_bench = time.perf_counter() - t0
        # Assume ~15 epochs with early stopping
        nn_est = nn_bench * (15 / 3) * scale_to_full * scale_to_all_folds * 8
        benchmark_results["nn"] = {
            "benchmark_n_rows": len(y_bench), "benchmark_n_folds": BENCHMARK_N_FOLDS,
            "benchmark_epochs": 3, "device": str(device),
            "benchmark_runtime_seconds": round(nn_bench, 3),
            "estimated_full_tuning_seconds": round(nn_est, 1),
            "estimated_full_tuning_minutes": round(nn_est / 60, 2), "grid_size": 8,
        }
        logger.info("NN | bench=%.2fs | device=%s | est=%.1fmin", nn_bench, device, nn_est / 60)
    except Exception as exc:
        logger.warning("NN benchmark failed: %s", exc)
        benchmark_results["nn"] = {"error": str(exc), "estimated_full_tuning_minutes": 0}

    # ------------------------------------------------------------------
    # Total and warning check
    # ------------------------------------------------------------------
    total_minutes = sum(
        r.get("estimated_full_tuning_minutes", 0)
        for r in benchmark_results.values()
    )
    logger.info("Total estimated tuning: %.1f minutes", total_minutes)

    within = total_minutes <= RUNTIME_WARNING_MINUTES
    if not within:
        msg = (
            f"RUNTIME WARNING: Estimated total {total_minutes:.1f} min "
            f"exceeds threshold {RUNTIME_WARNING_MINUTES:.1f} min. "
            "STOPPING. Review before running full tuning."
        )
        logger.warning(msg)
        print(f"\n[RUNTIME WARNING]\n{msg}\n")
    else:
        logger.info("Runtime WITHIN threshold. Proceeding with tuning is approved.")

    # Save
    import platform, torch as _torch
    report = {
        "experiment_id": EXPERIMENT_ID,
        "benchmark_n_rows": BENCHMARK_N_ROWS,
        "benchmark_n_folds": BENCHMARK_N_FOLDS,
        "train_sha256": train_sha256,
        "models": benchmark_results,
        "total_estimated_minutes": round(total_minutes, 2),
        "runtime_warning_threshold_minutes": RUNTIME_WARNING_MINUTES,
        "within_threshold": within,
        "environment": {
            "python": platform.python_version(),
            "torch": _torch.__version__,
            "cuda_available": _torch.cuda.is_available(),
        },
    }
    RUNTIME_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Runtime report saved: %s", RUNTIME_REPORT)
    logger.info("=== BENCHMARK COMPLETE ===")

    if not within:
        sys.exit(1)


if __name__ == "__main__":
    main()
