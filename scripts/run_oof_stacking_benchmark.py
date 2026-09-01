"""
scripts/run_oof_stacking_benchmark.py
---------------------------------------
Sprint 6 OOF stacking runtime benchmark.

Runs a representative subset:
    - 10,000 TRAIN rows (stratified subsample)
    - 2-fold OOF
    - 1 H1 seed (42)

Extrapolates to estimate full 3-seed runtime.
Writes results/stacking/EXP_OOF_STACK_V1/runtime_report.json.

CRITICAL: Does NOT read validation, development_test, protected_unseen_attack,
or excluded_train_backdoor data.
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import sklearn
from src.models.base_models.preprocessing import (
    load_selected_features,
    build_feature_matrix,
)
from src.models.stacking.oof_runner import (
    OOF_SEED,
    OOF_FIXED_EPOCH_COUNT,
    make_oof_folds,
    run_oof_seed,
)
from src.models.stacking.meta_learner import META_CONFIG
from src.models.stacking.artifacts import save_runtime_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("benchmark")

# Paths
TRAIN_PATH = ROOT / "data/splits/train.csv"
FEATURES_PATH = ROOT / "results/feature_selection/EXP_MI_V1_1/selected_features.json"
SVM_SCALER_PATH = ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_scaler.joblib"
NN_SCALER_PATH  = ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_scaler.joblib"
RESULTS_DIR = ROOT / "results/stacking/EXP_OOF_STACK_V1"

from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline  # noqa: E402

BENCHMARK_N = 10_000
BENCHMARK_FOLDS = 2
BENCHMARK_SEED = 42


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    logger.info("=== OOF STACKING BENCHMARK START ===")
    t_total = time.perf_counter()

    # Load full TRAIN (needed to fit pipeline for complete OHE)
    logger.info("Loading full TRAIN (%s) ...", TRAIN_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    logger.info("TRAIN shape: %s", train_df.shape)

    # Fit preprocessing pipeline on full TRAIN
    pipe = PreprocessingPipeline(experiment_id="EXP_OOF_STACK_V1")
    pipe.fit(train_df)
    ds_unscaled = pipe.transform(train_df, view="unscaled", split_name="train")
    encoded_df = pd.DataFrame(ds_unscaled.X, columns=ds_unscaled.feature_names)

    # Load features
    features = load_selected_features(FEATURES_PATH)
    X_full = build_feature_matrix(encoded_df, features)
    y_full = train_df["label"].to_numpy(dtype=np.int64)

    # Subsample 10,000 rows stratified
    logger.info("Subsampling %d rows for benchmark ...", BENCHMARK_N)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=None,
                                  train_size=BENCHMARK_N, random_state=0)
    sub_idx, _ = next(sss.split(X_full, y_full))
    X_bench = X_full[sub_idx]
    y_bench = y_full[sub_idx]
    logger.info("Benchmark subsample | shape=%s | class_dist=%s",
                X_bench.shape,
                dict(zip(*np.unique(y_bench, return_counts=True))))

    # Load frozen Sprint 5 scalers
    logger.info("Loading frozen Sprint 5 scalers ...")
    svm_scaler = joblib.load(SVM_SCALER_PATH)
    nn_scaler = joblib.load(NN_SCALER_PATH)

    # Create 2-fold benchmark folds
    folds = make_oof_folds(y_bench, n_splits=BENCHMARK_FOLDS, seed=OOF_SEED)

    # Run one seed
    logger.info("Running benchmark OOF (seed=%d, folds=%d) ...", BENCHMARK_SEED, BENCHMARK_FOLDS)
    t0 = time.perf_counter()
    _ = run_oof_seed(
        h1_seed=BENCHMARK_SEED,
        folds=folds,
        X_unscaled=X_bench,
        y=y_bench,
        svm_scaler=svm_scaler,
        nn_scaler=nn_scaler,
    )
    benchmark_elapsed = time.perf_counter() - t0

    # Extrapolate
    scale_n = 162_395 / BENCHMARK_N
    scale_folds = 5 / BENCHMARK_FOLDS
    extrapolated_per_seed = benchmark_elapsed * scale_n * scale_folds
    extrapolated_3seeds = extrapolated_per_seed * 3
    threshold_minutes = 240.0
    within_threshold = (extrapolated_3seeds / 60) <= threshold_minutes

    logger.info(
        "Benchmark | elapsed=%.2fs | extrapolated_per_seed=%.2fs | "
        "extrapolated_3seeds=%.2fmin | within_%dmin=%s",
        benchmark_elapsed,
        extrapolated_per_seed,
        extrapolated_3seeds / 60,
        int(threshold_minutes),
        within_threshold,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    report = {
        "experiment_id": "EXP_OOF_STACK_V1",
        "benchmark": {
            "n_rows": BENCHMARK_N,
            "n_folds": BENCHMARK_FOLDS,
            "h1_seed": BENCHMARK_SEED,
            "elapsed_seconds": benchmark_elapsed,
        },
        "extrapolation": {
            "full_n_rows": 162_395,
            "full_n_folds": 5,
            "full_n_seeds": 3,
            "estimated_per_seed_seconds": extrapolated_per_seed,
            "estimated_per_seed_minutes": extrapolated_per_seed / 60,
            "estimated_3seed_minutes": extrapolated_3seeds / 60,
            "threshold_minutes": threshold_minutes,
            "within_threshold": within_threshold,
        },
        "environment": {
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": device,
            "gpu": gpu_name,
        },
        "total_benchmark_runtime_seconds": time.perf_counter() - t_total,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_runtime_report(report, RESULTS_DIR)

    print(f"\nBenchmark elapsed: {benchmark_elapsed:.1f}s")
    print(f"Estimated per-seed (full): {extrapolated_per_seed / 60:.1f} min")
    print(f"Estimated total (3 seeds): {extrapolated_3seeds / 60:.1f} min")
    print(f"Within {int(threshold_minutes)}-min threshold: {within_threshold}")
    logger.info("=== OOF STACKING BENCHMARK DONE ===")


if __name__ == "__main__":
    main()
