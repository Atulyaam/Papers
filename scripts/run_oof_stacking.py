"""
scripts/run_oof_stacking.py
----------------------------
Sprint 6: Run OOF stacking for a single H1 seed.

Usage:
    python run_oof_stacking.py --seed 42
    python run_oof_stacking.py --seed 123
    python run_oof_stacking.py --seed 2024

Produces:
    results/stacking/EXP_OOF_STACK_V1/seed_{S}/
        oof_predictions.csv
        fold_assignments.csv
        metrics.json
        metadata.json
        logs/

    results/checkpoints/EXP_OOF_STACK_V1/seed_{S}/
        meta_learner.joblib
        metadata.json

DATA ACCESS:
    Reads ONLY: data/splits/train.csv
    FORBIDDEN:  validation.csv, development_test.csv,
                protected_unseen_attack.csv, excluded_train_backdoor.csv
"""

import argparse
import datetime
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.base_models.preprocessing import (
    load_selected_features,
    build_feature_matrix,
)
from src.models.stacking.oof_runner import (
    OOF_SEED,
    OOF_N_SPLITS,
    OOF_FIXED_EPOCH_COUNT,
    OOF_POS_WEIGHT,
    SCALING_LIMITATION_TEXT,
    H1_SEEDS,
    make_oof_folds,
    run_oof_seed,
)
from src.models.stacking.meta_learner import (
    META_CONFIG,
    META_EVALUATION_LIMITATION_TEXT,
    SPRINT5_RF_REFERENCE,
    SPRINT5_RF_REFERENCE_LABEL,
    train_meta_learner,
    compute_oof_metrics,
)
from src.models.stacking.artifacts import (
    save_oof_predictions,
    save_fold_assignments,
    save_seed_metrics,
    save_seed_metadata,
    save_meta_learner_checkpoint,
    save_meta_learner_metadata,
)
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRAIN_PATH   = ROOT / "data/splits/train.csv"
FEATURES_PATH = ROOT / "results/feature_selection/EXP_MI_V1_1/selected_features.json"
SVM_SCALER_PATH = ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_scaler.joblib"
NN_SCALER_PATH  = ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_scaler.joblib"

RESULTS_DIR  = ROOT / "results/stacking/EXP_OOF_STACK_V1"
CKPT_ROOT    = ROOT / "results/checkpoints/EXP_OOF_STACK_V1"

FROZEN_TRAIN_SHA = "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c"

# Resolved dataset paths (from Step 0)
RESOLVED_PATHS = {
    "train": {
        "path": str(ROOT / "data/splits/train.csv"),
        "sha256": "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c",
        "rows": 162395,
        "sprint6_access": "ALLOWED",
    },
    "validation": {
        "path": str(ROOT / "data/splits/validation.csv"),
        "sha256": "13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180",
        "rows": 11200,
        "sprint6_access": "FORBIDDEN",
    },
    "development_test": {
        "path": str(ROOT / "data/splits/development_test.csv"),
        "sha256": "04725e85732ab2fc6d9eaaa6105418b22b083b5c651067e7b0785464f414e508",
        "rows": 81749,
        "sprint6_access": "FORBIDDEN",
    },
    "protected_backdoor": {
        "path": str(ROOT / "data/splits/protected_unseen_attack.csv"),
        "sha256": "6ffd23479b575e438ad90678268f40f674a663c2b9507aaf65089623397a9d91",
        "rows": 583,
        "sprint6_access": "FORBIDDEN",
    },
    "excluded_backdoor": {
        "path": str(ROOT / "data/splits/excluded_train_backdoor.csv"),
        "sha256": "b3f6e7e60c9815a53f40eb2d41df8b67d29f884b922a487c3fe83c02e0db0a02",
        "rows": 1746,
        "sprint6_access": "FORBIDDEN",
    },
}


def _setup_logging(seed: int, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"oof_seed_{seed}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_train(path: Path) -> None:
    logger = logging.getLogger("verify")
    logger.info("Verifying TRAIN hash ...")
    actual = _sha256(path)
    if actual != FROZEN_TRAIN_SHA:
        raise RuntimeError(
            f"TRAIN SHA-256 mismatch!\n"
            f"  expected: {FROZEN_TRAIN_SHA}\n"
            f"  actual:   {actual}\n"
            f"  path:     {path}"
        )
    logger.info("TRAIN hash verified: %s", actual[:16] + "...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OOF stacking for one H1 seed.")
    parser.add_argument("--seed", type=int, required=True, choices=H1_SEEDS,
                        help="H1 seed: 42, 123, or 2024")
    args = parser.parse_args()
    h1_seed = args.seed

    seed_dir  = RESULTS_DIR / f"seed_{h1_seed}"
    ckpt_dir  = CKPT_ROOT / f"seed_{h1_seed}"
    log_dir   = seed_dir / "logs"
    _setup_logging(h1_seed, log_dir)
    logger = logging.getLogger("run_oof_stacking")

    t_total = time.perf_counter()
    logger.info("=== OOF STACKING | seed=%d | experiment=EXP_OOF_STACK_V1 ===", h1_seed)
    logger.info("SCALING LIMITATION: %s", SCALING_LIMITATION_TEXT)

    # ── Step 0: verify TRAIN ─────────────────────────────────────────────
    _verify_train(TRAIN_PATH)

    # ── Load TRAIN ────────────────────────────────────────────────────────
    logger.info("Loading TRAIN (%s) ...", TRAIN_PATH)
    t0 = time.perf_counter()
    train_df = pd.read_csv(TRAIN_PATH)
    logger.info("TRAIN loaded | shape=%s | time=%.2fs", train_df.shape, time.perf_counter() - t0)

    n_rows = len(train_df)
    class_counts = train_df["label"].value_counts().to_dict()
    logger.info("Class distribution: %s", class_counts)

    # ── Fit preprocessing pipeline on full TRAIN ─────────────────────────
    logger.info("Fitting preprocessing pipeline on full TRAIN ...")
    t0 = time.perf_counter()
    pipe = PreprocessingPipeline(experiment_id="EXP_OOF_STACK_V1")
    pipe.fit(train_df)
    ds_unscaled = pipe.transform(train_df, view="unscaled", split_name="train")
    encoded_df = pd.DataFrame(ds_unscaled.X, columns=ds_unscaled.feature_names)
    logger.info("Pipeline fit+transform done | time=%.2fs", time.perf_counter() - t0)

    # ── Load frozen 75 features ───────────────────────────────────────────
    logger.info("Loading frozen 75 features from EXP_MI_V1_1 ...")
    features = load_selected_features(FEATURES_PATH)
    X_unscaled = build_feature_matrix(encoded_df, features)
    y = ds_unscaled.y.to_numpy(dtype=np.int64)
    logger.info("Feature matrix | shape=%s", X_unscaled.shape)

    # ── Load frozen Sprint 5 scalers ─────────────────────────────────────
    logger.info("Loading frozen Sprint 5 SVM scaler (%s) ...", SVM_SCALER_PATH)
    svm_scaler = joblib.load(SVM_SCALER_PATH)
    logger.info("Loading frozen Sprint 5 NN scaler (%s) ...", NN_SCALER_PATH)
    nn_scaler = joblib.load(NN_SCALER_PATH)
    logger.info("Scalers loaded | svm n_features=%d | nn n_features=%d",
                svm_scaler.n_features_in_, nn_scaler.n_features_in_)

    # ── Create OOF folds (ONCE — same folds across all H1 seeds) ─────────
    logger.info("Creating OOF folds | n_splits=%d | seed=%d ...", OOF_N_SPLITS, OOF_SEED)
    folds = make_oof_folds(y, n_splits=OOF_N_SPLITS, seed=OOF_SEED)

    # Save fold assignments
    save_fold_assignments(folds, n_rows, seed_dir)

    # ── Run OOF for this seed ─────────────────────────────────────────────
    logger.info("Running OOF | h1_seed=%d | fixed_epochs=%d | pos_weight=%.8f",
                h1_seed, OOF_FIXED_EPOCH_COUNT, OOF_POS_WEIGHT)
    t_oof = time.perf_counter()
    oof_df = run_oof_seed(
        h1_seed=h1_seed,
        folds=folds,
        X_unscaled=X_unscaled,
        y=y,
        svm_scaler=svm_scaler,
        nn_scaler=nn_scaler,
    )
    oof_elapsed = time.perf_counter() - t_oof
    logger.info("OOF done | shape=%s | elapsed=%.2fs", oof_df.shape, oof_elapsed)

    # Verify completeness
    assert len(oof_df) == n_rows, f"OOF row count mismatch: {len(oof_df)} != {n_rows}"
    for col in ["dt_attack_probability", "rf_attack_probability",
                "svm_decision_score", "nn_attack_probability"]:
        assert not oof_df[col].isna().any(), f"NaN in OOF column {col}"
    logger.info("OOF completeness verified: %d rows, no NaNs", len(oof_df))

    # Save OOF predictions
    save_oof_predictions(oof_df, seed_dir)

    # ── Train meta-learner ────────────────────────────────────────────────
    logger.info("Training meta-learner | seed=%d | config=%s ...", h1_seed, META_CONFIG)
    t_meta = time.perf_counter()
    clf = train_meta_learner(oof_df, h1_seed)
    meta_elapsed = time.perf_counter() - t_meta
    logger.info("Meta-learner trained | elapsed=%.2fs", meta_elapsed)

    # ── OOF self-evaluation ───────────────────────────────────────────────
    metrics = compute_oof_metrics(clf, oof_df, h1_seed)
    metrics["oof_runtime_seconds"] = oof_elapsed
    metrics["meta_runtime_seconds"] = meta_elapsed
    logger.info(
        "OOF meta-eval | seed=%d | macro_f1=%.6f | IN-SAMPLE",
        h1_seed, metrics["macro_f1"],
    )

    # ── Save artifacts ────────────────────────────────────────────────────
    save_seed_metrics(metrics, seed_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unavailable"

    seed_metadata = {
        "experiment_id": "EXP_OOF_STACK_V1",
        "h1_seed": h1_seed,
        "oof_seed": OOF_SEED,
        "oof_n_splits": OOF_N_SPLITS,
        "feature_set": "EXP_MI_V1_1",
        "feature_count": 75,
        "train_sha256": FROZEN_TRAIN_SHA,
        "train_rows": n_rows,
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
        "oof_fixed_epoch_count": OOF_FIXED_EPOCH_COUNT,
        "oof_pos_weight": OOF_POS_WEIGHT,
        "resolved_dataset_paths": RESOLVED_PATHS,
        "meta_config": {**META_CONFIG, "random_state": h1_seed},
        "scaling_limitation": SCALING_LIMITATION_TEXT,
        "meta_evaluation_limitation": META_EVALUATION_LIMITATION_TEXT,
        "sprint5_reference": {
            "model": "RF",
            "macro_f1": SPRINT5_RF_REFERENCE,
            "label": SPRINT5_RF_REFERENCE_LABEL,
        },
        "environment": {
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": device,
            "gpu": gpu_name,
        },
        "git_commit": git_commit,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runtimes": {
            "oof_seconds": oof_elapsed,
            "meta_seconds": meta_elapsed,
            "total_seconds": time.perf_counter() - t_total,
        },
    }
    save_seed_metadata(seed_metadata, seed_dir)

    # Meta-learner checkpoint
    save_meta_learner_checkpoint(clf, ckpt_dir)
    save_meta_learner_metadata(
        {
            "experiment_id": "EXP_OOF_STACK_V1",
            "h1_seed": h1_seed,
            "meta_config": {**META_CONFIG, "random_state": h1_seed},
            "feature_set": "EXP_MI_V1_1",
            "feature_count": 75,
            "train_sha256": FROZEN_TRAIN_SHA,
            "meta_feature_cols": [
                "dt_attack_probability", "rf_attack_probability",
                "svm_decision_score", "nn_attack_probability",
            ],
            "row_id_excluded": True,
            "scaling_limitation": SCALING_LIMITATION_TEXT,
            "meta_evaluation_limitation": META_EVALUATION_LIMITATION_TEXT,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        ckpt_dir,
    )

    total_elapsed = time.perf_counter() - t_total
    logger.info(
        "=== OOF STACKING DONE | seed=%d | macro_f1=%.6f | total=%.2fs ===",
        h1_seed, metrics["macro_f1"], total_elapsed,
    )
    print(f"\nSeed {h1_seed}: macro_f1={metrics['macro_f1']:.6f} (IN-SAMPLE)")
    print(f"Total runtime: {total_elapsed:.1f}s")
    print(f"Artifacts: {seed_dir}")


if __name__ == "__main__":
    main()
