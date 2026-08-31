"""
scripts/run_mi_feature_selection.py
--------------------------------------
Official Sprint 4 execution script: Mutual Information feature selection.

Data access rule:
    READS:   data/splits/train.csv ONLY
    WRITES:  results/feature_selection/EXP_MI_V1/

NEVER reads:
    validation.csv
    development_test.csv
    protected_unseen_attack.csv
    excluded_train_backdoor.csv

Run from project root:
    .venv\\Scripts\\python.exe scripts/run_mi_feature_selection.py

Output files:
    mi_scores.csv
    feature_ranking.csv
    selected_features.json
    k_selection_results.csv
    metadata.json
    config.yaml
"""

import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.feature_selection.k_selector import (
    InnerCVConfig,
    check_selection_sanity,
    run_k_selection_cv,
)
from src.feature_selection.mi_selector import (
    MIConfig,
    build_family_report,
    compute_mi_scores,
    select_top_k,
)
from src.preprocessing.cleaning import (
    CATEGORICAL_COLS,
    separate_target_and_features,
)
from src.preprocessing.encoding import fit_encoder, get_feature_names, transform_encoder
from src.utils.hashing import sha256_file
from src.utils.reproducibility import set_all_seeds

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "EXP_MI_V1"
TRAIN_SHA256  = "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c"

OUT_DIR  = PROJECT_ROOT / "results" / "feature_selection" / EXPERIMENT_ID
LOG_DIR  = PROJECT_ROOT / "results" / "logs" / EXPERIMENT_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "run.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _lib_versions() -> dict:
    return {
        "python":   platform.python_version(),
        "numpy":    np.__version__,
        "pandas":   pd.__version__,
        "sklearn":  sklearn.__version__,
    }


def _verify_train_hash(train_path: Path) -> None:
    actual = _sha256(train_path)
    if actual != TRAIN_SHA256:
        logger.error(
            "TRAIN hash mismatch! Expected=%s Got=%s", TRAIN_SHA256, actual
        )
        sys.exit(1)
    logger.info("TRAIN SHA-256: MATCH (%s)", actual)


def _build_encoded_train(
    train_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build encoded-unscaled feature matrix from TRAIN.

    Returns (X_encoded, y, feature_names).
    Encoder fitted on TRAIN ONLY.
    Scaler NOT fitted (MI uses unscaled representation).
    """
    cleaned = separate_target_and_features(train_df, split_name="mi_train")
    y = cleaned.y.to_numpy(dtype=np.int64)

    cat_cols = list(cleaned.categorical_cols)
    num_cols = list(cleaned.numeric_cols)

    fitted_enc = fit_encoder(cleaned.X_raw[cat_cols], cat_cols)
    X_ohe = transform_encoder(fitted_enc, cleaned.X_raw[cat_cols])
    X_num = cleaned.X_raw[num_cols].to_numpy(dtype=np.float64)

    X = np.concatenate([X_ohe, X_num], axis=1)
    ohe_names = get_feature_names(fitted_enc)
    feature_names = ohe_names + num_cols

    logger.info(
        "Encoded TRAIN | shape=%s | OHE=%d | numeric=%d",
        X.shape, len(ohe_names), len(num_cols),
    )
    return X, y, feature_names


def _runtime_benchmark(
    train_df: pd.DataFrame,
    n_benchmark_rows: int = 5000,
    n_benchmark_features: int = 50,
) -> tuple[float, float]:
    """
    Run a representative MI benchmark on a small subset.

    Returns (benchmark_seconds, estimated_full_seconds).
    The benchmark has NO effect on K selection or feature ranking.
    """
    logger.info("=== RUNTIME BENCHMARK START (n=%d rows) ===", n_benchmark_rows)
    rng = np.random.default_rng(42)
    subset = train_df.sample(
        n=min(n_benchmark_rows, len(train_df)), random_state=42
    )
    X_sub, y_sub, feat_sub = _build_encoded_train(subset)

    # Use first n_benchmark_features columns to keep benchmark fast
    X_bm = X_sub[:, :min(n_benchmark_features, X_sub.shape[1])]
    feat_bm = feat_sub[:min(n_benchmark_features, len(feat_sub))]

    from src.feature_selection.mi_selector import compute_mi_scores, MIConfig
    config = MIConfig(n_neighbors=3, random_state=42)

    t0 = time.perf_counter()
    compute_mi_scores(X_bm, y_sub, feat_bm, config=config)
    benchmark_s = time.perf_counter() - t0

    # Estimate: scale by (full_rows / bm_rows) * (full_features / bm_features)
    full_rows    = len(train_df)
    full_feats   = X_sub.shape[1]
    bm_rows      = X_sub.shape[0]
    bm_feats     = X_bm.shape[1]
    scale        = (full_rows / bm_rows) * (full_feats / bm_feats)
    # CV adds 5 folds × 5 K + 1 final = 26 MI computations
    estimated_s  = benchmark_s * scale * 26

    logger.info(
        "Benchmark: %.2fs | Estimated full runtime: %.1fs (~%.1f min)",
        benchmark_s, estimated_s, estimated_s / 60,
    )
    return benchmark_s, estimated_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.perf_counter()
    now = datetime.now(timezone.utc)
    logger.info("=== %s START | %s ===", EXPERIMENT_ID, now.isoformat())

    set_all_seeds(42)
    git_commit = _git_commit()
    logger.info("Git commit: %s", git_commit)

    # --- Verify data access boundary ---
    train_path = PROJECT_ROOT / "data" / "splits" / "train.csv"
    if not train_path.exists():
        logger.error("FATAL: train.csv not found at %s", train_path)
        sys.exit(1)

    _verify_train_hash(train_path)
    logger.info("Loading train.csv...")
    train_df = pd.read_csv(train_path)
    logger.info("TRAIN shape: %s", train_df.shape)

    # --- Class balance (descriptive — no rebalancing) ---
    y_all_raw = train_df["label"]
    n_normal  = int((y_all_raw == 0).sum())
    n_attack  = int((y_all_raw == 1).sum())
    total     = len(train_df)
    logger.info(
        "Class balance | Normal=%d (%.2f%%) | Attack=%d (%.2f%%)",
        n_normal, 100 * n_normal / total,
        n_attack, 100 * n_attack / total,
    )

    # --- Runtime benchmark ---
    bm_s, est_s = _runtime_benchmark(train_df)

    # --- K-selection CV ---
    cv_config = InnerCVConfig(
        candidate_k=(10, 20, 30, 40, 50),
        n_splits=5,
        shuffle=True,
        cv_random_state=42,
        stratify_col="label",
        mi_n_neighbors=3,
        mi_random_state=42,
    )

    logger.info("=== INNER CV K-SELECTION START ===")
    t_cv_start = time.perf_counter()
    k_result = run_k_selection_cv(train_df, config=cv_config)
    cv_elapsed = time.perf_counter() - t_cv_start
    logger.info("CV elapsed: %.1fs", cv_elapsed)

    # --- Sanity check ---
    sanity = check_selection_sanity(k_result.summary_df)
    if sanity.status == "REVIEW_REQUIRED":
        logger.warning("SELECTION SANITY: REVIEW_REQUIRED | %s", sanity.reason)

    selected_k = k_result.selected_k
    logger.info("Selected K*=%d", selected_k)

    # --- Save k_selection_results.csv ---
    fold_df = pd.DataFrame(
        [{"k": r.k, "fold": r.fold, "macro_f1": r.macro_f1}
         for r in k_result.fold_records]
    )
    fold_path = OUT_DIR / "k_selection_results.csv"
    fold_df.to_csv(fold_path, index=False)
    logger.info("k_selection_results.csv saved (%d rows)", len(fold_df))

    # --- Final MI refit on complete TRAIN (no val/test) ---
    logger.info("=== FINAL MI REFIT ON COMPLETE TRAIN ===")
    t_refit = time.perf_counter()

    X_train, y_train, feature_names = _build_encoded_train(train_df)
    mi_config = MIConfig(n_neighbors=3, random_state=42)
    mi_result = compute_mi_scores(X_train, y_train, feature_names, config=mi_config)

    refit_elapsed = time.perf_counter() - t_refit
    logger.info("Final MI refit elapsed: %.1fs", refit_elapsed)
    logger.info("Encoded feature count: %d", mi_result.n_features)

    # --- Apply K* selection ---
    ranking_df = select_top_k(mi_result.ranking_df, selected_k, feature_names)
    selected_names = ranking_df.loc[ranking_df["selected"], "feature"].tolist()
    logger.info("Selected %d features", len(selected_names))

    # --- Source-family report ---
    family_report = build_family_report(ranking_df)
    logger.info("Source-family selected: %s", family_report["selected"])

    # --- Save mi_scores.csv ---
    mi_scores_df = mi_result.ranking_df.copy()
    (OUT_DIR / "mi_scores.csv").parent.mkdir(parents=True, exist_ok=True)
    mi_scores_df.to_csv(OUT_DIR / "mi_scores.csv", index=False)

    # --- Save feature_ranking.csv (with selected column from K* applied) ---
    ranking_df.to_csv(OUT_DIR / "feature_ranking.csv", index=False)
    logger.info("mi_scores.csv and feature_ranking.csv saved")

    # --- Save selected_features.json ---
    selected_json = {
        "experiment_id": EXPERIMENT_ID,
        "target": "label",
        "selected_k": selected_k,
        "selection_rule": "highest mean macro-F1 across 5 inner folds; smaller K on tie",
        "selection_metric": "macro_f1",
        "inner_cv_n_splits": 5,
        "inner_cv_random_state": 42,
        "features": selected_names,
        "feature_ordering": "by MI score descending (rank 1 = highest MI)",
        "ae_note": (
            "The Autoencoder (Sprint 6) will use this SAME selected feature set. "
            "MI is optimized for binary Normal-vs-Attack discrimination. "
            "Sharing this feature space with the AE prioritizes: one common feature space, "
            "architectural simplicity, reproducibility, consistent feature provenance, "
            "and consistent downstream SHAP mapping. "
            "Tradeoff acknowledged: the feature set is not independently optimized "
            "for benign-manifold reconstruction."
        ),
    }
    with open(OUT_DIR / "selected_features.json", "w") as f:
        json.dump(selected_json, f, indent=2)
    logger.info("selected_features.json saved")

    # --- Config snapshot ---
    config_dict = {
        "experiment_id": EXPERIMENT_ID,
        "mi": {
            "method": "mutual_info_classif",
            "n_neighbors": 3,
            "random_state": 42,
        },
        "candidate_k": [10, 20, 30, 40, 50],
        "inner_cv": {
            "method": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "stratify_on": "label",
        },
        "evaluator": {
            "model": "LogisticRegression",
            "solver": "liblinear",
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "evaluator_scaling": {
            "method": "StandardScaler",
            "fit_scope": "inner_train_only",
        },
        "selection": {
            "metric": "macro_f1",
            "tie_break": "smaller_k",
        },
        "runtime_benchmark": {
            "enabled": True,
            "affects_selection": False,
        },
        "selection_sanity": {
            "flat_range_tolerance": 1e-3,
            "flat_range_tolerance_note": (
                "REVIEW_REQUIRED if max(mean_macro_f1) - min(mean_macro_f1) <= 0.001"
            ),
            "monotonic_tolerance": 1e-4,
            "monotonic_tolerance_note": (
                "REVIEW_REQUIRED if F1 is non-decreasing across all K "
                "within tolerance of 1e-4 per step"
            ),
        },
    }
    with open(OUT_DIR / "config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    logger.info("config.yaml saved")

    # --- Metadata ---
    actual_elapsed = time.perf_counter() - t_start

    # K-selection summary as plain dict for JSON
    k_summary = []
    for _, row in k_result.summary_df.iterrows():
        k_summary.append({
            "k": int(row["k"]),
            "mean_macro_f1": float(row["mean_macro_f1"]),
            "std_macro_f1": float(row["std_macro_f1"]),
        })

    # Discrete mask summary
    dm = mi_result.discrete_mask
    discrete_count = int(dm.sum())
    continuous_count = int((~dm).sum())

    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": "1.0",
        "dataset": "UNSW-NB15",
        "created_at": now.isoformat(),
        "git_commit": git_commit,

        # Input
        "input_train_file": "data/splits/train.csv",
        "input_train_sha256": TRAIN_SHA256,
        "input_train_row_count": int(total),
        "input_encoded_feature_count": int(mi_result.n_features),
        "discrete_feature_count": discrete_count,
        "continuous_feature_count": continuous_count,

        # Columns
        "categorical_cols": list(CATEGORICAL_COLS),
        "numeric_col_count": len([f for f in feature_names
                                  if not any(f.startswith(p + "_")
                                             for p in ("proto", "service", "state"))]),

        # Class balance
        "class_balance": {
            "normal_count": n_normal,
            "attack_count": n_attack,
            "normal_pct": round(100 * n_normal / total, 4),
            "attack_pct": round(100 * n_attack / total, 4),
            "note": "No rebalancing applied. MI uses actual frozen TRAIN distribution.",
        },

        # MI config
        "mi_estimator": "mutual_info_classif",
        "mi_n_neighbors": 3,
        "mi_random_state": 42,
        "mi_target": "label",
        "mi_representation": "encoded_unscaled",

        # Candidate K
        "candidate_k": list(cv_config.candidate_k),

        # CV
        "cv_method": "StratifiedKFold",
        "cv_n_splits": 5,
        "cv_shuffle": True,
        "cv_random_state": 42,
        "cv_stratify_on": "label",

        # Evaluator
        "evaluator": {
            "model": "LogisticRegression",
            "solver": "liblinear",
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
            "role": "Fixed K-selection evaluator only. NOT a research model.",
        },

        # Selection
        "selection_metric": "macro_f1",
        "tie_break": "smaller_k",
        "selected_k": selected_k,
        "k_selection_summary": k_summary,

        # Source families
        "source_family_report": family_report,

        # Sanity
        "selection_sanity": {
            "status": sanity.status,
            "reason": sanity.reason,
            "flat_range": round(sanity.flat_range, 8),
            "flat_tolerance": sanity.flat_tolerance,
            "is_monotonic": sanity.is_monotonic,
            "monotonic_tolerance": sanity.monotonic_tolerance,
        },

        # Runtimes
        "runtime_benchmark": {
            "enabled": True,
            "affects_selection": False,
            "benchmark_duration_s": round(bm_s, 3),
            "estimated_full_runtime_s": round(est_s, 1),
            "actual_full_runtime_s": round(actual_elapsed, 1),
        },

        # AE tradeoff note
        "ae_feature_space_decision": (
            "The Autoencoder will use the SAME MI-selected feature set. "
            "MI is optimized for binary Normal-vs-Attack discrimination. "
            "Sharing this space prioritizes a common feature space and reproducibility. "
            "Tradeoff: not independently optimized for benign-manifold reconstruction."
        ),

        # Versions
        "library_versions": _lib_versions(),
    }

    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("metadata.json saved")

    # --- Summary ---
    logger.info("")
    logger.info("=== %s SUMMARY ===", EXPERIMENT_ID)
    logger.info("  TRAIN rows:          %d", total)
    logger.info("  Encoded features:    %d", mi_result.n_features)
    logger.info("  Discrete / Cont:     %d / %d", discrete_count, continuous_count)
    logger.info("  Selected K*:         %d", selected_k)
    logger.info("  Sanity:              %s", sanity.status)
    logger.info("  Benchmark:           %.2fs", bm_s)
    logger.info("  Estimated runtime:   %.1fs", est_s)
    logger.info("  Actual runtime:      %.1fs", actual_elapsed)

    for row in k_summary:
        logger.info(
            "  K=%d | mean_f1=%.6f | std_f1=%.6f",
            row["k"], row["mean_macro_f1"], row["std_macro_f1"]
        )
    logger.info("  Family selected: %s", family_report["selected"])
    logger.info("=== %s COMPLETE ===", EXPERIMENT_ID)

    if sanity.status == "REVIEW_REQUIRED":
        print("\nSELECTION SANITY: REVIEW_REQUIRED")
        print(sanity.reason)
        print("DO NOT freeze automatically. STOP for human review.")
    else:
        print("\nSELECTION SANITY: PASS")

    print(f"\nSTATUS: COMPLETE")
    print(f"  Selected K*:     {selected_k}")
    print(f"  Encoded feats:   {mi_result.n_features}")
    print(f"  Actual runtime:  {actual_elapsed:.1f}s")


if __name__ == "__main__":
    main()
