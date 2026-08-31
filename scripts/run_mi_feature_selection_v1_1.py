"""
scripts/run_mi_feature_selection_v1_1.py
------------------------------------------
Sprint 4 MI Experiment — Protocol v1.1
Experiment ID: EXP_MI_V1_1

Protocol amendment: K grid extended from {10,20,30,40,50} to
{10,20,30,40,50,75,100,150} after v1.0 produced a monotonic
sanity flag (REVIEW_REQUIRED). Human approval granted.

Original v1.0 results are preserved at:
    results/feature_selection/EXP_MI_V1/

This script writes ONLY to:
    results/feature_selection/EXP_MI_V1_1/

All methodology parameters are IDENTICAL to v1.0 except candidate_k.

Data access rule:
    READS:   data/splits/train.csv ONLY
    WRITES:  results/feature_selection/EXP_MI_V1_1/

Run from project root:
    .venv\\Scripts\\python.exe scripts/run_mi_feature_selection_v1_1.py
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
from src.preprocessing.cleaning import CATEGORICAL_COLS
from src.preprocessing.encoding import fit_encoder, get_feature_names, transform_encoder
from src.preprocessing.cleaning import separate_target_and_features
from src.utils.reproducibility import set_all_seeds

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID   = "EXP_MI_V1_1"
PROTOCOL_VERSION = "1.1"
TRAIN_SHA256    = "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c"

# Protocol amendment: extended K grid
CANDIDATE_K = (10, 20, 30, 40, 50, 75, 100, 150)

# v1.0 preserved results (for provenance recording — never used for selection)
V1_RESULTS = {
    10:  0.824852,
    20:  0.864436,
    30:  0.897442,
    40:  0.916198,
    50:  0.919560,
}

OUT_DIR = PROJECT_ROOT / "results" / "feature_selection" / EXPERIMENT_ID
LOG_DIR = PROJECT_ROOT / "results" / "logs" / EXPERIMENT_ID
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
# Helpers (identical to v1.0 script)
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
        "python":  platform.python_version(),
        "numpy":   np.__version__,
        "pandas":  pd.__version__,
        "sklearn": sklearn.__version__,
    }


def _verify_train_hash(train_path: Path) -> None:
    actual = _sha256(train_path)
    if actual != TRAIN_SHA256:
        logger.error("TRAIN hash mismatch! Expected=%s Got=%s", TRAIN_SHA256, actual)
        sys.exit(1)
    logger.info("TRAIN SHA-256: MATCH (%s)", actual)


def _build_encoded_train(
    train_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
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


def _runtime_benchmark(train_df: pd.DataFrame) -> tuple[float, float]:
    logger.info("=== RUNTIME BENCHMARK START ===")
    subset = train_df.sample(n=min(5000, len(train_df)), random_state=42)
    X_sub, y_sub, feat_sub = _build_encoded_train(subset)
    X_bm = X_sub[:, :min(50, X_sub.shape[1])]
    feat_bm = feat_sub[:min(50, len(feat_sub))]

    config = MIConfig(n_neighbors=3, random_state=42)
    t0 = time.perf_counter()
    compute_mi_scores(X_bm, y_sub, feat_bm, config=config)
    benchmark_s = time.perf_counter() - t0

    full_rows  = len(train_df)
    full_feats = X_sub.shape[1]
    bm_rows    = X_sub.shape[0]
    bm_feats   = X_bm.shape[1]
    scale      = (full_rows / bm_rows) * (full_feats / bm_feats)
    # 5 folds + 1 final refit = 6 MI computations
    estimated_s = benchmark_s * scale * 6

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
    logger.info("=== %s START | %s | protocol_version=%s ===",
                EXPERIMENT_ID, now.isoformat(), PROTOCOL_VERSION)

    set_all_seeds(42)
    git_commit = _git_commit()
    logger.info("Git commit: %s", git_commit)

    # --- Verify v1.0 results are preserved ---
    v1_dir = PROJECT_ROOT / "results" / "feature_selection" / "EXP_MI_V1"
    v1_meta = v1_dir / "metadata.json"
    if not v1_meta.exists():
        logger.error(
            "FATAL: Original v1.0 results not found at %s. "
            "Original results must be preserved before running v1.1.",
            v1_meta,
        )
        sys.exit(1)
    logger.info("v1.0 original results confirmed present at %s", v1_dir)

    # --- Data access ---
    train_path = PROJECT_ROOT / "data" / "splits" / "train.csv"
    if not train_path.exists():
        logger.error("FATAL: train.csv not found at %s", train_path)
        sys.exit(1)

    _verify_train_hash(train_path)
    logger.info("Loading train.csv...")
    train_df = pd.read_csv(train_path)
    logger.info("TRAIN shape: %s", train_df.shape)

    # --- Class balance ---
    y_raw    = train_df["label"]
    n_normal = int((y_raw == 0).sum())
    n_attack = int((y_raw == 1).sum())
    total    = len(train_df)
    logger.info(
        "Class balance | Normal=%d (%.2f%%) | Attack=%d (%.2f%%)",
        n_normal, 100 * n_normal / total,
        n_attack, 100 * n_attack / total,
    )

    # --- Benchmark ---
    bm_s, est_s = _runtime_benchmark(train_df)

    # --- K-selection CV (expanded grid) ---
    cv_config = InnerCVConfig(
        candidate_k=CANDIDATE_K,
        n_splits=5,
        shuffle=True,
        cv_random_state=42,
        stratify_col="label",
        mi_n_neighbors=3,
        mi_random_state=42,
    )

    logger.info("=== INNER CV K-SELECTION START | candidate_k=%s ===",
                list(CANDIDATE_K))
    t_cv_start = time.perf_counter()
    k_result = run_k_selection_cv(train_df, config=cv_config)
    cv_elapsed = time.perf_counter() - t_cv_start
    logger.info("CV elapsed: %.1fs", cv_elapsed)

    # --- Sanity check ---
    sanity = check_selection_sanity(k_result.summary_df)
    if sanity.status == "REVIEW_REQUIRED":
        logger.warning("SELECTION SANITY: REVIEW_REQUIRED | %s", sanity.reason)
    else:
        logger.info("SELECTION SANITY: PASS | %s", sanity.reason)

    selected_k = k_result.selected_k
    logger.info("Selected K*=%d", selected_k)

    # --- Save k_selection_results.csv ---
    fold_df = pd.DataFrame(
        [{"k": r.k, "fold": r.fold, "macro_f1": r.macro_f1}
         for r in k_result.fold_records]
    )
    fold_df.to_csv(OUT_DIR / "k_selection_results.csv", index=False)
    logger.info("k_selection_results.csv saved (%d rows)", len(fold_df))

    # --- Conditional final refit ---
    final_refit_done = False
    selected_names   = []
    family_report    = {}
    mi_result_final  = None
    ranking_df       = None

    if sanity.status != "REVIEW_REQUIRED":
        logger.info("=== FINAL MI REFIT ON COMPLETE TRAIN ===")
        t_refit = time.perf_counter()

        X_train, y_train, feature_names = _build_encoded_train(train_df)
        mi_config = MIConfig(n_neighbors=3, random_state=42)
        mi_result_final = compute_mi_scores(X_train, y_train, feature_names, config=mi_config)
        refit_elapsed = time.perf_counter() - t_refit
        logger.info("Final MI refit elapsed: %.1fs", refit_elapsed)

        ranking_df   = select_top_k(mi_result_final.ranking_df, selected_k, feature_names)
        selected_names = ranking_df.loc[ranking_df["selected"], "feature"].tolist()
        family_report  = build_family_report(ranking_df)

        mi_result_final.ranking_df.to_csv(OUT_DIR / "mi_scores.csv", index=False)
        ranking_df.to_csv(OUT_DIR / "feature_ranking.csv", index=False)

        selected_json = {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "target": "label",
            "selected_k": selected_k,
            "selection_rule": "highest mean macro-F1 across 5 inner folds; smaller K on tie",
            "selection_metric": "macro_f1",
            "inner_cv_n_splits": 5,
            "inner_cv_random_state": 42,
            "features": selected_names,
            "feature_ordering": "by MI score descending (rank 1 = highest MI)",
            "protocol_amendment_ref": "results/feature_selection/EXP_MI_V1_1/protocol_amendment.md",
            "ae_note": (
                "The Autoencoder will use this SAME MI-selected feature set. "
                "MI is optimized for binary Normal-vs-Attack discrimination. "
                "Sharing this feature space prioritizes a common feature space, "
                "reproducibility, and consistent SHAP mapping. "
                "Tradeoff: not independently optimized for benign-manifold reconstruction."
            ),
        }
        with open(OUT_DIR / "selected_features.json", "w") as f:
            json.dump(selected_json, f, indent=2)

        logger.info("mi_scores.csv, feature_ranking.csv, selected_features.json saved")
        final_refit_done = True
    else:
        logger.warning(
            "SANITY REVIEW_REQUIRED: skipping final refit. "
            "Do NOT freeze automatically. Stop for human review."
        )
        # Save placeholder files so the output directory is complete
        pd.DataFrame(columns=["feature", "mi_score", "rank", "source_family",
                               "selected"]).to_csv(OUT_DIR / "mi_scores.csv", index=False)
        pd.DataFrame(columns=["rank", "feature", "mi_score", "source_family",
                               "selected"]).to_csv(OUT_DIR / "feature_ranking.csv", index=False)
        with open(OUT_DIR / "selected_features.json", "w") as f:
            json.dump({"status": "REVIEW_REQUIRED", "selected_features": None}, f, indent=2)

    # --- Config snapshot ---
    actual_elapsed = time.perf_counter() - t_start
    config_dict = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_amendment": "results/feature_selection/EXP_MI_V1_1/protocol_amendment.md",
        "mi": {"method": "mutual_info_classif", "n_neighbors": 3, "random_state": 42},
        "candidate_k": list(CANDIDATE_K),
        "previous_candidate_k": [10, 20, 30, 40, 50],
        "inner_cv": {
            "method": "StratifiedKFold", "n_splits": 5,
            "shuffle": True, "random_state": 42, "stratify_on": "label",
        },
        "evaluator": {
            "model": "LogisticRegression", "solver": "liblinear",
            "C": 1.0, "max_iter": 1000,
            "class_weight": "balanced", "random_state": 42,
        },
        "evaluator_scaling": {"method": "StandardScaler", "fit_scope": "inner_train_only"},
        "selection": {"metric": "macro_f1", "tie_break": "smaller_k"},
        "runtime_benchmark": {"enabled": True, "affects_selection": False},
        "selection_sanity": {
            "flat_range_tolerance": 1e-3,
            "monotonic_tolerance": 1e-4,
            "stop_if_still_monotonic_at_k150": True,
        },
    }
    with open(OUT_DIR / "config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # --- K summary ---
    k_summary = [
        {
            "k": int(row["k"]),
            "mean_macro_f1": float(row["mean_macro_f1"]),
            "std_macro_f1": float(row["std_macro_f1"]),
        }
        for _, row in k_result.summary_df.iterrows()
    ]

    # --- Metadata ---
    dm = None
    if mi_result_final is not None:
        dm = mi_result_final.discrete_mask
        discrete_count    = int(dm.sum())
        continuous_count  = int((~dm).sum())
        n_encoded         = int(mi_result_final.n_features)
    else:
        # Compute from a quick peek at shape
        X_peek, _, fn_peek = _build_encoded_train(train_df.head(100))
        from src.feature_selection.mi_selector import build_discrete_mask
        dm_peek = build_discrete_mask(fn_peek)
        discrete_count    = int(dm_peek.sum())
        continuous_count  = int((~dm_peek).sum())
        n_encoded         = len(fn_peek)

    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "UNSW-NB15",
        "created_at": now.isoformat(),
        "git_commit": git_commit,

        "protocol_amendment": {
            "file": "results/feature_selection/EXP_MI_V1_1/protocol_amendment.md",
            "previous_candidate_k": [10, 20, 30, 40, 50],
            "new_candidate_k": list(CANDIDATE_K),
            "reason": (
                "Initial K grid produced a monotonic mean inner-CV Macro-F1 curve "
                "through its upper boundary; the project sanity check therefore "
                "required human review. The grid was expanded once to include "
                "higher-dimensional candidate feature sets before final K selection."
            ),
            "timing": (
                "Before rerunning the expanded K-selection experiment; no TEST, "
                "outer VALIDATION, protected Backdoor, or archived Backdoor data "
                "were used."
            ),
            "v1_results_preserved_at": "results/feature_selection/EXP_MI_V1/",
        },

        "v1_original_results": {
            "experiment_id": "EXP_MI_V1",
            "k_results": [
                {"k": k, "mean_macro_f1": f1} for k, f1 in V1_RESULTS.items()
            ],
            "sanity_status": "REVIEW_REQUIRED",
            "sanity_condition": "MONOTONIC",
        },

        "input_train_file": "data/splits/train.csv",
        "input_train_sha256": TRAIN_SHA256,
        "input_train_row_count": int(total),
        "input_encoded_feature_count": n_encoded,
        "discrete_feature_count": discrete_count,
        "continuous_feature_count": continuous_count,
        "categorical_cols": list(CATEGORICAL_COLS),

        "class_balance": {
            "normal_count": n_normal,
            "attack_count": n_attack,
            "normal_pct": round(100 * n_normal / total, 4),
            "attack_pct": round(100 * n_attack / total, 4),
            "note": "No rebalancing applied.",
        },

        "mi_estimator": "mutual_info_classif",
        "mi_n_neighbors": 3,
        "mi_random_state": 42,
        "mi_target": "label",
        "mi_representation": "encoded_unscaled",

        "candidate_k": list(CANDIDATE_K),
        "cv_method": "StratifiedKFold",
        "cv_n_splits": 5,
        "cv_shuffle": True,
        "cv_random_state": 42,
        "cv_stratify_on": "label",

        "evaluator": {
            "model": "LogisticRegression", "solver": "liblinear",
            "C": 1.0, "max_iter": 1000,
            "class_weight": "balanced", "random_state": 42,
            "role": "Fixed K-selection evaluator only. NOT a research model.",
        },

        "selection_metric": "macro_f1",
        "tie_break": "smaller_k",
        "selected_k": selected_k if final_refit_done else None,
        "final_refit_done": final_refit_done,
        "k_selection_summary": k_summary,
        "source_family_report": family_report if family_report else None,

        "selection_sanity": {
            "status": sanity.status,
            "reason": sanity.reason,
            "flat_range": round(sanity.flat_range, 8),
            "flat_tolerance": sanity.flat_tolerance,
            "is_monotonic": sanity.is_monotonic,
            "monotonic_tolerance": sanity.monotonic_tolerance,
            "stop_policy": (
                "If still monotonic at K=150: REVIEW_REQUIRED, stop, do not add K=175/193."
            ),
        },

        "runtime_benchmark": {
            "enabled": True,
            "affects_selection": False,
            "benchmark_duration_s": round(bm_s, 3),
            "estimated_full_runtime_s": round(est_s, 1),
            "actual_full_runtime_s": round(actual_elapsed, 1),
        },

        "library_versions": _lib_versions(),
    }

    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("metadata.json saved")

    # --- Summary ---
    logger.info("")
    logger.info("=== %s SUMMARY ===", EXPERIMENT_ID)
    logger.info("  Protocol version:    %s", PROTOCOL_VERSION)
    logger.info("  TRAIN rows:          %d", total)
    logger.info("  Encoded features:    %d", n_encoded)
    logger.info("  Candidate K:         %s", list(CANDIDATE_K))
    logger.info("  Sanity:              %s", sanity.status)
    logger.info("  Actual runtime:      %.1fs", actual_elapsed)
    for row in sorted(k_summary, key=lambda r: r["k"]):
        logger.info("  K=%-3d | mean_f1=%.6f | std_f1=%.6f",
                    row["k"], row["mean_macro_f1"], row["std_macro_f1"])
    if final_refit_done:
        logger.info("  Selected K*:         %d", selected_k)
        logger.info("  Family selected:     %s", family_report.get("selected", {}))
    else:
        logger.info("  Selected K*:         NOT SELECTED (REVIEW_REQUIRED)")
    logger.info("=== %s COMPLETE ===", EXPERIMENT_ID)

    if sanity.status == "REVIEW_REQUIRED":
        print("\nSELECTION SANITY: REVIEW_REQUIRED")
        print(sanity.reason)
        print("\nDO NOT freeze automatically. STOP for human review.")
        print("Do NOT add K=175 or K=193 without explicit human approval.")
    else:
        print("\nSELECTION SANITY: PASS")
        print(f"Selected K*: {selected_k}")

    print(f"\nSTATUS: COMPLETE")
    print(f"  Experiment ID:   {EXPERIMENT_ID}")
    print(f"  Protocol:        v{PROTOCOL_VERSION}")
    print(f"  Sanity:          {sanity.status}")
    print(f"  Actual runtime:  {actual_elapsed:.1f}s")


if __name__ == "__main__":
    main()
