"""
scripts/run_mi_final_refit_k75.py
-----------------------------------
Sprint 4 Final MI Refit — K=75, human-approved.

Human review decision (2026-08-31):
    - EXP_MI_V1_1 inner-CV produced monotonic sanity flag (REVIEW_REQUIRED).
    - Practical plateau begins at K=75.
    - K=75 is the highest mean macro-F1 under the predefined selection rule.
    - K=75 is ACCEPTED by human review.

This script performs ONLY the final refit on complete frozen TRAIN.
It does NOT rerun K-selection CV. That was completed in EXP_MI_V1_1.

Reads:   data/splits/train.csv ONLY
Writes:  results/feature_selection/EXP_MI_V1_1/ (updates artifacts)
         Specifically: mi_scores.csv, feature_ranking.csv,
         selected_features.json, metadata.json (updated)

Never reads: validation.csv, development_test.csv,
             protected_unseen_attack.csv, excluded_train_backdoor.csv

Run from project root:
    .venv\\Scripts\\python.exe scripts/run_mi_final_refit_k75.py
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sklearn

from src.feature_selection.mi_selector import (
    MIConfig,
    build_family_report,
    compute_mi_scores,
    select_top_k,
)
from src.preprocessing.cleaning import CATEGORICAL_COLS, separate_target_and_features
from src.preprocessing.encoding import fit_encoder, get_feature_names, transform_encoder
from src.utils.reproducibility import set_all_seeds

# ---------------------------------------------------------------------------
# Constants — frozen
# ---------------------------------------------------------------------------

EXPERIMENT_ID    = "EXP_MI_V1_1"
PROTOCOL_VERSION = "1.1"
FINAL_K          = 75
TRAIN_SHA256     = "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c"

# v1.1 K-selection results (from completed CV run) — frozen, not recomputed
V1_1_K_SUMMARY = [
    {"k": 10,  "mean_macro_f1": 0.824852, "std_macro_f1": 0.003435},
    {"k": 20,  "mean_macro_f1": 0.864436, "std_macro_f1": 0.002428},
    {"k": 30,  "mean_macro_f1": 0.897442, "std_macro_f1": 0.000917},
    {"k": 40,  "mean_macro_f1": 0.916198, "std_macro_f1": 0.002122},
    {"k": 50,  "mean_macro_f1": 0.919560, "std_macro_f1": 0.002323},
    {"k": 75,  "mean_macro_f1": 0.919799, "std_macro_f1": None},  # populated from metadata
    {"k": 100, "mean_macro_f1": 0.919775, "std_macro_f1": None},
    {"k": 150, "mean_macro_f1": 0.919750, "std_macro_f1": None},
]

OUT_DIR = PROJECT_ROOT / "results" / "feature_selection" / EXPERIMENT_ID
LOG_DIR = PROJECT_ROOT / "results" / "logs" / EXPERIMENT_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "final_refit.log", mode="w", encoding="utf-8"),
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


def _verify_original_v1_intact() -> None:
    v1_dir = PROJECT_ROOT / "results" / "feature_selection" / "EXP_MI_V1"
    required = ["mi_scores.csv", "feature_ranking.csv", "selected_features.json",
                "k_selection_results.csv", "metadata.json", "config.yaml"]
    for fname in required:
        p = v1_dir / fname
        if not p.exists():
            logger.error("FATAL: Original v1.0 artifact missing: %s", p)
            sys.exit(1)
    logger.info("v1.0 original results confirmed intact at %s", v1_dir)


def _verify_k_selection_complete() -> dict:
    """Load and verify the completed v1.1 K-selection metadata."""
    meta_path = OUT_DIR / "metadata.json"
    if not meta_path.exists():
        logger.error(
            "FATAL: EXP_MI_V1_1 metadata.json not found. "
            "Run scripts/run_mi_feature_selection_v1_1.py first."
        )
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)
    logger.info("v1.1 metadata loaded | experiment_id=%s | protocol_version=%s",
                meta.get("experiment_id"), meta.get("protocol_version"))
    return meta


def _build_encoded_train(
    train_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cleaned  = separate_target_and_features(train_df, split_name="final_refit")
    y        = cleaned.y.to_numpy(dtype=np.int64)
    cat_cols = list(cleaned.categorical_cols)
    num_cols = list(cleaned.numeric_cols)

    fitted_enc   = fit_encoder(cleaned.X_raw[cat_cols], cat_cols)
    X_ohe        = transform_encoder(fitted_enc, cleaned.X_raw[cat_cols])
    X_num        = cleaned.X_raw[num_cols].to_numpy(dtype=np.float64)
    X            = np.concatenate([X_ohe, X_num], axis=1)
    ohe_names    = get_feature_names(fitted_enc)
    feature_names = ohe_names + num_cols

    logger.info(
        "Encoded TRAIN | shape=%s | OHE=%d | numeric=%d",
        X.shape, len(ohe_names), len(num_cols),
    )
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.perf_counter()
    now     = datetime.now(timezone.utc)

    logger.info(
        "=== %s FINAL REFIT | K=%d | protocol_version=%s | %s ===",
        EXPERIMENT_ID, FINAL_K, PROTOCOL_VERSION, now.isoformat(),
    )
    logger.info("Human review decision: ACCEPT K=%d", FINAL_K)
    logger.info(
        "Sanity status: REVIEW_REQUIRED (monotonic) — explicitly human-reviewed and approved"
    )

    set_all_seeds(42)
    git_commit = _git_commit()
    logger.info("Git commit: %s", git_commit)

    # --- Pre-flight checks ---
    _verify_original_v1_intact()
    existing_meta = _verify_k_selection_complete()

    # Load completed K-summary from existing metadata
    k_summary = existing_meta.get("k_selection_summary", V1_1_K_SUMMARY)

    # Verify K=75 was indeed the selection rule winner
    best_row = max(k_summary, key=lambda r: (r["mean_macro_f1"], -r["k"]))
    if best_row["k"] != FINAL_K:
        logger.error(
            "CONSISTENCY ERROR: best K by rule is %d, but FINAL_K=%d. "
            "Check k_selection_results.csv.",
            best_row["k"], FINAL_K,
        )
        sys.exit(1)
    logger.info(
        "Selection rule verified: K=%d has highest mean_macro_f1=%.6f",
        FINAL_K, best_row["mean_macro_f1"],
    )

    # --- Load TRAIN ---
    train_path = PROJECT_ROOT / "data" / "splits" / "train.csv"
    _verify_train_hash(train_path)
    logger.info("Loading train.csv...")
    train_df = pd.read_csv(train_path)
    logger.info("TRAIN shape: %s", train_df.shape)

    total    = len(train_df)
    n_normal = int((train_df["label"] == 0).sum())
    n_attack = int((train_df["label"] == 1).sum())
    logger.info(
        "Class balance | Normal=%d (%.2f%%) | Attack=%d (%.2f%%)",
        n_normal, 100 * n_normal / total,
        n_attack, 100 * n_attack / total,
    )

    # --- Final MI refit on complete TRAIN ---
    logger.info("=== FINAL MI REFIT START | K=%d ===", FINAL_K)
    t_refit = time.perf_counter()

    X_train, y_train, feature_names = _build_encoded_train(train_df)

    mi_config = MIConfig(n_neighbors=3, random_state=42)
    mi_result = compute_mi_scores(X_train, y_train, feature_names, config=mi_config)

    refit_elapsed = time.perf_counter() - t_refit
    logger.info("Final MI refit elapsed: %.1fs", refit_elapsed)
    logger.info("Total encoded features: %d", mi_result.n_features)

    # --- Select top-K=75 ---
    ranking_df     = select_top_k(mi_result.ranking_df, FINAL_K, feature_names)
    selected_names = ranking_df.loc[ranking_df["selected"], "feature"].tolist()
    family_report  = build_family_report(ranking_df)

    logger.info("Selected %d features", len(selected_names))
    logger.info("Top-5 selected: %s", selected_names[:5])
    logger.info("Source-family distribution: %s", family_report.get("selected", {}))

    assert len(selected_names) == FINAL_K, (
        f"Expected {FINAL_K} selected features, got {len(selected_names)}"
    )

    # --- Save mi_scores.csv ---
    mi_result.ranking_df.to_csv(OUT_DIR / "mi_scores.csv", index=False)
    logger.info("mi_scores.csv saved (%d rows)", len(mi_result.ranking_df))

    # --- Save feature_ranking.csv ---
    ranking_df.to_csv(OUT_DIR / "feature_ranking.csv", index=False)
    logger.info("feature_ranking.csv saved (%d rows, %d selected)",
                len(ranking_df), int(ranking_df["selected"].sum()))

    # --- Save selected_features.json ---
    selected_json = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "target": "label",
        "selected_k": FINAL_K,
        "selection_rule": "highest mean macro-F1 across 5 inner folds; smaller K on tie",
        "selection_metric": "macro_f1",
        "inner_cv_n_splits": 5,
        "inner_cv_random_state": 42,

        # Sanity and human review
        "sanity_status": "REVIEW_REQUIRED",
        "sanity_condition": "MONOTONIC",
        "sanity_tolerance": 1e-4,
        "human_review": "APPROVED",
        "human_review_rationale": (
            "K=75 has the highest mean inner-CV macro-F1 (0.919799) under the predefined "
            "selection rule. The curve is technically monotonically non-decreasing through "
            "K=150 within tolerance=1e-4, but the practical improvement beyond K=75 is "
            "negligible: K=75 (0.919799) vs K=100 (0.919775) vs K=150 (0.919750). "
            "The predefined selection rule selected K=75 correctly. "
            "Human review confirms the selection is sound despite the REVIEW_REQUIRED flag."
        ),

        # Feature list
        "features": selected_names,
        "feature_ordering": "by MI score descending (rank 1 = highest MI)",
        "feature_count": FINAL_K,

        # Protocol history
        "protocol_amendment_ref": "results/feature_selection/EXP_MI_V1_1/protocol_amendment.md",
        "v1_selected_k": 50,
        "v1_sanity": "REVIEW_REQUIRED",
        "amendment_reason": (
            "v1.0 K grid {10,20,30,40,50} was monotonically increasing; "
            "grid extended to {10,20,30,40,50,75,100,150} by human approval."
        ),

        # AE note
        "ae_note": (
            "The Autoencoder (Sprint 8) will use this SAME selected feature set. "
            "MI is optimized for binary Normal-vs-Attack discrimination. "
            "Sharing this feature space prioritizes a common feature space, "
            "reproducibility, and consistent SHAP mapping. "
            "Tradeoff: not independently optimized for benign-manifold reconstruction."
        ),

        # Source-family distribution
        "source_family": family_report.get("selected", {}),
    }

    with open(OUT_DIR / "selected_features.json", "w") as f:
        json.dump(selected_json, f, indent=2)
    logger.info("selected_features.json saved")

    # --- Update metadata.json ---
    actual_elapsed = time.perf_counter() - t_start

    dm = mi_result.discrete_mask
    discrete_count   = int(dm.sum())
    continuous_count = int((~dm).sum())

    # Reload existing metadata to preserve all CV fields, then update
    meta_path = OUT_DIR / "metadata.json"
    with open(meta_path) as f:
        existing_meta_full = json.load(f)

    existing_meta_full.update({
        "final_refit_done": True,
        "final_refit_timestamp": now.isoformat(),
        "final_refit_git_commit": git_commit,
        "selected_k": FINAL_K,
        "input_encoded_feature_count": int(mi_result.n_features),
        "discrete_feature_count": discrete_count,
        "continuous_feature_count": continuous_count,
        "source_family_report": family_report,
        "library_versions": _lib_versions(),

        "human_review": {
            "decision": "APPROVED",
            "selected_k": FINAL_K,
            "rationale": (
                "K=75 is the highest mean inner-CV macro-F1 under the predefined rule. "
                "Practical plateau begins at K=75. Improvements beyond K=75 are negligible."
            ),
            "sanity_status_retained": "REVIEW_REQUIRED",
            "monotonic_flag_retained": True,
            "timestamp": now.isoformat(),
        },

        "selection_sanity": {
            "status": "REVIEW_REQUIRED",
            "condition": "MONOTONIC",
            "reason": (
                "Mean macro-F1 is monotonically non-decreasing through K=150 "
                "within tolerance=1e-4. REVIEW_REQUIRED flag is retained and documented. "
                "Human review: APPROVED. K=75 accepted as final selection."
            ),
            "flat_range": existing_meta_full.get("selection_sanity", {}).get("flat_range", None),
            "flat_tolerance": 1e-3,
            "is_monotonic": True,
            "monotonic_tolerance": 1e-4,
            "human_reviewed": True,
            "human_decision": "APPROVED",
        },

        "runtime_final_refit": {
            "refit_duration_s": round(refit_elapsed, 1),
            "total_script_duration_s": round(actual_elapsed, 1),
        },
    })

    with open(meta_path, "w") as f:
        json.dump(existing_meta_full, f, indent=2)
    logger.info("metadata.json updated with final refit provenance")

    # --- Summary ---
    logger.info("")
    logger.info("=== FINAL REFIT SUMMARY ===")
    logger.info("  Experiment ID:     %s", EXPERIMENT_ID)
    logger.info("  Protocol version:  %s", PROTOCOL_VERSION)
    logger.info("  TRAIN rows:        %d", total)
    logger.info("  Encoded features:  %d", mi_result.n_features)
    logger.info("  Discrete/Cont:     %d / %d", discrete_count, continuous_count)
    logger.info("  Selected K*:       %d", FINAL_K)
    logger.info("  Sanity:            REVIEW_REQUIRED (human-approved)")
    logger.info("  Top-5 features:    %s", selected_names[:5])
    logger.info("  Family selected:   %s", family_report.get("selected", {}))
    logger.info("  Refit runtime:     %.1fs", refit_elapsed)
    logger.info("=== FINAL REFIT COMPLETE ===")

    print(f"\n{'='*60}")
    print("SPRINT 4 — FINAL REFIT COMPLETE")
    print(f"{'='*60}")
    print(f"  Experiment ID:   {EXPERIMENT_ID}")
    print(f"  Protocol:        v{PROTOCOL_VERSION}")
    print(f"  Selected K*:     {FINAL_K}")
    print(f"  Encoded feats:   {mi_result.n_features}")
    print(f"  Sanity:          REVIEW_REQUIRED — HUMAN APPROVED")
    print(f"  Top-5:           {selected_names[:5]}")
    print(f"  Family:          {family_report.get('selected', {})}")
    print(f"  Refit runtime:   {refit_elapsed:.1f}s")
    print(f"{'='*60}")
    print("\nSTATUS: READY FOR FREEZE REVIEW")
    print("  Do NOT create Git commit until human issues final freeze prompt.")


if __name__ == "__main__":
    main()
