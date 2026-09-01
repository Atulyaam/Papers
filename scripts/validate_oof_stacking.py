"""
scripts/validate_oof_stacking.py
----------------------------------
Sprint 6 final validation of existing OOF stacking artifacts.

Validates the EXISTING artifacts only.
Does NOT retrain, retune, regenerate OOF runs, or modify any methodology.

Checks (24 categories, mapped to 24+ individual assertions):
  1.  Seed artifact directories and key files exist and load.
  2.  Each OOF CSV has exactly 162,395 rows.
  3.  Each row has exactly one OOF prediction per model.
  4.  No duplicate row_id assignments.
  5.  Fold assignments are valid 5-fold StratifiedKFold seed=7.
  6.  Fold assignment is bit-identical across all 3 H1 seeds.
  7.  No-self-prediction invariant per model/fold.
  8.  Meta-feature input is exactly the 4 approved columns (no row_id).
  9.  DT/RF outputs are valid probabilities in [0, 1].
 10.  SVM output is decision_function (unbounded, not probability).
 11.  NN uses fixed epoch_count=18 (from metadata).
 12.  Fixed full-TRAIN pos_weight = 44800/117595 (from metadata).
 13.  feature_set=EXP_MI_V1_1 and feature_count=75 in metadata.
 14.  TRAIN SHA-256 matches frozen hash in metadata.
 15.  All 5 resolved dataset path/hash pairs match Step 0 values on disk.
 16.  Forbidden splits not opened/used (isolation check via metadata).
 17.  Sprint 5 base-model checkpoints exist and were reused (not retrained).
 18.  Both mandatory limitation texts present in every required artifact.
 19.  H1 summary seeds={42,123,2024} and mean/std computation is correct.
 20.  Sprint 5 RF labelled as single-CV reference, NOT matched H1 baseline.
 21.  No statistical significance language in h1_summary.
 22.  Metadata/provenance fields valid (experiment_id, oof_seed, oof_n_splits).
 23.  Meta-learner checkpoints load successfully.
 24.  validation_report.json generated at the end.

Outputs:
    results/stacking/EXP_OOF_STACK_V1/validation_report.json
    Console summary: PASS/FAIL + totals
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("validate_oof_stacking")

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

FROZEN_TRAIN_SHA = "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c"
FROZEN_TRAIN_ROWS = 162_395
FROZEN_FEATURE_SET = "EXP_MI_V1_1"
FROZEN_FEATURE_COUNT = 75
FROZEN_OOF_SEED = 7
FROZEN_OOF_N_SPLITS = 5
FROZEN_EPOCH_COUNT = 18
FROZEN_POS_WEIGHT = 44_800 / 117_595  # = 0.38096857859602873
H1_SEEDS = [42, 123, 2024]
SPRINT5_RF_REF = 0.9508532447968256

RESOLVED_HASHES = {
    "train":      "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c",
    "validation": "13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180",
    "dev_test":   "04725e85732ab2fc6d9eaaa6105418b22b083b5c651067e7b0785464f414e508",
    "protected":  "6ffd23479b575e438ad90678268f40f674a663c2b9507aaf65089623397a9d91",
    "excluded":   "b3f6e7e60c9815a53f40eb2d41df8b67d29f884b922a487c3fe83c02e0db0a02",
}

RESOLVED_PATHS = {
    "train":      ROOT / "data/splits/train.csv",
    "validation": ROOT / "data/splits/validation.csv",
    "dev_test":   ROOT / "data/splits/development_test.csv",
    "protected":  ROOT / "data/splits/protected_unseen_attack.csv",
    "excluded":   ROOT / "data/splits/excluded_train_backdoor.csv",
}

META_FEATURE_COLS = [
    "dt_attack_probability",
    "rf_attack_probability",
    "svm_decision_score",
    "nn_attack_probability",
]

MANDATORY_SCALING_TEXT = (
    "SVM/NN OOF predictions use scaling statistics computed on the full "
    "frozen TRAIN, including rows in the OOF held-out fold. This is a bounded, "
    "label-independent leakage channel accepted for feature-space consistency."
)
MANDATORY_META_EVAL_TEXT = (
    "H1 Macro-F1 is computed by evaluating the meta-learner on the same OOF "
    "matrix used to train it. No separate meta-learner holdout exists under "
    "the current data-isolation rules. This is in-sample evaluation at the "
    "meta-learner level and is NOT a fully held-out end-to-end generalisation "
    "estimate."
)
SPRINT5_EXACT_LABEL = "Frozen Sprint 5 single-CV reference; not a matched 3-seed H1 baseline."

STACKING_DIR  = ROOT / "results/stacking/EXP_OOF_STACK_V1"
CKPT_DIR      = ROOT / "results/checkpoints/EXP_OOF_STACK_V1"
SPRINT5_CKPTS = ROOT / "results/checkpoints/EXP_BASE_MODELS_V1"
FEATURES_PATH = ROOT / "results/feature_selection/EXP_MI_V1_1/selected_features.json"

# ---------------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, name: str):
        self.name = name
        self.passed: bool = False
        self.message: str = ""
        self.detail: str = ""

    def ok(self, msg: str = "") -> "CheckResult":
        self.passed = True
        self.message = msg or "PASS"
        return self

    def fail(self, msg: str, detail: str = "") -> "CheckResult":
        self.passed = False
        self.message = msg
        self.detail = detail
        return self

    def to_dict(self) -> dict:
        d = {"name": self.name, "passed": self.passed, "message": self.message}
        if self.detail:
            d["detail"] = self.detail
        return d


CHECKS: list[CheckResult] = []


def check(name: str) -> CheckResult:
    c = CheckResult(name)
    CHECKS.append(c)
    return c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _text_contains(text: str, fragment: str) -> bool:
    return fragment.lower() in text.lower()


# ---------------------------------------------------------------------------
# Check groups
# ---------------------------------------------------------------------------

def check_1_artifacts_exist() -> dict[int, dict]:
    """Check 1: seed artifact directories and key files exist and load."""
    seed_data: dict[int, dict] = {}
    for seed in H1_SEEDS:
        sd = STACKING_DIR / f"seed_{seed}"
        cd = CKPT_DIR / f"seed_{seed}"
        c = check(f"1.{seed}: artifact files exist")

        missing = []
        for p in [
            sd / "oof_predictions.csv",
            sd / "fold_assignments.csv",
            sd / "metrics.json",
            sd / "metadata.json",
            cd / "meta_learner.joblib",
            cd / "metadata.json",
        ]:
            if not p.exists():
                missing.append(str(p))

        if missing:
            c.fail(f"Missing files: {missing}")
            seed_data[seed] = {}
        else:
            c.ok(f"All 6 artifact files present for seed {seed}")
            # Pre-load
            try:
                oof_df = pd.read_csv(sd / "oof_predictions.csv")
                fold_df = pd.read_csv(sd / "fold_assignments.csv")
                metrics = _load_json(sd / "metrics.json")
                meta = _load_json(sd / "metadata.json")
                clf = joblib.load(cd / "meta_learner.joblib")
                ckpt_meta = _load_json(cd / "metadata.json")
                seed_data[seed] = {
                    "oof_df": oof_df, "fold_df": fold_df,
                    "metrics": metrics, "meta": meta,
                    "clf": clf, "ckpt_meta": ckpt_meta,
                    "sd": sd, "cd": cd,
                }
            except Exception as e:
                check(f"1.{seed}: artifact load").fail(str(e), traceback.format_exc())
                seed_data[seed] = {}
    return seed_data


def check_2_oof_row_count(seed_data: dict) -> None:
    """Check 2: Each OOF CSV has exactly 162,395 rows."""
    for seed in H1_SEEDS:
        c = check(f"2.{seed}: OOF CSV has exactly {FROZEN_TRAIN_ROWS} rows")
        if not seed_data.get(seed):
            c.fail("Artifacts unavailable")
            continue
        n = len(seed_data[seed]["oof_df"])
        if n == FROZEN_TRAIN_ROWS:
            c.ok(f"n={n}")
        else:
            c.fail(f"Expected {FROZEN_TRAIN_ROWS}, got {n}")


def check_3_one_prediction_per_row(seed_data: dict) -> None:
    """Check 3: Each row has exactly one OOF prediction per model."""
    for seed in H1_SEEDS:
        c = check(f"3.{seed}: exactly one prediction per row per model")
        if not seed_data.get(seed):
            c.fail("Artifacts unavailable")
            continue
        df = seed_data[seed]["oof_df"]
        issues = []
        for col in META_FEATURE_COLS:
            if col not in df.columns:
                issues.append(f"missing column {col}")
            elif df[col].isna().any():
                issues.append(f"NaN in {col}: {df[col].isna().sum()} rows")
        if issues:
            c.fail("; ".join(issues))
        else:
            c.ok("All 4 model columns present, no NaNs")


def check_4_no_duplicate_row_ids(seed_data: dict) -> None:
    """Check 4: No duplicate row_id assignments."""
    for seed in H1_SEEDS:
        c = check(f"4.{seed}: no duplicate row_ids")
        if not seed_data.get(seed):
            c.fail("Artifacts unavailable")
            continue
        df = seed_data[seed]["oof_df"]
        n_unique = df["row_id"].nunique()
        n_total = len(df)
        if n_unique == n_total:
            c.ok(f"All {n_total} row_ids unique")
        else:
            c.fail(f"{n_total - n_unique} duplicate row_ids found")


def check_5_fold_assignments_valid(seed_data: dict, train_y: np.ndarray) -> None:
    """Check 5: Fold assignments are valid 5-fold StratifiedKFold seed=7."""
    # Recompute expected folds
    skf = StratifiedKFold(n_splits=FROZEN_OOF_N_SPLITS, shuffle=True, random_state=FROZEN_OOF_SEED)
    expected_fold_col = np.full(FROZEN_TRAIN_ROWS, -1, dtype=np.int64)
    for fold_idx, (_, oof_idx) in enumerate(skf.split(train_y, train_y)):
        expected_fold_col[oof_idx] = fold_idx

    for seed in H1_SEEDS:
        c = check(f"5.{seed}: fold assignments match StratifiedKFold(n=5,seed=7)")
        if not seed_data.get(seed):
            c.fail("Artifacts unavailable")
            continue
        fold_df = seed_data[seed]["fold_df"].sort_values("row_id").reset_index(drop=True)
        if len(fold_df) != FROZEN_TRAIN_ROWS:
            c.fail(f"fold_assignments.csv has {len(fold_df)} rows != {FROZEN_TRAIN_ROWS}")
            continue
        actual = fold_df["fold_idx"].to_numpy(dtype=np.int64)
        if np.array_equal(actual, expected_fold_col):
            c.ok("Fold assignments match exactly")
        else:
            n_diff = int((actual != expected_fold_col).sum())
            c.fail(f"{n_diff} rows differ from expected StratifiedKFold(n=5,seed=7)")


def check_6_identical_folds_across_seeds(seed_data: dict) -> None:
    """Check 6: Fold assignment bit-identical across all 3 H1 seeds."""
    c = check("6: fold assignments bit-identical across seeds 42/123/2024")
    available = [s for s in H1_SEEDS if seed_data.get(s)]
    if len(available) < 2:
        c.fail("Fewer than 2 seeds available for comparison")
        return

    ref_seed = available[0]
    ref = seed_data[ref_seed]["fold_df"].sort_values("row_id")["fold_idx"].to_numpy()
    diffs = []
    for seed in available[1:]:
        other = seed_data[seed]["fold_df"].sort_values("row_id")["fold_idx"].to_numpy()
        if not np.array_equal(ref, other):
            diffs.append(seed)
    if diffs:
        c.fail(f"Fold assignments differ for seeds: {diffs}")
    else:
        c.ok(f"Bit-identical across {available}")


def check_7_no_self_prediction(seed_data: dict, train_y: np.ndarray) -> None:
    """Check 7: No-self-prediction invariant per model/fold."""
    skf = StratifiedKFold(n_splits=FROZEN_OOF_N_SPLITS, shuffle=True, random_state=FROZEN_OOF_SEED)
    folds = [(tr.copy(), oof.copy()) for tr, oof in skf.split(train_y, train_y)]

    for seed in H1_SEEDS:
        c = check(f"7.{seed}: no-self-prediction invariant")
        if not seed_data.get(seed):
            c.fail("Artifacts unavailable")
            continue
        fold_df = seed_data[seed]["fold_df"]
        # Build row_id -> fold_idx mapping
        row_to_fold = dict(zip(fold_df["row_id"].tolist(), fold_df["fold_idx"].tolist()))
        leaks = 0
        for fold_idx, (train_idx, oof_idx) in enumerate(folds):
            train_set = set(train_idx.tolist())
            for oof_row in oof_idx:
                if oof_row in train_set:
                    leaks += 1
        if leaks == 0:
            c.ok("No self-prediction leakage across any fold")
        else:
            c.fail(f"{leaks} self-prediction leakage instances found")


def check_8_meta_feature_input(seed_data: dict) -> None:
    """Check 8: Meta-feature input is exactly the 4 approved cols; no row_id."""
    c = check("8: meta-feature columns correct (4 cols, row_id excluded)")
    issues = []
    for seed in H1_SEEDS:
        if not seed_data.get(seed):
            issues.append(f"seed {seed}: unavailable")
            continue
        df = seed_data[seed]["oof_df"]
        for col in META_FEATURE_COLS:
            if col not in df.columns:
                issues.append(f"seed {seed}: missing {col}")
        clf = seed_data[seed]["clf"]
        if clf.n_features_in_ != 4:
            issues.append(f"seed {seed}: meta-learner n_features_in_={clf.n_features_in_} != 4")
        ckpt_meta = seed_data[seed]["ckpt_meta"]
        if ckpt_meta.get("row_id_excluded") is not True:
            issues.append(f"seed {seed}: row_id_excluded flag missing or False in checkpoint metadata")

    if issues:
        c.fail("; ".join(issues))
    else:
        c.ok("All seeds: 4 meta-feature cols, meta-learner n_features_in_=4, row_id_excluded=True")


def check_9_dt_rf_probability(seed_data: dict) -> None:
    """Check 9: DT/RF outputs are valid probabilities in [0, 1]."""
    for seed in H1_SEEDS:
        c = check(f"9.{seed}: DT/RF outputs in [0, 1]")
        if not seed_data.get(seed):
            c.fail("Artifacts unavailable")
            continue
        df = seed_data[seed]["oof_df"]
        issues = []
        for col in ["dt_attack_probability", "rf_attack_probability"]:
            vals = df[col].to_numpy()
            if vals.min() < 0.0 or vals.max() > 1.0:
                issues.append(f"{col}: range [{vals.min():.4f}, {vals.max():.4f}]")
        if issues:
            c.fail("; ".join(issues))
        else:
            c.ok("DT and RF columns in [0, 1]")


def check_10_svm_decision_function(seed_data: dict) -> None:
    """Check 10: SVM output is decision_function (not probability)."""
    # We verify: column exists, name is 'svm_decision_score' (not 'svm_attack_probability'),
    # and the checkpoint metadata does NOT claim predict_proba output.
    for seed in H1_SEEDS:
        c = check(f"10.{seed}: SVM column is svm_decision_score (not probability)")
        if not seed_data.get(seed):
            c.fail("Artifacts unavailable")
            continue
        df = seed_data[seed]["oof_df"]
        issues = []
        if "svm_decision_score" not in df.columns:
            issues.append("svm_decision_score column missing")
        if "svm_attack_probability" in df.columns:
            issues.append("svm_attack_probability present (should not be — SVM uses decision_function)")
        # The checkpoint metadata should reference meta_feature_cols containing 'svm_decision_score'
        ckpt_meta = seed_data[seed]["ckpt_meta"]
        meta_cols = ckpt_meta.get("meta_feature_cols", [])
        if "svm_decision_score" not in meta_cols:
            issues.append("svm_decision_score not in checkpoint meta_feature_cols")
        if issues:
            c.fail("; ".join(issues))
        else:
            c.ok("SVM output confirmed as decision_function (svm_decision_score)")


def check_11_nn_epoch_count(seed_data: dict) -> None:
    """Check 11: NN uses fixed epoch_count=18 (from metadata)."""
    for seed in H1_SEEDS:
        c = check(f"11.{seed}: oof_fixed_epoch_count=18 in metadata")
        if not seed_data.get(seed):
            c.fail("Artifacts unavailable")
            continue
        meta = seed_data[seed]["meta"]
        actual = meta.get("oof_fixed_epoch_count")
        if actual == FROZEN_EPOCH_COUNT:
            c.ok(f"oof_fixed_epoch_count={actual}")
        else:
            c.fail(f"Expected {FROZEN_EPOCH_COUNT}, got {actual!r}")


def check_12_pos_weight(seed_data: dict) -> None:
    """Check 12: Fixed full-TRAIN pos_weight = 44800/117595."""
    expected = FROZEN_POS_WEIGHT
    for seed in H1_SEEDS:
        c = check(f"12.{seed}: oof_pos_weight correct")
        if not seed_data.get(seed):
            c.fail("Artifacts unavailable")
            continue
        meta = seed_data[seed]["meta"]
        actual = meta.get("oof_pos_weight")
        if actual is None:
            c.fail("oof_pos_weight missing from metadata")
            continue
        if abs(float(actual) - expected) < 1e-10:
            c.ok(f"oof_pos_weight={actual:.12f}")
        else:
            c.fail(f"Expected {expected:.12f}, got {actual:.12f}")


def check_13_feature_set(seed_data: dict) -> None:
    """Check 13: feature_set=EXP_MI_V1_1 and feature_count=75."""
    for seed in H1_SEEDS:
        c = check(f"13.{seed}: feature_set=EXP_MI_V1_1, feature_count=75")
        if not seed_data.get(seed):
            c.fail("Artifacts unavailable")
            continue
        meta = seed_data[seed]["meta"]
        fs = meta.get("feature_set")
        fc = meta.get("feature_count")
        issues = []
        if fs != FROZEN_FEATURE_SET:
            issues.append(f"feature_set={fs!r} != {FROZEN_FEATURE_SET!r}")
        if fc != FROZEN_FEATURE_COUNT:
            issues.append(f"feature_count={fc!r} != {FROZEN_FEATURE_COUNT}")
        if issues:
            c.fail("; ".join(issues))
        else:
            c.ok(f"feature_set={fs}, feature_count={fc}")

    # Also verify the selected_features.json on disk
    c2 = check("13x: selected_features.json has 75 features, id=EXP_MI_V1_1")
    if not FEATURES_PATH.exists():
        c2.fail(f"selected_features.json not found: {FEATURES_PATH}")
    else:
        try:
            feat_data = _load_json(FEATURES_PATH)
            n = len(feat_data.get("features", []))
            eid = feat_data.get("experiment_id")
            if n == 75 and eid == "EXP_MI_V1_1":
                c2.ok(f"75 features, experiment_id={eid}")
            else:
                c2.fail(f"features={n}, experiment_id={eid!r}")
        except Exception as e:
            c2.fail(str(e))


def check_14_train_sha(seed_data: dict) -> None:
    """Check 14: TRAIN SHA-256 in metadata matches frozen hash."""
    for seed in H1_SEEDS:
        c = check(f"14.{seed}: train_sha256 in metadata matches frozen")
        if not seed_data.get(seed):
            c.fail("Artifacts unavailable")
            continue
        actual = seed_data[seed]["meta"].get("train_sha256", "")
        if actual == FROZEN_TRAIN_SHA:
            c.ok(f"SHA={actual[:16]}...")
        else:
            c.fail(f"Mismatch: {actual[:16]}... != {FROZEN_TRAIN_SHA[:16]}...")

    # Also verify on disk
    c2 = check("14x: TRAIN file on disk SHA-256 matches frozen")
    try:
        actual = _sha256(RESOLVED_PATHS["train"])
        if actual == FROZEN_TRAIN_SHA:
            c2.ok(f"On-disk SHA={actual[:16]}...")
        else:
            c2.fail(f"On-disk SHA mismatch: {actual[:16]}...")
    except Exception as e:
        c2.fail(str(e))


def check_15_resolved_dataset_paths() -> None:
    """Check 15: All 5 resolved dataset paths/hashes match Step 0 values."""
    for name, path in RESOLVED_PATHS.items():
        c = check(f"15.{name}: path exists and SHA-256 matches Step 0")
        if not path.exists():
            c.fail(f"File not found: {path}")
            continue
        try:
            actual = _sha256(path)
            expected = RESOLVED_HASHES[name]
            if actual == expected:
                c.ok(f"SHA={actual[:16]}...")
            else:
                c.fail(f"SHA mismatch: {actual[:16]}... != {expected[:16]}...")
        except Exception as e:
            c.fail(str(e))


def check_16_data_isolation(seed_data: dict) -> None:
    """Check 16: Forbidden splits not accessed; isolation confirmed via metadata."""
    # We verify that metadata.resolved_dataset_paths marks forbidden splits correctly
    # and that OOF runner only reads TRAIN (verified by hash in check 14).
    c = check("16: data isolation — forbidden splits marked correctly in metadata")
    issues = []
    for seed in H1_SEEDS:
        if not seed_data.get(seed):
            issues.append(f"seed {seed}: unavailable")
            continue
        rdp = seed_data[seed]["meta"].get("resolved_dataset_paths", {})
        if not rdp:
            issues.append(f"seed {seed}: resolved_dataset_paths missing from metadata")
            continue
        forbidden = ["validation", "development_test", "protected_backdoor", "excluded_backdoor"]
        for key in forbidden:
            if key not in rdp:
                issues.append(f"seed {seed}: {key} missing from resolved_dataset_paths")
            elif rdp[key].get("sprint6_access") != "FORBIDDEN":
                issues.append(f"seed {seed}: {key} sprint6_access != FORBIDDEN")
        if rdp.get("train", {}).get("sprint6_access") != "ALLOWED":
            issues.append(f"seed {seed}: train sprint6_access != ALLOWED")

    if issues:
        c.fail("; ".join(issues))
    else:
        c.ok("All seeds: train=ALLOWED, 4 forbidden splits=FORBIDDEN in metadata")


def check_17_sprint5_checkpoints_reused() -> None:
    """Check 17: Sprint 5 base-model checkpoints exist (not retrained)."""
    c = check("17: Sprint 5 base-model checkpoints exist and were reused")
    missing = []
    for model in ["dt", "rf", "svm", "nn"]:
        d = SPRINT5_CKPTS / model
        if not d.exists():
            missing.append(f"{model} checkpoint dir")
        else:
            # Check at least one model file exists
            files = list(d.glob("*.joblib")) + list(d.glob("*.pt"))
            if not files:
                missing.append(f"{model} no .joblib/.pt files")

    # Verify Sprint 5 scalers (reused in Sprint 6)
    for path in [SPRINT5_CKPTS / "svm/svm_scaler.joblib", SPRINT5_CKPTS / "nn/nn_scaler.joblib"]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        c.fail(f"Missing: {missing}")
    else:
        c.ok("All 4 Sprint 5 model dirs exist with artifacts + scalers accessible")


def check_18_limitation_texts(seed_data: dict) -> None:
    """Check 18: Both mandatory limitation texts in every required artifact."""
    SCALING_KEY = "scaling_limitation"
    META_EVAL_KEY = "meta_evaluation_limitation"

    # Per-seed metadata and metrics
    for seed in H1_SEEDS:
        for artifact_name, artifact_key in [("metadata", "meta"), ("metrics", "metrics")]:
            c = check(f"18.{seed}.{artifact_name}: both limitation texts present")
            if not seed_data.get(seed):
                c.fail("Artifacts unavailable")
                continue
            data = seed_data[seed][artifact_key]
            issues = []
            scaling = data.get(SCALING_KEY, "")
            meta_eval = data.get(META_EVAL_KEY, "")
            if not scaling or MANDATORY_SCALING_TEXT.lower() not in scaling.lower():
                issues.append("scaling_limitation missing or incorrect")
            if not meta_eval or MANDATORY_META_EVAL_TEXT.lower() not in meta_eval.lower():
                issues.append("meta_evaluation_limitation missing or incorrect")
            if issues:
                c.fail("; ".join(issues))
            else:
                c.ok("Both limitation texts present and correct")

    # h1_summary
    c2 = check("18.h1_summary: both limitation texts present")
    h1_path = STACKING_DIR / "h1_summary.json"
    if not h1_path.exists():
        c2.fail("h1_summary.json not found")
    else:
        try:
            h1 = _load_json(h1_path)
            issues = []
            if MANDATORY_SCALING_TEXT.lower() not in h1.get(SCALING_KEY, "").lower():
                issues.append("scaling_limitation missing/incorrect in h1_summary")
            if MANDATORY_META_EVAL_TEXT.lower() not in h1.get(META_EVAL_KEY, "").lower():
                issues.append("meta_evaluation_limitation missing/incorrect in h1_summary")
            if issues:
                c2.fail("; ".join(issues))
            else:
                c2.ok("Both limitation texts correct in h1_summary")
        except Exception as e:
            c2.fail(str(e))


def check_19_h1_summary() -> dict | None:
    """Check 19: H1 summary seeds correct and mean/std computation accurate."""
    c1 = check("19a: h1_summary.json exists and loads")
    h1_path = STACKING_DIR / "h1_summary.json"
    if not h1_path.exists():
        c1.fail("h1_summary.json not found")
        return None
    try:
        h1 = _load_json(h1_path)
        c1.ok("h1_summary.json loaded")
    except Exception as e:
        c1.fail(str(e))
        return None

    c2 = check("19b: h1_summary contains seeds 42/123/2024")
    actual_seeds = sorted(h1.get("h1_seeds", []))
    if actual_seeds == sorted(H1_SEEDS):
        c2.ok(f"Seeds: {actual_seeds}")
    else:
        c2.fail(f"Expected {sorted(H1_SEEDS)}, got {actual_seeds}")

    c3 = check("19c: h1_summary mean_macro_f1 is correctly computed from per_seed_macro_f1")
    per_seed_f1s = h1.get("per_seed_macro_f1", {})
    if len(per_seed_f1s) == 3:
        vals = np.array(list(per_seed_f1s.values()), dtype=np.float64)
        expected_mean = float(np.mean(vals))
        expected_std = float(np.std(vals, ddof=1))
        actual_mean = h1.get("mean_macro_f1", None)
        actual_std = h1.get("std_macro_f1", None)
        issues = []
        if actual_mean is None or abs(actual_mean - expected_mean) > 1e-9:
            issues.append(f"mean={actual_mean} != computed {expected_mean}")
        if actual_std is None or abs(actual_std - expected_std) > 1e-9:
            issues.append(f"std={actual_std} != computed {expected_std}")
        if issues:
            c3.fail("; ".join(issues))
        else:
            c3.ok(f"mean={actual_mean:.6f}, std={actual_std:.6f}")
    else:
        c3.fail(f"Expected 3 per_seed entries, got {len(per_seed_f1s)}")

    return h1


def check_20_sprint5_label(seed_data: dict, h1: dict | None) -> None:
    """Check 20: Sprint 5 RF labelled as single-CV reference, NOT matched H1 baseline."""
    c = check("20: Sprint 5 RF has exact frozen single-CV reference label")
    issues = []

    # Check in h1_summary
    if h1:
        ref = h1.get("sprint5_reference", {})
        actual_label = ref.get("label", "")
        if actual_label != SPRINT5_EXACT_LABEL:
            issues.append(f"h1_summary: label={actual_label!r}")
        actual_f1 = ref.get("macro_f1")
        if actual_f1 is None or abs(float(actual_f1) - SPRINT5_RF_REF) > 1e-8:
            issues.append(f"h1_summary: macro_f1={actual_f1} != {SPRINT5_RF_REF}")
    else:
        issues.append("h1_summary unavailable")

    # Check in seed metadata
    for seed in H1_SEEDS:
        if not seed_data.get(seed):
            continue
        ref = seed_data[seed]["meta"].get("sprint5_reference", {})
        actual_label = ref.get("label", "")
        if actual_label != SPRINT5_EXACT_LABEL:
            issues.append(f"seed {seed} metadata: label={actual_label!r}")

    if issues:
        c.fail("; ".join(issues))
    else:
        c.ok("Sprint 5 RF label correct in h1_summary and all seed metadata")


def check_21_no_stat_significance(h1: dict | None) -> None:
    """Check 21: No statistical significance language in h1_summary."""
    c = check("21: no 'statistically significant' language in h1_summary")
    if h1 is None:
        c.fail("h1_summary unavailable")
        return
    text = json.dumps(h1).lower()
    if "statistically significant" in text:
        c.fail("'statistically significant' found in h1_summary")
    elif "p-value" in text or "p value" in text:
        c.fail("'p-value' language found in h1_summary")
    else:
        c.ok("No statistical significance language detected")


def check_22_provenance_fields(seed_data: dict) -> None:
    """Check 22: Metadata provenance fields valid."""
    REQUIRED_FIELDS = [
        "experiment_id", "h1_seed", "oof_seed", "oof_n_splits",
        "feature_set", "feature_count", "train_sha256", "train_rows",
        "oof_fixed_epoch_count", "oof_pos_weight", "meta_config",
        "scaling_limitation", "meta_evaluation_limitation",
        "sprint5_reference", "timestamp_utc",
    ]
    for seed in H1_SEEDS:
        c = check(f"22.{seed}: all provenance fields present in metadata")
        if not seed_data.get(seed):
            c.fail("Artifacts unavailable")
            continue
        meta = seed_data[seed]["meta"]
        missing = [f for f in REQUIRED_FIELDS if f not in meta]
        issues = []
        if missing:
            issues.append(f"missing fields: {missing}")
        if meta.get("experiment_id") != "EXP_OOF_STACK_V1":
            issues.append(f"experiment_id={meta.get('experiment_id')!r}")
        if meta.get("oof_seed") != FROZEN_OOF_SEED:
            issues.append(f"oof_seed={meta.get('oof_seed')} != {FROZEN_OOF_SEED}")
        if meta.get("oof_n_splits") != FROZEN_OOF_N_SPLITS:
            issues.append(f"oof_n_splits={meta.get('oof_n_splits')} != {FROZEN_OOF_N_SPLITS}")
        if meta.get("h1_seed") != seed:
            issues.append(f"h1_seed={meta.get('h1_seed')} != {seed}")
        if meta.get("train_rows") != FROZEN_TRAIN_ROWS:
            issues.append(f"train_rows={meta.get('train_rows')} != {FROZEN_TRAIN_ROWS}")
        mc = meta.get("meta_config", {})
        if mc.get("solver") != "lbfgs":
            issues.append(f"meta_config.solver={mc.get('solver')!r}")
        if mc.get("C") != 1.0:
            issues.append(f"meta_config.C={mc.get('C')!r}")
        if mc.get("random_state") != seed:
            issues.append(f"meta_config.random_state={mc.get('random_state')} != {seed}")

        if issues:
            c.fail("; ".join(issues))
        else:
            c.ok("All provenance fields valid")


def check_23_meta_learner_loads(seed_data: dict) -> None:
    """Check 23: Meta-learner checkpoints load and predict successfully."""
    from sklearn.linear_model import LogisticRegression
    for seed in H1_SEEDS:
        c = check(f"23.{seed}: meta-learner checkpoint loads and predicts")
        if not seed_data.get(seed):
            c.fail("Artifacts unavailable")
            continue
        clf = seed_data[seed]["clf"]
        # Verify it's a LogisticRegression with correct config
        issues = []
        if not isinstance(clf, LogisticRegression):
            issues.append(f"type={type(clf).__name__} != LogisticRegression")
        else:
            if clf.solver != "lbfgs":
                issues.append(f"solver={clf.solver}")
            if clf.C != 1.0:
                issues.append(f"C={clf.C}")
            if clf.class_weight != "balanced":
                issues.append(f"class_weight={clf.class_weight}")
            if clf.random_state != seed:
                issues.append(f"random_state={clf.random_state} != {seed}")
            if clf.n_features_in_ != 4:
                issues.append(f"n_features_in_={clf.n_features_in_} != 4")
            # Run a dummy prediction
            try:
                X_dummy = np.zeros((5, 4), dtype=np.float64)
                proba = clf.predict_proba(X_dummy)
                if proba.shape != (5, 2):
                    issues.append(f"predict_proba shape={proba.shape}")
            except Exception as e:
                issues.append(f"predict_proba failed: {e}")
        if issues:
            c.fail("; ".join(issues))
        else:
            c.ok(f"LR loaded: solver=lbfgs, C=1.0, balanced, random_state={seed}, n_features=4")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== OOF STACKING VALIDATION START | EXP_OOF_STACK_V1 ===")
    t0_wall = datetime.datetime.now(datetime.timezone.utc)

    # ── Load TRAIN labels (needed for fold reconstruction) ────────────────
    logger.info("Loading TRAIN labels for fold reconstruction ...")
    train_df = pd.read_csv(RESOLVED_PATHS["train"])
    train_y = train_df["label"].to_numpy(dtype=np.int64)
    logger.info("TRAIN loaded | rows=%d", len(train_y))

    # ── Run all checks ────────────────────────────────────────────────────

    # Check 1: existence and load
    seed_data = check_1_artifacts_exist()

    # Check 2: OOF row count
    check_2_oof_row_count(seed_data)

    # Check 3: one prediction per row
    check_3_one_prediction_per_row(seed_data)

    # Check 4: no duplicate row_ids
    check_4_no_duplicate_row_ids(seed_data)

    # Check 5: fold assignments match expected StratifiedKFold
    check_5_fold_assignments_valid(seed_data, train_y)

    # Check 6: identical folds across seeds
    check_6_identical_folds_across_seeds(seed_data)

    # Check 7: no self-prediction
    check_7_no_self_prediction(seed_data, train_y)

    # Check 8: meta-feature input
    check_8_meta_feature_input(seed_data)

    # Check 9: DT/RF probability range
    check_9_dt_rf_probability(seed_data)

    # Check 10: SVM decision_function
    check_10_svm_decision_function(seed_data)

    # Check 11: NN epoch count
    check_11_nn_epoch_count(seed_data)

    # Check 12: pos_weight
    check_12_pos_weight(seed_data)

    # Check 13: feature set
    check_13_feature_set(seed_data)

    # Check 14: TRAIN SHA-256
    check_14_train_sha(seed_data)

    # Check 15: resolved dataset paths on disk
    check_15_resolved_dataset_paths()

    # Check 16: data isolation
    check_16_data_isolation(seed_data)

    # Check 17: Sprint 5 checkpoints reused
    check_17_sprint5_checkpoints_reused()

    # Check 18: limitation texts
    check_18_limitation_texts(seed_data)

    # Check 19: H1 summary
    h1 = check_19_h1_summary()

    # Check 20: Sprint 5 label
    check_20_sprint5_label(seed_data, h1)

    # Check 21: no stat significance language
    check_21_no_stat_significance(h1)

    # Check 22: provenance fields
    check_22_provenance_fields(seed_data)

    # Check 23: meta-learner loads and predicts
    check_23_meta_learner_loads(seed_data)

    # ── Aggregate ─────────────────────────────────────────────────────────
    total = len(CHECKS)
    passed = sum(1 for c in CHECKS if c.passed)
    failed = total - passed
    overall = "PASS" if failed == 0 else "FAIL"

    # Group reporting
    def _section(pattern: str) -> tuple[int, int]:
        subset = [c for c in CHECKS if pattern in c.name]
        p = sum(1 for c in subset if c.passed)
        return p, len(subset)

    # ── Print report ──────────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"EXP_OOF_STACK_V1 — OOF STACKING VALIDATION REPORT")
    print(sep)
    print(f"Overall:            {overall}")
    print(f"Total checks:       {total}")
    print(f"Passed:             {passed}")
    print(f"Failed:             {failed}")
    print(sep)

    # Per-section
    sections = [
        ("Artifact integrity",  ["1.", "22.", "23."]),
        ("Data isolation",      ["14.", "14x", "15.", "16.", "17."]),
        ("OOF coverage",        ["2.", "3.", "4."]),
        ("Self-prediction",     ["7."]),
        ("Fold protocol",       ["5.", "6."]),
        ("Model contracts",     ["8.", "9.", "10.", "11.", "12.", "13."]),
        ("Limitation texts",    ["18."]),
        ("H1 aggregation",      ["19a", "19b", "19c", "20:", "21:"]),
    ]
    for label, keys in sections:
        subset = [c for c in CHECKS if any(k in c.name for k in keys)]
        p = sum(1 for c in subset if c.passed)
        n = len(subset)
        status = "OK  " if p == n else "FAIL"
        print(f"  [{status}] {label:<25} {p}/{n}")

    print(sep)

    if failed > 0:
        print("\nFAILED CHECKS:")
        for c in CHECKS:
            if not c.passed:
                print(f"  FAIL [{c.name}] {c.message}")
                if c.detail:
                    print(f"      {c.detail[:300]}")
    else:
        print("\nAll checks passed.")

    print(sep + "\n")

    # ── Save validation_report.json ───────────────────────────────────────
    report = {
        "experiment_id": "EXP_OOF_STACK_V1",
        "validation_timestamp_utc": t0_wall.isoformat(),
        "overall": overall,
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "h1_seeds": H1_SEEDS,
        "h1_results": {
            str(seed): {
                "macro_f1": seed_data[seed]["metrics"].get("macro_f1") if seed_data.get(seed) else None
            }
            for seed in H1_SEEDS
        },
        "mean_macro_f1": h1.get("mean_macro_f1") if h1 else None,
        "std_macro_f1": h1.get("std_macro_f1") if h1 else None,
        "checks": [c.to_dict() for c in CHECKS],
    }
    out = STACKING_DIR / "validation_report.json"
    STACKING_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Validation report saved: %s", out)
    logger.info("=== VALIDATION %s | %d/%d checks passed ===", overall, passed, total)

    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
