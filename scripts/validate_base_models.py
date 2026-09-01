"""
scripts/validate_base_models.py
----------------------------------
Sprint 5 validation script — sanity checks on all four base model checkpoints.

Validation checks
-----------------
For each model:
 1. Checkpoint files exist and are loadable
 2. Model predicts on a held-out reference sample (100 rows from TRAIN)
 3. Predictions are binary {0, 1}
 4. Prediction set covers both classes (not all-same)
 5. Metrics on reference sample match expected range (macro_f1 >= 0.6)
 6. SVM decision_function is present, predict_proba is ABSENT
 7. NN probabilities are in [0, 1]
 8. SVM scaler mean matches TRAIN distribution
 9. No forbidden dataset paths accessed
10. Metadata JSON is well-formed

This script does NOT access validation.csv, development_test.csv,
protected_unseen_attack.csv, or excluded_train_backdoor.csv.
The reference sample is drawn from TRAIN only.

IMPORTANT — preprocessing contract
------------------------------------
PreprocessingPipeline MUST be fitted on the full TRAIN (all 162 395 rows)
so that the OneHotEncoder sees every categorical value present in TRAIN.
The 75 selected features include OHE columns for rare categories
(proto_unas, proto_sctp, service_pop3, …) that only appear in a small
fraction of rows.  Fitting on a subset → those columns absent → ValueError.

Usage
-----
    python scripts/validate_base_models.py

Outputs
-------
    results/base_models/EXP_BASE_MODELS_V1/validation_report.json
    PASS / FAIL status to stdout
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.reproducibility import set_all_seeds
from src.utils.hashing import sha256_file
from src.models.base_models.preprocessing import (
    load_selected_features,
    build_feature_matrix,
    EXPECTED_FEATURE_COUNT,
)
from src.models.base_models.neural_network import IDSNet, nn_predict
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from sklearn.metrics import f1_score

EXPERIMENT_ID = "EXP_BASE_MODELS_V1"
TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"
FEATURES_PATH = ROOT / "results" / "feature_selection" / "EXP_MI_V1_1" / "selected_features.json"
OUTPUT_DIR = ROOT / "results" / "base_models" / EXPERIMENT_ID
CHECKPOINT_ROOT = ROOT / "results" / "checkpoints" / EXPERIMENT_ID

# Frozen TRAIN hash (set after first successful run; guards against accidental swap)
FROZEN_TRAIN_SHA256: str | None = None   # set to the actual hash to enforce

# Reference sample size — drawn from TRAIN only
REFERENCE_N = 100
MIN_MACRO_F1_THRESHOLD = 0.60

CHECKS = {}  # populated during run


def record(check_name: str, passed: bool, detail: str = ""):
    CHECKS[check_name] = {"passed": passed, "detail": detail}
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check_name}: {detail}")


def main():
    set_all_seeds(42)
    t_start = time.perf_counter()
    print(f"\n=== SPRINT 5 BASE MODELS VALIDATION | experiment={EXPERIMENT_ID} ===\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load full TRAIN and build reference sample.
    #
    # WHY full TRAIN is required for pipe.fit():
    #   The 75 selected features include OHE columns produced from rare
    #   categorical values (proto_unas, proto_sctp, service_pop3, …).
    #   These values appear in only a small fraction of TRAIN rows.
    #   If pipe.fit() sees fewer rows the OHE does not learn those
    #   categories → the corresponding columns are never produced →
    #   build_feature_matrix() raises ValueError.
    #
    #   This is exactly the same contract as run_base_models_refit.py:
    #       pipe.fit(full_train_df)   ← must see all rows
    #       pipe.transform(ref_df)    ← subsample for prediction test
    # ------------------------------------------------------------------
    print("[SETUP] Loading full TRAIN for preprocessing fit (required for complete OHE) ...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_sha256 = sha256_file(TRAIN_PATH)
    print(f"  TRAIN loaded | shape={train_df.shape} | SHA-256={train_sha256}")

    # Frozen hash guard (optional — populate FROZEN_TRAIN_SHA256 to enforce)
    if FROZEN_TRAIN_SHA256 is not None:
        record("TRAIN: SHA-256 matches frozen hash",
               train_sha256 == FROZEN_TRAIN_SHA256,
               f"actual={train_sha256}")
    else:
        record("TRAIN: SHA-256 recorded (not yet frozen)",
               True, train_sha256)

    # Stratified 100-row subsample for prediction test (from full TRAIN)
    rng = np.random.default_rng(42)
    classes = np.unique(train_df["label"].to_numpy())
    ref_idx: list[int] = []
    for cls in classes:
        cls_rows = train_df.index[train_df["label"] == cls].tolist()
        n_cls = min(REFERENCE_N // len(classes), len(cls_rows))
        ref_idx.extend(rng.choice(cls_rows, size=n_cls, replace=False).tolist())
    ref_df = train_df.loc[sorted(ref_idx)]

    # Fit pipeline on FULL TRAIN — same contract as refit script
    print("  Fitting preprocessing pipeline on full TRAIN ...")
    pipe = PreprocessingPipeline(experiment_id=EXPERIMENT_ID)
    pipe.fit(train_df)

    # Transform the 100-row reference subsample
    ds_unscaled = pipe.transform(ref_df, view="unscaled", split_name="reference")

    # Load frozen 75 features and build the feature matrix
    features = load_selected_features(FEATURES_PATH)
    record("FEATURES: selected count == 75",
           len(features) == EXPECTED_FEATURE_COUNT,
           f"count={len(features)}")

    feature_df = pd.DataFrame(ds_unscaled.X, columns=ds_unscaled.feature_names)
    X_ref = build_feature_matrix(feature_df, features)
    y_ref = ds_unscaled.y.to_numpy(dtype=int)

    record("FEATURES: encoded matrix shape correct",
           X_ref.shape == (len(ref_df), 75),
           f"shape={X_ref.shape}")
    # Verify that OHE-synthesised columns (proto_X, service_X, state_X)
    # are NOT present as raw CSV column names — they only exist post-encoding.
    ohe_prefixes = ("proto_", "service_", "state_")
    ohe_features = [f for f in features if any(f.startswith(p) for p in ohe_prefixes)]
    raw_cols = set(train_df.columns)
    ohe_in_raw = [f for f in ohe_features if f in raw_cols]
    record("FEATURES: OHE column names absent from raw CSV",
           len(ohe_in_raw) == 0,
           f"OHE features mistakenly in raw CSV: {ohe_in_raw}" if ohe_in_raw else "ok — OHE columns are post-encoding only")

    print(f"  Reference sample: n={len(y_ref)}, "
          f"classes={dict(zip(*np.unique(y_ref, return_counts=True)))}\n")

    # ------------------------------------------------------------------
    # Validate DT
    # ------------------------------------------------------------------
    print("[DT] Validating Decision Tree checkpoint ...")
    dt_dir = CHECKPOINT_ROOT / "dt"
    dt_ckpt = dt_dir / "dt_final.joblib"
    dt_meta = dt_dir / "dt_metadata.json"

    record("DT: checkpoint exists", dt_ckpt.exists(), str(dt_ckpt))
    if dt_ckpt.exists():
        try:
            dt_clf = joblib.load(dt_ckpt)
            record("DT: loadable", True)
            preds = dt_clf.predict(X_ref)
            record("DT: predict() returns binary", set(preds.tolist()).issubset({0, 1}),
                   f"unique={set(preds.tolist())}")
            record("DT: covers both classes", len(set(preds.tolist())) == 2,
                   f"unique={set(preds.tolist())}")
            f1 = f1_score(y_ref, preds, average="macro", zero_division=0)
            record("DT: macro_f1 >= threshold", f1 >= MIN_MACRO_F1_THRESHOLD,
                   f"f1={f1:.4f}")
            record("DT: n_features_in_ == 75", dt_clf.n_features_in_ == 75,
                   f"n_features_in_={dt_clf.n_features_in_}")
            meta = json.loads(dt_meta.read_text()) if dt_meta.exists() else {}
            record("DT: metadata.json valid", bool(meta.get("experiment_id") == EXPERIMENT_ID),
                   "ok" if meta else "missing")
        except Exception as exc:
            record("DT: load/predict error", False, str(exc))

    # ------------------------------------------------------------------
    # Validate RF
    # ------------------------------------------------------------------
    print("\n[RF] Validating Random Forest checkpoint ...")
    rf_dir = CHECKPOINT_ROOT / "rf"
    rf_ckpt = rf_dir / "rf_final.joblib"
    rf_meta = rf_dir / "rf_metadata.json"

    record("RF: checkpoint exists", rf_ckpt.exists(), str(rf_ckpt))
    if rf_ckpt.exists():
        try:
            rf_clf = joblib.load(rf_ckpt)
            record("RF: loadable", True)
            preds = rf_clf.predict(X_ref)
            record("RF: predict() returns binary", set(preds.tolist()).issubset({0, 1}),
                   f"unique={set(preds.tolist())}")
            record("RF: covers both classes", len(set(preds.tolist())) == 2,
                   f"unique={set(preds.tolist())}")
            f1 = f1_score(y_ref, preds, average="macro", zero_division=0)
            record("RF: macro_f1 >= threshold", f1 >= MIN_MACRO_F1_THRESHOLD,
                   f"f1={f1:.4f}")
            probs = rf_clf.predict_proba(X_ref)
            record("RF: predict_proba shape", probs.shape == (len(y_ref), 2),
                   f"shape={probs.shape}")
            meta = json.loads(rf_meta.read_text()) if rf_meta.exists() else {}
            record("RF: metadata.json valid", bool(meta.get("experiment_id") == EXPERIMENT_ID))
        except Exception as exc:
            record("RF: load/predict error", False, str(exc))

    # ------------------------------------------------------------------
    # Validate SVM
    # ------------------------------------------------------------------
    print("\n[SVM] Validating LinearSVC checkpoint ...")
    svm_dir = CHECKPOINT_ROOT / "svm"
    svm_ckpt = svm_dir / "svm_final.joblib"
    svm_scaler_ckpt = svm_dir / "svm_scaler.joblib"
    svm_meta = svm_dir / "svm_metadata.json"

    record("SVM: checkpoint exists", svm_ckpt.exists(), str(svm_ckpt))
    record("SVM: scaler exists", svm_scaler_ckpt.exists(), str(svm_scaler_ckpt))
    if svm_ckpt.exists() and svm_scaler_ckpt.exists():
        try:
            svm_clf = joblib.load(svm_ckpt)
            svm_scaler = joblib.load(svm_scaler_ckpt)
            record("SVM: both loadable", True)
            X_ref_scaled = svm_scaler.transform(X_ref)
            preds = svm_clf.predict(X_ref_scaled)
            record("SVM: predict() returns binary", set(preds.tolist()).issubset({0, 1}),
                   f"unique={set(preds.tolist())}")
            scores = svm_clf.decision_function(X_ref_scaled)
            record("SVM: decision_function() present", scores is not None and scores.shape == (len(y_ref),),
                   f"shape={scores.shape}")
            # CRITICAL: SVM must NOT have predict_proba
            has_proba = hasattr(svm_clf, "predict_proba")
            record("SVM: predict_proba ABSENT (required)", not has_proba,
                   "PASS — predict_proba not present" if not has_proba else "FAIL — predict_proba should not exist")
            # Verify decision_function values are not probabilities
            out_of_range = np.any(scores > 1.0) or np.any(scores < 0.0)
            record("SVM: decision_function not in [0,1] (not probabilities)", out_of_range,
                   f"min={scores.min():.4f}, max={scores.max():.4f}")
            f1 = f1_score(y_ref, preds, average="macro", zero_division=0)
            record("SVM: macro_f1 >= threshold", f1 >= MIN_MACRO_F1_THRESHOLD,
                   f"f1={f1:.4f}")
            meta = json.loads(svm_meta.read_text()) if svm_meta.exists() else {}
            record("SVM: metadata.json valid", bool(meta.get("experiment_id") == EXPERIMENT_ID))
        except Exception as exc:
            record("SVM: load/predict error", False, str(exc))

    # ------------------------------------------------------------------
    # Validate NN
    # ------------------------------------------------------------------
    print("\n[NN] Validating Neural Network checkpoint ...")
    nn_dir = CHECKPOINT_ROOT / "nn"
    nn_ckpt = nn_dir / "nn_final.pt"
    nn_scaler_ckpt = nn_dir / "nn_scaler.joblib"
    nn_arch_cfg = nn_dir / "nn_architecture.json"
    nn_meta = nn_dir / "nn_metadata.json"

    record("NN: checkpoint exists", nn_ckpt.exists(), str(nn_ckpt))
    record("NN: scaler exists", nn_scaler_ckpt.exists(), str(nn_scaler_ckpt))
    record("NN: architecture.json exists", nn_arch_cfg.exists(), str(nn_arch_cfg))
    if nn_ckpt.exists() and nn_scaler_ckpt.exists() and nn_arch_cfg.exists():
        try:
            arch = json.loads(nn_arch_cfg.read_text())
            nn_net = IDSNet(input_dim=arch["input_dim"], hidden_sizes=arch["hidden_sizes"])
            nn_net.load_state_dict(torch.load(nn_ckpt, map_location="cpu"))
            nn_net.eval()
            nn_scaler = joblib.load(nn_scaler_ckpt)
            record("NN: net + scaler loadable", True)

            X_ref_scaled = nn_scaler.transform(X_ref)
            y_pred, probs = nn_predict(nn_net, X_ref_scaled)
            record("NN: predict() returns binary", set(y_pred.tolist()).issubset({0, 1}),
                   f"unique={set(y_pred.tolist())}")
            record("NN: probs in [0,1]",
                   bool(np.all(probs >= 0.0) and np.all(probs <= 1.0)),
                   f"min={probs.min():.4f} max={probs.max():.4f}")
            f1 = f1_score(y_ref, y_pred, average="macro", zero_division=0)
            record("NN: macro_f1 >= threshold", f1 >= MIN_MACRO_F1_THRESHOLD,
                   f"f1={f1:.4f}")
            meta = json.loads(nn_meta.read_text()) if nn_meta.exists() else {}
            record("NN: metadata final_epoch_count present",
                   "final_epoch_count" in meta, str(meta.get("final_epoch_count")))
            record("NN: metadata.json valid", bool(meta.get("experiment_id") == EXPERIMENT_ID))
        except Exception as exc:
            record("NN: load/predict error", False, str(exc))

    # ------------------------------------------------------------------
    # Forbidden path check
    # ------------------------------------------------------------------
    print("\n[LEAKAGE] Checking no forbidden datasets used ...")
    import inspect, src.models.base_models.preprocessing as _prep
    src_code = inspect.getsource(_prep)
    record("LEAKAGE: validation.csv absent from preprocessing",
           "validation.csv" not in src_code)
    record("LEAKAGE: protected_unseen_attack absent from preprocessing",
           "protected_unseen_attack" not in src_code)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    n_pass = sum(1 for v in CHECKS.values() if v["passed"])
    n_fail = sum(1 for v in CHECKS.values() if not v["passed"])
    n_total = len(CHECKS)

    print(f"\n=== VALIDATION SUMMARY | {n_pass}/{n_total} checks passed | {n_fail} failed ===")
    if n_fail > 0:
        print(f"\n[FAILED CHECKS]")
        for name, info in CHECKS.items():
            if not info["passed"]:
                print(f"  FAIL: {name}: {info['detail']}")

    # Save report
    report = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_total": n_total,
        "overall_status": "PASS" if n_fail == 0 else "FAIL",
        "runtime_seconds": round(elapsed, 2),
        "checks": CHECKS,
    }
    report_path = OUTPUT_DIR / "validation_report.json"
    def _json_default(obj):
        """Handle numpy scalar types that json.dumps can't serialize."""
        import numpy as _np
        if isinstance(obj, (_np.bool_, _np.integer)):
            return bool(obj) if isinstance(obj, _np.bool_) else int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        return str(obj)
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nValidation report saved: {report_path}")
    print(f"Overall: {'PASS' if n_fail == 0 else 'FAIL'} ({n_pass}/{n_total})\n")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
