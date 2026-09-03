"""
scripts/run_determinism_check.py

Sprint 9 Deterministic Verification Run (T-DETERMINISTIC)

PURPOSE
-------
Performs a second, identical evaluation pass using the corrected
evaluate_sprint9.py implementation. Outputs are written EXCLUSIVELY to:

    results/evaluation/EXP_H123_V1/determinism_check/

The original Sprint 9 artifacts in EXP_H123_V1/ are NEVER overwritten.

RULES
-----
- Same frozen checkpoints (verified via SHA-256)
- Same frozen datasets (verified via SHA-256)
- Same config (epsilon, tau, seeds, supported_threshold, fpr_cap)
- Same preprocessing pipeline (fitted fresh on TRAIN as in original run)
- Corrected DD-7 H1 boundary: diff < -epsilon -> NOT_SUPPORTED
- H3 is a re-presentation of frozen Sprint 8 evidence (not new inference)

OUTPUT
------
determinism_check/rerun_h1_results.json
determinism_check/rerun_h2_results.json
determinism_check/rerun_h3_results.json
determinism_check/rerun_runtime_report.json
determinism_check/comparison.json
determinism_check/README.md
"""

import sys
import json
import hashlib
import time
import logging
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.autoencoder.ae_model import Autoencoder
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.models.base_models.preprocessing import load_selected_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("sprint9.determinism_check")

# Output directory — NEVER the main EXP_H123_V1/ root
ORIG_DIR = ROOT / "results/evaluation/EXP_H123_V1"
CHECK_DIR = ORIG_DIR / "determinism_check"

# -----------------------------------------------------------------------
# LOCKED CONFIG (identical to original run)
# -----------------------------------------------------------------------
CONFIG = {
    "h1_seeds":             [42, 123, 2024],
    "h1_epsilon":           0.005,
    "h2_tau":               11.160062745213509,
    "h2_threshold_id":      "mean+3sigma",
    "h2_multiplier":        3,
    "h3_fpr_cap":           0.02,
    "n_features":           75,
    "feature_set":          "EXP_MI_V1_1",
    "n_dev_test":           81749,
    "n_prot":               583,
    "n_validation":         11200,
}

DATASET_HASHES = {
    "train":              "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c",
    "validation":         "13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180",
    "development_test":   "04725e85732ab2fc6d9eaaa6105418b22b083b5c651067e7b0785464f414e508",
    "protected_backdoor": "6ffd23479b575e438ad90678268f40f674a663c2b9507aaf65089623397a9d91",
}

CHECKPOINT_HASHES = {
    "EXP_BASE_MODELS_V1": [
        ("dt_final_joblib",  "results/checkpoints/EXP_BASE_MODELS_V1/dt/dt_final.joblib",  "748261c8106e5b12a93decb4de7df435e09dd587b03294dba3837e20c8a2e4a3"),
        ("rf_final_joblib",  "results/checkpoints/EXP_BASE_MODELS_V1/rf/rf_final.joblib",  "f1f873ef4bd7f09c03885ffbbc4c9ec51306dc2aecc0f48e4584fddd7a97a68f"),
        ("svm_final_joblib", "results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_final.joblib","f325d57525dda5bd92cc20c5393a38fa1b9ca055001b0c24fc9402bdbece990c"),
        ("svm_scaler_joblib","results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_scaler.joblib","a85eeeb74d34bed8cead09cc7506c4bbac6522bb1df0467d6904178996bdaa85"),
        ("nn_final_pt",      "results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_final.pt",       "7f3dcdfa59cbd084fcd952645db3b14fa67554769500551f06737d42e5e058ae"),
        ("nn_scaler_joblib", "results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_scaler.joblib",  "a85eeeb74d34bed8cead09cc7506c4bbac6522bb1df0467d6904178996bdaa85"),
    ],
    "EXP_OOF_STACK_V1": [
        ("seed_42_meta",   "results/checkpoints/EXP_OOF_STACK_V1/seed_42/meta_learner.joblib",   "e5b776680a99ffee3271624445f7f52593f8f94037d20ba56e9f4b54a848ef19"),
        ("seed_123_meta",  "results/checkpoints/EXP_OOF_STACK_V1/seed_123/meta_learner.joblib",  "f6517b59fac54864b82db07f3da35139f21f400e2a7664ef56ee29b09fcd6672"),
        ("seed_2024_meta", "results/checkpoints/EXP_OOF_STACK_V1/seed_2024/meta_learner.joblib", "f6139a79f3e7c96bb2c6610f22907184df117a06dd110ea74d6eb1897aeada74"),
    ],
    "EXP_AE_V1": [
        ("ae_final_pt",              "results/checkpoints/EXP_AE_V1/ae_final.pt",                                        "4ab66af8d4a6e61212ef5d78360f30a8caa68aa85dac3d54042218e010f9a1d6"),
        ("ae_scaler_joblib",         "results/checkpoints/EXP_AE_V1/ae_scaler.joblib",                                   "c0128d42ed9ef5be695f261be75155e7de4ddf8e51b926e3ce516c4a88ad8211"),
        ("threshold_calibration_json","results/autoencoder/EXP_AE_V1/threshold/threshold_calibration.json",              "29bd47b8a0dd886383d312e1364320c9ada62d78989c4c5f847a96f8c1882971"),
    ],
}

DATASET_PATHS = {
    "train":              ROOT / "data/splits/train.csv",
    "validation":         ROOT / "data/splits/validation.csv",
    "development_test":   ROOT / "data/splits/development_test.csv",
    "protected_backdoor": ROOT / "data/splits/protected_unseen_attack.csv",
}

FP_TOL = 1e-8  # np.allclose tolerance per Final Design §13 T-DETERMINISTIC

def get_hash(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def run():
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    logger.info("=== T-DETERMINISTIC second evaluation run ===")

    # ------------------------------------------------------------------
    # STEP 1: Verify all checkpoint and dataset hashes (identical to original)
    # ------------------------------------------------------------------
    logger.info("STEP 1: Hash verification")
    for exp, files in CHECKPOINT_HASHES.items():
        for key, rel_path, expected in files:
            sha = get_hash(ROOT / rel_path)
            assert sha == expected, f"HALT: checkpoint hash mismatch {rel_path}: got {sha}"

    for name, expected in DATASET_HASHES.items():
        sha = get_hash(DATASET_PATHS[name])
        assert sha == expected, f"HALT: dataset hash mismatch {name}: got {sha}"

    logger.info("All 12 checkpoint + 4 dataset hashes verified")

    # ------------------------------------------------------------------
    # Load models and preprocessing (identical to original run)
    # ------------------------------------------------------------------
    features = load_selected_features()
    assert len(features) == 75, f"Feature count mismatch: {len(features)}"

    train_raw = pd.read_csv(DATASET_PATHS["train"])
    pipeline = PreprocessingPipeline()
    pipeline.fit(train_raw)

    # Base models — exact same loading as evaluate_sprint9.py
    dt         = joblib.load(ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/dt/dt_final.joblib")
    rf         = joblib.load(ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/rf/rf_final.joblib")
    svm        = joblib.load(ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_final.joblib")
    svm_scaler = joblib.load(ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_scaler.joblib")
    nn_scaler  = joblib.load(ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_scaler.joblib")

    from src.models.base_models.neural_network import IDSNet
    nn_model = IDSNet(input_dim=75, hidden_sizes=[128, 64])
    nn_model.load_state_dict(torch.load(
        ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_final.pt",
        map_location="cpu", weights_only=True))
    nn_model.eval()

    # AE
    ae_scaler = joblib.load(ROOT / "results/checkpoints/EXP_AE_V1/ae_scaler.joblib")
    ae_model  = Autoencoder(input_dim=75)
    ae_model.load_state_dict(torch.load(
        ROOT / "results/checkpoints/EXP_AE_V1/ae_final.pt",
        map_location="cpu", weights_only=True))
    ae_model.eval()

    # Inference — exactly mirrors evaluate_sprint9.py
    # X_dev is the raw unscaled 75-feature matrix used for DT/RF/NN-scaler
    # SVM uses svm_scaler; NN uses nn_scaler
    # rf_prob reused: same array for RF binary baseline AND stacking meta-feature column (T-RF-PREDICTION-REUSE)

    def get_ae_re(df_enc):
        X = df_enc[features].values
        X_sc = ae_scaler.transform(X).astype(np.float32)
        res = []
        with torch.no_grad():
            for i in range(0, len(X_sc), 1024):
                x_t = torch.tensor(X_sc[i:i+1024])
                x_h = ae_model(x_t)
                res.append(((x_t - x_h) ** 2).mean(dim=1).cpu().numpy())
        return np.concatenate(res)

    # ------------------------------------------------------------------
    # H2: AE Normal VALIDATION FPR check (before any Protected Backdoor access)
    # ------------------------------------------------------------------
    logger.info("H2: AE VAL FPR check")
    val_raw = pd.read_csv(DATASET_PATHS["validation"])
    val_enc = pipeline.transform(val_raw, view="unscaled", split_name="validation")
    val_df  = pd.DataFrame(val_enc.X, columns=val_enc.feature_names)
    val_re  = get_ae_re(val_df)
    val_flagged = int((val_re > CONFIG["h2_tau"]).sum())
    assert val_flagged == 7, f"HALT T-AE-VAL-FPR-CONSISTENCY: ae_val_flagged={val_flagged} != 7"
    ae_val_fpr = val_flagged / len(val_df)
    logger.info(f"ae_val_flagged={val_flagged}, ae_val_fpr={ae_val_fpr}")

    # ------------------------------------------------------------------
    # H1: Dev TEST stacking inference (3 seeds)
    # ------------------------------------------------------------------
    logger.info("H1: Dev TEST inference")
    dev_raw = pd.read_csv(DATASET_PATHS["development_test"])
    dev_enc = pipeline.transform(dev_raw, view="unscaled", split_name="development_test")
    dev_df  = pd.DataFrame(dev_enc.X, columns=dev_enc.feature_names)
    dev_labels = dev_enc.y

    assert len(dev_labels) == CONFIG["n_dev_test"], f"n_dev_test mismatch: {len(dev_labels)}"

    X_dev = dev_df[features].values

    # T-RF-PREDICTION-REUSE: compute exactly once
    rf_prob  = rf.predict_proba(X_dev)[:, 1]
    rf_preds = (rf_prob >= 0.5).astype(int)
    rf_macro_f1 = f1_score(dev_labels, rf_preds, average="macro")

    # Other base model scores/probabilities
    dt_prob  = dt.predict_proba(X_dev)[:, 1]
    X_svm    = svm_scaler.transform(X_dev)
    svm_score = svm.decision_function(X_svm)

    X_nn = nn_scaler.transform(X_dev)
    with torch.no_grad():
        nn_prob = torch.sigmoid(nn_model(torch.tensor(X_nn, dtype=torch.float32))).numpy()

    # Meta-feature matrix: same columns as original (DT prob, RF prob, SVM score, NN prob)
    meta_X = np.column_stack([dt_prob, rf_prob, svm_score, nn_prob])

    stacking_macro_f1s, stacking_weighted_f1s, stacking_balanced_accs, stacking_accs = [], [], [], []
    for seed in CONFIG["h1_seeds"]:
        meta_lr = joblib.load(ROOT / f"results/checkpoints/EXP_OOF_STACK_V1/seed_{seed}/meta_learner.joblib")
        preds = meta_lr.predict(meta_X)
        stacking_macro_f1s.append(f1_score(dev_labels, preds, average="macro"))
        stacking_weighted_f1s.append(f1_score(dev_labels, preds, average="weighted"))
        stacking_balanced_accs.append(balanced_accuracy_score(dev_labels, preds))
        stacking_accs.append(accuracy_score(dev_labels, preds))
    mean_macro_f1  = np.mean(stacking_macro_f1s)
    diff           = mean_macro_f1 - rf_macro_f1

    # Corrected DD-7 three-way verdict function
    if diff > CONFIG["h1_epsilon"]:
        h1_verdict = "SUPPORTED"
    elif diff < -CONFIG["h1_epsilon"]:   # DD-7 LOCKED
        h1_verdict = "NOT_SUPPORTED"
    else:
        h1_verdict = "INCONCLUSIVE"

    rerun_h1 = {
        "stacking_macro_f1_seed_42":   float(stacking_macro_f1s[0]),
        "stacking_macro_f1_seed_123":  float(stacking_macro_f1s[1]),
        "stacking_macro_f1_seed_2024": float(stacking_macro_f1s[2]),
        "stacking_mean_macro_f1":      float(mean_macro_f1),
        "stacking_std_macro_f1":       float(np.std(stacking_macro_f1s)),   # ddof=0
        "rf_dev_test_macro_f1":        float(rf_macro_f1),
        "diff":                        float(diff),
        "h1_verdict":                  h1_verdict,
        "epsilon":                     CONFIG["h1_epsilon"],
        "seeds":                       CONFIG["h1_seeds"],
    }
    with open(CHECK_DIR / "rerun_h1_results.json", "w") as f:
        json.dump(rerun_h1, f, indent=2)
    logger.info(f"H1 rerun: diff={diff:.10f}, verdict={h1_verdict}")

    # ------------------------------------------------------------------
    # H2: AE-only Protected Backdoor detection
    # ------------------------------------------------------------------
    logger.info("H2: AE Protected Backdoor")
    prot_raw = pd.read_csv(DATASET_PATHS["protected_backdoor"])
    prot_enc = pipeline.transform(prot_raw, view="unscaled", split_name="protected_unseen_attack")
    prot_df  = pd.DataFrame(prot_enc.X, columns=prot_enc.feature_names)
    prot_re  = get_ae_re(prot_df)
    ae_detected = int((prot_re > CONFIG["h2_tau"]).sum())

    if ae_detected >= 2:
        h2_verdict = "SUPPORTED"
    elif ae_detected == 1:
        h2_verdict = "INCONCLUSIVE"
    else:
        h2_verdict = "NOT_SUPPORTED"

    rerun_h2 = {
        "ae_detected_count":       ae_detected,
        "n_prot":                  CONFIG["n_prot"],
        "tau":                     CONFIG["h2_tau"],
        "threshold_id":            CONFIG["h2_threshold_id"],
        "ae_val_fpr_recomputed":   float(ae_val_fpr),
        "supported_threshold_locked": True,
        "h2_verdict":              h2_verdict,
    }
    with open(CHECK_DIR / "rerun_h2_results.json", "w") as f:
        json.dump(rerun_h2, f, indent=2)
    logger.info(f"H2 rerun: detected={ae_detected}, verdict={h2_verdict}")

    # ------------------------------------------------------------------
    # H3: Re-presentation of frozen Sprint 8 evidence (no new inference)
    # ------------------------------------------------------------------
    logger.info("H3: inherited frozen evidence")
    c01_detected, c06_detected = 582, 582
    c01_fpr, c06_fpr = 0.191892, 0.192243
    fpr_delta = c06_fpr - c01_fpr

    if c06_detected > c01_detected and fpr_delta <= CONFIG["h3_fpr_cap"]:
        h3_verdict = "SUPPORTED"
    elif c06_detected <= c01_detected:
        h3_verdict = "NOT_SUPPORTED"
    else:
        h3_verdict = "INCONCLUSIVE"

    rerun_h3 = {
        "c01_detected":   c01_detected,
        "c06_detected":   c06_detected,
        "c01_dev_test_fpr": c01_fpr,
        "c06_dev_test_fpr": c06_fpr,
        "fpr_delta":      fpr_delta,
        "fpr_cap":        CONFIG["h3_fpr_cap"],
        "n_prot":         CONFIG["n_prot"],
        "h3_verdict":     h3_verdict,
        "evidence_source": "EXP_FUSION_V1 frozen artifacts — no new H3 inference",
    }
    with open(CHECK_DIR / "rerun_h3_results.json", "w") as f:
        json.dump(rerun_h3, f, indent=2)
    logger.info(f"H3 rerun: verdict={h3_verdict}")

    # ------------------------------------------------------------------
    # Runtime report
    # ------------------------------------------------------------------
    t_end = time.time()
    with open(CHECK_DIR / "rerun_runtime_report.json", "w") as f:
        json.dump({"total_seconds": t_end - t_start,
                   "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}, f, indent=2)

    # ------------------------------------------------------------------
    # Field-by-field comparison against original stored artifacts
    # ------------------------------------------------------------------
    logger.info("Comparing rerun vs original artifacts")

    orig_h1 = json.load(open(ORIG_DIR / "h1_results.json"))
    orig_h2 = json.load(open(ORIG_DIR / "h2_results.json"))
    orig_h3 = json.load(open(ORIG_DIR / "h3_results.json"))

    def cmp_field(field, orig, rerun, is_float=False):
        if is_float:
            match = np.isclose(orig, rerun, rtol=FP_TOL, atol=FP_TOL)
            diff_val = abs(orig - rerun)
        else:
            match = orig == rerun
            diff_val = None
        return {
            "field": field,
            "original": orig,
            "rerun": rerun,
            "match": bool(match),
            "diff": diff_val,
            "tolerance": FP_TOL if is_float else "exact",
            "status": "PASS" if match else "FAIL",
        }

    comparisons = []

    # H1 fields
    for key in ["stacking_macro_f1_seed_42", "stacking_macro_f1_seed_123",
                "stacking_macro_f1_seed_2024", "stacking_mean_macro_f1",
                "stacking_std_macro_f1", "rf_dev_test_macro_f1", "diff"]:
        comparisons.append(cmp_field(f"h1.{key}", orig_h1[key], rerun_h1[key], is_float=True))
    comparisons.append(cmp_field("h1.h1_verdict", orig_h1["h1_verdict"], rerun_h1["h1_verdict"]))
    comparisons.append(cmp_field("h1.epsilon", orig_h1["epsilon"], rerun_h1["epsilon"], is_float=True))

    # H2 fields
    comparisons.append(cmp_field("h2.ae_detected_count", orig_h2["ae_detected_count"], rerun_h2["ae_detected_count"]))
    comparisons.append(cmp_field("h2.tau",               orig_h2["tau"],               rerun_h2["tau"], is_float=True))
    comparisons.append(cmp_field("h2.ae_val_fpr_recomputed", orig_h2["ae_val_fpr_recomputed"], rerun_h2["ae_val_fpr_recomputed"], is_float=True))
    comparisons.append(cmp_field("h2.h2_verdict",        orig_h2["h2_verdict"],        rerun_h2["h2_verdict"]))

    # H3 fields (inherited — all must match exactly)
    comparisons.append(cmp_field("h3.c01_detected", orig_h3["c01_detected"], rerun_h3["c01_detected"]))
    comparisons.append(cmp_field("h3.c06_detected", orig_h3["c06_detected"], rerun_h3["c06_detected"]))
    comparisons.append(cmp_field("h3.fpr_delta",    orig_h3["fpr_delta"],    rerun_h3["fpr_delta"], is_float=True))
    comparisons.append(cmp_field("h3.h3_verdict",   orig_h3["h3_verdict"],   rerun_h3["h3_verdict"]))

    n_pass = sum(1 for c in comparisons if c["status"] == "PASS")
    n_fail = sum(1 for c in comparisons if c["status"] == "FAIL")
    t_deterministic = "PASS" if n_fail == 0 else "FAIL"

    comparison_doc = {
        "run_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "fp_tolerance": FP_TOL,
        "t_deterministic": t_deterministic,
        "total_fields": len(comparisons),
        "pass_count": n_pass,
        "fail_count": n_fail,
        "fields": comparisons,
        "provenance_note": (
            "Original Sprint 9 result artifacts were generated using the pre-correction "
            "implementation of evaluate_sprint9.py. A non-verdict-affecting H1 boundary defect "
            "was subsequently corrected to implement DD-7 exactly. "
            "The observed Sprint 9 diff (0.01222799051528034) lies outside the affected interval "
            "(-0.005, 0), so the correction does not change the stored H1 verdict. "
            "This deterministic verification run was performed using the corrected implementation "
            "solely to verify reproducibility and future-evaluation correctness."
        ),
    }
    with open(CHECK_DIR / "comparison.json", "w") as f:
        json.dump(comparison_doc, f, indent=2)

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    with open(CHECK_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(f"""# Sprint 9 — T-DETERMINISTIC Verification

## Purpose
Second identical evaluation run to verify T-DETERMINISTIC per Final Design §13.

## Result
**T-DETERMINISTIC = {t_deterministic}**
- {n_pass} fields PASS / {n_fail} fields FAIL
- FP tolerance: {FP_TOL} (rtol and atol)

## Provenance Note
Original Sprint 9 result artifacts were generated using the pre-correction
implementation of evaluate_sprint9.py. A non-verdict-affecting H1 boundary defect
was subsequently corrected to implement DD-7 exactly.
The observed Sprint 9 diff (0.01222799051528034) lies outside the affected interval
(-0.005, 0), so the correction does not change the stored H1 verdict.
This deterministic verification run was performed using the corrected implementation
solely to verify reproducibility and future-evaluation correctness.

## Artifacts
- `rerun_h1_results.json` — H1 rerun output
- `rerun_h2_results.json` — H2 rerun output
- `rerun_h3_results.json` — H3 rerun (inherited Sprint 8 evidence; no new inference)
- `rerun_runtime_report.json` — timing
- `comparison.json` — field-by-field match against original EXP_H123_V1/ artifacts
""")

    logger.info(f"=== T-DETERMINISTIC: {t_deterministic} ({n_pass} PASS / {n_fail} FAIL) ===")

    if n_fail > 0:
        print("\nMISMATCHES DETECTED:")
        for c in comparisons:
            if c["status"] == "FAIL":
                print(f"  FAIL: {c['field']}: original={c['original']} rerun={c['rerun']}")
        sys.exit(1)
    else:
        print(f"\nT-DETERMINISTIC = PASS ({n_pass}/{len(comparisons)} fields match within tolerance)")


if __name__ == "__main__":
    run()
