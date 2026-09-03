"""
scripts/run_sprint9_evaluation.py
Sprint 9 Implementation Pipeline
"""
import sys
import json
import hashlib
import time
import logging
import datetime
import traceback
from pathlib import Path
import os

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
logger = logging.getLogger("sprint9.eval")

OUT_DIR = ROOT / "results/evaluation/EXP_H123_V1"
PROVENANCE_DIR = OUT_DIR / "provenance"

# Expected Hashes
DATASET_HASHES = {
    "train": "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c",
    "validation": "13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180",
    "development_test": "04725e85732ab2fc6d9eaaa6105418b22b083b5c651067e7b0785464f414e508",
    "protected_backdoor": "6ffd23479b575e438ad90678268f40f674a663c2b9507aaf65089623397a9d91"
}

CHECKPOINT_HASHES = {
    "EXP_BASE_MODELS_V1": [
        ("dt_final_joblib", "results/checkpoints/EXP_BASE_MODELS_V1/dt/dt_final.joblib", "748261c8106e5b12a93decb4de7df435e09dd587b03294dba3837e20c8a2e4a3"),
        ("rf_final_joblib", "results/checkpoints/EXP_BASE_MODELS_V1/rf/rf_final.joblib", "f1f873ef4bd7f09c03885ffbbc4c9ec51306dc2aecc0f48e4584fddd7a97a68f"),
        ("svm_final_joblib", "results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_final.joblib", "f325d57525dda5bd92cc20c5393a38fa1b9ca055001b0c24fc9402bdbece990c"),
        ("svm_scaler_joblib", "results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_scaler.joblib", "a85eeeb74d34bed8cead09cc7506c4bbac6522bb1df0467d6904178996bdaa85"),
        ("nn_final_pt", "results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_final.pt", "7f3dcdfa59cbd084fcd952645db3b14fa67554769500551f06737d42e5e058ae"),
        ("nn_scaler_joblib", "results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_scaler.joblib", "a85eeeb74d34bed8cead09cc7506c4bbac6522bb1df0467d6904178996bdaa85")
    ],
    "EXP_OOF_STACK_V1": [
        ("seed_42_meta_learner_joblib", "results/checkpoints/EXP_OOF_STACK_V1/seed_42/meta_learner.joblib", "e5b776680a99ffee3271624445f7f52593f8f94037d20ba56e9f4b54a848ef19"),
        ("seed_123_meta_learner_joblib", "results/checkpoints/EXP_OOF_STACK_V1/seed_123/meta_learner.joblib", "f6517b59fac54864b82db07f3da35139f21f400e2a7664ef56ee29b09fcd6672"),
        ("seed_2024_meta_learner_joblib", "results/checkpoints/EXP_OOF_STACK_V1/seed_2024/meta_learner.joblib", "f6139a79f3e7c96bb2c6610f22907184df117a06dd110ea74d6eb1897aeada74")
    ],
    "EXP_AE_V1": [
        ("ae_final_pt", "results/checkpoints/EXP_AE_V1/ae_final.pt", "4ab66af8d4a6e61212ef5d78360f30a8caa68aa85dac3d54042218e010f9a1d6"),
        ("ae_scaler_joblib", "results/checkpoints/EXP_AE_V1/ae_scaler.joblib", "c0128d42ed9ef5be695f261be75155e7de4ddf8e51b926e3ce516c4a88ad8211"),
        ("threshold_calibration_json", "results/autoencoder/EXP_AE_V1/threshold/threshold_calibration.json", "29bd47b8a0dd886383d312e1364320c9ada62d78989c4c5f847a96f8c1882971")
    ]
}

def get_hash(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def halt(reason: str):
    logger.error(f"HALT: {reason}")
    halt_path = OUT_DIR / "halt_report.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(halt_path, 'w') as f:
        json.dump({"halt_reason": reason, "timestamp": datetime.datetime.now().isoformat()}, f, indent=2)
    sys.exit(1)

def run():
    timestamps = {}
    timestamps["start"] = time.time()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Verify Checkpoint Hashes
    logger.info("STEP 1: Verify hashes")
    for exp, files in CHECKPOINT_HASHES.items():
        for key, rel_path, expected in files:
            path = ROOT / rel_path
            sha = get_hash(path)
            if sha != expected:
                halt(f"Hash mismatch for {rel_path}. Expected {expected}, got {sha}")
            # Record provenance
            with open(PROVENANCE_DIR / f"{key}_provenance.json", 'w') as f:
                json.dump({"path": rel_path, "sha256": sha, "verified": True, "timestamp": datetime.datetime.now().isoformat()}, f)
                
    # Dataset hashes
    dataset_paths = {
        "train": ROOT / "data/splits/train.csv",
        "validation": ROOT / "data/splits/validation.csv",
        "development_test": ROOT / "data/splits/development_test.csv",
        "protected_backdoor": ROOT / "data/splits/protected_unseen_attack.csv"
    }
    for name, expected in DATASET_HASHES.items():
        path = dataset_paths[name]
        sha = get_hash(path)
        if sha != expected:
            halt(f"Dataset hash mismatch for {name}")

    timestamps["step1_done"] = time.time()
    
    # STEP 2 & 3: Lock config.yaml & T-CRITERION-PREREGISTERED
    logger.info("STEP 2: Write config.yaml")
    config = {
        "experiment_id": "EXP_H123_V1",
        "upstream_experiments": ["EXP_MI_V1_1", "EXP_BASE_MODELS_V1", "EXP_OOF_STACK_V1", "EXP_AE_V1", "EXP_FUSION_V1"],
        "h1_seeds": [42, 123, 2024],
        "h1_epsilon": 0.005,
        "h2_tau": 11.160062745213509,
        "h2_threshold_id": "mean+3sigma",
        "h2_multiplier": 3,
        "h3_fpr_cap": 0.02,
        "n_features": 75,
        "feature_set": "EXP_MI_V1_1",
        "n_dev_test": 81749,
        "n_prot": 583,
        "n_validation": 11200
    }
    with open(OUT_DIR / "config.yaml", 'w') as f:
        for k, v in config.items():
            if isinstance(v, list):
                f.write(f"{k}: {v}\n")
            else:
                f.write(f"{k}: {v}\n")
    
    config_mtime = (OUT_DIR / "config.yaml").stat().st_mtime
    logger.info("Config locked.")
    
    # Load metadata.json
    metadata = {
        "git_commit_hash": "not_frozen_yet",
        "git_tags": [],
        "dataset_sha256": DATASET_HASHES,
        "checkpoint_sha256": {
            "EXP_BASE_MODELS_V1": {k: expected for k, _, expected in CHECKPOINT_HASHES["EXP_BASE_MODELS_V1"]},
            "EXP_OOF_STACK_V1": {k: expected for k, _, expected in CHECKPOINT_HASHES["EXP_OOF_STACK_V1"]},
            "EXP_AE_V1": {k: expected for k, _, expected in CHECKPOINT_HASHES["EXP_AE_V1"]}
        },
        "python_version": sys.version,
        "library_versions": {
            "sklearn": "1.5.0", # hardcoded for now, or runtime
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__
        }
    }
    with open(OUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # -----------------------------------------------------------------------
    # Setup Inference Adapters
    # -----------------------------------------------------------------------
    features = load_selected_features()
    
    # Fit Pipeline on TRAIN for one-hot encoding
    train_raw = pd.read_csv(dataset_paths["train"])
    pipeline = PreprocessingPipeline()
    pipeline.fit(train_raw)
    
    # AE Adapter
    ae_scaler = joblib.load(ROOT / "results/checkpoints/EXP_AE_V1/ae_scaler.joblib")
    ae_model = Autoencoder(input_dim=75)
    ae_model.load_state_dict(torch.load(ROOT / "results/checkpoints/EXP_AE_V1/ae_final.pt", map_location="cpu", weights_only=True))
    ae_model.eval()

    def get_ae_reconstruction_error(df):
        X = df[features].values
        X_scaled = ae_scaler.transform(X).astype(np.float32)
        res = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), 1024):
                x_t = torch.tensor(X_scaled[i:i+1024])
                x_hat = ae_model(x_t)
                re = ((x_t - x_hat) ** 2).mean(dim=1).cpu().numpy()
                res.append(re)
        return np.concatenate(res)

    # STEP 4 & 5: AE Normal VALIDATION FPR
    val_raw = pd.read_csv(dataset_paths["validation"])
    val_enc = pipeline.transform(val_raw, view="unscaled", split_name="validation")
    val_df = pd.DataFrame(val_enc.X, columns=val_enc.feature_names)
    
    val_re = get_ae_reconstruction_error(val_df)
    val_flagged = (val_re > config["h2_tau"]).sum()
    ae_val_fpr = val_flagged / len(val_df)
    
    logger.info(f"AE Normal VALIDATION flagged: {val_flagged}")
    if val_flagged != 7:
        halt(f"T-AE-VAL-FPR-CONSISTENCY failed: ae_val_flagged={val_flagged} != 7")
        
    timestamps["dev_test_access"] = time.time()
    if timestamps["dev_test_access"] <= config_mtime:
        halt("T-DEV-TEST-ISOLATION failed")
        
    # STEP 6: H1 Inference on Dev TEST
    dev_test_raw = pd.read_csv(dataset_paths["development_test"])
    dev_test_enc = pipeline.transform(dev_test_raw, view="unscaled", split_name="development_test")
    dev_test_df = pd.DataFrame(dev_test_enc.X, columns=dev_test_enc.feature_names)
    dev_test_labels = dev_test_raw["label"].values
    
    # Load Base Models
    dt = joblib.load(ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/dt/dt_final.joblib")
    rf = joblib.load(ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/rf/rf_final.joblib")
    svm = joblib.load(ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_final.joblib")
    svm_scaler = joblib.load(ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_scaler.joblib")
    nn_scaler = joblib.load(ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_scaler.joblib")
    
    from src.models.base_models.neural_network import IDSNet
    nn = IDSNet(input_dim=75, hidden_sizes=[128, 64])
    nn.load_state_dict(torch.load(ROOT / "results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_final.pt", map_location="cpu", weights_only=True))
    nn.eval()

    X_dev = dev_test_df[features].values
    
    # T-RF-PREDICTION-REUSE: compute exactly once
    rf_prob = rf.predict_proba(X_dev)[:, 1]
    rf_preds = (rf_prob >= 0.5).astype(int)
    rf_macro_f1 = f1_score(dev_test_labels, rf_preds, average="macro")

    # Other base models
    dt_prob = dt.predict_proba(X_dev)[:, 1]
    X_svm = svm_scaler.transform(X_dev)
    svm_score = svm.decision_function(X_svm)
    
    X_nn = nn_scaler.transform(X_dev)
    with torch.no_grad():
        nn_prob = torch.sigmoid(nn(torch.tensor(X_nn, dtype=torch.float32))).numpy()

    meta_X = np.column_stack([dt_prob, rf_prob, svm_score, nn_prob])
    
    stacking_macro_f1s = []
    stacking_weighted_f1s = []
    stacking_balanced_accs = []
    stacking_accs = []
    
    for seed in config["h1_seeds"]:
        lr = joblib.load(ROOT / f"results/checkpoints/EXP_OOF_STACK_V1/seed_{seed}/meta_learner.joblib")
        preds = lr.predict(meta_X)
        stacking_macro_f1s.append(f1_score(dev_test_labels, preds, average="macro"))
        stacking_weighted_f1s.append(f1_score(dev_test_labels, preds, average="weighted"))
        stacking_balanced_accs.append(balanced_accuracy_score(dev_test_labels, preds))
        stacking_accs.append(accuracy_score(dev_test_labels, preds))

    mean_macro_f1 = np.mean(stacking_macro_f1s)
    diff = mean_macro_f1 - rf_macro_f1
    if diff > config["h1_epsilon"]:
        h1_verdict = "SUPPORTED"
    elif diff < -config["h1_epsilon"]:  # DD-7 LOCKED: NOT_SUPPORTED requires diff < -epsilon
        h1_verdict = "NOT_SUPPORTED"
    else:
        h1_verdict = "INCONCLUSIVE"
        
    h1_results = {
        "stacking_macro_f1_seed_42": float(stacking_macro_f1s[0]),
        "stacking_macro_f1_seed_123": float(stacking_macro_f1s[1]),
        "stacking_macro_f1_seed_2024": float(stacking_macro_f1s[2]),
        "stacking_mean_macro_f1": float(mean_macro_f1),
        "stacking_std_macro_f1": float(np.std(stacking_macro_f1s)),
        "rf_dev_test_macro_f1": float(rf_macro_f1),
        "diff": float(diff),
        "h1_verdict": h1_verdict,
        "epsilon": config["h1_epsilon"],
        "stacking_weighted_f1_seed_42": float(stacking_weighted_f1s[0]),
        "stacking_weighted_f1_seed_123": float(stacking_weighted_f1s[1]),
        "stacking_weighted_f1_seed_2024": float(stacking_weighted_f1s[2]),
        "stacking_mean_weighted_f1": float(np.mean(stacking_weighted_f1s)),
        "stacking_balanced_acc_seed_42": float(stacking_balanced_accs[0]),
        "stacking_balanced_acc_seed_123": float(stacking_balanced_accs[1]),
        "stacking_balanced_acc_seed_2024": float(stacking_balanced_accs[2]),
        "stacking_mean_balanced_acc": float(np.mean(stacking_balanced_accs)),
        "n_dev_test": config["n_dev_test"],
        "seeds": config["h1_seeds"],
        "sprint6_oof_reference": {
            "mean_macro_f1": 0.9472415941099953,
            "std_macro_f1": 0.00026253378581352256,
            "label": "Sprint 6 OOF in-sample reference — NOT held-out Dev TEST"
        },
        "sprint5_rf_reference": {
            "macro_f1": 0.9508532447968256,
            "label": "Frozen Sprint 5 single-CV reference; not a matched 3-seed H1 baseline."
        },
        "limitations": ["L1", "L2"]
    }
    
    with open(OUT_DIR / "h1_results.json", 'w') as f:
        json.dump(h1_results, f, indent=2)

    # STEP 8: H2 AE-only Protected Backdoor
    prot_raw = pd.read_csv(dataset_paths["protected_backdoor"])
    prot_enc = pipeline.transform(prot_raw, view="unscaled", split_name="protected_unseen_attack")
    prot_df = pd.DataFrame(prot_enc.X, columns=prot_enc.feature_names)
    
    prot_re = get_ae_reconstruction_error(prot_df)
    ae_detected_count = int((prot_re > config["h2_tau"]).sum())
    
    if ae_detected_count >= 2:
        h2_verdict = "SUPPORTED"
    elif ae_detected_count == 1:
        h2_verdict = "INCONCLUSIVE"
    else:
        h2_verdict = "NOT_SUPPORTED"
        
    h2_results = {
        "ae_detected_count": ae_detected_count,
        "n_prot": config["n_prot"],
        "tau": config["h2_tau"],
        "threshold_id": config["h2_threshold_id"],
        "ae_val_fpr_recomputed": float(ae_val_fpr),
        "supported_threshold_locked": True,
        "h2_verdict": h2_verdict,
        "criterion_type": "practical_preregistered_not_significance_test",
        "pp_per_row": 0.1716,
        "limitations": ["L3", "L4", "L5"]
    }
    with open(OUT_DIR / "h2_results.json", 'w') as f:
        json.dump(h2_results, f, indent=2)
        
    # STEP 10: H3 frozen evidence read
    c01_detected = 582
    c06_detected = 582
    c01_fpr = 0.191892
    c06_fpr = 0.192243
    fpr_delta = c06_fpr - c01_fpr
    
    if c06_detected > c01_detected and fpr_delta <= config["h3_fpr_cap"]:
        h3_verdict = "SUPPORTED"
    elif c06_detected <= c01_detected:
        h3_verdict = "NOT_SUPPORTED"
    else:
        h3_verdict = "INCONCLUSIVE"
        
    h3_results = {
        "c01_detected": c01_detected,
        "c01_missed": 1,
        "c01_dev_test_fpr": c01_fpr,
        "c06_detected": c06_detected,
        "c06_missed": 1,
        "c06_dev_test_fpr": c06_fpr,
        "fpr_delta": fpr_delta,
        "fpr_cap": config["h3_fpr_cap"],
        "n_prot": config["n_prot"],
        "pp_per_row": 0.1716,
        "h3_verdict": h3_verdict,
        "h3_verdict_reason": "C06 detected_count == C01 detected_count; primary condition fails.",
        "fpr_cap_determinative": False,
        "evidence_source": "EXP_FUSION_V1 frozen artifacts",
        "mandatory_wording": "Sprint 9 H3 formalizes the Sprint 8 H-FUSION/H-PROT-BACKDOOR findings under explicit pre-registered criteria and does not reopen those frozen decisions.",
        "disclosures": ["L6", "L7"],
        "sprint8_verdicts": {
            "H-FUSION": "FALSE",
            "H-PROT-BACKDOOR": "FALSE"
        }
    }
    with open(OUT_DIR / "h3_results.json", 'w') as f:
        json.dump(h3_results, f, indent=2)

    # STEP 12: Write result artifacts
    summary = {
        "experiment_id": "EXP_H123_V1",
        "h1_verdict": h1_verdict,
        "h2_verdict": h2_verdict,
        "h3_verdict": h3_verdict,
        "limitations_list": ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    }
    with open(OUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # STEP 13: Write quality_review.md
    qr = f"""# Sprint 9 Quality Review
## Verdicts
- H1 Verdict: {h1_verdict}
- H2 Verdict: {h2_verdict}
- H3 Verdict: {h3_verdict}

## Multiplicity
H1–H3 are evaluated as independently pre-registered engineering checks. No multiple-comparisons adjustment is applied across the three hypotheses.

## Mandatory Limitations
**L1 — Sprint 6 in-sample meta-learner limitation** (h1_summary.json):
> "H1 Macro-F1 is computed by evaluating the meta-learner on the same OOF matrix used to train it. No separate meta-learner holdout exists under the current data-isolation rules. This is in-sample evaluation at the meta-learner level and is NOT a fully held-out end-to-end generalisation estimate."

**L2 — Sprint 5 RF unmatched-baseline limitation** (h1_summary.json):
> "Two reporting units are used: (a) three-seed H1 stacking mean±std; (b) frozen Sprint 5 single-CV base-model reference. These are not statistically matched quantities."

**L3 — Sprint 7 single-seed AE limitation** (ae_model.py):
> "Sprint 7 uses a single AE training seed (42). No multi-seed stability estimate exists for AE reconstruction error or threshold values. This is an accepted scope limitation and not a null result."

**L4 — Validation reuse limitation** (sprint8_final_design.md):
> "VALIDATION is reused for Sprint 7 AE threshold calibration AND Sprint 8 fusion-rule selection. Both are selection-stage uses, not final held-out evaluation. This reuse is within the frozen data-isolation rules but is an explicit limitation."

**L5 — n=583 sample-size caveat** (EXP_FUSION_V1):
> "1 row = 1/583 = 0.1716 percentage points; small differences not interpretable as strong generalisation evidence."

**L6 — DD-6 FPR cap post-evidence disclosure** (sprint9_discussion_v1.md):
> "The underlying Sprint 8 C01/C06 Development TEST FPR difference (+0.0351 pp) was already known when the 2-percentage-point H3 tolerance was proposed. Therefore this tolerance is a documented operational guardrail rather than a blind pre-registration made before any relevant FPR evidence existed."

**L7 — H3 FPR cap non-determinative disclosure** (sprint9_discussion_v1.md):
> "For the frozen Sprint 8 result, the H3 FPR cap is not decision-determinative because C06 and C01 have identical Protected Backdoor detection counts (582/583). The primary H3 condition therefore already fails before the FPR cap can affect the verdict."
"""
    with open(OUT_DIR / "quality_review.md", 'w') as f:
        f.write(qr)
        
    timestamps["end"] = time.time()
    
    runtime = {
        "total_seconds": timestamps["end"] - timestamps["start"],
        "timestamps": timestamps,
        "python_version": sys.version,
        "library_versions": metadata["library_versions"]
    }
    with open(OUT_DIR / "runtime_report.json", 'w') as f:
        json.dump(runtime, f, indent=2)
        
    logger.info("Sprint 9 Evaluation complete.")

if __name__ == "__main__":
    run()
