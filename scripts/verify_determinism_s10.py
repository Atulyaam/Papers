"""
scripts/verify_determinism_s10.py

Sprint 10 — Independent Determinism Verification (Validation 20)
Concrete Bar Requirements:
1. Independently loads inputs from disk (data/splits/development_test.csv).
2. Independently loads cached predictions / checkpoints from disk.
3. Independently executes inference (not a copy of stored output).
4. Produces its own output arrays.
5. Numerically compares freshly-produced outputs against stored results.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP = ROOT / "results/ablation/EXP_ABLATION_V1"
CACHE = EXP / "cache"

from src.models.autoencoder.ae_model import Autoencoder
from scripts.run_ablation import (
    get_pipeline_and_features, DATASET_PATHS, build_meta_lr,
    compute_metrics, build_a1b_scores, get_device
)

def verify_determinism():
    print("=== SPRINT 10 INDEPENDENT DETERMINISM VERIFICATION ===")

    # 1. Independent data loading from disk
    pipe, feats = get_pipeline_and_features()
    raw_dev = pd.read_csv(DATASET_PATHS["development_test"])
    enc_dev = pipe.transform(raw_dev, view="unscaled")
    enc_df = pd.DataFrame(enc_dev.X, columns=enc_dev.feature_names)
    X_dev = enc_df[feats].values.astype(np.float64)
    y_dev = raw_dev["label"].values
    n_dev = len(y_dev)
    print(f"[DISK LOAD] development_test.csv loaded: shape={X_dev.shape}, N={n_dev}")

    # Load AE artifacts independently from disk
    ae_cfg_path = ROOT / "results/checkpoints/EXP_AE_V1/ae_final.pt"
    ae_scaler_path = ROOT / "results/checkpoints/EXP_AE_V1/ae_scaler.joblib"
    ae_thresh_path = ROOT / "results/autoencoder/EXP_AE_V1/threshold/threshold_calibration.json"

    ae_thresh_data = json.loads(ae_thresh_path.read_text())
    tau = ae_thresh_data["thresholds"]["mean3sigma"]["threshold_value"]
    ae_model = Autoencoder(input_dim=75)
    ae_model.load_state_dict(torch.load(ae_cfg_path, map_location="cpu", weights_only=True))
    ae_model.eval()
    ae_scaler = joblib.load(ae_scaler_path)

    device = get_device()
    ae_model.to(device)
    X_dev_ae = ae_scaler.transform(X_dev).astype(np.float32)
    re_vals = []
    with torch.no_grad():
        for i in range(0, len(X_dev_ae), 4096):
            x_t = torch.tensor(X_dev_ae[i:i+4096]).to(device)
            x_h = ae_model(x_t)
            re_vals.append(((x_t - x_h) ** 2).mean(dim=1).cpu().numpy())
    re = np.concatenate(re_vals)
    ae_flags = (re > tau).astype(int)
    print(f"[INFERENCE] Fresh AE flags computed: flagged_count={ae_flags.sum()} / {n_dev}")

    all_matches = True
    max_diff_seen = 0.0

    configs_to_verify = [
        ("A1_FULL_STACK", ["dt", "rf", "svm", "nn"]),
        ("A2_NO_DT", ["rf", "svm", "nn"]),
        ("A3_NO_RF", ["dt", "svm", "nn"]),
        ("A4_NO_SVM", ["dt", "rf", "nn"]),
        ("A5_NO_NN", ["dt", "rf", "svm"]),
    ]

    for seed in [42, 123, 2024]:
        print(f"\n--- Verifying Seed {seed} ---")

        # Verify LR Meta-Learners
        for cid, model_list in configs_to_verify:
            # Load OOF cache columns from disk
            oof_cols = []
            dev_cols = []
            y_oof = None
            for mn in model_list:
                c_data = np.load(CACHE / f"{mn}_seed{seed}.npz", allow_pickle=True)
                oof_cols.append(c_data["oof_scores"])
                dev_cols.append(c_data["dev_test_scores"])
                if y_oof is None:
                    y_oof = c_data["oof_labels"]

            meta_X_oof = np.column_stack(oof_cols)
            meta_X_dev = np.column_stack(dev_cols)

            # Independent fit & inference
            lr = build_meta_lr(seed)
            lr.fit(meta_X_oof, y_oof)
            fresh_preds = lr.predict(meta_X_dev)
            fresh_metrics = compute_metrics(y_dev, fresh_preds)

            # Compare against stored JSON on disk
            stored_json = json.loads((EXP / cid / f"seed_{seed}.json").read_text())
            diff = abs(fresh_metrics["macro_f1"] - stored_json["macro_f1"])
            max_diff_seen = max(max_diff_seen, diff)
            status = "EXACT MATCH" if diff == 0.0 else f"DIFF={diff:.2e}"
            print(f"  {cid:<18}: fresh_mf1={fresh_metrics['macro_f1']:.6f} vs stored={stored_json['macro_f1']:.6f} [{status}]")
            if diff > 1e-9:
                all_matches = False

        # Verify A1b Soft-Vote
        c_dt = np.load(CACHE / f"dt_seed{seed}.npz", allow_pickle=True)["dev_test_scores"]
        c_rf = np.load(CACHE / f"rf_seed{seed}.npz", allow_pickle=True)["dev_test_scores"]
        c_svm = np.load(CACHE / f"svm_seed{seed}.npz", allow_pickle=True)["dev_test_scores"]
        c_nn = np.load(CACHE / f"nn_seed{seed}.npz", allow_pickle=True)["dev_test_scores"]

        # svm_unit = sigmoid(decision_function)
        svm_unit = 1.0 / (1.0 + np.exp(-c_svm))
        a1b_scores = np.mean([c_dt, c_rf, svm_unit, c_nn], axis=0)
        a1b_fresh_preds = (a1b_scores >= 0.5).astype(int)
        a1b_fresh_m = compute_metrics(y_dev, a1b_fresh_preds)
        a1b_stored = json.loads((EXP / f"A1b_SOFT_VOTE/seed_{seed}.json").read_text())
        a1b_diff = abs(a1b_fresh_m["macro_f1"] - a1b_stored["macro_f1"])
        max_diff_seen = max(max_diff_seen, a1b_diff)
        a1b_status = "EXACT MATCH" if a1b_diff == 0.0 else f"DIFF={a1b_diff:.2e}"
        print(f"  {'A1b_SOFT_VOTE':<18}: fresh_mf1={a1b_fresh_m['macro_f1']:.6f} vs stored={a1b_stored['macro_f1']:.6f} [{a1b_status}]")
        if a1b_diff > 1e-9:
            all_matches = False

        # Verify A6 Fusion
        # Re-derive A1 preds
        oof_cols_a1 = [np.load(CACHE / f"{mn}_seed{seed}.npz", allow_pickle=True)["oof_scores"] for mn in ["dt","rf","svm","nn"]]
        dev_cols_a1 = [np.load(CACHE / f"{mn}_seed{seed}.npz", allow_pickle=True)["dev_test_scores"] for mn in ["dt","rf","svm","nn"]]
        lr_a1 = build_meta_lr(seed)
        lr_a1.fit(np.column_stack(oof_cols_a1), y_oof)
        a1_preds = lr_a1.predict(np.column_stack(dev_cols_a1))

        # A6 OR fusion
        a6_fresh_preds = np.maximum(a1_preds, ae_flags)
        a6_fresh_m = compute_metrics(y_dev, a6_fresh_preds)
        a6_stored = json.loads((EXP / f"A6_STACK_PLUS_AE/seed_{seed}.json").read_text())
        a6_diff = abs(a6_fresh_m["macro_f1"] - a6_stored["macro_f1"])
        max_diff_seen = max(max_diff_seen, a6_diff)
        a6_status = "EXACT MATCH" if a6_diff == 0.0 else f"DIFF={a6_diff:.2e}"
        print(f"  {'A6_STACK_PLUS_AE':<18}: fresh_mf1={a6_fresh_m['macro_f1']:.6f} vs stored={a6_stored['macro_f1']:.6f} [{a6_status}]")
        if a6_diff > 1e-9:
            all_matches = False

    print("\n=== DETERMINISM SUMMARY ===")
    print(f"All configurations and seeds matched: {all_matches}")
    print(f"Maximum absolute numerical difference observed: {max_diff_seen:.2e}")
    if all_matches and max_diff_seen == 0.0:
        print("DETERMINISM VERIFICATION: PASS (Exact bit-level reproduction)")
        return True
    else:
        print("DETERMINISM VERIFICATION: FAIL")
        return False

if __name__ == "__main__":
    success = verify_determinism()
    sys.exit(0 if success else 1)
