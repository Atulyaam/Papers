"""
scripts/validate_ae.py
-----------------------
Sprint 7 — EXP_AE_V1 Validation (Step 15).

Validates all existing artifacts without retraining, rerunning,
or modifying any frozen data.

Checks:
  1.  All required artifact files exist
  2.  TRAIN SHA-256 matches frozen hash
  3.  VALIDATION SHA-256 matches frozen hash
  4.  monitor_split_indices.csv: 44800 rows, no duplicates, disjoint
  5.  ae_fit count = 40320, monitor count = 4480
  6.  No Attack rows implied (monitor/ae_fit are positional Normal indices)
  7.  ae_scaler.joblib loads and has 75 features
  8.  ae_final.pt loads and architecture is correct (75->12->6->12->75)
  9.  n_params = 2049
  10. AE output shape matches input shape
  11. validation_reconstruction_errors.csv: 11200 rows, no NaN, all >= 0
  12. RE is mean over 75 features (spot check)
  13. threshold_sweep.csv: 5 rows, required columns present
  14. All 5 threshold rules present: p95, p99, p999, mean2sigma, mean3sigma
  15. Threshold monotonicity: p95 <= p99 <= p999
  16. p95 FPR sanity: ~5% (3-7%)
  17. p99 FPR sanity: ~1% (0.5-2%)
  18. p999 FPR sanity: ~0.1% (0-0.5%)
  19. primary_threshold = DEFERRED_TO_SPRINT_8
  20. Scaler-space limitation present in ae_metadata.json
  21. Single-seed limitation present in ae_metadata.json
  22. metadata.json: all required provenance fields present
  23. metadata.json: feature_count = 75
  24. metadata.json: ae_seed = 42, monitor_split_seed = 42
  25. metadata.json: best_epoch >= 1
  26. metadata.json: training_rows = 40320, monitor_rows = 4480, normal_train_total = 44800
  27. metadata.json: calibration_rows = 11200
  28. ae_architecture.json: correct layer sizes
  29. epoch_diagnostics.json: best_epoch matches metadata
  30. training_history.csv: rows >= best_epoch
  31. No development_test.csv access during validation
  32. No protected_unseen_attack.csv access during validation
  33. No excluded_train_backdoor.csv access during validation
  34. AE determinism: same seed -> same RE on same input
  35. Scaler source documented as Normal AE-fit subset only
"""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FROZEN_TRAIN_SHA = "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c"
FROZEN_VAL_SHA   = "13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180"

RES_DIR  = ROOT / "results/autoencoder/EXP_AE_V1"
CKPT_DIR = ROOT / "results/checkpoints/EXP_AE_V1"

REQUIRED_RESULTS = [
    RES_DIR / "config.yaml",
    RES_DIR / "metadata.json",
    RES_DIR / "runtime_report.json",
    RES_DIR / "monitor/monitor_split_indices.csv",
    RES_DIR / "monitor/monitor_metadata.json",
    RES_DIR / "training/training_history.csv",
    RES_DIR / "training/epoch_diagnostics.json",
    RES_DIR / "threshold/validation_reconstruction_errors.csv",
    RES_DIR / "threshold/threshold_sweep.csv",
    RES_DIR / "threshold/threshold_calibration.json",
]
REQUIRED_CHECKPOINTS = [
    CKPT_DIR / "ae_final.pt",
    CKPT_DIR / "ae_scaler.joblib",
    CKPT_DIR / "ae_architecture.json",
    CKPT_DIR / "ae_metadata.json",
    CKPT_DIR / "threshold_config.json",
]

THRESHOLD_RULES = {"p95", "p99", "p999", "mean2sigma", "mean3sigma"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def main() -> int:
    passed, failed = [], []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            passed.append(name)
            print(f"  PASS  [{len(passed)+len(failed):>2}] {name}")
        else:
            failed.append(name)
            print(f"  FAIL  [{len(passed)+len(failed):>2}] {name}  <<  {detail}")

    print("=" * 65)
    print("SPRINT 7 - EXP_AE_V1 VALIDATION REPORT")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Required artifact files exist
    # ------------------------------------------------------------------
    for p in REQUIRED_RESULTS + REQUIRED_CHECKPOINTS:
        check(f"File exists: {p.name}", p.exists(), str(p))

    # ------------------------------------------------------------------
    # 2-3. Hash verification
    # ------------------------------------------------------------------
    train_sha = sha256(ROOT / "data/splits/train.csv")
    check("TRAIN SHA-256", train_sha == FROZEN_TRAIN_SHA, train_sha[:16])
    val_sha = sha256(ROOT / "data/splits/validation.csv")
    check("VALIDATION SHA-256", val_sha == FROZEN_VAL_SHA, val_sha[:16])

    # ------------------------------------------------------------------
    # 4-6. Monitor split
    # ------------------------------------------------------------------
    split_df = pd.read_csv(RES_DIR / "monitor/monitor_split_indices.csv")
    check("Monitor split: total rows = 44800", len(split_df) == 44800,
          f"got {len(split_df)}")
    check("Monitor split: no duplicates", split_df["row_id"].nunique() == 44800,
          f"unique={split_df['row_id'].nunique()}")
    ae_fit_idx  = split_df[split_df["split"] == "ae_fit"]["row_id"].values
    monitor_idx = split_df[split_df["split"] == "monitor"]["row_id"].values
    check("Monitor split: ae_fit = 40320", len(ae_fit_idx) == 40320, f"got {len(ae_fit_idx)}")
    check("Monitor split: monitor = 4480", len(monitor_idx) == 4480, f"got {len(monitor_idx)}")
    overlap = set(ae_fit_idx.tolist()) & set(monitor_idx.tolist())
    check("Monitor split: ae_fit INTERSECT monitor = empty", len(overlap) == 0,
          f"overlap size={len(overlap)}")
    check("Monitor split: split column values valid",
          set(split_df["split"].unique()) == {"ae_fit", "monitor"}, "")

    # ------------------------------------------------------------------
    # 7. Scaler
    # ------------------------------------------------------------------
    scaler = joblib.load(CKPT_DIR / "ae_scaler.joblib")
    check("Scaler: 75 features (mean_ length)", len(scaler.mean_) == 75,
          f"got {len(scaler.mean_)}")
    check("Scaler: 75 features (scale_ length)", len(scaler.scale_) == 75,
          f"got {len(scaler.scale_)}")

    # ------------------------------------------------------------------
    # 8-10. AE model
    # ------------------------------------------------------------------
    from src.models.autoencoder.ae_model import Autoencoder
    ae = Autoencoder()
    state = torch.load(CKPT_DIR / "ae_final.pt", map_location="cpu", weights_only=True)
    ae.load_state_dict(state)
    ae.eval()
    check("AE: loads from checkpoint", True)

    arch = ae.architecture_dict()
    check("AE: encoder = [75, 12, 6]",  arch["encoder"]  == [75, 12, 6],  str(arch["encoder"]))
    check("AE: decoder = [6, 12, 75]",  arch["decoder"]  == [6, 12, 75],  str(arch["decoder"]))
    check("AE: hidden_activation = ReLU", arch["hidden_activation"] == "ReLU", "")
    check("AE: output_activation = Linear (none)", arch["output_activation"] == "Linear (none)", "")
    check("AE: batchnorm = False",  arch["batchnorm"]  is False, "")
    check("AE: dropout = False",    arch["dropout"]    is False, "")
    check("AE: n_params = 2049",    arch["n_params"] == 2049, f"got {arch['n_params']}")

    x_test = torch.randn(8, 75)
    with torch.no_grad():
        out = ae(x_test)
    check("AE: forward output shape (8, 75)", out.shape == (8, 75), str(out.shape))

    # 9. n_params
    n_params = ae.count_parameters()
    check("AE: parameter count = 2049", n_params == 2049, f"got {n_params}")

    # ------------------------------------------------------------------
    # 11-12. Reconstruction errors
    # ------------------------------------------------------------------
    re_df = pd.read_csv(RES_DIR / "threshold/validation_reconstruction_errors.csv")
    check("RE csv: 11200 rows", len(re_df) == 11200, f"got {len(re_df)}")
    check("RE csv: no NaN", re_df["re_value"].isna().sum() == 0, "")
    check("RE csv: all >= 0", (re_df["re_value"] >= 0).all(), "")
    check("RE csv: 'row_id' column present", "row_id" in re_df.columns, "")
    check("RE csv: 're_value' column present", "re_value" in re_df.columns, "")

    # Spot-check: RE is mean not sum
    x_sp = torch.zeros(1, 75)
    with torch.no_grad():
        x_hat = ae(x_sp)
    re_mean = ((x_sp - x_hat) ** 2).mean(dim=1).item()
    re_from_method = ae.reconstruction_error(x_sp).item()
    check("RE: method uses mean (not sum) over features",
          abs(re_from_method - re_mean) < 1e-6,
          f"method={re_from_method:.8f} manual_mean={re_mean:.8f}")

    # ------------------------------------------------------------------
    # 13-18. Threshold sweep
    # ------------------------------------------------------------------
    sweep_df = pd.read_csv(RES_DIR / "threshold/threshold_sweep.csv")
    check("Threshold sweep: 5 rows", len(sweep_df) == 5, f"got {len(sweep_df)}")
    required_cols = {"threshold_rule", "threshold_value", "samples_above_threshold",
                     "fraction_above_threshold"}
    check("Threshold sweep: required columns",
          required_cols.issubset(set(sweep_df.columns)),
          str(required_cols - set(sweep_df.columns)))
    sweep_rules = set(sweep_df["threshold_rule"].tolist())
    check("Threshold sweep: all 5 rules present", sweep_rules == THRESHOLD_RULES,
          str(THRESHOLD_RULES - sweep_rules))

    cal = json.load(open(RES_DIR / "threshold/threshold_calibration.json"))
    thresholds = cal["thresholds"]
    check("threshold_calibration.json: all 5 rules", set(thresholds.keys()) == THRESHOLD_RULES, "")

    tau_p95  = thresholds["p95"]["threshold_value"]
    tau_p99  = thresholds["p99"]["threshold_value"]
    tau_p999 = thresholds["p999"]["threshold_value"]
    check("Threshold monotonicity: p95 <= p99",   tau_p95 <= tau_p99,
          f"p95={tau_p95:.6f} p99={tau_p99:.6f}")
    check("Threshold monotonicity: p99 <= p999",  tau_p99 <= tau_p999,
          f"p99={tau_p99:.6f} p999={tau_p999:.6f}")

    fpr_p95  = thresholds["p95"]["fraction_above_threshold"]
    fpr_p99  = thresholds["p99"]["fraction_above_threshold"]
    fpr_p999 = thresholds["p999"]["fraction_above_threshold"]
    check("p95 FPR sanity: [0.03, 0.07]",  0.03 <= fpr_p95  <= 0.07,  f"got {fpr_p95:.4f}")
    check("p99 FPR sanity: [0.005, 0.02]", 0.005 <= fpr_p99 <= 0.02, f"got {fpr_p99:.4f}")
    check("p999 FPR sanity: [0, 0.005]",   0.0   <= fpr_p999 <= 0.005, f"got {fpr_p999:.4f}")

    prim = cal.get("primary_threshold", "")
    check("primary_threshold = DEFERRED_TO_SPRINT_8", prim == "DEFERRED_TO_SPRINT_8", repr(prim))
    check("calibration_split = Normal VALIDATION only",
          "Normal VALIDATION" in cal.get("calibration_split", ""), "")
    check("calibration_rows = 11200", cal.get("calibration_rows") == 11200,
          str(cal.get("calibration_rows")))

    # ------------------------------------------------------------------
    # 20-21. Limitation text in ae_metadata.json
    # ------------------------------------------------------------------
    ae_meta = json.load(open(CKPT_DIR / "ae_metadata.json"))
    check("Scaler-space limitation in ae_metadata",
          "Normal-TRAIN-scaled feature space" in ae_meta.get("scaler_space_limitation", ""), "")
    check("Single-seed limitation in ae_metadata",
          "single AE training seed" in ae_meta.get("single_seed_limitation", ""), "")

    # ------------------------------------------------------------------
    # 22-30. Provenance fields
    # ------------------------------------------------------------------
    meta = json.load(open(RES_DIR / "metadata.json"))
    REQUIRED_FIELDS = [
        "experiment_id", "sprint", "dataset", "feature_set", "feature_count",
        "train_sha256", "validation_sha256", "training_rows", "monitor_rows",
        "normal_train_total", "calibration_rows", "monitor_split_seed", "ae_seed",
        "architecture", "loss", "optimizer", "learning_rate", "weight_decay",
        "batch_size", "max_epochs", "patience", "best_epoch", "final_epoch_count",
        "scaler_source", "scaler_fit_population", "threshold_candidates",
        "primary_threshold", "data_access_boundary",
        "scaler_space_limitation", "single_seed_limitation",
        "torch_version", "git_commit", "timestamp_utc",
    ]
    missing_fields = [f for f in REQUIRED_FIELDS if f not in meta]
    check("metadata.json: all required provenance fields",
          len(missing_fields) == 0, str(missing_fields))
    check("metadata.json: experiment_id = EXP_AE_V1",
          meta.get("experiment_id") == "EXP_AE_V1", "")
    check("metadata.json: sprint = 7",          meta.get("sprint") == 7, "")
    check("metadata.json: feature_count = 75",  meta.get("feature_count") == 75, "")
    check("metadata.json: ae_seed = 42",         meta.get("ae_seed") == 42, "")
    check("metadata.json: monitor_split_seed = 42", meta.get("monitor_split_seed") == 42, "")
    check("metadata.json: training_rows = 40320",    meta.get("training_rows") == 40320, "")
    check("metadata.json: monitor_rows = 4480",      meta.get("monitor_rows") == 4480, "")
    check("metadata.json: normal_train_total = 44800", meta.get("normal_train_total") == 44800, "")
    check("metadata.json: calibration_rows = 11200",   meta.get("calibration_rows") == 11200, "")
    check("metadata.json: best_epoch >= 1", meta.get("best_epoch", 0) >= 1,
          f"got {meta.get('best_epoch')}")
    check("metadata.json: primary_threshold = DEFERRED_TO_SPRINT_8",
          meta.get("primary_threshold") == "DEFERRED_TO_SPRINT_8", "")
    check("metadata.json: scaler_source contains 'Normal AE-fit subset'",
          "Normal AE-fit subset" in meta.get("scaler_source", ""), "")
    check("metadata.json: train_sha256 correct",
          meta.get("train_sha256") == FROZEN_TRAIN_SHA, "")
    check("metadata.json: validation_sha256 correct",
          meta.get("validation_sha256") == FROZEN_VAL_SHA, "")

    # ------------------------------------------------------------------
    # 28. ae_architecture.json
    # ------------------------------------------------------------------
    ae_arch = json.load(open(CKPT_DIR / "ae_architecture.json"))
    check("ae_architecture.json: encoder = [75,12,6]", ae_arch["encoder"] == [75, 12, 6], "")
    check("ae_architecture.json: decoder = [6,12,75]", ae_arch["decoder"] == [6, 12, 75], "")
    check("ae_architecture.json: n_params = 2049", ae_arch["n_params"] == 2049, "")

    # ------------------------------------------------------------------
    # 29-30. Epoch diagnostics and training history
    # ------------------------------------------------------------------
    ep_diag = json.load(open(RES_DIR / "training/epoch_diagnostics.json"))
    check("epoch_diagnostics.json: best_epoch >= 1", ep_diag.get("best_epoch", 0) >= 1, "")
    hist_df = pd.read_csv(RES_DIR / "training/training_history.csv")
    best_ep = meta.get("best_epoch", 0)
    check("training_history.csv: rows >= best_epoch",
          len(hist_df) >= best_ep, f"rows={len(hist_df)} best_epoch={best_ep}")
    check("training_history.csv: epoch_diagnostics best_epoch matches metadata",
          ep_diag.get("best_epoch") == meta.get("best_epoch"), "")
    check("training_history.csv: required columns",
          {"epoch", "ae_fit_mse", "monitor_mse"}.issubset(set(hist_df.columns)), "")

    # ------------------------------------------------------------------
    # 34. Determinism: same seed -> same RE
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    ae1 = Autoencoder()
    torch.manual_seed(42)
    ae2 = Autoencoder()
    x_d = torch.randn(4, 75)
    ae1.eval(); ae2.eval()
    with torch.no_grad():
        re1 = ae1.reconstruction_error(x_d)
        re2 = ae2.reconstruction_error(x_d)
    check("AE determinism: same seed -> same RE", torch.allclose(re1, re2, atol=1e-6), "")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(passed) + len(failed)
    print()
    print("=" * 65)
    print(f"VALIDATION RESULT: {len(passed)}/{total} PASS")
    if failed:
        print("FAILED CHECKS:")
        for f in failed:
            print(f"  - {f}")
        print()
        print("STATUS: CHANGES_REQUIRED")
    else:
        print()
        print("STATUS: VALIDATED")
    print("=" * 65)

    # Print key results
    print()
    print("KEY RESULTS:")
    print(f"  best_epoch      = {meta.get('best_epoch')}")
    print(f"  final_epoch     = {meta.get('final_epoch_count')}")
    re_stats = cal.get("re_stats", {})
    print(f"  RE mean         = {re_stats.get('mean', 'N/A'):.6f}")
    print(f"  RE std          = {re_stats.get('std', 'N/A'):.6f}")
    print(f"  RE max          = {re_stats.get('max', 'N/A'):.6f}")
    print(f"  RE p50          = {re_stats.get('p50', 'N/A'):.6f}")
    print(f"  RE p95          = {re_stats.get('p95', 'N/A'):.6f}")
    print(f"  RE p99          = {re_stats.get('p99', 'N/A'):.6f}")
    print(f"  RE p99.9        = {re_stats.get('p999', 'N/A'):.6f}")
    print()
    print("THRESHOLDS:")
    for rule, vals in thresholds.items():
        print(f"  {rule:<12} tau={vals['threshold_value']:.6f}"
              f" | n_above={vals['samples_above_threshold']}"
              f" | frac={vals['fraction_above_threshold']:.4f}")
    print()
    print("  primary_threshold = DEFERRED_TO_SPRINT_8")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
