"""
Smoke test for Sprint 2 preprocessing pipeline on real UNSW-NB15 data.
Run from project root:
    .venv\Scripts\python.exe scripts/run_preprocessing_smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import pandas as pd
import numpy as np

from src.preprocessing.loader import load_raw_unswnb15
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.utils.hashing import sha256_file

TRAIN_HASH_EXPECTED = "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa"
TEST_HASH_EXPECTED  = "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559"

print("=== SPRINT 2 REAL-DATA SMOKE TEST ===")
print()

# --- Raw file integrity pre-run ---
train_hash_pre = sha256_file("data/raw/UNSW_NB15_training-set.csv")
test_hash_pre  = sha256_file("data/raw/UNSW_NB15_testing-set.csv")
train_match = train_hash_pre == TRAIN_HASH_EXPECTED
test_match  = test_hash_pre  == TEST_HASH_EXPECTED
print(f"Train hash pre-run: {'MATCH' if train_match else 'MISMATCH'}")
print(f"Test hash pre-run:  {'MATCH' if test_match else 'MISMATCH'}")
if not (train_match and test_match):
    print("FATAL: Raw file hash mismatch. Aborting.")
    sys.exit(1)

# --- Load ---
with open("configs/project_config.yaml") as f:
    config = yaml.safe_load(f)

splits = load_raw_unswnb15(config)
train_df = splits["train"]
print(f"Train shape: {train_df.shape}")

dev_test_df  = pd.read_csv("data/splits/development_test.csv", low_memory=False)
protected_df = pd.read_csv("data/splits/protected_unseen_attack.csv", low_memory=False)
print(f"Dev test shape:  {dev_test_df.shape}")
print(f"Protected shape: {protected_df.shape}")

# --- Fit on TRAIN only ---
pp = PreprocessingPipeline("SMOKE_TEST")
pp.fit(train_df)
print()
print(f"Fitted | n_features={len(pp.feature_names)}")
print(f"Categorical cols: {pp._categorical_cols}")
print(f"Numeric col count: {len(pp._numeric_cols)}")
print(f"Feature names [0:8]: {pp.feature_names[:8]}")

# --- Transform TRAIN (both views) ---
ds_train_u, ds_train_s = pp.transform_both_views(train_df, split_name="train")
assert ds_train_u.feature_names == ds_train_s.feature_names, "Feature names differ between views!"
assert ds_train_u.n_rows == len(train_df), "Row count mismatch TRAIN!"
assert not np.allclose(ds_train_u.X, ds_train_s.X), "Scaled and unscaled X are identical!"
print(f"TRAIN unscaled: {ds_train_u.X.shape}  PASS")
print(f"TRAIN scaled:   {ds_train_s.X.shape}  PASS")

# --- Transform development_test (NO REFITTING) ---
ds_dev_u = pp.transform(dev_test_df, view="unscaled", split_name="development_test")
ds_dev_s = pp.transform(dev_test_df, view="scaled",   split_name="development_test")
assert ds_dev_u.n_rows == len(dev_test_df), "Row count mismatch dev test!"
backdoor_in_dev = (ds_dev_s.attack_cat == "Backdoor").sum()
assert backdoor_in_dev == 0, f"Backdoor found in dev test: {backdoor_in_dev} rows!"
print(f"Dev test unscaled: {ds_dev_u.X.shape}  Backdoor=0  PASS")
print(f"Dev test scaled:   {ds_dev_s.X.shape}  Backdoor=0  PASS")

# --- Transform protected (NO REFITTING) ---
ds_prot_s = pp.transform(protected_df, view="scaled",   split_name="protected_unseen_attack")
ds_prot_u = pp.transform(protected_df, view="unscaled", split_name="protected_unseen_attack")
assert ds_prot_s.n_rows == len(protected_df), "Row count mismatch protected!"
all_backdoor = (ds_prot_s.attack_cat == "Backdoor").all()
assert all_backdoor, "Protected set must be all Backdoor!"
assert ds_prot_s.n_features == len(pp.feature_names), "Feature count mismatch protected!"
print(f"Protected scaled:   {ds_prot_s.X.shape}  all_Backdoor=True  PASS")

# --- Feature names consistent across all splits ---
assert ds_train_u.feature_names == ds_dev_u.feature_names == ds_prot_u.feature_names
print("Feature names consistent TRAIN/dev_test/protected: PASS")

# --- label/attack_cat/id not in features ---
for forbidden in ("label", "attack_cat", "id"):
    assert forbidden not in pp.feature_names, f"{forbidden} in feature_names!"
print("label/attack_cat/id not in feature_names: PASS")

# --- Scaler statistics ---
scaler_mean_min = float(pp.fitted_scaler.train_mean.min())
scaler_mean_max = float(pp.fitted_scaler.train_mean.max())
print(f"Scaler TRAIN mean range: [{scaler_mean_min:.4f}, {scaler_mean_max:.4f}]")

# --- Raw file integrity post-run ---
train_hash_post = sha256_file("data/raw/UNSW_NB15_training-set.csv")
test_hash_post  = sha256_file("data/raw/UNSW_NB15_testing-set.csv")
assert train_hash_pre == train_hash_post, "FATAL: Train file hash changed post-run!"
assert test_hash_pre  == test_hash_post,  "FATAL: Test file hash changed post-run!"
print("Raw file hashes unchanged post-run: PASS")

# --- Protected set unchanged ---
protected_df2 = pd.read_csv("data/splits/protected_unseen_attack.csv", low_memory=False)
assert len(protected_df2) == len(protected_df), "Protected CSV row count changed!"
assert (protected_df2["attack_cat"] == "Backdoor").all(), "Protected CSV tampered!"
print("Protected unseen set unchanged: PASS")

print()
print("=== SMOKE TEST SUMMARY ===")
print(f"  n_features:        {len(pp.feature_names)}")
print(f"  TRAIN rows:        {ds_train_s.n_rows}")
print(f"  Dev test rows:     {ds_dev_s.n_rows}")
print(f"  Protected rows:    {ds_prot_s.n_rows}")
print(f"  Backdoor in dev:   0")
print(f"  Protected Backdoor: {ds_prot_s.n_rows}")
print(f"  Two model views:   PASS (scaled != unscaled)")
print(f"  No refitting:      PASS")
print(f"  Raw integrity:     PASS")
print()
print("STATUS: ALL PASS")
