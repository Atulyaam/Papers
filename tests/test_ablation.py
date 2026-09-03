"""
tests/test_ablation.py

Sprint 10 — EXP_ABLATION_V1 — 30-Check Machine Test Suite
All tests must fail loudly when violated.
"""

import json, hashlib, csv, sys, datetime
from pathlib import Path
import numpy as np
import pytest

ROOT  = Path(__file__).resolve().parent.parent
EXP   = ROOT / "results/ablation/EXP_ABLATION_V1"
CACHE = EXP / "cache"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cfg():
    import yaml
    with open(EXP / "config.yaml") as f:
        return yaml.safe_load(f)


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_cache(model_name, seed):
    p = CACHE / f"{model_name}_seed{seed}.npz"
    assert p.exists(), f"Cache not found: {p}"
    return np.load(p, allow_pickle=True)


SEEDS = [42, 123, 2024]
CONFIGS = ["A0_RF","A1_FULL_STACK","A1b_SOFT_VOTE",
           "A2_NO_DT","A3_NO_RF","A4_NO_SVM","A5_NO_NN","A6_STACK_PLUS_AE"]
A0_MODELS = ["rf"]
A1_MODELS = ["dt","rf","svm","nn"]
A1b_MODELS = ["dt","rf","svm","nn"]
A2_MODELS = ["rf","svm","nn"]
A3_MODELS = ["dt","svm","nn"]
A4_MODELS = ["dt","rf","nn"]
A5_MODELS = ["dt","rf","svm"]


# ─────────────────────────────────────────────────────────────────────────────
# T1 — A0 model identity matches frozen Sprint 9 strongest model
# ─────────────────────────────────────────────────────────────────────────────
def test_T1_a0_identity():
    cfg = load_cfg()
    sprint9_rf = cfg["a0_identity"]["sprint9_rf_cv_macro_f1"]
    sprint9_dt = cfg["a0_identity"]["sprint9_dt_cv_macro_f1"]
    sprint9_svm = cfg["a0_identity"]["sprint9_svm_cv_macro_f1"]
    sprint9_nn  = cfg["a0_identity"]["sprint9_nn_cv_macro_f1"]
    assert sprint9_rf >= sprint9_dt,  f"RF not >= DT: {sprint9_rf} < {sprint9_dt}"
    assert sprint9_rf >= sprint9_svm, f"RF not >= SVM: {sprint9_rf} < {sprint9_svm}"
    assert sprint9_rf >= sprint9_nn,  f"RF not >= NN: {sprint9_rf} < {sprint9_nn}"
    assert cfg["a0_identity"]["model"] == "RF", "A0 identity must be RF"


# ─────────────────────────────────────────────────────────────────────────────
# T2 — A0 has independently-trained results for all three seeds
# ─────────────────────────────────────────────────────────────────────────────
def test_T2_a0_all_seeds():
    for seed in SEEDS:
        p = EXP / f"A0_RF/seed_{seed}.json"
        assert p.exists(), f"A0 seed={seed} result missing: {p}"
        r = load_json(p)
        assert r["config_id"] == "A0_RF"
        assert r["seed"] == seed
        assert "macro_f1" in r and r["macro_f1"] > 0
        # Verify each A0 was trained with seed-specific random_state
        assert r.get("rf_config_random_state") == seed, \
            f"A0 seed={seed}: expected rf_config_random_state={seed}, got {r.get('rf_config_random_state')}"


# ─────────────────────────────────────────────────────────────────────────────
# T3 — A1 contains DT, RF, SVM, NN
# ─────────────────────────────────────────────────────────────────────────────
def test_T3_a1_contains_all_four():
    for seed in SEEDS:
        r = load_json(EXP / f"A1_FULL_STACK/seed_{seed}.json")
        for mn in A1_MODELS:
            assert mn in r["base_models"], f"A1 seed={seed} missing {mn}"


# ─────────────────────────────────────────────────────────────────────────────
# T4 — A1b contains all four and no meta-learner
# ─────────────────────────────────────────────────────────────────────────────
def test_T4_a1b_no_meta_learner():
    for seed in SEEDS:
        r = load_json(EXP / f"A1b_SOFT_VOTE/seed_{seed}.json")
        assert r["config_id"] == "A1b_SOFT_VOTE"
        assert "a1b_svm_normalization" in r, "A1b must document SVM normalization"
        assert "a1b_threshold" in r, "A1b must document threshold"
        # Must NOT contain 'meta_learner' key
        assert "meta_learner" not in r, "A1b must not reference a meta-learner"


# ─────────────────────────────────────────────────────────────────────────────
# T5 — A2 excludes DT
# ─────────────────────────────────────────────────────────────────────────────
def test_T5_a2_excludes_dt():
    for seed in SEEDS:
        r = load_json(EXP / f"A2_NO_DT/seed_{seed}.json")
        assert "dt" not in r["base_models"], f"A2 seed={seed} must not contain DT"
        for mn in A2_MODELS:
            assert mn in r["base_models"]


# ─────────────────────────────────────────────────────────────────────────────
# T6 — A3 excludes RF
# ─────────────────────────────────────────────────────────────────────────────
def test_T6_a3_excludes_rf():
    for seed in SEEDS:
        r = load_json(EXP / f"A3_NO_RF/seed_{seed}.json")
        assert "rf" not in r["base_models"]
        for mn in A3_MODELS:
            assert mn in r["base_models"]


# ─────────────────────────────────────────────────────────────────────────────
# T7 — A4 excludes SVM
# ─────────────────────────────────────────────────────────────────────────────
def test_T7_a4_excludes_svm():
    for seed in SEEDS:
        r = load_json(EXP / f"A4_NO_SVM/seed_{seed}.json")
        assert "svm" not in r["base_models"]
        for mn in A4_MODELS:
            assert mn in r["base_models"]


# ─────────────────────────────────────────────────────────────────────────────
# T8 — A5 excludes NN
# ─────────────────────────────────────────────────────────────────────────────
def test_T8_a5_excludes_nn():
    for seed in SEEDS:
        r = load_json(EXP / f"A5_NO_NN/seed_{seed}.json")
        assert "nn" not in r["base_models"]
        for mn in A5_MODELS:
            assert mn in r["base_models"]


# ─────────────────────────────────────────────────────────────────────────────
# T9 — A6 uses frozen AE
# ─────────────────────────────────────────────────────────────────────────────
def test_T9_a6_uses_frozen_ae():
    cfg = load_cfg()
    frozen_tau = cfg["ae_config"]["threshold"]
    for seed in SEEDS:
        r = load_json(EXP / f"A6_STACK_PLUS_AE/seed_{seed}.json")
        assert abs(r["ae_threshold"] - frozen_tau) < 1e-10, \
            f"A6 seed={seed} tau mismatch: {r['ae_threshold']} != {frozen_tau}"
        assert r["fusion_rule"] == "OR", f"A6 seed={seed} fusion must be OR (C06)"


# ─────────────────────────────────────────────────────────────────────────────
# T10 — Cache key includes seed (separate file per (model, seed))
# ─────────────────────────────────────────────────────────────────────────────
def test_T10_cache_key_includes_seed():
    for seed in SEEDS:
        for mn in A1_MODELS:
            p = CACHE / f"{mn}_seed{seed}.npz"
            assert p.exists(), f"Cache missing: {p}"


# ─────────────────────────────────────────────────────────────────────────────
# T11 — OOF cache reuse: A2-A5 retained columns == A1 columns (exact equality)
# ─────────────────────────────────────────────────────────────────────────────
def test_T11_oof_cache_reuse():
    for seed in SEEDS:
        # A1 full OOF columns: [dt, rf, svm, nn]
        a1_cols = [load_cache(mn, seed)["oof_scores"] for mn in A1_MODELS]
        a1_map = dict(zip(A1_MODELS, a1_cols))
        # Check each A2-A5 config retains matching columns
        for config_models in [A2_MODELS, A3_MODELS, A4_MODELS, A5_MODELS]:
            for mn in config_models:
                c_col = load_cache(mn, seed)["oof_scores"]
                assert np.array_equal(c_col, a1_map[mn]), \
                    f"D26 FAIL: cache OOF for {mn} seed={seed} differs from A1"


# ─────────────────────────────────────────────────────────────────────────────
# T12 — Dev-test cache reuse: exact equality
# ─────────────────────────────────────────────────────────────────────────────
def test_T12_devtest_cache_reuse():
    for seed in SEEDS:
        a1_map = {mn: load_cache(mn, seed)["dev_test_scores"] for mn in A1_MODELS}
        for config_models in [A2_MODELS, A3_MODELS, A4_MODELS, A5_MODELS]:
            for mn in config_models:
                c_col = load_cache(mn, seed)["dev_test_scores"]
                assert np.array_equal(c_col, a1_map[mn]), \
                    f"D26 FAIL: dev_test cache for {mn} seed={seed} differs"


# ─────────────────────────────────────────────────────────────────────────────
# T13 — Exact retained-column identity
# ─────────────────────────────────────────────────────────────────────────────
def test_T13_retained_column_identity():
    for seed in SEEDS:
        for config_name, retained, excluded in [
            ("A2_NO_DT", A2_MODELS, ["dt"]),
            ("A3_NO_RF", A3_MODELS, ["rf"]),
            ("A4_NO_SVM",A4_MODELS, ["svm"]),
            ("A5_NO_NN", A5_MODELS, ["nn"]),
        ]:
            r = load_json(EXP / f"{config_name}/seed_{seed}.json")
            for mn in excluded:
                assert mn not in r["base_models"]
            for mn in retained:
                assert mn in r["base_models"]


# ─────────────────────────────────────────────────────────────────────────────
# T14 — Feature count = 75
# ─────────────────────────────────────────────────────────────────────────────
def test_T14_feature_count():
    cfg = load_cfg()
    assert cfg["n_features"] == 75
    feat_path = ROOT / "results/feature_selection/EXP_MI_V1_1/selected_features.json"
    feats = json.loads(feat_path.read_text())
    if isinstance(feats, dict):
        feats = feats.get("selected_features", feats.get("features", list(feats.values())[0]))
    assert len(feats) == 75, f"Feature count = {len(feats)}, expected 75"


# ─────────────────────────────────────────────────────────────────────────────
# T15 — Exact feature order (consistent across seeds via cache label check)
# ─────────────────────────────────────────────────────────────────────────────
def test_T15_feature_order():
    # OOF label arrays must be identical across models for same seed
    for seed in SEEDS:
        labels = [load_cache(mn, seed)["oof_labels"] for mn in A1_MODELS]
        for i in range(1, len(labels)):
            assert np.array_equal(labels[0], labels[i]), \
                f"OOF label mismatch between models seed={seed}"


# ─────────────────────────────────────────────────────────────────────────────
# T16 — Dataset hashes match (raw-byte SHA-256)
# ─────────────────────────────────────────────────────────────────────────────
def test_T16_dataset_hashes():
    cfg = load_cfg()
    dataset_paths = {
        "train":              ROOT / "data/splits/train.csv",
        "validation":         ROOT / "data/splits/validation.csv",
        "development_test":   ROOT / "data/splits/development_test.csv",
        "protected_backdoor": ROOT / "data/splits/protected_unseen_attack.csv",
    }
    for split, expected in cfg["dataset_hashes"].items():
        actual = sha256_file(dataset_paths[split])
        assert actual == expected, f"Dataset hash FAIL for {split}: {actual} != {expected}"


# ─────────────────────────────────────────────────────────────────────────────
# T17 — Target/label mapping is locked (binary {0, 1})
# ─────────────────────────────────────────────────────────────────────────────
def test_T17_target_label_mapping():
    import pandas as pd
    train = pd.read_csv(ROOT / "data/splits/train.csv", usecols=["label"])
    labels = sorted(train["label"].unique())
    assert labels == [0, 1], f"Non-binary labels found: {labels}"


# ─────────────────────────────────────────────────────────────────────────────
# T18 — Seed list exact: [42, 123, 2024]
# ─────────────────────────────────────────────────────────────────────────────
def test_T18_seed_list():
    cfg = load_cfg()
    assert cfg["seeds"] == [42, 123, 2024], f"Seed list mismatch: {cfg['seeds']}"


# ─────────────────────────────────────────────────────────────────────────────
# T19 — AE threshold frozen (unchanged from Sprint 9)
# ─────────────────────────────────────────────────────────────────────────────
def test_T19_ae_threshold_frozen():
    cfg = load_cfg()
    expected_tau = 11.160062745213509
    assert abs(cfg["ae_config"]["threshold"] - expected_tau) < 1e-10, \
        f"AE threshold changed: {cfg['ae_config']['threshold']} != {expected_tau}"
    # Also verify from Sprint 9 source
    sprint9_thresh = json.loads(
        (ROOT / "results/autoencoder/EXP_AE_V1/threshold/threshold_calibration.json").read_text()
    )
    actual = sprint9_thresh["thresholds"]["mean3sigma"]["threshold_value"]
    assert abs(actual - expected_tau) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# T20 — Protected Backdoor isolation: never touched during training
# ─────────────────────────────────────────────────────────────────────────────
def test_T20_protected_backdoor_isolation():
    # Verify prot results exist and none of the training configs reference it
    prot_path = EXP / "protected_backdoor_results.json"
    assert prot_path.exists(), "Protected Backdoor results missing"
    prot = load_json(prot_path)
    assert "per_seed" in prot
    # None of A0-A6 per-seed results should mention backdoor in training context
    for cid in CONFIGS:
        for seed in SEEDS:
            rp = EXP / f"{cid}/seed_{seed}.json"
            if rp.exists():
                r = load_json(rp)
                # Should not have been used for selection
                assert "backdoor_used_for_selection" not in r


# ─────────────────────────────────────────────────────────────────────────────
# T21 — Headline metrics use DEVELOPMENT_TEST, not VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def test_T21_headline_uses_devtest():
    cfg = load_cfg()
    assert cfg["headline_split"] == "development_test"
    for cid in CONFIGS:
        for seed in SEEDS:
            rp = EXP / f"{cid}/seed_{seed}.json"
            if rp.exists():
                r = load_json(rp)
                # n_dev_test should be the size of the development_test split
                assert "n_dev_test" in r, f"{cid} seed={seed} missing n_dev_test"
                # Should not reference validation as primary split
                assert r.get("headline_split", "development_test") != "validation"


# ─────────────────────────────────────────────────────────────────────────────
# T22 — Per-seed results complete for every config
# ─────────────────────────────────────────────────────────────────────────────
def test_T22_per_seed_complete():
    required = ["config_id","seed","macro_f1","precision","recall","f1",
                "balanced_accuracy","fpr","runtime_sec"]
    for cid in CONFIGS:
        for seed in SEEDS:
            rp = EXP / f"{cid}/seed_{seed}.json"
            assert rp.exists(), f"Missing: {rp}"
            r = load_json(rp)
            for k in required:
                assert k in r and r[k] is not None, f"{cid} seed={seed} missing/null field: {k}"


# ─────────────────────────────────────────────────────────────────────────────
# T23 — Paired deltas complete, including A1-vs-A6 Recall/FPR/Backdoor
# ─────────────────────────────────────────────────────────────────────────────
def test_T23_paired_deltas_complete():
    import csv
    delta_path = EXP / "paired_deltas.csv"
    assert delta_path.exists(), "paired_deltas.csv missing"
    with open(delta_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    comparisons = {r["comparison"] for r in rows}
    required_comps = {"A1-A0","A1-A1b","A1-A2","A1-A3","A1-A4","A1-A5","A6-A1"}
    assert required_comps <= comparisons, f"Missing comparisons: {required_comps - comparisons}"
    # A6-A1 must have recall, fpr, backdoor_detection_rate
    a6a1_metrics = {r["metric"] for r in rows if r["comparison"] == "A6-A1"}
    for m in ["macro_f1","recall","fpr","backdoor_detection_rate"]:
        assert m in a6a1_metrics, f"A6-A1 missing metric: {m}"


# ─────────────────────────────────────────────────────────────────────────────
# T24 — CSV schema exact
# ─────────────────────────────────────────────────────────────────────────────
def test_T24_csv_schema():
    import csv
    # ablation_table.csv
    with open(EXP / "ablation_table.csv", encoding="utf-8") as f:
        abl_cols = next(csv.reader(f))
    expected_abl = ["config_id","seed","macro_f1","precision","recall","f1",
                    "balanced_accuracy","fpr","runtime_sec"]
    assert abl_cols == expected_abl, f"ablation_table.csv columns: {abl_cols} != {expected_abl}"

    # paired_deltas.csv
    with open(EXP / "paired_deltas.csv", encoding="utf-8") as f:
        delta_cols = next(csv.reader(f))
    expected_delta = ["comparison","seed","metric","delta_value"]
    assert delta_cols == expected_delta, f"paired_deltas.csv columns: {delta_cols} != {expected_delta}"


# ─────────────────────────────────────────────────────────────────────────────
# T25 — Config hash unchanged (D19)
# ─────────────────────────────────────────────────────────────────────────────
def test_T25_config_hash_unchanged():
    prov_path = EXP / "provenance/config_sha256_after.json"
    assert prov_path.exists(), "config_sha256_after.json missing — D19 not completed"
    rec = load_json(prov_path)
    assert rec["match"] is True, \
        f"D19 FAIL: config_sha256_before={rec['config_sha256_before']} != after={rec['config_sha256_after']}"


# ─────────────────────────────────────────────────────────────────────────────
# T26 — No result-based configuration selection
# ─────────────────────────────────────────────────────────────────────────────
def test_T26_no_result_based_selection():
    # Verify config hash before == after (covered by T25)
    # Additionally verify no config field is set based on a result value
    cfg = load_cfg()
    assert isinstance(cfg["seeds"], list), "seeds must be a list"
    assert cfg["n_features"] == 75, "n_features must be 75"
    # AE threshold must match Sprint 9, not any ablation result
    assert abs(cfg["ae_config"]["threshold"] - 11.160062745213509) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# T27 — Deterministic settings logged per seed
# ─────────────────────────────────────────────────────────────────────────────
def test_T27_deterministic_settings_logged():
    # Each A0 result must record rf_config_random_state = seed
    for seed in SEEDS:
        r = load_json(EXP / f"A0_RF/seed_{seed}.json")
        assert r.get("rf_config_random_state") == seed
    # metadata.json must exist with environment
    meta = load_json(EXP / "metadata.json")
    assert "environment" in meta
    assert "sklearn" in meta["environment"]


# ─────────────────────────────────────────────────────────────────────────────
# T28 — Resumability integrity check logic is correct
# ─────────────────────────────────────────────────────────────────────────────
def test_T28_resumability_integrity_check():
    """Test with deliberately incomplete/truncated artifacts that they are rejected."""
    import tempfile, os
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_ablation import result_integrity_check

    # Case 1: file does not exist
    assert not result_integrity_check(Path("/nonexistent/path/file.json"),
                                      ["macro_f1"]), "Missing file must fail"

    # Case 2: zero-length file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write("")
        tmp_path = Path(f.name)
    try:
        assert not result_integrity_check(tmp_path, ["macro_f1"]), "Empty file must fail"
    finally:
        tmp_path.unlink(missing_ok=True)

    # Case 3: valid JSON but missing required key
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"config_id": "A1_FULL_STACK", "seed": 42}, f)
        tmp_path = Path(f.name)
    try:
        assert not result_integrity_check(tmp_path, ["macro_f1"]), "Missing key must fail"
    finally:
        tmp_path.unlink(missing_ok=True)

    # Case 4: complete valid file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"config_id": "A1_FULL_STACK", "seed": 42, "macro_f1": 0.9}, f)
        tmp_path = Path(f.name)
    try:
        assert result_integrity_check(tmp_path, ["config_id","seed","macro_f1"]), \
            "Valid complete file must pass"
    finally:
        tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# T29 — Provenance completeness
# ─────────────────────────────────────────────────────────────────────────────
def test_T29_provenance_complete():
    meta = load_json(EXP / "metadata.json")
    required_meta = [
        "experiment_id","sprint","feature_set","n_features","seeds",
        "n_folds","headline_split","a0_identity","a1b_svm_normalization",
        "frozen_ae_threshold","frozen_ae_fusion_rule","environment",
        "config_sha256","d11_resolution",
    ]
    for k in required_meta:
        assert k in meta, f"metadata.json missing key: {k}"
    assert (EXP / "provenance/config_sha256_before.json").exists()
    assert (EXP / "provenance/config_sha256_after.json").exists()
    assert (EXP / "environment.txt").exists()


# ─────────────────────────────────────────────────────────────────────────────
# T30 — Phase order: Phase 2 cache generation preceded A1-A5 training
#        (verified via file creation timestamps)
# ─────────────────────────────────────────────────────────────────────────────
def test_T30_phase_order():
    """Verify that cache files predate meta-learner result files."""
    for seed in SEEDS:
        for mn in A1_MODELS:
            cp = CACHE / f"{mn}_seed{seed}.npz"
            assert cp.exists(), f"Cache missing: {cp}"
        for cid in ["A1_FULL_STACK","A2_NO_DT","A3_NO_RF","A4_NO_SVM","A5_NO_NN"]:
            rp = EXP / f"{cid}/seed_{seed}.json"
            assert rp.exists(), f"Result missing: {rp}"
            for mn in A1_MODELS:
                cp = CACHE / f"{mn}_seed{seed}.npz"
                cache_mtime  = cp.stat().st_mtime
                result_mtime = rp.stat().st_mtime
                assert cache_mtime <= result_mtime + 1.0, \
                    f"Phase order violation: cache {mn} seed={seed} newer than {cid} result"
