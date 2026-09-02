"""
tests/test_fusion_isolation.py
================================
Sprint 8 — EXP_FUSION_V1 Focused Fusion Isolation Tests (STEP 2).

Tests T-CANDIDATE-COUNT, T-CANDIDATE-IDS, T-CANDIDATE-DETERMINISTIC,
T-75-FEATURES, T-THRESHOLD-MATCH, T-SIGMA-CAUTION, T-AE-STRICT,
T-RE-DEFINITION, T-VAL-NORMAL, T-VAL-FPR-ONLY, T-S6-CANONICAL-CHECKPOINT.
"""

import json
import pathlib
import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).parent.parent

# Import the script under test
import sys
sys.path.insert(0, str(ROOT))
from scripts.run_fusion_evaluation import (
    build_candidate_configs,
    FROZEN_TAU,
    WITHIN_RULE_PRIORITY,
    FPR_GATE,
    PROT_N,
    run_selection,
    fuse,
    compute_fpr_only,
    AEAdapter,
    SupervisedAdapter,
)


# ---------------------------------------------------------------------------
# T-CANDIDATE-COUNT / T-CANDIDATE-IDS / T-CANDIDATE-DETERMINISTIC
# ---------------------------------------------------------------------------

def test_candidate_count():
    """T-CANDIDATE-COUNT: exactly 11 configurations."""
    configs = build_candidate_configs()
    assert len(configs) == 11, f"Expected 11 configs, got {len(configs)}"


def test_candidate_ids():
    """T-CANDIDATE-IDS: exactly C01..C11."""
    configs = build_candidate_configs()
    ids = {c.config_id for c in configs}
    expected = {f"C{i:02d}" for i in range(1, 12)}
    assert ids == expected, f"Unexpected ids: {ids ^ expected}"


def test_candidate_deterministic():
    """T-CANDIDATE-DETERMINISTIC: same order on every call."""
    c1 = [c.config_id for c in build_candidate_configs()]
    c2 = [c.config_id for c in build_candidate_configs()]
    assert c1 == c2


def test_c01_is_supervised_only():
    """C01 must be supervised_only with no threshold."""
    c01 = build_candidate_configs()[0]
    assert c01.config_id == "C01"
    assert c01.rule == "supervised_only"
    assert c01.tau is None
    assert c01.threshold_key is None


def test_or_configs():
    """C02-C06 must all be OR rule."""
    configs = build_candidate_configs()
    or_configs = [c for c in configs if c.rule == "OR"]
    assert len(or_configs) == 5
    assert {c.config_id for c in or_configs} == {"C02","C03","C04","C05","C06"}


def test_and_configs():
    """C07-C11 must all be AND rule."""
    configs = build_candidate_configs()
    and_configs = [c for c in configs if c.rule == "AND"]
    assert len(and_configs) == 5
    assert {c.config_id for c in and_configs} == {"C07","C08","C09","C10","C11"}


# ---------------------------------------------------------------------------
# T-THRESHOLD-MATCH
# ---------------------------------------------------------------------------

def test_threshold_match():
    """T-THRESHOLD-MATCH: Sprint 8 tau values match Sprint 7 frozen values."""
    thr = json.load(open(ROOT / "results/checkpoints/EXP_AE_V1/threshold_config.json"))
    thresholds = thr["thresholds"]

    mapping = {
        "p95":        ("p95",        "threshold_value"),
        "p99":        ("p99",        "threshold_value"),
        "p999":       ("p999",       "threshold_value"),
        "mean2sigma": ("mean2sigma", "threshold_value"),
        "mean3sigma": ("mean3sigma", "threshold_value"),
    }
    for key, (thr_key, val_key) in mapping.items():
        expected = thresholds[thr_key][val_key]
        actual   = FROZEN_TAU[key]
        assert abs(actual - expected) < 1e-9, \
            f"Tau mismatch for {key}: expected={expected} actual={actual}"


def test_tau_ordering():
    """Verify tau ordering: p95 < p99 < mean2sigma < p999 < mean3sigma."""
    assert FROZEN_TAU["p95"] < FROZEN_TAU["p99"]
    assert FROZEN_TAU["p99"] < FROZEN_TAU["mean2sigma"]
    assert FROZEN_TAU["mean2sigma"] < FROZEN_TAU["p999"]
    assert FROZEN_TAU["p999"] < FROZEN_TAU["mean3sigma"]


def test_within_rule_priority_conservative_first():
    """OD-4b: within-rule priority is conservative-first (descending tau)."""
    tau_order = [FROZEN_TAU[k] for k in WITHIN_RULE_PRIORITY]
    # Must be descending
    for i in range(len(tau_order) - 1):
        assert tau_order[i] > tau_order[i+1], \
            f"Priority not descending at position {i}: {tau_order[i]} <= {tau_order[i+1]}"


# ---------------------------------------------------------------------------
# T-SIGMA-CAUTION
# ---------------------------------------------------------------------------

def test_sigma_caution_flags():
    """T-SIGMA-CAUTION: mean2sigma and mean3sigma configs are flagged."""
    configs = build_candidate_configs()
    for cfg in configs:
        if cfg.threshold_key in ("mean2sigma", "mean3sigma"):
            assert cfg.outlier_influenced is True, \
                f"{cfg.config_id} should have outlier_influenced=True"
        elif cfg.threshold_key in ("p95", "p99", "p999", None):
            assert cfg.outlier_influenced is False, \
                f"{cfg.config_id} should have outlier_influenced=False"


# ---------------------------------------------------------------------------
# T-AE-STRICT / T-RE-DEFINITION
# ---------------------------------------------------------------------------

def test_ae_strict_boundary():
    """T-AE-STRICT: RE == tau classified as Normal (flag=0), not anomaly."""
    from scripts.run_fusion_evaluation import AEAdapter
    adapter = AEAdapter()
    re = np.array([0.5, 1.0, 1.5])
    tau = 1.0
    flags = adapter.ae_flag(re, tau)
    # RE=1.0 == tau → Normal (0)
    assert flags[1] == 0, f"RE==tau must be Normal, got {flags[1]}"
    # RE=0.5 < tau → Normal
    assert flags[0] == 0
    # RE=1.5 > tau → Anomaly
    assert flags[2] == 1


def test_fuse_or():
    """OR rule: flag if supervised OR ae_flag."""
    sup  = np.array([0, 1, 0, 1])
    ae_f = np.array([0, 0, 1, 1])
    out  = fuse(sup, ae_f, "OR")
    np.testing.assert_array_equal(out, [0, 1, 1, 1])


def test_fuse_and():
    """AND rule: flag only if supervised AND ae_flag."""
    sup  = np.array([0, 1, 0, 1])
    ae_f = np.array([0, 0, 1, 1])
    out  = fuse(sup, ae_f, "AND")
    np.testing.assert_array_equal(out, [0, 0, 0, 1])


def test_fuse_supervised_only():
    """supervised_only: AE flag ignored."""
    sup  = np.array([0, 1, 0, 1])
    ae_f = np.array([1, 1, 1, 1])
    out  = fuse(sup, ae_f, "supervised_only")
    np.testing.assert_array_equal(out, [0, 1, 0, 1])


# ---------------------------------------------------------------------------
# T-VAL-NORMAL / T-VAL-FPR-ONLY
# ---------------------------------------------------------------------------

def test_validation_normal_only():
    """T-VAL-NORMAL: VALIDATION split contains only label=0 rows."""
    val = pd.read_csv(ROOT / "data/splits/validation.csv")
    assert (val["label"] == 0).all(), "VALIDATION must be all Normal (label=0)"
    assert len(val) == 11200, f"Expected 11200 validation rows, got {len(val)}"


def test_compute_fpr_only_rejects_attack_rows():
    """T-VAL-FPR-ONLY: compute_fpr_only must assert all y_true==0."""
    y_true_bad = np.array([0, 1, 0])  # has a label=1
    y_pred     = np.array([0, 0, 0])
    with pytest.raises(AssertionError):
        compute_fpr_only(y_true_bad, y_pred)


def test_compute_fpr_only_correct():
    """FPR calculation: 2 FP out of 10 Normal = 0.2."""
    y_true = np.zeros(10, dtype=int)
    y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    fpr, fp = compute_fpr_only(y_true, y_pred)
    assert fp == 2
    assert abs(fpr - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# T-S6-CANONICAL-CHECKPOINT
# ---------------------------------------------------------------------------

def test_s6_canonical_checkpoint_exists():
    """T-S6-CANONICAL-CHECKPOINT: EXP_OOF_STACK_V1 seed-42 checkpoint exists."""
    lr_path = ROOT / "results/checkpoints/EXP_OOF_STACK_V1/seed_42/meta_learner.joblib"
    assert lr_path.exists(), f"Seed-42 meta-learner checkpoint not found: {lr_path}"


def test_s6_seed42_metadata():
    """Seed-42 metadata confirms h1_seed=42 and feature_count=75."""
    meta = json.load(open(
        ROOT / "results/checkpoints/EXP_OOF_STACK_V1/seed_42/metadata.json"
    ))
    assert meta.get("h1_seed") == 42, f"h1_seed mismatch: {meta.get('h1_seed')}"
    assert meta.get("feature_count") == 75, f"feature_count mismatch: {meta.get('feature_count')}"
    assert meta.get("experiment_id") == "EXP_OOF_STACK_V1"


# ---------------------------------------------------------------------------
# T-75-FEATURES
# ---------------------------------------------------------------------------

def test_75_features():
    """T-75-FEATURES: frozen feature list has exactly 75 features."""
    from src.models.base_models.preprocessing import load_selected_features
    features = load_selected_features()
    assert len(features) == 75, f"Expected 75 features, got {len(features)}"
    assert len(set(features)) == 75, "Duplicate features detected"


# ---------------------------------------------------------------------------
# Selection function unit tests
# ---------------------------------------------------------------------------

def test_selection_od5_fallback():
    """OD-5: if no config passes gate, C01 selected with fallback_triggered=True."""
    configs = build_candidate_configs()
    # All FPR > gate
    fpr_values = {c.config_id: 0.99 for c in configs}
    result = run_selection(configs, fpr_values, gate=0.05)
    assert result["selected_config"] == "C01"
    assert result["fallback_triggered"] is True
    assert result["n_passing"] == 0


def test_selection_or_preferred_over_and():
    """OD-4: OR preferred over AND when both pass."""
    configs = build_candidate_configs()
    # All pass gate
    fpr_values = {c.config_id: 0.01 for c in configs}
    result = run_selection(configs, fpr_values, gate=0.05)
    selected = result["selected_config"]
    cfg = next(c for c in configs if c.config_id == selected)
    assert cfg.rule == "OR", f"Expected OR, got {cfg.rule}"


def test_selection_conservative_first_within_or():
    """OD-4b: within OR, largest tau selected first."""
    configs = build_candidate_configs()
    # All pass gate
    fpr_values = {c.config_id: 0.01 for c in configs}
    result = run_selection(configs, fpr_values, gate=0.05)
    # C06 = OR + mean3sigma (largest tau among OR) should be selected
    assert result["selected_config"] == "C06", \
        f"Expected C06 (OR+mean3sigma), got {result['selected_config']}"


def test_selection_falls_to_and_if_no_or_passes():
    """If no OR config passes gate, AND is preferred over supervised_only."""
    configs = build_candidate_configs()
    or_ids = {c.config_id for c in configs if c.rule == "OR"}
    # OR configs fail, AND configs pass
    fpr_values = {c.config_id: (0.10 if c.config_id in or_ids else 0.01)
                  for c in configs}
    result = run_selection(configs, fpr_values, gate=0.05)
    cfg = next(c for c in configs if c.config_id == result["selected_config"])
    assert cfg.rule == "AND", f"Expected AND, got {cfg.rule}"


def test_selection_n_candidates_always_11():
    """n_candidates in selection output is always 11."""
    configs = build_candidate_configs()
    fpr_values = {c.config_id: 0.01 for c in configs}
    result = run_selection(configs, fpr_values)
    assert result["n_candidates"] == 11


def test_selection_outlier_flag_for_sigma_configs():
    """outlier_influenced=True when a mean±sigma config is selected."""
    configs = build_candidate_configs()
    # Make only mean3sigma OR pass
    sigma_or_ids = {c.config_id for c in configs
                    if c.rule == "OR" and c.threshold_key == "mean3sigma"}
    # All pass at 0.01
    fpr_values = {c.config_id: 0.01 for c in configs}
    result = run_selection(configs, fpr_values)
    # C06 (mean3sigma OR) should be selected — it is outlier_influenced
    assert result["outlier_influenced"] is True


def test_selection_c01_is_always_baseline():
    """baseline_config must always be C01."""
    configs = build_candidate_configs()
    fpr_values = {c.config_id: 0.01 for c in configs}
    result = run_selection(configs, fpr_values)
    assert result["baseline_config"] == "C01"


def test_fpr_gate_value():
    """OD-3: FPR gate is exactly 5%."""
    assert FPR_GATE == 0.05


def test_prot_n():
    """OD-7: Protected Backdoor n=583."""
    assert PROT_N == 583
