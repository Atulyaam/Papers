"""
scripts/run_fusion_evaluation.py
=================================
Sprint 8 — EXP_FUSION_V1 Fusion Evaluation

Implements the fully-specified protocol from the approved Final Design.

Approved decisions applied:
  OD-1a  : LR.predict() at frozen 0.5 boundary, EXP_OOF_STACK_V1 seed-42
  OD-2a  : RE > tau  (strict greater-than)
  OD-3   : Validation FPR gate = 5%
  OD-4   : Option A  — gate only; rule priority OR > AND > Supervised-only
  OD-4b  : Conservative-first within-rule: mean3sigma > p999 > mean2sigma > p99 > p95
  OD-5   : C01 fallback if no config passes gate
  OD-6   : Dev TEST metric hierarchy: Macro-F1 primary
  OD-7   : Protected Backdoor: counts + rate, n=583, 0.1716 pp/row caveat
  OD-8   : Per-config RST/FIN subgroup FPR analysis
  OD-9   : 11 unique configurations (C01-C11)
  OD-10  : H-FUSION, H-PROT-BACKDOOR

Execution order (from Final Design §13):
  STEP 0: Verify upstream artifacts (called externally)
  STEP 1: Load inference adapters
  STEP 2-3: Tests called externally
  STEP 4: Enumerate 11 configurations
  STEP 5: Compute Validation FPR + subgroup FPR
  STEP 6: Apply selection function
  STEP 7: Freeze ONE configuration
  STEP 8: Dev TEST (single-shot)
  STEP 9: Protected Backdoor (single-shot)
  STEP 10: Comparison analyses
  STEP 11: Exploratory all-11 (INFORMATIONAL ONLY)
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import datetime
from dataclasses import asdict, dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.autoencoder.ae_model import Autoencoder
from src.models.base_models.preprocessing import (
    load_selected_features,
    build_feature_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("sprint8.fusion")


# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------

def load_and_encode_splits(
    split_names: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Load raw CSV splits and encode them via PreprocessingPipeline.

    The pipeline is fit ONCE on TRAIN (same as all prior sprints).
    Then each requested split is transformed to produce the encoded
    feature DataFrame compatible with build_feature_matrix(df, features).

    Parameters
    ----------
    split_names : list of str
        E.g. ["validation", "development_test", "protected_unseen_attack"]
        "train" is always loaded internally for fitting.

    Returns
    -------
    dict mapping split name → encoded DataFrame with 75 feature columns
    plus 'label' and optionally 'row_id'.
    """
    from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

    logger.info("Loading and encoding splits: %s", split_names)

    # Fit pipeline on TRAIN (never on any other split)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    pipeline = PreprocessingPipeline()
    pipeline.fit(train_raw)
    logger.info("  PreprocessingPipeline fitted on TRAIN (%d rows)", len(train_raw))

    features = load_selected_features()

    results: dict[str, pd.DataFrame] = {}
    for name in split_names:
        fname = "protected_unseen_attack.csv" if name == "protected_unseen_attack" \
                else f"{name}.csv"
        raw = pd.read_csv(DATA_DIR / fname)
        enc = pipeline.transform(raw, view="unscaled", split_name=name)
        feat_df = pd.DataFrame(enc.X, columns=enc.feature_names)
        # Attach label and row_id for downstream use
        feat_df["label"] = raw["label"].values
        if "id" in raw.columns:
            feat_df["row_id"] = raw["id"].values
        elif "row_id" in raw.columns:
            feat_df["row_id"] = raw["row_id"].values
        else:
            feat_df["row_id"] = feat_df.index.values
        # Verify all 75 features present
        missing = [f for f in features if f not in feat_df.columns]
        assert not missing, f"Missing features in {name}: {missing[:5]}"
        results[name] = feat_df
        logger.info("  Encoded split '%s': %d rows, %d cols",
                    name, len(feat_df), feat_df.shape[1])

    # Store for RST/FIN subgroup detection (raw state col)
    results["_train_raw"] = train_raw
    return results


def get_raw_state_col(raw_df: pd.DataFrame) -> pd.Series:
    """Return the raw 'state' column from a raw split for subgroup detection."""
    return raw_df.get("state", pd.Series(dtype=str))


# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

SPRINT8_EXPERIMENT_ID = "EXP_FUSION_V1"

# Frozen canonical tau values from EXP_AE_V1 threshold_config.json
FROZEN_TAU: dict[str, float] = {
    "p95":        0.5673861503601074,
    "p99":        1.5121638774871826,
    "mean2sigma": 7.5151087790727615,
    "p999":       10.69687557220459,
    "mean3sigma": 11.160062745213509,
}

# OD-4b: conservative-first within-rule priority (descending tau)
WITHIN_RULE_PRIORITY: list[str] = [
    "mean3sigma",   # tau=11.160  (largest, least aggressive)
    "p999",         # tau=10.697
    "mean2sigma",   # tau=7.515  ⚠ outlier-influenced
    "p99",          # tau=1.512
    "p95",          # tau=0.567  (smallest, most aggressive)
]

# OD-3: FPR gate
FPR_GATE: float = 0.05

# Data paths
DATA_DIR = ROOT / "data/splits"
CKPT_AE = ROOT / "results/checkpoints/EXP_AE_V1"
CKPT_S6_SEED42 = ROOT / "results/checkpoints/EXP_OOF_STACK_V1/seed_42"
CKPT_BASE = ROOT / "results/checkpoints/EXP_BASE_MODELS_V1"
OUTPUT_DIR = ROOT / "results/fusion/EXP_FUSION_V1"

# OD-7: Protected Backdoor size
PROT_N = 583
PROT_PP_PER_ROW = round(1 / PROT_N, 6)   # 0.001716...


# ---------------------------------------------------------------------------
# 11 Candidate Configuration Definitions  (OD-9)
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    config_id: str
    rule: str                  # "supervised_only", "OR", "AND"
    threshold_key: Optional[str]  # None for C01
    tau: Optional[float]
    outlier_influenced: bool

    def __post_init__(self):
        assert self.config_id in {f"C{i:02d}" for i in range(1, 12)}, \
            f"Invalid config_id {self.config_id}"


def build_candidate_configs() -> list[FusionConfig]:
    """
    OD-9: Build exactly 11 unique configurations.
    C01 = Supervised-only
    C02-C06 = OR (ordered by tau ascending for table clarity; selection priority is OD-4b)
    C07-C11 = AND (same)
    """
    outlier_keys = {"mean2sigma", "mean3sigma"}

    configs = [
        # C01 — Supervised-only (OD-5 fallback)
        FusionConfig("C01", "supervised_only", None, None, False),
        # C02-C06 — OR configurations (tau ascending)
        FusionConfig("C02", "OR", "p95",        FROZEN_TAU["p95"],        False),
        FusionConfig("C03", "OR", "p99",        FROZEN_TAU["p99"],        False),
        FusionConfig("C04", "OR", "mean2sigma", FROZEN_TAU["mean2sigma"], True),
        FusionConfig("C05", "OR", "p999",       FROZEN_TAU["p999"],       False),
        FusionConfig("C06", "OR", "mean3sigma", FROZEN_TAU["mean3sigma"], True),
        # C07-C11 — AND configurations (tau ascending)
        FusionConfig("C07", "AND", "p95",        FROZEN_TAU["p95"],        False),
        FusionConfig("C08", "AND", "p99",        FROZEN_TAU["p99"],        False),
        FusionConfig("C09", "AND", "mean2sigma", FROZEN_TAU["mean2sigma"], True),
        FusionConfig("C10", "AND", "p999",       FROZEN_TAU["p999"],       False),
        FusionConfig("C11", "AND", "mean3sigma", FROZEN_TAU["mean3sigma"], True),
    ]

    assert len(configs) == 11, f"Expected 11 configs, got {len(configs)}"
    assert configs[0].config_id == "C01"
    return configs


# ---------------------------------------------------------------------------
# Inference Adapters (STEP 1)
# ---------------------------------------------------------------------------

class SupervisedAdapter:
    """
    Frozen EXP_OOF_STACK_V1 seed-42 meta-learner inference adapter.

    OD-1a: LR.predict() at frozen 0.5 boundary.
    Produces binary predictions: 0 (Normal) or 1 (Attack).

    Meta-feature columns (fixed):
        dt_attack_probability, rf_attack_probability,
        svm_decision_score, nn_attack_probability
    """

    META_COLS = [
        "dt_attack_probability",
        "rf_attack_probability",
        "svm_decision_score",
        "nn_attack_probability",
    ]

    def __init__(self) -> None:
        self._lr: "LogisticRegression | None" = None
        self._dt: "DecisionTreeClassifier | None" = None
        self._rf: "RandomForestClassifier | None" = None
        self._svm: "LinearSVC | None" = None
        self._svm_scaler: "StandardScaler | None" = None
        self._nn: "IDSNet | None" = None
        self._nn_scaler: "StandardScaler | None" = None
        self._features: list[str] | None = None

    def load(self) -> "SupervisedAdapter":
        """Load all frozen Sprint 5/6 artifacts."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
        from src.models.base_models.neural_network import IDSNet
        logger.info("Loading supervised adapter (EXP_OOF_STACK_V1 seed-42)...")

        # Sprint 6 meta-learner (seed-42)
        lr_path = CKPT_S6_SEED42 / "meta_learner.joblib"
        self._lr = joblib.load(lr_path)
        logger.info("  Loaded meta-learner: %s", lr_path)

        # Base models
        self._dt = joblib.load(CKPT_BASE / "dt/dt_final.joblib")
        self._rf = joblib.load(CKPT_BASE / "rf/rf_final.joblib")
        self._svm = joblib.load(CKPT_BASE / "svm/svm_final.joblib")
        self._svm_scaler = joblib.load(CKPT_BASE / "svm/svm_scaler.joblib")
        self._nn_scaler = joblib.load(CKPT_BASE / "nn/nn_scaler.joblib")

        # NN
        nn_arch = json.load(open(CKPT_BASE / "nn/nn_architecture.json"))
        hidden = nn_arch.get("hidden_sizes", [128, 64])
        self._nn = IDSNet(input_dim=75, hidden_sizes=hidden)
        self._nn.load_state_dict(torch.load(
            CKPT_BASE / "nn/nn_final.pt", map_location="cpu", weights_only=True
        ))
        self._nn.eval()

        # Feature list
        self._features = load_selected_features()
        assert len(self._features) == 75, "Feature count mismatch"

        logger.info("  Supervised adapter loaded. Features=%d", len(self._features))
        return self

    def get_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 4 meta-features for any split DataFrame (no retraining).
        """
        assert self._features is not None, "Call .load() before get_meta_features()"
        assert self._dt is not None and self._rf is not None
        assert self._svm is not None and self._svm_scaler is not None
        assert self._nn is not None and self._nn_scaler is not None

        X = build_feature_matrix(df, self._features)

        # DT
        dt_prob = self._dt.predict_proba(X)[:, 1]

        # RF
        rf_prob = self._rf.predict_proba(X)[:, 1]

        # SVM
        X_svm = self._svm_scaler.transform(X)
        svm_score = self._svm.decision_function(X_svm)

        # NN
        X_nn = self._nn_scaler.transform(X)
        with torch.no_grad():
            X_t = torch.tensor(X_nn, dtype=torch.float32)
            logits = self._nn(X_t)           # shape (N,) — IDSNet output
            nn_prob = torch.sigmoid(logits).numpy()

        meta = pd.DataFrame({
            "dt_attack_probability": dt_prob,
            "rf_attack_probability": rf_prob,
            "svm_decision_score":    svm_score,
            "nn_attack_probability": nn_prob,
        })
        return meta

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        OD-1a: LR.predict() at frozen 0.5 boundary.
        Returns binary array: 0=Normal, 1=Attack.
        """
        assert self._lr is not None, "Call .load() before predict()"
        meta = self.get_meta_features(df)
        X_meta = meta[self.META_COLS].to_numpy()
        preds = self._lr.predict(X_meta)        # uses sklearn default 0.5
        return preds.astype(int)


class AEAdapter:
    """
    Frozen EXP_AE_V1 Autoencoder inference adapter.

    OD-2a: RE > tau → anomaly flag = 1; RE <= tau → Normal flag = 0.
    RE = mean((x_scaled - AE(x_scaled))^2) over 75 features per row.
    """

    def __init__(self) -> None:
        self._model: Autoencoder | None = None
        self._scaler: "StandardScaler | None" = None
        self._features: list[str] | None = None

    def load(self) -> "AEAdapter":
        """Load frozen AE weights and AE-fit scaler."""
        logger.info("Loading AE adapter (EXP_AE_V1)...")

        self._scaler = joblib.load(CKPT_AE / "ae_scaler.joblib")

        ae_arch = json.load(open(CKPT_AE / "ae_architecture.json"))
        # Autoencoder has fixed architecture (75→12→6→12→75); input_dim only
        self._model = Autoencoder(input_dim=ae_arch.get("input_dim", 75))
        self._model.load_state_dict(torch.load(
            CKPT_AE / "ae_final.pt", map_location="cpu", weights_only=True
        ))
        self._model.eval()

        self._features = load_selected_features()
        assert len(self._features) == 75

        logger.info("  AE adapter loaded.")
        return self

    def reconstruction_errors(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute RE = mean((x_scaled - x_hat)^2) per row. Shape: (n,)
        Uses AE-fit Normal-TRAIN scaler (NOT full-TRAIN scaler).
        """
        assert self._features is not None, "Call .load() before reconstruction_errors()"
        assert self._scaler is not None and self._model is not None

        X = build_feature_matrix(df, self._features)
        X_scaled = self._scaler.transform(X).astype(np.float32)

        batch_size = 1024
        res = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), batch_size):
                x_t = torch.tensor(X_scaled[i:i+batch_size])
                x_hat = self._model(x_t)
                re = ((x_t - x_hat) ** 2).mean(dim=1).cpu().numpy()
                res.append(re)
        return np.concatenate(res)

    def ae_flag(self, re: np.ndarray, tau: float) -> np.ndarray:
        """OD-2a: RE > tau → 1 (anomaly); RE <= tau → 0 (Normal)."""
        return (re > tau).astype(int)


# ---------------------------------------------------------------------------
# Fusion prediction
# ---------------------------------------------------------------------------

def fuse(sup_pred: np.ndarray, ae_flag: np.ndarray, rule: str) -> np.ndarray:
    """
    Apply fusion rule to binary supervised and AE predictions.
    rule: "supervised_only" | "OR" | "AND"
    """
    if rule == "supervised_only":
        return sup_pred.copy()
    elif rule == "OR":
        return np.logical_or(sup_pred, ae_flag).astype(int)
    elif rule == "AND":
        return np.logical_and(sup_pred, ae_flag).astype(int)
    else:
        raise ValueError(f"Unknown rule: {rule}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

from sklearn.metrics import (
    f1_score, balanced_accuracy_score, confusion_matrix,
    recall_score,
)

def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """OD-6: Full metric set for Development TEST."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[1,1])
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "macro_f1":        round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1":     round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "balanced_acc":    round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "fpr":             round(float(fpr), 6),
        "recall":          round(float(recall), 6),
        "fnr":             round(float(fnr), 6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

def compute_fpr_only(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:
    """Validation FPR (Normal-only). Returns (fpr, fp_count)."""
    assert (y_true == 0).all(), "Validation must be Normal-only (all label=0)"
    fp_count = int((y_pred == 1).sum())
    n = len(y_true)
    return round(fp_count / n, 6), fp_count

def compute_prot_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """OD-7: Protected Backdoor metrics."""
    detected = int(((y_pred == 1) & (y_true == 1)).sum())
    missed = PROT_N - detected
    rate = round(detected / PROT_N, 6)
    return {
        "detected_count": detected,
        "missed_count":   missed,
        "detection_rate": rate,
        "n_prot":         PROT_N,
        "pp_per_row":     PROT_PP_PER_ROW,
        "caveat": f"1 row = 1/{PROT_N} = {PROT_PP_PER_ROW:.4f} pp; "
                  "small differences must not be over-interpreted.",
    }


# ---------------------------------------------------------------------------
# Selection Function (STEP 6)  — OD-4 Option A + OD-4b conservative-first
# ---------------------------------------------------------------------------

def run_selection(
    configs: list[FusionConfig],
    fpr_values: dict[str, float],
    gate: float = FPR_GATE,
) -> dict:
    """
    Apply canonical selection function.

    Returns dict with selection provenance for validation_selection.json.
    """
    # STEP 1 — Gate
    passing = [c for c in configs if fpr_values[c.config_id] <= gate]

    if not passing:
        # OD-5: fallback
        c01 = next(c for c in configs if c.config_id == "C01")
        return {
            "n_candidates": 11,
            "fpr_gate": gate,
            "n_passing": 0,
            "passing_configs": [],
            "fpr_values": fpr_values,
            "selected_config": "C01",
            "selection_rule": "OD-5 fallback (no config passed gate)",
            "selection_option": "A+OD-4b",
            "outlier_influenced": False,
            "fallback_triggered": True,
            "only_baseline_passed": False,
            "baseline_config": "C01",
            "baseline_fpr": fpr_values["C01"],
        }

    # STEP 2+3 — Rule priority: OR > AND > Supervised-only
    # Within rule: OD-4b conservative-first (largest tau first)
    selected = None
    for rule in ["OR", "AND", "supervised_only"]:
        rule_passing = [c for c in passing if c.rule == rule]
        if rule_passing:
            # Sort by tau descending (conservative-first = largest tau first)
            # None tau (C01) handled by rule order — "supervised_only" is last
            rule_passing_sorted = sorted(
                rule_passing,
                key=lambda c: c.tau if c.tau is not None else -1,
                reverse=True,
            )
            selected = rule_passing_sorted[0]
            break

    if selected is None:
        # Safety: should not be reachable if passing is non-empty
        selected = next(c for c in configs if c.config_id == "C01")

    only_baseline = (len(passing) == 1 and selected.config_id == "C01")

    return {
        "n_candidates": 11,
        "fpr_gate": gate,
        "n_passing": len(passing),
        "passing_configs": [c.config_id for c in passing],
        "fpr_values": {k: round(v, 6) for k, v in fpr_values.items()},
        "selected_config": selected.config_id,
        "selected_rule": selected.rule,
        "selected_threshold": selected.threshold_key,
        "selected_tau": selected.tau,
        "selection_rule": "OD-4 Option A + OD-4b conservative-first",
        "within_rule_priority": "conservative-first (largest tau)",
        "selection_option": "A+OD-4b",
        "outlier_influenced": selected.outlier_influenced,
        "fallback_triggered": False,
        "only_baseline_passed": only_baseline,
        "baseline_config": "C01",
        "baseline_fpr": fpr_values["C01"],
        "h_fusion_hypothesis": "H-FUSION",
        "h_prot_backdoor_hypothesis": "H-PROT-BACKDOOR",
    }


# ---------------------------------------------------------------------------
# RST/FIN Subgroup  (OD-8)
# ---------------------------------------------------------------------------

def get_rstfin_subgroup_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Identify the Sprint 7 RST/FIN Normal subgroup in VALIDATION.
    Matches on encoded state columns (OHE from preprocessing pipeline).
    Falls back to RE-based heuristic if OHE columns absent.
    """
    state_cols_rst = [c for c in df.columns if "state_RST" in c or "state_FIN" in c]
    if state_cols_rst:
        mask = (df[state_cols_rst].sum(axis=1) > 0).to_numpy()
    else:
        # Fallback: rows where all OHE state columns indicate RST/FIN pattern
        # This is a conservative heuristic
        logger.warning("RST/FIN OHE columns not found; subgroup mask may be empty")
        mask = np.zeros(len(df), dtype=bool)
    return mask


def compute_subgroup_fpr(y_pred: np.ndarray, mask: np.ndarray) -> dict:
    """Compute subgroup FPR for RST/FIN Normal rows."""
    sub_n = int(mask.sum())
    if sub_n == 0:
        return {"subgroup_n": 0, "subgroup_fp": 0, "subgroup_fpr": None}
    sub_fp = int(y_pred[mask].sum())   # all are Normal; any pred=1 is FP
    return {
        "subgroup_n":   sub_n,
        "subgroup_fp":  sub_fp,
        "subgroup_fpr": round(sub_fp / sub_n, 6),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    logger.info("=" * 70)
    logger.info("Sprint 8 — EXP_FUSION_V1 — %s", ts)
    logger.info("=" * 70)

    # ── Create output directories ───────────────────────────────────────
    for sub in ["validation", "development_test", "protected_backdoor",
                "comparison", "exploratory"]:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Load inference adapters ────────────────────────────────
    logger.info("\nSTEP 1: Loading inference adapters...")
    sup = SupervisedAdapter().load()
    ae  = AEAdapter().load()

    # ── STEP 4: Enumerate 11 configurations ────────────────────────────
    logger.info("\nSTEP 4: Enumerating 11 candidate configurations...")
    configs = build_candidate_configs()
    config_map = {c.config_id: c for c in configs}
    assert len(configs) == 11
    logger.info("  Confirmed 11 configurations: %s", [c.config_id for c in configs])

    # ── STEP 5: Compute Validation FPR for all 11 configs ──────────────
    logger.info("\nSTEP 1b: Encoding all data splits via PreprocessingPipeline...")
    splits = load_and_encode_splits([
        "validation", "development_test", "protected_unseen_attack"
    ])
    val_df_enc   = splits["validation"]
    test_df_enc  = splits["development_test"]
    prot_df_enc  = splits["protected_unseen_attack"]

    # Load raw validation CSV for RST/FIN subgroup detection (raw 'state' col)
    val_raw = pd.read_csv(DATA_DIR / "validation.csv")

    logger.info("\nSTEP 5: Computing Validation FPR (all 11 configs)...")
    assert (val_df_enc["label"] == 0).all(), "VALIDATION must be Normal-only"
    assert len(val_df_enc) == 11200, f"Expected 11200 rows, got {len(val_df_enc)}"
    y_val = val_df_enc["label"].to_numpy()

    # Supervised predictions on VALIDATION (encoded)
    sup_val = sup.predict(val_df_enc)

    # AE reconstruction errors on VALIDATION (encoded)
    ae_re_val = ae.reconstruction_errors(val_df_enc)
    logger.info("  AE RE on VALIDATION: mean=%.4f std=%.4f max=%.4f",
                ae_re_val.mean(), ae_re_val.std(), ae_re_val.max())

    # RST/FIN subgroup mask — use raw 'state' column from raw CSV
    raw_state_val = val_raw["state"] if "state" in val_raw.columns else pd.Series(dtype=str)
    rstfin_mask_val = raw_state_val.isin(["RST", "FIN"]).to_numpy()
    logger.info("  RST/FIN subgroup size: %d rows", rstfin_mask_val.sum())


    # Compute FPR + subgroup FPR for all 11 configs
    fpr_values: dict[str, float] = {}
    fp_counts:  dict[str, int] = {}
    subgroup_results: dict[str, dict] = {}
    all_val_preds: dict[str, np.ndarray] = {}

    rows = []
    for cfg in configs:
        if cfg.rule == "supervised_only":
            y_pred = fuse(sup_val, np.zeros_like(sup_val), "supervised_only")
        else:
            assert cfg.tau is not None, f"{cfg.config_id} has no tau but rule={cfg.rule}"
            ae_flag_vals = ae.ae_flag(ae_re_val, cfg.tau)
            y_pred = fuse(sup_val, ae_flag_vals, cfg.rule)

        fpr, fp_cnt = compute_fpr_only(y_val, y_pred)
        sub_res = compute_subgroup_fpr(y_pred, rstfin_mask_val)

        fpr_values[cfg.config_id] = fpr
        fp_counts[cfg.config_id] = fp_cnt
        subgroup_results[cfg.config_id] = sub_res
        all_val_preds[cfg.config_id] = y_pred

        rows.append({
            "config_id":        cfg.config_id,
            "rule":             cfg.rule,
            "threshold":        cfg.threshold_key or "—",
            "tau":              cfg.tau,
            "val_fpr":          fpr,
            "val_fp_count":     fp_cnt,
            "passes_gate":      fpr <= FPR_GATE,
            "outlier_influenced": cfg.outlier_influenced,
            "subgroup_n":       sub_res["subgroup_n"],
            "subgroup_fp":      sub_res["subgroup_fp"],
            "subgroup_fpr":     sub_res["subgroup_fpr"],
        })
        logger.info("  %s  rule=%-16s  tau=%-10s  FPR=%.4f  (%s)  sub_fpr=%s",
                    cfg.config_id, cfg.rule,
                    f"{cfg.tau:.4f}" if cfg.tau else "—",
                    fpr,
                    "PASS" if fpr <= FPR_GATE else "FAIL",
                    sub_res.get("subgroup_fpr"))

    # Save fusion_candidate_results.csv
    cand_df = pd.DataFrame(rows)
    cand_df.to_csv(OUTPUT_DIR / "validation/fusion_candidate_results.csv", index=False)
    logger.info("  Saved fusion_candidate_results.csv")

    # ── STEP 6: Selection function ─────────────────────────────────────
    logger.info("\nSTEP 6: Applying canonical selection function...")
    sel = run_selection(configs, fpr_values)
    logger.info("  Selected: %s  (fallback=%s, outlier_influenced=%s)",
                sel["selected_config"], sel["fallback_triggered"],
                sel["outlier_influenced"])
    logger.info("  Passing configs (%d/%d): %s",
                sel["n_passing"], 11, sel["passing_configs"])

    # ── STEP 7: Freeze ONE configuration ───────────────────────────────
    logger.info("\nSTEP 7: Freezing selected configuration...")
    sel["frozen_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    sel["experiment_id"] = SPRINT8_EXPERIMENT_ID

    with open(OUTPUT_DIR / "validation/validation_selection.json", "w") as f:
        json.dump(sel, f, indent=2)
    logger.info("  Written: validation_selection.json")

    selected_cfg = config_map[sel["selected_config"]]
    logger.info("  Frozen config: %s (%s, tau=%s)",
                selected_cfg.config_id, selected_cfg.rule, selected_cfg.tau)

    # ── STEP 8: Development TEST (single-shot) ─────────────────────────
    logger.info("\nSTEP 8: Running Development TEST (single-shot, frozen config)...")
    y_test = test_df_enc["label"].to_numpy()
    logger.info("  Dev TEST rows=%d, attack=%d, normal=%d",
                len(test_df_enc), (y_test == 1).sum(), (y_test == 0).sum())

    # Supervised predictions (encoded)
    sup_test = sup.predict(test_df_enc)
    # AE reconstruction errors (encoded)
    ae_re_test = ae.reconstruction_errors(test_df_enc)

    def run_config_on_test(cfg: FusionConfig) -> np.ndarray:
        if cfg.rule == "supervised_only":
            return fuse(sup_test, np.zeros_like(sup_test), "supervised_only")
        assert cfg.tau is not None, f"{cfg.config_id} has no tau but rule={cfg.rule}"
        ae_f = ae.ae_flag(ae_re_test, cfg.tau)
        return fuse(sup_test, ae_f, cfg.rule)

    # Primary: selected config
    y_pred_sel = run_config_on_test(selected_cfg)
    metrics_sel = compute_full_metrics(y_test, y_pred_sel)

    # Reference: C01
    c01_cfg = config_map["C01"]
    y_pred_c01 = run_config_on_test(c01_cfg)
    metrics_c01 = compute_full_metrics(y_test, y_pred_c01)

    # Save primary results
    test_preds = pd.DataFrame({
        "row_id": test_df_enc.get("row_id", pd.RangeIndex(len(test_df_enc))),
        "label":  y_test,
        "pred":   y_pred_sel,
        "c01_pred": y_pred_c01,
    })
    test_preds.to_csv(OUTPUT_DIR / "development_test/predictions.csv", index=False)

    test_metrics_out = {
        "selected_config": selected_cfg.config_id,
        "metrics": metrics_sel,
        "baseline_c01_metrics": metrics_c01,
    }
    with open(OUTPUT_DIR / "development_test/metrics.json", "w") as f:
        json.dump(test_metrics_out, f, indent=2)

    cm_out = {"config_id": selected_cfg.config_id, **metrics_sel["confusion_matrix"]}
    with open(OUTPUT_DIR / "development_test/confusion_matrix.json", "w") as f:
        json.dump(cm_out, f, indent=2)

    logger.info("  Dev TEST %s: Macro-F1=%.4f  FPR=%.4f  Recall=%.4f",
                selected_cfg.config_id,
                metrics_sel["macro_f1"], metrics_sel["fpr"], metrics_sel["recall"])
    logger.info("  Dev TEST C01:          Macro-F1=%.4f  FPR=%.4f  Recall=%.4f",
                metrics_c01["macro_f1"], metrics_c01["fpr"], metrics_c01["recall"])

    # ── STEP 9: Protected Backdoor (single-shot) ────────────────────────
    logger.info("\nSTEP 9: Running Protected Backdoor evaluation (single-shot)...")
    y_prot = prot_df_enc["label"].to_numpy()
    assert len(prot_df_enc) == PROT_N, f"Expected {PROT_N} rows, got {len(prot_df_enc)}"
    logger.info("  Protected Backdoor rows=%d", len(prot_df_enc))

    sup_prot = sup.predict(prot_df_enc)
    ae_re_prot = ae.reconstruction_errors(prot_df_enc)

    def run_config_on_prot(cfg: FusionConfig) -> np.ndarray:
        if cfg.rule == "supervised_only":
            return fuse(sup_prot, np.zeros_like(sup_prot), "supervised_only")
        assert cfg.tau is not None, f"{cfg.config_id} has no tau but rule={cfg.rule}"
        ae_f = ae.ae_flag(ae_re_prot, cfg.tau)
        return fuse(sup_prot, ae_f, cfg.rule)

    y_pred_prot_sel = run_config_on_prot(selected_cfg)
    prot_metrics_sel = compute_prot_metrics(y_prot, y_pred_prot_sel)

    y_pred_prot_c01 = run_config_on_prot(c01_cfg)
    prot_metrics_c01 = compute_prot_metrics(y_prot, y_pred_prot_c01)

    prot_preds = pd.DataFrame({
        "row_id": prot_df_enc.get("row_id", pd.RangeIndex(len(prot_df_enc))),
        "label":  y_prot,
        "pred":   y_pred_prot_sel,
        "c01_pred": y_pred_prot_c01,
    })
    prot_preds.to_csv(OUTPUT_DIR / "protected_backdoor/predictions.csv", index=False)

    prot_metrics_out = {
        "selected_config": selected_cfg.config_id,
        "metrics": prot_metrics_sel,
        "baseline_c01_metrics": prot_metrics_c01,
    }
    with open(OUTPUT_DIR / "protected_backdoor/metrics.json", "w") as f:
        json.dump(prot_metrics_out, f, indent=2)
    with open(OUTPUT_DIR / "protected_backdoor/confusion_matrix.json", "w") as f:
        json.dump({"config_id": selected_cfg.config_id,
                   "detected": prot_metrics_sel["detected_count"],
                   "missed":   prot_metrics_sel["missed_count"]}, f, indent=2)

    logger.info("  Protected Backdoor %s: detected=%d/%d (%.2f%%)",
                selected_cfg.config_id,
                prot_metrics_sel["detected_count"], PROT_N,
                prot_metrics_sel["detection_rate"] * 100)
    logger.info("  Protected Backdoor C01:   detected=%d/%d (%.2f%%)",
                prot_metrics_c01["detected_count"], PROT_N,
                prot_metrics_c01["detection_rate"] * 100)
    logger.info("  Caveat: %s", prot_metrics_sel["caveat"])

    # ── STEP 10: Comparison analyses (primary only) ─────────────────────
    logger.info("\nSTEP 10: Generating comparison analyses (primary only)...")

    # supervised_vs_fusion.csv
    sup_vs_fusion = pd.DataFrame([
        {"config": "C01",                 "split": "development_test", **metrics_c01},
        {"config": selected_cfg.config_id, "split": "development_test", **metrics_sel},
    ])
    sup_vs_fusion.drop(columns=["confusion_matrix"], errors="ignore", inplace=True)
    sup_vs_fusion.to_csv(OUTPUT_DIR / "comparison/supervised_vs_fusion.csv", index=False)

    # rf_vs_stack_vs_fusion.csv — cross-sprint macro-F1 comparison
    # Sprint 5 RF OOF and Sprint 6 stack OOF from h1_summary.json
    h1_path = ROOT / "results/stacking/EXP_OOF_STACK_V1/h1_summary.json"
    h1 = json.load(open(h1_path)) if h1_path.exists() else {}
    stack_macro_f1 = h1.get("macro_f1", {}).get("mean", None)

    # Sprint 5 RF CV ref (from stacking metadata)
    s5_rf_macro_f1 = None
    for seed_dir in (ROOT / "results/stacking/EXP_OOF_STACK_V1").iterdir():
        if seed_dir.is_dir() and "seed" in seed_dir.name:
            mpath = seed_dir / "metrics.json"
            if mpath.exists():
                m = json.load(open(mpath))
                s5_rf_macro_f1 = m.get("rf_oof_macro_f1")
                break

    xsprint = pd.DataFrame([
        {"sprint": 5, "method": "RF OOF (seed-42)",
         "macro_f1": s5_rf_macro_f1, "note": "Sprint 5 frozen CV reference"},
        {"sprint": 6, "method": "OOF Stack (3-seed mean)",
         "macro_f1": stack_macro_f1,  "note": "Sprint 6 H1 in-sample OOF"},
        {"sprint": 8, "method": f"Fusion {selected_cfg.config_id}",
         "macro_f1": metrics_sel["macro_f1"], "note": "Sprint 8 held-out Dev TEST"},
        {"sprint": 8, "method": "Supervised-only (C01)",
         "macro_f1": metrics_c01["macro_f1"], "note": "Sprint 8 held-out Dev TEST"},
    ])
    xsprint.to_csv(OUTPUT_DIR / "comparison/rf_vs_stack_vs_fusion.csv", index=False)
    logger.info("  Saved comparison CSVs.")

    # ── STEP 11: Exploratory all-11 (INFORMATIONAL ONLY) ───────────────
    logger.info("\nSTEP 11: Exploratory all-11 analysis (INFORMATIONAL ONLY)...")

    # Dev TEST for all 11 configs
    exp_test_rows = []
    exp_prot_rows = []
    for cfg in configs:
        y_pred_t = run_config_on_test(cfg)
        m_t = compute_full_metrics(y_test, y_pred_t)
        exp_test_rows.append({
            "config_id": cfg.config_id,
            "rule": cfg.rule,
            "threshold": cfg.threshold_key or "—",
            "tau": cfg.tau,
            "macro_f1": m_t["macro_f1"],
            "weighted_f1": m_t["weighted_f1"],
            "balanced_acc": m_t["balanced_acc"],
            "fpr": m_t["fpr"],
            "recall": m_t["recall"],
            "fnr": m_t["fnr"],
            "tp": m_t["tp"], "fp": m_t["fp"], "tn": m_t["tn"], "fn": m_t["fn"],
            "is_primary_selected": cfg.config_id == selected_cfg.config_id,
            "outlier_influenced": cfg.outlier_influenced,
            "informational_only": True,
        })

        y_pred_p = run_config_on_prot(cfg)
        m_p = compute_prot_metrics(y_prot, y_pred_p)
        exp_prot_rows.append({
            "config_id": cfg.config_id,
            "rule": cfg.rule,
            "threshold": cfg.threshold_key or "—",
            "tau": cfg.tau,
            "detected_count": m_p["detected_count"],
            "missed_count": m_p["missed_count"],
            "detection_rate": m_p["detection_rate"],
            "n_prot": PROT_N,
            "pp_per_row": PROT_PP_PER_ROW,
            "is_primary_selected": cfg.config_id == selected_cfg.config_id,
            "informational_only": True,
        })

    pd.DataFrame(exp_test_rows).to_csv(
        OUTPUT_DIR / "exploratory/all_11_development_test_metrics.csv", index=False)
    pd.DataFrame(exp_prot_rows).to_csv(
        OUTPUT_DIR / "exploratory/all_11_protected_backdoor_metrics.csv", index=False)
    logger.info("  Saved exploratory all-11 CSVs (INFORMATIONAL ONLY).")

    # ── STEP 12: Write metadata.json ────────────────────────────────────
    metadata = {
        "experiment_id": SPRINT8_EXPERIMENT_ID,
        "sprint": 8,
        "status": "IMPLEMENTED",
        "upstream_experiments": [
            "EXP_MI_V1_1", "EXP_BASE_MODELS_V1", "EXP_OOF_STACK_V1", "EXP_AE_V1"
        ],
        "od_1": "OD-1a — LR.predict() at frozen 0.5 boundary, EXP_OOF_STACK_V1 seed-42 checkpoint only",
        "od_2": "OD-2a — RE > tau (strict greater-than)",
        "od_3": 0.05,
        "od_4": "Option A — gate-only, OR > AND > Supervised-only",
        "od_4_sub": "OD-4b — conservative-first (largest tau)",
        "od_5": "C01 fallback",
        "od_6": "Macro-F1 primary",
        "od_7": "counts + rate + n=583 caveat",
        "od_8": "RST/FIN subgroup FPR per config",
        "od_9": "11 unique configurations",
        "od_10": "H-FUSION / H-PROT-BACKDOOR",
        "s6_canonical_checkpoint": "EXP_OOF_STACK_V1 / seed=42 / frozen LR meta-learner",
        "s6_no_seed_averaging": True,
        "n_candidates": 11,
        "selected_config": sel["selected_config"],
        "fpr_gate": FPR_GATE,
        "development_test_runs": 1,
        "protected_backdoor_runs": 1,
        "protected_backdoor_n": PROT_N,
        "protected_backdoor_caveat": f"1 row = 1/{PROT_N} = {PROT_PP_PER_ROW:.4f} pp",
        "exploratory_all_11_computed": True,
        "exploratory_note": "INFORMATIONAL ONLY — post-hoc, not used for selection or verdicts",
        "validation_reuse_limitation": (
            "VALIDATION is reused for Sprint 7 AE threshold calibration AND "
            "Sprint 8 fusion-rule selection. Both are selection-stage uses, not "
            "final held-out evaluation."
        ),
        "scaler_space_limitation": (
            "AE operates in a Normal-TRAIN-scaled feature space distinct from "
            "the full-TRAIN-scaled space used by DT/RF/SVM/NN."
        ),
        "outlier_note": "row_id 10737 (RE~269.09) and 10731 (RE~269.03) are legitimate Normal rows (RST/FIN TCP). Not filtered.",
        "frozen_at_utc": ts,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("  Saved metadata.json")

    # ── Final summary ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SPRINT 8 IMPLEMENTATION COMPLETE")
    logger.info("=" * 70)
    logger.info("  Selected config:    %s (%s, tau=%s)",
                sel["selected_config"], selected_cfg.rule, selected_cfg.tau)
    logger.info("  Passing configs:    %d / 11", sel["n_passing"])
    logger.info("  Outlier-influenced: %s", sel["outlier_influenced"])
    logger.info("  Fallback triggered: %s", sel["fallback_triggered"])
    logger.info("")
    logger.info("  Dev TEST %s:  Macro-F1=%.4f | Recall=%.4f | FPR=%.4f",
                sel["selected_config"],
                metrics_sel["macro_f1"], metrics_sel["recall"], metrics_sel["fpr"])
    logger.info("  Dev TEST C01:        Macro-F1=%.4f | Recall=%.4f | FPR=%.4f",
                metrics_c01["macro_f1"], metrics_c01["recall"], metrics_c01["fpr"])
    logger.info("")
    logger.info("  Prot Backdoor %s: %d/%d detected (%.2f%%)",
                sel["selected_config"],
                prot_metrics_sel["detected_count"], PROT_N,
                prot_metrics_sel["detection_rate"] * 100)
    logger.info("  Prot Backdoor C01:   %d/%d detected (%.2f%%)",
                prot_metrics_c01["detected_count"], PROT_N,
                prot_metrics_c01["detection_rate"] * 100)
    logger.info("")
    logger.info("  Output dir: %s", OUTPUT_DIR)
    logger.info("  STEP 13: Awaiting human review before freeze commit/tag.")
    logger.info("=" * 70)

    return sel, metrics_sel, prot_metrics_sel


if __name__ == "__main__":
    main()
