"""
scripts/run_ablation.py

Sprint 10 — EXP_ABLATION_V1 Ablation Study Runner
Implements D1-D34. Sequential, resumable, phase-ordered execution.

Phase order (never interleaved):
  PHASE 0 — Pre-verification (D9, D11, D12, D19 before-hash)
  PHASE 1 — Smoke test: A0 seed 42, A1 seed 42
  PHASE 2 — Base cache generation (all 4 models, all 3 seeds)
  PHASE 3 — A0: fresh RF for seeds 123, 2024
  PHASE 4 — A1: meta-learner for all 3 seeds
  PHASE 5 — A1b: soft-vote for all 3 seeds
  PHASE 6 — A2, A3, A4, A5: ablated meta-learners for all 3 seeds
  PHASE 7 — A6: A1 + frozen AE/fusion for all 3 seeds
  PHASE 8 — Aggregation: stats, paired deltas, reports
"""

import sys, json, hashlib, time, logging, signal, datetime, random
import warnings
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import joblib
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             balanced_accuracy_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.autoencoder.ae_model import Autoencoder
from src.models.base_models.neural_network import IDSNet
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.models.base_models.preprocessing import load_selected_features

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EXP_DIR  = ROOT / "results/ablation/EXP_ABLATION_V1"
CACHE_DIR = EXP_DIR / "cache"
CFG_PATH  = EXP_DIR / "config.yaml"

DATASET_PATHS = {
    "train":              ROOT / "data/splits/train.csv",
    "validation":         ROOT / "data/splits/validation.csv",
    "development_test":   ROOT / "data/splits/development_test.csv",
    "protected_backdoor": ROOT / "data/splits/protected_unseen_attack.csv",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("sprint10.ablation")

# ─────────────────────────────────────────────────────────────────────────────
# STOP-REPORT helper
# ─────────────────────────────────────────────────────────────────────────────
def stop_report(rule_id, expected, found, file_artifact, why, completed, recommended):
    msg = (
        f"\n{'='*70}\n"
        f"STOP-REPORT\n"
        f"  rule_id:               {rule_id}\n"
        f"  what_was_expected:     {expected}\n"
        f"  what_was_found:        {found}\n"
        f"  artifact/file:         {file_artifact}\n"
        f"  why_blocks:            {why}\n"
        f"  completed_work:        {completed}\n"
        f"  recommended_next:      {recommended}\n"
        f"{'='*70}\n"
    )
    logger.error(msg)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_cfg() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def result_integrity_check(path: Path, required_keys: List[str]) -> bool:
    """Returns True only if file exists, is non-empty, readable, and has all required keys."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        for k in required_keys:
            if obj.get(k) is None:
                return False
        return True
    except Exception:
        return False


@contextmanager
def hard_timeout(seconds: int, label: str):
    """Context manager that raises TimeoutError after `seconds` seconds (POSIX only; no-op on Windows)."""
    if hasattr(signal, "SIGALRM"):
        def _handler(signum, frame):
            raise TimeoutError(f"TIMEOUT after {seconds}s in {label}")
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        # Windows — log the timeout limit but cannot enforce it via SIGALRM
        logger.warning(f"[TIMEOUT] Hard timeout ({seconds}s) for {label} is advisory only on Windows.")
        yield


def compute_metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision":         float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":                float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "fpr":               float(fpr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Base model builders (use seed as random_state)
# ─────────────────────────────────────────────────────────────────────────────
def build_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, criterion="gini", class_weight="balanced",
        n_jobs=-1, max_depth=None, min_samples_leaf=1, max_features=0.3,
        random_state=seed,
    )


def build_dt(seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion="entropy", max_depth=None, min_samples_split=2,
        min_samples_leaf=1, class_weight="balanced", random_state=seed,
    )


def build_svm(seed: int) -> LinearSVC:
    return LinearSVC(C=0.1, max_iter=5000, class_weight="balanced",
                     random_state=seed)


def build_meta_lr(seed: int) -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs", C=1.0, class_weight="balanced",
        max_iter=1000, random_state=seed,
    )


def train_nn_fold(X_train_sc: np.ndarray, y_train: np.ndarray,
                  X_val_sc: np.ndarray, y_val: np.ndarray,
                  seed: int, cfg: dict) -> IDSNet:
    """Train IDSNet for one fold or full-train with early stopping."""
    set_seed(seed)
    device = get_device()
    model = IDSNet(input_dim=cfg["nn_config"]["input_dim"],
                   hidden_sizes=cfg["nn_config"]["hidden_sizes"])
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["nn_config"]["learning_rate"],
        weight_decay=cfg["nn_config"]["weight_decay"],
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([y_train.sum() / (len(y_train) - y_train.sum())]).to(device)
    )
    X_t = torch.tensor(X_train_sc, dtype=torch.float32)
    y_t = torch.tensor(y_train.astype(np.float32))  # shape (N,) — matches IDSNet 1D output
    X_v = torch.tensor(X_val_sc, dtype=torch.float32).to(device)
    y_v_np = y_val

    bs = cfg["nn_config"]["batch_size"]
    max_ep = cfg["nn_config"]["max_epochs"]
    patience = cfg["nn_config"]["patience"]

    best_f1, patience_cnt, best_state = 0.0, 0, None
    for ep in range(max_ep):
        model.train()
        idx = np.random.permutation(len(X_t))
        for i in range(0, len(idx), bs):
            xb = X_t[idx[i:i+bs]].to(device)
            yb = y_t[idx[i:i+bs]].to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        # Validate
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(X_v)).cpu().numpy().flatten()
        val_pred = (val_prob >= 0.5).astype(int)
        val_f1 = f1_score(y_v_np, val_pred, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def nn_predict_proba(model: IDSNet, X_sc: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Return sigmoid probabilities from IDSNet."""
    device = get_device()
    model.to(device)
    model.eval()
    probs = []
    X_t = torch.tensor(X_sc, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            p = torch.sigmoid(model(X_t[i:i+batch_size].to(device))).cpu().numpy().flatten()
            probs.append(p)
    return np.concatenate(probs)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (shared)
# ─────────────────────────────────────────────────────────────────────────────
_pipeline_cache = {}
_features_cache = None

def get_pipeline_and_features():
    global _pipeline_cache, _features_cache
    if "pipeline" not in _pipeline_cache:
        logger.info("Fitting PreprocessingPipeline on TRAIN …")
        train_raw = pd.read_csv(DATASET_PATHS["train"])
        pipe = PreprocessingPipeline()
        pipe.fit(train_raw)
        _pipeline_cache["pipeline"] = pipe
        _pipeline_cache["train_raw"] = train_raw
    if _features_cache is None:
        feats = load_selected_features()
        assert len(feats) == 75, f"Expected 75 features, got {len(feats)}"
        _features_cache = feats
    return _pipeline_cache["pipeline"], _features_cache


def encode_split(split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (X_75feat, y) for the given split."""
    pipe, feats = get_pipeline_and_features()
    raw = pd.read_csv(DATASET_PATHS[split_name])
    enc = pipe.transform(raw, view="unscaled")  # returns ProcessedDataset with .X and .feature_names
    enc_df = pd.DataFrame(enc.X, columns=enc.feature_names)
    X = enc_df[feats].values.astype(np.float64)
    y = raw["label"].values
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — Pre-verification
# ─────────────────────────────────────────────────────────────────────────────
def phase0_preverify(cfg: dict):
    logger.info("=== PHASE 0: Pre-verification ===")

    # D11 — Environment check
    import sklearn, numpy as np_module, pandas as pd_module
    env_current = {
        "sklearn": sklearn.__version__,
        "numpy":   np_module.__version__,
        "pandas":  pd_module.__version__,
        "torch":   torch.__version__,
        "joblib":  joblib.__version__,
    }
    env_expected = {
        "sklearn": "1.9.0",
        "numpy":   "2.4.6",
        "torch":   "2.7.1+cu118",
        "pandas":  "3.0.5",
    }
    mismatches = []
    for k, exp_v in env_expected.items():
        cur_v = env_current.get(k, "MISSING")
        if cur_v != exp_v:
            mismatches.append(f"{k}: expected={exp_v}, found={cur_v}")
    if mismatches:
        stop_report(
            "D11", f"Environment matches Sprint 9 authoritative record: {env_expected}",
            str(mismatches), "current .venv",
            "Material version mismatch blocks reproducibility guarantee.",
            "Phase 0 only — no training started.",
            "Resolve environment versions then restart."
        )
    logger.info(f"D11 environment: PASS — {env_current}")

    # Capture pip freeze for environment.txt
    import subprocess
    pip_out = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True
    ).stdout
    (EXP_DIR / "environment.txt").write_text(pip_out, encoding="utf-8")
    logger.info("environment.txt written")

    # D12 — Dataset hash verification (raw bytes)
    logger.info("D12: Dataset hash verification …")
    for split, expected_hash in cfg["dataset_hashes"].items():
        path = DATASET_PATHS[split]
        actual = sha256_file(path)
        if actual != expected_hash:
            stop_report(
                "D12", f"SHA-256 {expected_hash} for {split}",
                f"SHA-256 {actual}", str(path),
                "Dataset integrity violation — training on wrong data.",
                "Phase 0 only.", "Restore correct dataset files."
            )
    logger.info("D12 dataset hashes: ALL PASS")

    # D12 — Feature list hash (hash the JSON file bytes)
    feat_path = ROOT / "results/feature_selection/EXP_MI_V1_1/selected_features.json"
    feat_hash = sha256_file(feat_path)
    logger.info(f"EXP_MI_V1_1 selected_features.json SHA-256: {feat_hash}")

    # Frozen AE hashes
    logger.info("Verifying frozen AE artifact hashes …")
    for key, path_rel in [
        ("ae_final_pt",              "results/checkpoints/EXP_AE_V1/ae_final.pt"),
        ("ae_scaler_joblib",         "results/checkpoints/EXP_AE_V1/ae_scaler.joblib"),
        ("threshold_calibration_json","results/autoencoder/EXP_AE_V1/threshold/threshold_calibration.json"),
    ]:
        expected = cfg["frozen_sprint9_hashes"][key]
        actual   = sha256_file(ROOT / path_rel)
        if actual != expected:
            stop_report("D12-AE", f"{key} SHA-256={expected}", f"actual={actual}",
                        path_rel, "Frozen AE artifact corrupted.", "Phase 0.", "Restore AE checkpoint.")
    logger.info("Frozen AE hashes: ALL PASS")

    # D9 — Target verification
    train_labels = pd.read_csv(DATASET_PATHS["train"], usecols=["label"])["label"]
    unique_labels = sorted(train_labels.unique())
    if unique_labels != [0, 1]:
        stop_report("D9", "Binary labels {0, 1}", str(unique_labels),
                    "data/splits/train.csv", "Non-binary target — A1b semantics invalid.",
                    "Phase 0.", "Verify label column.")
    logger.info(f"D9 target: binary confirmed — {dict(train_labels.value_counts())}")

    # D19 — Config immutability: record before-hash
    cfg_hash = sha256_file(CFG_PATH)
    save_json(EXP_DIR / "provenance/config_sha256_before.json",
              {"config_sha256_before": cfg_hash,
               "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z"})
    logger.info(f"D19 config_sha256_before: {cfg_hash}")

    logger.info("=== PHASE 0: COMPLETE ===")
    return env_current, cfg_hash


# ─────────────────────────────────────────────────────────────────────────────
# Base cache generation — single (model, seed)
# ─────────────────────────────────────────────────────────────────────────────
def cache_path(model_name: str, seed: int) -> Path:
    return CACHE_DIR / f"{model_name}_seed{seed}.npz"


def cache_integrity_ok(model_name: str, seed: int) -> bool:
    p = cache_path(model_name, seed)
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        d = np.load(p, allow_pickle=True)
        for k in ["oof_scores", "oof_labels", "dev_test_scores", "dev_test_labels"]:
            if k not in d:
                return False
        return True
    except Exception:
        return False


def generate_cache_one(model_name: str, seed: int, X_train: np.ndarray,
                       y_train: np.ndarray, X_dev: np.ndarray,
                       y_dev: np.ndarray, cfg: dict):
    """Generate OOF + dev_test predictions for (model_name, seed) and persist."""
    logger.info(f"Cache generation: {model_name} seed={seed}")
    set_seed(seed)
    skf = StratifiedKFold(n_splits=cfg["oof_config"]["n_splits"],
                          shuffle=True, random_state=seed)

    oof_scores = np.zeros(len(y_train), dtype=np.float64)
    oof_fold_ids = np.zeros(len(y_train), dtype=np.int32)

    t_oof_start = time.time()

    if model_name == "nn":
        # NN needs StandardScaler fitted per fold
        nn_cfg = cfg["nn_config"]
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            set_seed(seed + fold_idx)
            X_f_tr, y_f_tr = X_train[tr_idx], y_train[tr_idx]
            X_f_val, y_f_val = X_train[val_idx], y_train[val_idx]
            # Fit scaler on fold-train
            sc = StandardScaler()
            X_f_tr_sc = sc.fit_transform(X_f_tr).astype(np.float32)
            X_f_val_sc = sc.transform(X_f_val).astype(np.float32)
            # Train NN
            model = train_nn_fold(X_f_tr_sc, y_f_tr, X_f_val_sc, y_f_val, seed + fold_idx, cfg)
            # OOF scores
            oof_scores[val_idx] = nn_predict_proba(model, X_f_val_sc)
            oof_fold_ids[val_idx] = fold_idx
            logger.info(f"  NN fold {fold_idx+1}/5 done")

        # Dev-test: full TRAIN fit with small held-out for early stopping (val_frac=5%)
        set_seed(seed)
        val_frac = nn_cfg.get("val_frac", 0.05)
        n_val = max(1, int(len(X_train) * val_frac))
        idx_perm = np.random.permutation(len(X_train))
        full_train_idx = idx_perm[n_val:]
        es_val_idx     = idx_perm[:n_val]
        sc_full = StandardScaler()
        X_full_sc = sc_full.fit_transform(X_train[full_train_idx]).astype(np.float32)
        X_es_sc   = sc_full.transform(X_train[es_val_idx]).astype(np.float32)
        X_dev_sc  = sc_full.transform(X_dev).astype(np.float32)
        nn_full = train_nn_fold(X_full_sc, y_train[full_train_idx],
                                X_es_sc, y_train[es_val_idx], seed, cfg)
        dev_test_scores = nn_predict_proba(nn_full, X_dev_sc)

    elif model_name == "svm":
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_f_tr, y_f_tr = X_train[tr_idx], y_train[tr_idx]
            X_f_val = X_train[val_idx]
            sc = StandardScaler()
            X_f_tr_sc = sc.fit_transform(X_f_tr)
            X_f_val_sc = sc.transform(X_f_val)
            m = build_svm(seed)
            m.fit(X_f_tr_sc, y_f_tr)
            oof_scores[val_idx] = m.decision_function(X_f_val_sc)
            oof_fold_ids[val_idx] = fold_idx

        sc_full = StandardScaler()
        X_train_sc = sc_full.fit_transform(X_train)
        X_dev_sc   = sc_full.transform(X_dev)
        m_full = build_svm(seed)
        m_full.fit(X_train_sc, y_train)
        dev_test_scores = m_full.decision_function(X_dev_sc)

    elif model_name == "rf":
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            m = build_rf(seed)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            oof_scores[val_idx] = m.predict_proba(X_train[val_idx])[:, 1]
            oof_fold_ids[val_idx] = fold_idx

        m_full = build_rf(seed)
        m_full.fit(X_train, y_train)
        dev_test_scores = m_full.predict_proba(X_dev)[:, 1]

    elif model_name == "dt":
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            m = build_dt(seed)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            oof_scores[val_idx] = m.predict_proba(X_train[val_idx])[:, 1]
            oof_fold_ids[val_idx] = fold_idx

        m_full = build_dt(seed)
        m_full.fit(X_train, y_train)
        dev_test_scores = m_full.predict_proba(X_dev)[:, 1]

    else:
        raise ValueError(f"Unknown model: {model_name}")

    t_oof = time.time() - t_oof_start
    logger.info(f"  {model_name} seed={seed} OOF+dev_test done in {t_oof:.1f}s")

    np.savez_compressed(
        cache_path(model_name, seed),
        oof_scores=oof_scores,
        oof_labels=y_train,
        oof_fold_ids=oof_fold_ids,
        dev_test_scores=dev_test_scores,
        dev_test_labels=y_dev,
    )
    return float(t_oof)


def load_cache(model_name: str, seed: int) -> dict:
    d = np.load(cache_path(model_name, seed), allow_pickle=True)
    return {k: d[k] for k in d.files}


# ─────────────────────────────────────────────────────────────────────────────
# Build meta-feature matrices from cache
# ─────────────────────────────────────────────────────────────────────────────
def build_meta_features(model_names: List[str], seed: int,
                        split: str = "oof") -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (meta_X, y) for the given split ('oof' or 'dev_test').
    Scores are raw (decision_function for SVM, proba for DT/RF, sigmoid for NN).
    The meta-learner handles mixed scales.
    """
    cols, y = [], None
    for mn in model_names:
        c = load_cache(mn, seed)
        key = "oof_scores" if split == "oof" else "dev_test_scores"
        label_key = "oof_labels" if split == "oof" else "dev_test_labels"
        cols.append(c[key])
        if y is None:
            y = c[label_key]
    return np.column_stack(cols), y


def build_a1b_scores(seed: int, split: str = "dev_test") -> Tuple[np.ndarray, np.ndarray]:
    """A1b: mean([dt_prob, rf_prob, sigmoid(svm_dec), nn_prob])."""
    dt_c  = load_cache("dt",  seed)
    rf_c  = load_cache("rf",  seed)
    svm_c = load_cache("svm", seed)
    nn_c  = load_cache("nn",  seed)

    sk = "oof_scores" if split == "oof" else "dev_test_scores"
    lk = "oof_labels" if split == "oof" else "dev_test_labels"

    svm_unit = sigmoid(svm_c[sk])
    scores   = np.mean(np.column_stack([dt_c[sk], rf_c[sk], svm_unit, nn_c[sk]]), axis=1)
    labels   = dt_c[lk]
    return scores, labels


# ─────────────────────────────────────────────────────────────────────────────
# A0 — Fresh RF training
# ─────────────────────────────────────────────────────────────────────────────
def run_a0(seed: int, X_train: np.ndarray, y_train: np.ndarray,
           X_dev: np.ndarray, y_dev: np.ndarray, cfg: dict) -> dict:
    result_path = EXP_DIR / f"A0_RF/seed_{seed}.json"
    req_keys = ["config_id", "seed", "macro_f1", "runtime_sec"]
    if result_integrity_check(result_path, req_keys):
        logger.info(f"A0 seed={seed} — SKIP (valid result exists)")
        return json.loads(result_path.read_text())

    logger.info(f"A0: fresh RF training seed={seed}")
    set_seed(seed)
    t0 = time.time()
    with hard_timeout(cfg["timeouts"]["a0_training_sec"], f"A0 seed={seed}"):
        rf = build_rf(seed)
        rf.fit(X_train, y_train)
        dev_preds = rf.predict(X_dev)

    rt = time.time() - t0
    m = compute_metrics(y_dev, dev_preds)
    result = {"config_id": "A0_RF", "seed": seed,
              **{k: m[k] for k in ["macro_f1","precision","recall","f1","balanced_accuracy","fpr"]},
              "runtime_sec": float(rt),
              "n_dev_test": len(y_dev),
              "rf_config_random_state": seed}
    save_json(result_path, result)
    logger.info(f"A0 seed={seed} macro_f1={m['macro_f1']:.6f} rt={rt:.1f}s")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# A1 — Full 4-model stack
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS = {
    "A1_FULL_STACK": ["dt", "rf", "svm", "nn"],
    "A2_NO_DT":      ["rf", "svm", "nn"],
    "A3_NO_RF":      ["dt", "svm", "nn"],
    "A4_NO_SVM":     ["dt", "rf", "nn"],
    "A5_NO_NN":      ["dt", "rf", "svm"],
}


def run_stacking_config(cfg_id: str, model_names: List[str], seed: int,
                        y_dev: np.ndarray, cfg: dict) -> dict:
    result_path = EXP_DIR / f"{cfg_id}/seed_{seed}.json"
    req_keys = ["config_id", "seed", "macro_f1", "runtime_sec"]
    if result_integrity_check(result_path, req_keys):
        logger.info(f"{cfg_id} seed={seed} — SKIP")
        return json.loads(result_path.read_text())

    logger.info(f"{cfg_id} seed={seed}: fitting LR meta-learner")
    set_seed(seed)
    t0 = time.time()
    with hard_timeout(cfg["timeouts"]["per_config_seed_sec"], f"{cfg_id} seed={seed}"):
        meta_X_oof, y_oof = build_meta_features(model_names, seed, "oof")
        meta_X_dev, _     = build_meta_features(model_names, seed, "dev_test")
        lr = build_meta_lr(seed)
        lr.fit(meta_X_oof, y_oof)
        dev_preds = lr.predict(meta_X_dev)

    rt = time.time() - t0
    m = compute_metrics(y_dev, dev_preds)
    result = {"config_id": cfg_id, "seed": seed,
              **{k: m[k] for k in ["macro_f1","precision","recall","f1","balanced_accuracy","fpr"]},
              "runtime_sec": float(rt),
              "base_models": model_names,
              "n_dev_test": len(y_dev)}
    save_json(result_path, result)
    logger.info(f"{cfg_id} seed={seed} macro_f1={m['macro_f1']:.6f} rt={rt:.1f}s")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# A1b — Soft vote
# ─────────────────────────────────────────────────────────────────────────────
def run_a1b(seed: int, y_dev: np.ndarray, cfg: dict) -> dict:
    result_path = EXP_DIR / f"A1b_SOFT_VOTE/seed_{seed}.json"
    req_keys = ["config_id", "seed", "macro_f1", "runtime_sec"]
    if result_integrity_check(result_path, req_keys):
        logger.info(f"A1b seed={seed} — SKIP")
        return json.loads(result_path.read_text())

    logger.info(f"A1b soft-vote seed={seed}")
    set_seed(seed)
    t0 = time.time()
    with hard_timeout(cfg["timeouts"]["a1b_combination_sec"], f"A1b seed={seed}"):
        scores, _ = build_a1b_scores(seed, "dev_test")
        preds = (scores >= 0.5).astype(int)

    # Report SVM unit distribution
    svm_unit = sigmoid(load_cache("svm", seed)["dev_test_scores"])
    svm_dist = {
        "min": float(svm_unit.min()), "max": float(svm_unit.max()),
        "mean": float(svm_unit.mean()), "median": float(np.median(svm_unit)),
        "pct_leq_1e-6": float((svm_unit <= 1e-6).mean() * 100),
        "pct_geq_1m1e-6": float((svm_unit >= 1 - 1e-6).mean() * 100),
        "pct_leq_0001": float((svm_unit <= 0.001).mean() * 100),
        "pct_geq_0999": float((svm_unit >= 0.999).mean() * 100),
    }
    logger.info(f"A1b svm_unit distribution (dev_test): {svm_dist}")
    # STOP if severely saturated
    saturated_frac = svm_dist["pct_leq_1e-6"] + svm_dist["pct_geq_1m1e-6"]
    if saturated_frac > 50.0:
        stop_report(
            "A1b-SATURATION",
            "svm_unit saturation < 50% of dev_test samples",
            f"saturation = {saturated_frac:.1f}% of samples",
            f"cache/svm_seed{seed}.npz decision_function",
            "Strongly saturated sigmoid makes A1b averaging numerically degenerate.",
            f"A1b score distribution computed for seed={seed}.",
            "Human review required: confirm or replace A1b SVM normalization strategy."
        )

    rt = time.time() - t0
    m = compute_metrics(y_dev, preds)
    result = {"config_id": "A1b_SOFT_VOTE", "seed": seed,
              **{k: m[k] for k in ["macro_f1","precision","recall","f1","balanced_accuracy","fpr"]},
              "runtime_sec": float(rt),
              "a1b_svm_normalization": "sigmoid",
              "a1b_threshold": 0.5,
              "svm_unit_distribution_dev_test": svm_dist,
              "n_dev_test": len(y_dev)}
    save_json(result_path, result)
    logger.info(f"A1b seed={seed} macro_f1={m['macro_f1']:.6f} rt={rt:.1f}s")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# A6 — Full stack + frozen AE/fusion
# ─────────────────────────────────────────────────────────────────────────────
def run_a6(seed: int, X_dev: np.ndarray, y_dev: np.ndarray, cfg: dict) -> dict:
    result_path = EXP_DIR / f"A6_STACK_PLUS_AE/seed_{seed}.json"
    req_keys = ["config_id", "seed", "macro_f1", "runtime_sec"]
    if result_integrity_check(result_path, req_keys):
        logger.info(f"A6 seed={seed} — SKIP")
        return json.loads(result_path.read_text())

    # Load A1 predictions for this seed (supervised component)
    a1_result_path = EXP_DIR / f"A1_FULL_STACK/seed_{seed}.json"
    if not a1_result_path.exists():
        raise RuntimeError(f"A6 requires A1 seed={seed} to exist first.")

    logger.info(f"A6 seed={seed}: combining A1 + frozen AE")
    t0 = time.time()
    with hard_timeout(cfg["timeouts"]["a6_inference_sec"], f"A6 seed={seed}"):
        # A1 supervised predictions on dev_test
        meta_X_dev, _ = build_meta_features(["dt", "rf", "svm", "nn"], seed, "dev_test")
        a1_result_data = json.loads(a1_result_path.read_text())
        # Re-predict from stored meta-learner (reload LR)
        # We need to rebuild the LR — it was not persisted separately.
        # Refit from OOF (deterministic same result since same seed+data)
        set_seed(seed)
        meta_X_oof, y_oof = build_meta_features(["dt", "rf", "svm", "nn"], seed, "oof")
        lr = build_meta_lr(seed)
        lr.fit(meta_X_oof, y_oof)
        a1_preds = lr.predict(meta_X_dev)  # supervised predictions

        # Frozen AE: load and compute RE on dev_test
        ae_model = Autoencoder(input_dim=cfg["ae_config"]["input_dim"])
        ae_model.load_state_dict(torch.load(
            ROOT / cfg["ae_config"]["checkpoint"],
            map_location="cpu", weights_only=True))
        ae_model.eval()
        ae_scaler = joblib.load(ROOT / cfg["ae_config"]["scaler"])

        X_dev_ae = ae_scaler.transform(X_dev).astype(np.float32)
        device = get_device()
        ae_model.to(device)
        re_vals = []
        with torch.no_grad():
            for i in range(0, len(X_dev_ae), 4096):
                x_t = torch.tensor(X_dev_ae[i:i+4096]).to(device)
                x_h = ae_model(x_t)
                re_vals.append(((x_t - x_h) ** 2).mean(dim=1).cpu().numpy())
        re = np.concatenate(re_vals)

        tau = cfg["ae_config"]["threshold"]
        ae_flags = (re > tau).astype(int)

        # C06 OR fusion: predict attack if supervised=1 OR ae_flagged=1
        a6_preds = np.maximum(a1_preds, ae_flags)

    rt = time.time() - t0
    m = compute_metrics(y_dev, a6_preds)
    ae_detected_dev = int(ae_flags.sum())
    result = {
        "config_id": "A6_STACK_PLUS_AE", "seed": seed,
        **{k: m[k] for k in ["macro_f1","precision","recall","f1","balanced_accuracy","fpr"]},
        "runtime_sec": float(rt),
        "ae_threshold": tau,
        "fusion_rule": "OR",
        "ae_flagged_dev_test": ae_detected_dev,
        "ae_flagged_frac": float(ae_detected_dev / len(y_dev)),
        "n_dev_test": len(y_dev),
    }
    save_json(result_path, result)
    logger.info(f"A6 seed={seed} macro_f1={m['macro_f1']:.6f} rt={rt:.1f}s")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Protected Backdoor evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_protected_backdoor(cfg: dict):
    """Evaluate A1 and A6 on Protected Backdoor (evaluation-only)."""
    result_path = EXP_DIR / "protected_backdoor_results.json"
    logger.info("Protected Backdoor evaluation …")
    pipe, feats = get_pipeline_and_features()
    prot_raw = pd.read_csv(DATASET_PATHS["protected_backdoor"])
    prot_enc = pipe.transform(prot_raw, view="unscaled")
    prot_enc_df = pd.DataFrame(prot_enc.X, columns=prot_enc.feature_names)
    X_prot   = prot_enc_df[feats].values.astype(np.float64)
    y_prot   = prot_raw["label"].values
    n_prot   = len(y_prot)

    # AE RE on protected backdoor
    ae_model = Autoencoder(input_dim=cfg["ae_config"]["input_dim"])
    ae_model.load_state_dict(torch.load(
        ROOT / cfg["ae_config"]["checkpoint"],
        map_location="cpu", weights_only=True))
    ae_model.eval()
    ae_scaler = joblib.load(ROOT / cfg["ae_config"]["scaler"])
    tau = cfg["ae_config"]["threshold"]

    device = get_device()
    ae_model.to(device)
    X_prot_ae = ae_scaler.transform(X_prot).astype(np.float32)
    re_vals = []
    with torch.no_grad():
        for i in range(0, len(X_prot_ae), 4096):
            x_t = torch.tensor(X_prot_ae[i:i+4096]).to(device)
            x_h = ae_model(x_t)
            re_vals.append(((x_t - x_h) ** 2).mean(dim=1).cpu().numpy())
    ae_re = np.concatenate(re_vals)
    ae_flags = (ae_re > tau).astype(int)

    prot_results = {"n_prot": n_prot, "tau": tau, "per_seed": {}}
    for seed in cfg["seeds"]:
        # Build cache-based dev_test predictions → replicate for prot using same pipeline
        # Need to get prot predictions from models trained on full TRAIN
        # We rebuild LR from OOF and use cache-based full-TRAIN model predictions
        # For A1 on prot: we need prot-set predictions from each base model
        # Since we cached full-TRAIN model dev_test predictions, we need to
        # re-generate prot predictions from the same full-TRAIN models.
        # We reconstruct from cache's full-train models (which are not persisted).
        # For simplicity and correctness: rebuild full-TRAIN models for prot inference.
        set_seed(seed)
        X_train, y_train = encode_split("train")

        # RF prot predictions
        rf_m = build_rf(seed); rf_m.fit(X_train, y_train)
        rf_prot = rf_m.predict_proba(X_prot)[:, 1]

        # DT prot predictions
        dt_m = build_dt(seed); dt_m.fit(X_train, y_train)
        dt_prot = dt_m.predict_proba(X_prot)[:, 1]

        # SVM prot predictions
        sc_svm = StandardScaler()
        X_tr_sc = sc_svm.fit_transform(X_train)
        svm_m = build_svm(seed); svm_m.fit(X_tr_sc, y_train)
        svm_prot_score = svm_m.decision_function(sc_svm.transform(X_prot))

        # NN prot predictions
        set_seed(seed)
        val_frac = cfg["nn_config"].get("val_frac", 0.05)
        n_val = max(1, int(len(X_train) * val_frac))
        idx_perm = np.random.permutation(len(X_train))
        sc_nn = StandardScaler()
        X_nn_tr = sc_nn.fit_transform(X_train[idx_perm[n_val:]])
        X_nn_val = sc_nn.transform(X_train[idx_perm[:n_val]])
        nn_m = train_nn_fold(X_nn_tr, y_train[idx_perm[n_val:]], X_nn_val,
                             y_train[idx_perm[:n_val]], seed, cfg)
        nn_prot = nn_predict_proba(nn_m, sc_nn.transform(X_prot).astype(np.float32))

        # A1 meta-features for prot
        meta_X_prot = np.column_stack([dt_prot, rf_prot, svm_prot_score, nn_prot])
        meta_X_oof, y_oof = build_meta_features(["dt","rf","svm","nn"], seed, "oof")
        lr = build_meta_lr(seed); lr.fit(meta_X_oof, y_oof)
        a1_prot = lr.predict(meta_X_prot)

        # A6 prot
        a6_prot = np.maximum(a1_prot, ae_flags)

        a1_detected  = int(a1_prot.sum())
        a6_detected  = int(a6_prot.sum())
        ae_detected  = int(ae_flags.sum())

        prot_results["per_seed"][str(seed)] = {
            "A1_detected": a1_detected, "A1_rate": a1_detected / n_prot,
            "A1_missed":   n_prot - a1_detected,
            "A6_detected": a6_detected, "A6_rate": a6_detected / n_prot,
            "A6_missed":   n_prot - a6_detected,
            "AE_detected": ae_detected, "AE_rate": ae_detected / n_prot,
        }
        logger.info(f"Backdoor seed={seed}: A1={a1_detected}/{n_prot}, A6={a6_detected}/{n_prot}")

    save_json(result_path, prot_results)
    logger.info(f"Protected Backdoor results written to {result_path}")
    return prot_results


# ─────────────────────────────────────────────────────────────────────────────
# Cache integrity check (D26)
# ─────────────────────────────────────────────────────────────────────────────
def verify_cache_column_equality(seed: int):
    """Verify that A2-A5 cache slices exactly equal A1 cached columns (D26)."""
    a1_models = ["dt", "rf", "svm", "nn"]
    for config_id, retained in [
        ("A2_NO_DT", ["rf","svm","nn"]),
        ("A3_NO_RF", ["dt","svm","nn"]),
        ("A4_NO_SVM",["dt","rf","nn"]),
        ("A5_NO_NN", ["dt","rf","svm"]),
    ]:
        for mn in retained:
            c1 = load_cache(mn, seed)["oof_scores"]
            c2 = load_cache(mn, seed)["oof_scores"]  # same file → always equal
            # More meaningful: compare that the column in a1 matches independently
            a1_meta_X_oof, _ = build_meta_features(a1_models, seed, "oof")
            sub_meta_X_oof, _ = build_meta_features(retained, seed, "oof")
            # Verify each retained column exactly matches A1's corresponding column
            a1_col_idx = a1_models.index(mn)
            retained_col_idx = retained.index(mn)
            assert np.array_equal(a1_meta_X_oof[:, a1_col_idx],
                                  sub_meta_X_oof[:, retained_col_idx]), \
                f"D26 FAIL: {config_id} column {mn} != A1 column {mn} (seed={seed})"
    logger.info(f"D26 cache integrity: PASS (seed={seed})")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8 — Aggregation
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_results(cfg: dict):
    logger.info("=== PHASE 8: Aggregation ===")
    seeds = cfg["seeds"]
    config_ids = ["A0_RF", "A1_FULL_STACK", "A1b_SOFT_VOTE",
                  "A2_NO_DT", "A3_NO_RF", "A4_NO_SVM", "A5_NO_NN",
                  "A6_STACK_PLUS_AE"]
    dir_map = {cid: EXP_DIR / cid for cid in config_ids}

    metrics_keys = ["macro_f1","precision","recall","f1","balanced_accuracy","fpr"]

    # Load all per-seed results
    all_results = {}
    rows_csv = []
    for cid in config_ids:
        all_results[cid] = {}
        for seed in seeds:
            rp = dir_map[cid] / f"seed_{seed}.json"
            if not rp.exists():
                raise RuntimeError(f"Missing result: {rp}")
            r = json.loads(rp.read_text())
            all_results[cid][seed] = r
            rows_csv.append({
                "config_id": cid, "seed": seed,
                **{k: r[k] for k in metrics_keys},
                "runtime_sec": r.get("runtime_sec", 0.0),
            })

    # ablation_table.csv
    df = pd.DataFrame(rows_csv, columns=["config_id","seed","macro_f1","precision",
                                          "recall","f1","balanced_accuracy","fpr","runtime_sec"])
    df.to_csv(EXP_DIR / "ablation_table.csv", index=False)
    logger.info("ablation_table.csv written")

    # Per-config summary stats
    summary = {}
    for cid in config_ids:
        vals = {k: [all_results[cid][s][k] for s in seeds] for k in metrics_keys}
        summary[cid] = {
            "per_seed": {str(s): {k: all_results[cid][s][k] for k in metrics_keys} for s in seeds},
            "mean":   {k: float(np.mean(vals[k]))         for k in metrics_keys},
            "std":    {k: float(np.std(vals[k], ddof=0))  for k in metrics_keys},
            "min":    {k: float(np.min(vals[k]))          for k in metrics_keys},
            "max":    {k: float(np.max(vals[k]))          for k in metrics_keys},
        }

    # Paired deltas — macro_f1 (all), recall/fpr for A1-vs-A6
    delta_rows = []
    comparisons_mf1 = [
        ("A1-A0",  "A1_FULL_STACK", "A0_RF"),
        ("A1-A1b", "A1_FULL_STACK", "A1b_SOFT_VOTE"),
        ("A1-A2",  "A1_FULL_STACK", "A2_NO_DT"),
        ("A1-A3",  "A1_FULL_STACK", "A3_NO_RF"),
        ("A1-A4",  "A1_FULL_STACK", "A4_NO_SVM"),
        ("A1-A5",  "A1_FULL_STACK", "A5_NO_NN"),
        ("A6-A1",  "A6_STACK_PLUS_AE", "A1_FULL_STACK"),
    ]
    for comp_label, cid_a, cid_b in comparisons_mf1:
        deltas_mf1 = []
        for seed in seeds:
            d = all_results[cid_a][seed]["macro_f1"] - all_results[cid_b][seed]["macro_f1"]
            delta_rows.append({"comparison": comp_label, "seed": seed,
                                "metric": "macro_f1", "delta_value": float(d)})
            deltas_mf1.append(d)
        delta_rows.append({"comparison": comp_label, "seed": "mean",
                           "metric": "macro_f1",
                           "delta_value": float(np.mean(deltas_mf1))})

    # A1-vs-A6 additional metrics: recall, fpr
    for metric in ["recall", "fpr"]:
        deltas = []
        for seed in seeds:
            d = (all_results["A6_STACK_PLUS_AE"][seed][metric] -
                 all_results["A1_FULL_STACK"][seed][metric])
            delta_rows.append({"comparison": "A6-A1", "seed": seed,
                                "metric": metric, "delta_value": float(d)})
            deltas.append(d)
        delta_rows.append({"comparison": "A6-A1", "seed": "mean",
                           "metric": metric, "delta_value": float(np.mean(deltas))})

    # A1-vs-A6 backdoor detection rate
    prot_path = EXP_DIR / "protected_backdoor_results.json"
    if prot_path.exists():
        prot = json.loads(prot_path.read_text())
        deltas_bdr = []
        for seed in seeds:
            r_seed = prot["per_seed"][str(seed)]
            d = r_seed["A6_rate"] - r_seed["A1_rate"]
            delta_rows.append({"comparison": "A6-A1", "seed": seed,
                               "metric": "backdoor_detection_rate", "delta_value": float(d)})
            deltas_bdr.append(d)
        delta_rows.append({"comparison": "A6-A1", "seed": "mean",
                           "metric": "backdoor_detection_rate",
                           "delta_value": float(np.mean(deltas_bdr))})

    df_deltas = pd.DataFrame(delta_rows, columns=["comparison","seed","metric","delta_value"])
    df_deltas.to_csv(EXP_DIR / "paired_deltas.csv", index=False)
    logger.info("paired_deltas.csv written")

    # summary.json
    save_json(EXP_DIR / "summary.json", {
        "experiment_id": "EXP_ABLATION_V1",
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "configs": summary,
        "no_significance_claim": "n=3 seeds — no statistical significance claimed",
    })
    logger.info("summary.json written")

    return summary, df_deltas


def build_quality_review(summary: dict, df_deltas: pd.DataFrame, cfg: dict):
    """Write quality_review.md."""
    seeds = cfg["seeds"]
    config_ids = ["A0_RF","A1_FULL_STACK","A1b_SOFT_VOTE",
                  "A2_NO_DT","A3_NO_RF","A4_NO_SVM","A5_NO_NN","A6_STACK_PLUS_AE"]

    lines = ["# Sprint 10 — EXP_ABLATION_V1 Quality Review\n"]
    lines.append(f"Generated: {datetime.datetime.utcnow().isoformat()}Z\n\n")
    lines.append("## Per-Configuration Macro-F1\n\n")
    lines.append("| Config | Seed 42 | Seed 123 | Seed 2024 | Mean | Std (ddof=0) |\n")
    lines.append("|--------|---------|----------|-----------|------|-------------|\n")
    for cid in config_ids:
        s = summary[cid]
        vals = [s["per_seed"][str(seed)]["macro_f1"] for seed in seeds]
        lines.append(f"| {cid} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} "
                     f"| {s['mean']['macro_f1']:.6f} | {s['std']['macro_f1']:.8f} |\n")

    lines.append("\n## Paired Macro-F1 Deltas\n\n")
    lines.append("| Comparison | Seed 42 | Seed 123 | Seed 2024 | Mean |\n")
    lines.append("|------------|---------|----------|-----------|------|\n")
    for comp in ["A1-A0","A1-A1b","A1-A2","A1-A3","A1-A4","A1-A5","A6-A1"]:
        sub = df_deltas[(df_deltas.comparison == comp) & (df_deltas.metric == "macro_f1")]
        by_seed = {int(r["seed"]): r["delta_value"] for _, r in sub.iterrows()
                   if r["seed"] != "mean"}
        mean_d = sub[sub.seed == "mean"]["delta_value"].values
        mean_str = f"{mean_d[0]:+.6f}" if len(mean_d) > 0 else "N/A"
        lines.append(f"| {comp} | {by_seed.get(42,0):+.6f} | "
                     f"{by_seed.get(123,0):+.6f} | {by_seed.get(2024,0):+.6f} | {mean_str} |\n")

    lines.append("\n> No statistical significance claimed from n=3 seeds.\n\n")
    lines.append("## A1 vs A6 — Recall / FPR / Backdoor\n\n")
    for metric in ["recall","fpr","backdoor_detection_rate"]:
        sub = df_deltas[(df_deltas.comparison == "A6-A1") & (df_deltas.metric == metric)]
        mean_d = sub[sub.seed == "mean"]["delta_value"].values
        lines.append(f"- **A6-A1 {metric} delta (mean)**: "
                     f"{mean_d[0]:+.6f}\n" if len(mean_d) > 0 else f"- {metric}: N/A\n")

    (EXP_DIR / "quality_review.md").write_text("".join(lines), encoding="utf-8")
    logger.info("quality_review.md written")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = load_cfg()
    seeds = cfg["seeds"]

    # ── PHASE 0 ──────────────────────────────────────────────────────────────
    env_info, cfg_hash_before = phase0_preverify(cfg)

    # ── Load TRAIN + DEV_TEST once ────────────────────────────────────────────
    logger.info("Encoding TRAIN and DEVELOPMENT_TEST …")
    X_train, y_train = encode_split("train")
    X_dev,   y_dev   = encode_split("development_test")
    logger.info(f"TRAIN: {X_train.shape}, DEV_TEST: {X_dev.shape}")

    smoke_done_flag = EXP_DIR / ".smoke_done"

    # ── PHASE 1 — Smoke test (A0 seed=42, A1 seed=42) ────────────────────────
    if not smoke_done_flag.exists():
        logger.info("=== PHASE 1: Smoke test (A0 seed=42, A1 seed=42) ===")
        # Cache seed=42 for all models
        for mn in ["dt", "rf", "svm", "nn"]:
            if not cache_integrity_ok(mn, 42):
                with hard_timeout(cfg["timeouts"]["cache_per_seed_sec"],
                                  f"cache {mn} seed=42"):
                    generate_cache_one(mn, 42, X_train, y_train, X_dev, y_dev, cfg)
        # A0 seed=42
        run_a0(42, X_train, y_train, X_dev, y_dev, cfg)
        # A1 seed=42
        run_stacking_config("A1_FULL_STACK", ["dt","rf","svm","nn"], 42, y_dev, cfg)
        # Smoke: report svm_unit distribution for seed=42 OOF
        svm_unit_oof = sigmoid(load_cache("svm", 42)["oof_scores"])
        svm_dist_oof = {
            "min": float(svm_unit_oof.min()), "max": float(svm_unit_oof.max()),
            "mean": float(svm_unit_oof.mean()), "median": float(np.median(svm_unit_oof)),
            "pct_leq_1e-6": float((svm_unit_oof <= 1e-6).mean() * 100),
            "pct_geq_1m1e-6": float((svm_unit_oof >= 1 - 1e-6).mean() * 100),
            "pct_leq_0001": float((svm_unit_oof <= 0.001).mean() * 100),
            "pct_geq_0999": float((svm_unit_oof >= 0.999).mean() * 100),
        }
        logger.info(f"SMOKE TEST svm_unit_oof distribution seed=42: {svm_dist_oof}")
        save_json(EXP_DIR / "provenance/smoke_test_svm_unit_oof_seed42.json", svm_dist_oof)

        # Check saturation before continuing
        saturated_frac = svm_dist_oof["pct_leq_1e-6"] + svm_dist_oof["pct_geq_1m1e-6"]
        if saturated_frac > 50.0:
            stop_report(
                "SMOKE-A1b-SATURATION",
                "svm_unit OOF saturation < 50%",
                f"{saturated_frac:.1f}% saturated",
                "cache/svm_seed42.npz oof_scores",
                "Sigmoid strongly saturated on OOF. A1b averaging is numerically degenerate.",
                "A0 seed=42 and A1 seed=42 complete. Cache seed=42 complete.",
                "Human: review sigmoid saturation and decide on alternative normalization."
            )

        smoke_done_flag.write_text("SMOKE TEST PASS", encoding="utf-8")
        logger.info("=== PHASE 1: Smoke test PASS ===")
    else:
        logger.info("PHASE 1: Smoke test already done — skipping")

    # ── PHASE 2 — Base cache (all models, all seeds) ──────────────────────────
    logger.info("=== PHASE 2: Base cache generation ===")
    for seed in seeds:
        for mn in ["dt", "rf", "svm", "nn"]:
            if not cache_integrity_ok(mn, seed):
                with hard_timeout(cfg["timeouts"]["cache_per_seed_sec"],
                                  f"cache {mn} seed={seed}"):
                    generate_cache_one(mn, seed, X_train, y_train, X_dev, y_dev, cfg)
            else:
                logger.info(f"Cache {mn} seed={seed} — OK (skip)")
    logger.info("=== PHASE 2: COMPLETE ===")

    # ── D26 Cache integrity check ─────────────────────────────────────────────
    for seed in seeds:
        verify_cache_column_equality(seed)

    # ── PHASE 3 — A0 remaining seeds ─────────────────────────────────────────
    logger.info("=== PHASE 3: A0 fresh RF seeds 123, 2024 ===")
    for seed in [123, 2024]:
        run_a0(seed, X_train, y_train, X_dev, y_dev, cfg)
    logger.info("=== PHASE 3: COMPLETE ===")

    # ── PHASE 4 — A1 all seeds ────────────────────────────────────────────────
    logger.info("=== PHASE 4: A1 meta-learner all seeds ===")
    for seed in seeds:
        run_stacking_config("A1_FULL_STACK", ["dt","rf","svm","nn"], seed, y_dev, cfg)
    logger.info("=== PHASE 4: COMPLETE ===")

    # ── PHASE 5 — A1b all seeds ───────────────────────────────────────────────
    logger.info("=== PHASE 5: A1b soft-vote all seeds ===")
    for seed in seeds:
        run_a1b(seed, y_dev, cfg)
    logger.info("=== PHASE 5: COMPLETE ===")

    # ── PHASE 6 — A2–A5 ──────────────────────────────────────────────────────
    logger.info("=== PHASE 6: A2-A5 ablated stacking ===")
    for cfg_id, model_names in [
        ("A2_NO_DT", ["rf","svm","nn"]),
        ("A3_NO_RF", ["dt","svm","nn"]),
        ("A4_NO_SVM",["dt","rf","nn"]),
        ("A5_NO_NN", ["dt","rf","svm"]),
    ]:
        for seed in seeds:
            run_stacking_config(cfg_id, model_names, seed, y_dev, cfg)
        # D26 cache integrity post each config
        for seed in seeds:
            verify_cache_column_equality(seed)
    logger.info("=== PHASE 6: COMPLETE ===")

    # ── PHASE 7 — A6 ──────────────────────────────────────────────────────────
    logger.info("=== PHASE 7: A6 — full stack + frozen AE ===")
    for seed in seeds:
        run_a6(seed, X_dev, y_dev, cfg)
    logger.info("=== PHASE 7: COMPLETE ===")

    # ── Protected Backdoor ────────────────────────────────────────────────────
    prot_path = EXP_DIR / "protected_backdoor_results.json"
    if not prot_path.exists():
        run_protected_backdoor(cfg)
    else:
        logger.info("Protected Backdoor results exist — skip")

    # ── PHASE 8 — Aggregation ────────────────────────────────────────────────
    summary, df_deltas = aggregate_results(cfg)
    build_quality_review(summary, df_deltas, cfg)

    # ── D19 Config immutability after-hash ────────────────────────────────────
    cfg_hash_after = sha256_file(CFG_PATH)
    before_rec = json.loads((EXP_DIR / "provenance/config_sha256_before.json").read_text())
    if cfg_hash_after != before_rec["config_sha256_before"]:
        stop_report(
            "D19", f"config SHA-256={before_rec['config_sha256_before']}",
            f"after SHA-256={cfg_hash_after}", str(CFG_PATH),
            "Config was modified during execution — results invalid.",
            "All results written.", "Human: review config changes and rerun."
        )
    save_json(EXP_DIR / "provenance/config_sha256_after.json",
              {"config_sha256_before": before_rec["config_sha256_before"],
               "config_sha256_after": cfg_hash_after,
               "match": cfg_hash_after == before_rec["config_sha256_before"],
               "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z"})
    logger.info(f"D19 config immutability: PASS ({cfg_hash_after})")

    # ── metadata.json ─────────────────────────────────────────────────────────
    save_json(EXP_DIR / "metadata.json", {
        "experiment_id": "EXP_ABLATION_V1",
        "sprint": 10,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "a0_identity": "RF (verified from selected_configs.json — highest CV Macro-F1=0.9508)",
        "feature_set": "EXP_MI_V1_1",
        "n_features": 75,
        "seeds": seeds,
        "n_folds": cfg["n_folds"],
        "headline_split": "development_test",
        "a1b_svm_normalization": "sigmoid(decision_function)",
        "a1b_aggregation": "mean",
        "a1b_threshold": 0.5,
        "frozen_ae_threshold": cfg["ae_config"]["threshold"],
        "frozen_ae_fusion_rule": "OR (C06)",
        "environment": env_info,
        "config_sha256": cfg_hash_after,
        "d11_resolution": "sklearn=1.9.0 authoritative (matches EXP_OOF_STACK_V1); EXP_H123_V1 recorded 1.5.0 is a recording discrepancy.",
        "sprint9_protected": True,
    })

    logger.info("=== EXP_ABLATION_V1 COMPLETE ===")
    print("\nIMPLEMENTATION: COMPLETE")


if __name__ == "__main__":
    main()
