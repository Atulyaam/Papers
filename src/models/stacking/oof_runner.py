"""
src/models/stacking/oof_runner.py
-----------------------------------
Out-of-Fold (OOF) prediction runner for Sprint 6.

OOF Protocol (frozen)
----------------------
    StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    Stratification target: label

    Folds are created ONCE and reused for all H1 seeds.
    The same fold assignment is identical across seeds 42, 123, and 2024.

H1 Seeds: 42, 123, 2024
    For each seed S:
        DT  random_state = S
        RF  random_state = S
        SVM random_state = S
        NN  set_all_seeds(S)   [Python, NumPy, PyTorch, CUDA]
        Meta-learner random_state = S

    All other frozen Sprint 5 hyperparameters are unchanged.

Base-model OOF outputs
-----------------------
    DT:  predict_proba()[:, 1]          — attack probability in [0, 1]
    RF:  predict_proba()[:, 1]          — attack probability in [0, 1]
    SVM: decision_function()             — unbounded real score
    NN:  torch.sigmoid(logits)           — attack probability in [0, 1]

NN OOF rule
-----------
    Fixed final_epoch_count = 18 for every OOF fold.
    The OOF held-out fold is NOT used as a validation/early-stopping set.
    pos_weight = 0.38096857859602873  (full-TRAIN constant, not per-fold)

Scaling limitation (bounded, label-independent)
------------------------------------------------
SVM/NN OOF predictions use scaling statistics computed on the full
frozen TRAIN, including rows in the OOF held-out fold. This is a bounded,
label-independent leakage channel accepted for feature-space consistency.

This limitation does NOT apply to target leakage, self-prediction leakage,
protected-test leakage, or validation/test model-selection leakage — all of
which remain fully prevented.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_models.neural_network import IDSNet, NN_BATCH_SIZE, NN_INPUT_DIM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen OOF constants
# ---------------------------------------------------------------------------

OOF_SEED: int = 7           # StratifiedKFold seed — fixed across all H1 seeds
OOF_N_SPLITS: int = 5

# Frozen NN training parameter
OOF_FIXED_EPOCH_COUNT: int = 18   # median(best_epoch) from Sprint 5 CV

# pos_weight from full frozen TRAIN (NOT recomputed per fold)
OOF_POS_WEIGHT: float = 44_800 / 117_595   # = 0.38096857859602873

# H1 seeds
H1_SEEDS: list[int] = [42, 123, 2024]

# Mandatory limitation text (verbatim)
SCALING_LIMITATION_TEXT: str = (
    "SVM/NN OOF predictions use scaling statistics computed on the full "
    "frozen TRAIN, including rows in the OOF held-out fold. This is a bounded, "
    "label-independent leakage channel accepted for feature-space consistency."
)

# Frozen Sprint 5 selected configurations (hyperparams only — random_state overridden per H1 seed)
_DT_FROZEN_CONFIG: dict[str, Any] = {
    "criterion": "entropy",
    "max_depth": None,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
}

_RF_FROZEN_CONFIG: dict[str, Any] = {
    "n_estimators": 300,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": 0.3,
    "class_weight": "balanced",
    "n_jobs": -1,
}

_SVM_FROZEN_CONFIG: dict[str, Any] = {
    "C": 0.1,
    "class_weight": "balanced",
    "max_iter": 5000,
}

_NN_FROZEN_CONFIG: dict[str, Any] = {
    "hidden_sizes": [128, 64],
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
}


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------


def set_all_seeds(seed: int) -> None:
    """
    Set Python, NumPy, PyTorch, and CUDA seeds.

    Used before each NN OOF fold to ensure reproducible weight initialisation.
    Does NOT force full CUDA determinism (cuDNN benchmarking not disabled —
    documented policy from Sprint 5).

    Parameters
    ----------
    seed : int
        The H1 seed to apply.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.debug("set_all_seeds(%d) applied", seed)


# ---------------------------------------------------------------------------
# OOF fold factory — called ONCE, reused across all H1 seeds
# ---------------------------------------------------------------------------


def make_oof_folds(
    y: np.ndarray,
    n_splits: int = OOF_N_SPLITS,
    seed: int = OOF_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create OOF fold assignments. Must be called ONCE and reused for all H1 seeds.

    Parameters
    ----------
    y : np.ndarray
        Label array used for stratification (binary 0/1, shape (n,)).
    n_splits : int
        Number of folds (frozen: 5).
    seed : int
        StratifiedKFold random_state (frozen: 7).

    Returns
    -------
    list of (train_idx, oof_idx) tuples
        Length = n_splits. Each element is a pair of integer index arrays.

    Invariants
    ----------
    - The same ``y`` + ``n_splits`` + ``seed`` always produces bit-identical folds.
    - The returned list must be passed unchanged to run_oof_seed for every H1 seed.
    """
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if len(y) == 0:
        raise ValueError("y is empty — cannot create OOF folds.")
    unique = np.unique(y)
    if len(unique) < 2:
        raise ValueError(
            f"y contains only one class ({unique}). "
            "OOF stacking requires at least two classes."
        )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [(train_idx.copy(), oof_idx.copy()) for train_idx, oof_idx in skf.split(y, y)]
    logger.info(
        "OOF folds created | n_splits=%d | seed=%d | fold_sizes=%s",
        n_splits,
        seed,
        [len(o) for _, o in folds],
    )
    return folds


# ---------------------------------------------------------------------------
# Internal: train NN on one OOF fold (fixed epochs, no early stopping)
# ---------------------------------------------------------------------------


def _train_nn_oof_fold(
    X_scaled_tr: np.ndarray,
    y_tr: np.ndarray,
    h1_seed: int,
    device: torch.device,
) -> IDSNet:
    """
    Train IDSNet on one OOF fold's train split.

    Fixed epoch_count=18. No early stopping. No held-out validation split.
    pos_weight = OOF_POS_WEIGHT (full-TRAIN constant, not per-fold).

    Parameters
    ----------
    X_scaled_tr : np.ndarray, shape (n_tr, 75)
        Scaled (via frozen Sprint 5 scaler) training feature matrix.
    y_tr : np.ndarray, shape (n_tr,)
        Binary labels.
    h1_seed : int
        H1 seed — set_all_seeds(h1_seed) must be called by the caller
        before invoking this function.
    device : torch.device

    Returns
    -------
    IDSNet
        Trained model in eval() mode.
    """
    hidden_sizes = list(_NN_FROZEN_CONFIG["hidden_sizes"])
    lr = float(_NN_FROZEN_CONFIG["learning_rate"])
    wd = float(_NN_FROZEN_CONFIG["weight_decay"])

    net = IDSNet(input_dim=NN_INPUT_DIM, hidden_sizes=hidden_sizes).to(device)

    pos_weight_t = torch.tensor([OOF_POS_WEIGHT], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    X_t = torch.tensor(X_scaled_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.float32)
    train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=NN_BATCH_SIZE, shuffle=True)

    for epoch in range(1, OOF_FIXED_EPOCH_COUNT + 1):
        net.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = net(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0 or epoch == OOF_FIXED_EPOCH_COUNT:
            logger.debug("NN OOF fold | epoch=%d/%d", epoch, OOF_FIXED_EPOCH_COUNT)

    net.eval()
    return net


# ---------------------------------------------------------------------------
# Core OOF runner for a single H1 seed
# ---------------------------------------------------------------------------


def run_oof_seed(
    h1_seed: int,
    folds: list[tuple[np.ndarray, np.ndarray]],
    X_unscaled: np.ndarray,
    y: np.ndarray,
    svm_scaler: StandardScaler,
    nn_scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Run full OOF prediction for one H1 seed.

    Parameters
    ----------
    h1_seed : int
        One of {42, 123, 2024}. Propagated to DT/RF/SVM random_state and NN seeds.
    folds : list of (train_idx, oof_idx)
        Pre-built fold assignment from make_oof_folds(). MUST be identical across seeds.
    X_unscaled : np.ndarray, shape (162395, 75)
        Encoded, UNSCALED feature matrix for the full TRAIN.
    y : np.ndarray, shape (162395,)
        TRAIN labels.
    svm_scaler : StandardScaler
        Frozen Sprint 5 full-TRAIN scaler for SVM.
        (Fitted on all 162,395 rows — bounded label-independent leakage documented.)
    nn_scaler : StandardScaler
        Frozen Sprint 5 full-TRAIN scaler for NN.
        (Same limitation applies.)

    Returns
    -------
    pd.DataFrame
        Shape (162395, 6), columns:
        [row_id, dt_attack_probability, rf_attack_probability,
         svm_decision_score, nn_attack_probability, label]
        Ordered by row_id (original TRAIN row index).

    Invariants
    ----------
    - No TRAIN row is missing (OOF coverage = 162,395).
    - No TRAIN row appears more than once per model.
    - For every (row, model), the row is NOT in the model's training indices
      for the fold that generated that row's prediction (no self-prediction).
    """
    n = len(y)
    if n == 0:
        raise ValueError("TRAIN is empty — cannot run OOF.")

    # Pre-scale for SVM and NN (frozen full-TRAIN scalers — documented limitation)
    X_svm = svm_scaler.transform(X_unscaled)
    X_nn = nn_scaler.transform(X_unscaled)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accumulators
    dt_preds = np.full(n, np.nan, dtype=np.float64)
    rf_preds = np.full(n, np.nan, dtype=np.float64)
    svm_preds = np.full(n, np.nan, dtype=np.float64)
    nn_preds = np.full(n, np.nan, dtype=np.float64)

    logger.info(
        "OOF seed=%d | n_folds=%d | n_train=%d | device=%s",
        h1_seed, len(folds), n, device,
    )

    for fold_idx, (train_idx, oof_idx) in enumerate(folds):
        t0 = time.perf_counter()
        logger.info(
            "OOF seed=%d fold=%d | n_train_fold=%d | n_oof=%d",
            h1_seed, fold_idx, len(train_idx), len(oof_idx),
        )

        X_tr = X_unscaled[train_idx]
        y_tr = y[train_idx]

        # ── DT ────────────────────────────────────────────────────────────
        dt = DecisionTreeClassifier(
            **_DT_FROZEN_CONFIG,
            random_state=h1_seed,
        )
        dt.fit(X_tr, y_tr)
        dt_preds[oof_idx] = np.asarray(dt.predict_proba(X_unscaled[oof_idx]))[:, 1]

        # ── RF ────────────────────────────────────────────────────────────
        rf = RandomForestClassifier(
            **_RF_FROZEN_CONFIG,
            random_state=h1_seed,
        )
        rf.fit(X_tr, y_tr)
        rf_preds[oof_idx] = np.asarray(rf.predict_proba(X_unscaled[oof_idx]))[:, 1]

        # ── SVM ───────────────────────────────────────────────────────────
        # Uses frozen Sprint 5 full-TRAIN scaler (bounded label-indep. leakage)
        svm = LinearSVC(
            **_SVM_FROZEN_CONFIG,
            random_state=h1_seed,
        )
        svm.fit(X_svm[train_idx], y_tr)
        svm_preds[oof_idx] = svm.decision_function(X_svm[oof_idx])

        # ── NN ────────────────────────────────────────────────────────────
        # set_all_seeds before EVERY NN fold training
        set_all_seeds(h1_seed)
        net = _train_nn_oof_fold(X_nn[train_idx], y_tr, h1_seed, device)
        with torch.no_grad():
            X_oof_t = torch.tensor(X_nn[oof_idx], dtype=torch.float32).to(device)
            logits = net(X_oof_t)
            nn_preds[oof_idx] = torch.sigmoid(logits).cpu().numpy()

        elapsed = time.perf_counter() - t0
        logger.info(
            "OOF seed=%d fold=%d done | runtime=%.2fs",
            h1_seed, fold_idx, elapsed,
        )

    # Verify completeness (no NaNs)
    for name, arr in [
        ("DT", dt_preds), ("RF", rf_preds), ("SVM", svm_preds), ("NN", nn_preds)
    ]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            raise RuntimeError(
                f"OOF completeness failure: {n_nan} NaN predictions in {name} "
                f"(seed={h1_seed}). Some rows were not assigned to any OOF fold."
            )

    oof_df = pd.DataFrame({
        "row_id": np.arange(n, dtype=np.int64),
        "dt_attack_probability": dt_preds,
        "rf_attack_probability": rf_preds,
        "svm_decision_score": svm_preds,
        "nn_attack_probability": nn_preds,
        "label": y.astype(np.int64),
    })

    logger.info(
        "OOF seed=%d complete | shape=%s | dt_nan=%d rf_nan=%d svm_nan=%d nn_nan=%d",
        h1_seed, oof_df.shape,
        oof_df["dt_attack_probability"].isna().sum(),
        oof_df["rf_attack_probability"].isna().sum(),
        oof_df["svm_decision_score"].isna().sum(),
        oof_df["nn_attack_probability"].isna().sum(),
    )
    return oof_df
