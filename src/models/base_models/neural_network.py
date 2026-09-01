"""
src/models/base_models/neural_network.py
------------------------------------------
PyTorch Neural Network base model for Sprint 5.

Architecture candidates
-----------------------
A: 75 → 128 → 64 → 1   (hidden_sizes=[128, 64])
B: 75 → 256 → 128 → 1  (hidden_sizes=[256, 128])

Activation: ReLU between hidden layers.
Output: single logit (BCEWithLogitsLoss).
Sigmoid applied post-hoc to produce probabilities.

Fixed training parameters (frozen)
------------------------------------
optimizer:  Adam
batch_size: 256
patience:   5  (early stopping on inner_val loss)
loss:       BCEWithLogitsLoss

pos_weight (class imbalance)
-----------------------------
Computed ONCE from frozen TRAIN counts:
    N_normal  = 44,800
    N_attack  = 117,595
    pos_weight = 44800 / 117595  ≈ 0.380786...

This is computed outside the fold loop and recorded in metadata.
It is NOT recomputed per fold.

Tuning grid (frozen)
--------------------
hidden_sizes:   [128, 64]  |  [256, 128]
learning_rate:  0.001      |  0.0001
weight_decay:   0.0001     |  0.001
Total: 2 × 2 × 2 = 8 configurations.

Scaler isolation
----------------
A fresh StandardScaler is fit on inner_train ONLY inside each fold.
inner_val is only transformed (never fit).

Early stopping
--------------
Monitor: inner_val BCEWithLogitsLoss (with pos_weight).
patience = 5.
Save best model weights; restore after early stopping.
Record: best_epoch, best_val_loss, final_epoch.

Final refit
-----------
Train on full TRAIN for final_epoch_count = median(best_epoch across 5 folds).
No early stopping on final refit (no held-out monitor).

Output contract
---------------
predict()  -> class labels {0, 1}
sigmoid probability -> float in [0, 1]
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.models.base_models.cv_utils import (
    CVSummary,
    FoldMetrics,
    aggregate_cv_results,
    compute_fold_metrics,
    make_model_skf,
)
from src.models.base_models.preprocessing import fit_scaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class imbalance — fixed from frozen TRAIN
# ---------------------------------------------------------------------------

TRAIN_N_NORMAL = 44_800
TRAIN_N_ATTACK = 117_595

# pos_weight = N_negative / N_positive (fixed scalar, NOT per-fold)
TRAIN_POS_WEIGHT: float = TRAIN_N_NORMAL / TRAIN_N_ATTACK  # ≈ 0.38079

# ---------------------------------------------------------------------------
# Fixed training hyperparameters (frozen)
# ---------------------------------------------------------------------------

NN_BATCH_SIZE = 256
NN_PATIENCE = 5
NN_INPUT_DIM = 75

# ---------------------------------------------------------------------------
# Tuning grid (frozen)
# ---------------------------------------------------------------------------

NN_TUNING_GRID: dict[str, list[Any]] = {
    "hidden_sizes": [[128, 64], [256, 128]],
    "learning_rate": [0.001, 0.0001],
    "weight_decay": [0.0001, 0.001],
}

# Baseline configuration
NN_BASELINE_CONFIG: dict[str, Any] = {
    "hidden_sizes": [128, 64],
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
}

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NNConfig:
    """
    Neural Network hyperparameter configuration.
    """

    hidden_sizes: tuple[int, ...] = (128, 64)
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_sizes": list(self.hidden_sizes),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

    def validate(self) -> None:
        """Raise ValueError for out-of-protocol configurations."""
        if len(self.hidden_sizes) < 1:
            raise ValueError("hidden_sizes must have at least one layer.")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")


# ---------------------------------------------------------------------------
# PyTorch model definition
# ---------------------------------------------------------------------------


class IDSNet(nn.Module):
    """
    Simple feedforward binary classifier for the UNSW-NB15 IDS task.

    Architecture: input → [Linear → ReLU] × len(hidden_sizes) → Linear(1)

    Parameters
    ----------
    input_dim : int
        Number of input features (75 for MI-selected set).
    hidden_sizes : list[int]
        List of hidden layer widths, e.g. [128, 64] or [256, 128].
    """

    def __init__(self, input_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must have at least one element.")
        for h in hidden_sizes:
            if not isinstance(h, int) or h < 1:
                raise ValueError(f"Each hidden size must be a positive int, got {h!r}")

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor shape (batch, input_dim)

        Returns
        -------
        Tensor shape (batch,)  — raw logits (pre-sigmoid)
        """
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Epoch-level diagnostics container
# ---------------------------------------------------------------------------


@dataclass
class NNEpochDiagnostics:
    """
    Records early-stopping epoch information across CV folds for one config.
    """

    config: dict[str, Any]
    best_epochs: list[int] = field(default_factory=list)
    best_val_losses: list[float] = field(default_factory=list)
    final_epochs: list[int] = field(default_factory=list)

    @property
    def mean_best_epoch(self) -> float:
        return float(np.mean(self.best_epochs)) if self.best_epochs else float("nan")

    @property
    def median_best_epoch(self) -> float:
        return float(np.median(self.best_epochs)) if self.best_epochs else float("nan")

    @property
    def std_best_epoch(self) -> float:
        return (
            float(np.std(self.best_epochs, ddof=1))
            if len(self.best_epochs) > 1
            else 0.0
        )

    @property
    def min_best_epoch(self) -> int:
        return int(np.min(self.best_epochs)) if self.best_epochs else 0

    @property
    def max_best_epoch(self) -> int:
        return int(np.max(self.best_epochs)) if self.best_epochs else 0

    @property
    def range_best_epoch(self) -> int:
        return self.max_best_epoch - self.min_best_epoch

    @property
    def range_median_ratio(self) -> float:
        med = self.median_best_epoch
        if med == 0 or math.isnan(med):
            return float("nan")
        return self.range_best_epoch / med

    @property
    def diagnostic_flag(self) -> bool:
        """True if range/median > 1.0  (review trigger, not auto-action)."""
        ratio = self.range_median_ratio
        return (not math.isnan(ratio)) and ratio > 1.0

    @property
    def final_epoch_count(self) -> int:
        """Median best_epoch rounded to nearest int for final refit."""
        return int(round(self.median_best_epoch))

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "best_epochs": self.best_epochs,
            "best_val_losses": self.best_val_losses,
            "final_epochs": self.final_epochs,
            "mean_best_epoch": self.mean_best_epoch,
            "median_best_epoch": self.median_best_epoch,
            "std_best_epoch": self.std_best_epoch,
            "min_best_epoch": self.min_best_epoch,
            "max_best_epoch": self.max_best_epoch,
            "range_best_epoch": self.range_best_epoch,
            "range_median_ratio": self.range_median_ratio,
            "diagnostic_flag": self.diagnostic_flag,
            "final_epoch_count": self.final_epoch_count,
        }


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------


def _train_one_fold(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    config: dict[str, Any],
    pos_weight_value: float,
    device: torch.device,
) -> tuple[IDSNet, int, float, int]:
    """
    Train one fold with early stopping.

    Returns
    -------
    (best_model_state_net, best_epoch, best_val_loss, final_epoch)
    """
    hidden_sizes: list[int] = list(config.get("hidden_sizes", [128, 64]))
    lr: float = float(config.get("learning_rate", 0.001))
    wd: float = float(config.get("weight_decay", 0.0001))

    net = IDSNet(input_dim=NN_INPUT_DIM, hidden_sizes=hidden_sizes).to(device)

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    # Build dataloaders
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).to(device)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=NN_BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_weights = copy.deepcopy(net.state_dict())
    epoch = 0

    while True:
        epoch += 1
        net.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = net(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        net.eval()
        with torch.no_grad():
            val_logits = net(X_va_t)
            val_loss = criterion(val_logits, y_va_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_weights = copy.deepcopy(net.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= NN_PATIENCE:
            break

    net.load_state_dict(best_weights)
    return net, best_epoch, best_val_loss, epoch


# ---------------------------------------------------------------------------
# Single CV run for one NN configuration
# ---------------------------------------------------------------------------


def run_nn_cv(
    X_unscaled: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    pos_weight_value: float = TRAIN_POS_WEIGHT,
) -> tuple[CVSummary, NNEpochDiagnostics]:
    """
    Run 5-fold StratifiedKFold CV for one NN configuration.

    SCALER ISOLATION: A fresh StandardScaler is fit on inner_train ONLY
    inside each fold.

    EARLY STOPPING: inner_val is the early-stopping monitor ONLY.
    Outer validation/test are never accessed here.

    Parameters
    ----------
    X_unscaled : np.ndarray
        Feature matrix (encoded, UNSCALED), shape (n, 75).
    y : np.ndarray
        Binary labels 0/1.
    config : dict
        NN hyperparameter dict.
    pos_weight_value : float
        Fixed pos_weight from frozen TRAIN (default TRAIN_POS_WEIGHT).

    Returns
    -------
    (CVSummary, NNEpochDiagnostics)
    """
    _validate_nn_inputs(X_unscaled, y, config)

    device = _get_device()
    logger.info(
        "NN CV | device=%s | config=%s | pos_weight=%.6f",
        device,
        _config_str(config),
        pos_weight_value,
    )

    skf = make_model_skf()
    fold_metrics: list[FoldMetrics] = []
    diag = NNEpochDiagnostics(config=config)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_unscaled, y)):
        t0 = time.perf_counter()

        X_tr_raw, X_va_raw = X_unscaled[train_idx], X_unscaled[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # ----------------------------------------------------------------
        # SCALER: fit ONLY on inner_train.
        # ----------------------------------------------------------------
        scaler = fit_scaler(X_tr_raw)
        X_tr = scaler.transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)

        net, best_epoch, best_val_loss, final_epoch = _train_one_fold(
            X_tr, X_va, y_tr, y_va, config, pos_weight_value, device
        )

        # Record epoch diagnostics
        diag.best_epochs.append(best_epoch)
        diag.best_val_losses.append(best_val_loss)
        diag.final_epochs.append(final_epoch)

        # Evaluate
        net.eval()
        with torch.no_grad():
            X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
            logits = net(X_va_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_pred = (probs >= 0.5).astype(int)

        elapsed = time.perf_counter() - t0

        fm = compute_fold_metrics(
            y_true=y_va,
            y_pred=y_pred,
            fold_idx=fold_idx,
            runtime_seconds=elapsed,
            n_train=len(train_idx),
            n_val=len(val_idx),
        )
        fold_metrics.append(fm)

        logger.info(
            "NN CV fold=%d | macro_f1=%.6f | best_epoch=%d | final_epoch=%d | runtime=%.2fs",
            fold_idx,
            fm.macro_f1,
            best_epoch,
            final_epoch,
            elapsed,
        )

    summary = aggregate_cv_results(fold_metrics, config=config, model_type="nn")
    logger.info(
        "NN CV done | mean_f1=%.6f ± %.6f | median_best_epoch=%d | "
        "range/median=%.3f | diagnostic_flag=%s | config=%s",
        summary.mean_macro_f1,
        summary.std_macro_f1,
        diag.final_epoch_count,
        diag.range_median_ratio if not math.isnan(diag.range_median_ratio) else float("nan"),
        diag.diagnostic_flag,
        _config_str(config),
    )
    if diag.diagnostic_flag:
        logger.warning(
            "NN EPOCH DIAGNOSTIC FLAG: range/median=%.3f > 1.0 | "
            "best_epochs=%s | REVIEW REQUIRED (no automatic action).",
            diag.range_median_ratio,
            diag.best_epochs,
        )
    return summary, diag


# ---------------------------------------------------------------------------
# Phase A — Baseline
# ---------------------------------------------------------------------------


def run_nn_baseline(
    X_unscaled: np.ndarray,
    y: np.ndarray,
    pos_weight_value: float = TRAIN_POS_WEIGHT,
) -> tuple[CVSummary, NNEpochDiagnostics]:
    """
    Phase A: Run 5-fold CV with the frozen baseline NN configuration.

    Parameters
    ----------
    X_unscaled, y : np.ndarray
        Encoded UNSCALED TRAIN feature matrix and labels.
    pos_weight_value : float
        Fixed pos_weight from frozen TRAIN.

    Returns
    -------
    (CVSummary, NNEpochDiagnostics)
    """
    logger.info("=== NN BASELINE START | pos_weight=%.6f ===", pos_weight_value)
    t0 = time.perf_counter()
    result, diag = run_nn_cv(X_unscaled, y, NN_BASELINE_CONFIG, pos_weight_value)
    elapsed = time.perf_counter() - t0
    logger.info(
        "=== NN BASELINE DONE | mean_f1=%.6f | total_runtime=%.2fs ===",
        result.mean_macro_f1,
        elapsed,
    )
    return result, diag


# ---------------------------------------------------------------------------
# Phase B — Tuning grid
# ---------------------------------------------------------------------------


def run_nn_tuning(
    X_unscaled: np.ndarray,
    y: np.ndarray,
    pos_weight_value: float = TRAIN_POS_WEIGHT,
) -> list[tuple[CVSummary, NNEpochDiagnostics]]:
    """
    Phase B: Run 5-fold CV for all 8 NN grid configurations.

    Parameters
    ----------
    X_unscaled, y : np.ndarray
        Encoded UNSCALED TRAIN feature matrix and labels.
    pos_weight_value : float
        Fixed pos_weight from frozen TRAIN.

    Returns
    -------
    list[(CVSummary, NNEpochDiagnostics)]
        One entry per grid configuration (8 total).
    """
    grid_size = _nn_grid_size()
    logger.info("=== NN TUNING START | grid_size=%d ===", grid_size)
    t0 = time.perf_counter()

    results = []
    configs = _generate_nn_configs()

    for i, config in enumerate(configs):
        logger.info("NN tuning config %d/%d | %s", i + 1, len(configs), _config_str(config))
        result, diag = run_nn_cv(X_unscaled, y, config, pos_weight_value)
        results.append((result, diag))

    elapsed = time.perf_counter() - t0
    logger.info(
        "=== NN TUNING DONE | n_configs=%d | total_runtime=%.2fs ===",
        len(results),
        elapsed,
    )
    return results


# ---------------------------------------------------------------------------
# Phase D — Final refit on full TRAIN
# ---------------------------------------------------------------------------


def refit_nn(
    X_unscaled: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    final_epoch_count: int,
    pos_weight_value: float = TRAIN_POS_WEIGHT,
) -> tuple[IDSNet, StandardScaler]:
    """
    Phase D: Fit the final NN on the complete frozen TRAIN data.

    NO early stopping — trains for exactly ``final_epoch_count`` epochs.
    ``final_epoch_count`` must be median(best_epoch) from the 5 inner-CV folds.

    A fresh StandardScaler is fit on the complete TRAIN data.

    Parameters
    ----------
    X_unscaled : np.ndarray
        Full TRAIN feature matrix (encoded, UNSCALED), shape (n, 75).
    y : np.ndarray
        Full TRAIN labels.
    config : dict
        Selected best NN configuration.
    final_epoch_count : int
        Fixed number of training epochs = median(best_epoch from inner-CV).
    pos_weight_value : float
        Fixed pos_weight from frozen TRAIN.

    Returns
    -------
    (IDSNet, StandardScaler)
        Fitted model and scaler — both must be persisted in the checkpoint.

    Notes
    -----
    The returned model exposes:
        - predict(X_scaled_tensor)  -> class labels {0, 1}
        - sigmoid of forward()      -> probability in [0, 1]
    """
    _validate_nn_inputs(X_unscaled, y, config)
    if final_epoch_count < 1:
        raise ValueError(f"final_epoch_count must be >= 1, got {final_epoch_count}")

    logger.info(
        "NN final refit | config=%s | final_epochs=%d | n_train=%d | pos_weight=%.6f",
        _config_str(config),
        final_epoch_count,
        len(y),
        pos_weight_value,
    )
    t0 = time.perf_counter()
    device = _get_device()

    # Fresh scaler on complete TRAIN
    scaler = fit_scaler(X_unscaled)
    X_scaled = scaler.transform(X_unscaled)

    hidden_sizes: list[int] = list(config.get("hidden_sizes", [128, 64]))
    lr: float = float(config.get("learning_rate", 0.001))
    wd: float = float(config.get("weight_decay", 0.0001))

    net = IDSNet(input_dim=NN_INPUT_DIM, hidden_sizes=hidden_sizes).to(device)

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=NN_BATCH_SIZE, shuffle=True)

    for epoch in range(1, final_epoch_count + 1):
        net.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = net(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if epoch % 5 == 0 or epoch == final_epoch_count:
            logger.info(
                "NN refit | epoch=%d/%d | mean_batch_loss=%.6f",
                epoch,
                final_epoch_count,
                epoch_loss / max(n_batches, 1),
            )

    elapsed = time.perf_counter() - t0
    logger.info("NN final refit done | runtime=%.2fs", elapsed)
    return net, scaler


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def nn_predict(
    net: IDSNet,
    X_scaled: np.ndarray,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on scaled input.

    Parameters
    ----------
    net : IDSNet
    X_scaled : np.ndarray, shape (n, 75)
    device : optional device override.
        If None, auto-detected from the model's current parameter device.
    threshold : float
        Classification threshold (default 0.5).

    Returns
    -------
    (y_pred, probs)
        y_pred : np.ndarray, dtype int, values in {0, 1}
        probs  : np.ndarray, dtype float, values in [0, 1]
    """
    if device is None:
        # Detect device from model's parameters (handles both CPU and GPU)
        try:
            device = next(net.parameters()).device
        except StopIteration:
            device = _get_device()
    net.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        logits = net(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    y_pred = (probs >= threshold).astype(int)
    return y_pred, probs



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_pos_weight(n_normal: int, n_attack: int) -> float:
    """
    Compute pos_weight = N_negative / N_positive for BCEWithLogitsLoss.

    Parameters
    ----------
    n_normal : int   — count of Normal (negative) samples in TRAIN
    n_attack : int   — count of Attack (positive) samples in TRAIN

    Returns
    -------
    float
    """
    if n_attack <= 0:
        raise ValueError(f"n_attack must be > 0, got {n_attack}")
    if n_normal <= 0:
        raise ValueError(f"n_normal must be > 0, got {n_normal}")
    return float(n_normal) / float(n_attack)


def _generate_nn_configs() -> list[dict[str, Any]]:
    """Generate all 8 NN grid configurations in deterministic order."""
    configs = []
    for hidden_sizes, lr, wd in itertools.product(
        NN_TUNING_GRID["hidden_sizes"],
        NN_TUNING_GRID["learning_rate"],
        NN_TUNING_GRID["weight_decay"],
    ):
        configs.append(
            {
                "hidden_sizes": list(hidden_sizes),
                "learning_rate": lr,
                "weight_decay": wd,
            }
        )
    return configs


def _nn_grid_size() -> int:
    return (
        len(NN_TUNING_GRID["hidden_sizes"])
        * len(NN_TUNING_GRID["learning_rate"])
        * len(NN_TUNING_GRID["weight_decay"])
    )


def _config_str(config: dict[str, Any]) -> str:
    return (
        f"hidden_sizes={config.get('hidden_sizes')} "
        f"lr={config.get('learning_rate')} "
        f"wd={config.get('weight_decay')}"
    )


def _validate_nn_inputs(
    X: np.ndarray, y: np.ndarray, config: dict[str, Any]
) -> None:
    """Validate X/y shapes and basic NN config constraints."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X rows ({len(X)}) != y length ({len(y)})")
    if X.shape[0] == 0:
        raise ValueError("Empty training set (0 rows).")
    unique_cls = np.unique(y)
    if len(unique_cls) < 2:
        raise ValueError(
            f"Training set contains only one class: {unique_cls}. "
            "NN requires at least two classes."
        )
    lr = config.get("learning_rate", 0.001)
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError(f"learning_rate must be > 0, got {lr!r}")
    wd = config.get("weight_decay", 0.0001)
    if not isinstance(wd, (int, float)) or wd < 0:
        raise ValueError(f"weight_decay must be >= 0, got {wd!r}")
    hidden_sizes = config.get("hidden_sizes", [128, 64])
    if not isinstance(hidden_sizes, (list, tuple)) or len(hidden_sizes) < 1:
        raise ValueError(f"hidden_sizes must be a non-empty list, got {hidden_sizes!r}")
