"""
src/models/autoencoder/ae_trainer.py
--------------------------------------
Sprint 7 — EXP_AE_V1 AE Trainer.

Training protocol (FINAL):
    - Train on Normal AE-fit subset (40,320 rows)
    - Monitor on Normal monitor subset (4,480 rows) for early stopping
    - patience = 5
    - Max epochs = 100
    - Restore best monitor-loss weights

Final refit:
    - Train on ALL 44,800 Normal TRAIN rows for best_epoch epochs
    - No early stopping
    - No VALIDATION data used

Scaler:
    - Fresh StandardScaler fitted ONLY on AE-fit subset
    - Same scaler reused for: monitor transform, final refit, VAL calibration

Seed policy:
    set_all_seeds(42) before training.

Data-access invariant:
    No Attack-labeled rows enter AE training tensors.
    No VALIDATION data enters AE weights.
"""

from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.autoencoder.ae_model import Autoencoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen training constants
# ---------------------------------------------------------------------------

AE_SEED: int = 42
AE_LR: float = 0.001
AE_WEIGHT_DECAY: float = 0.0001
AE_BATCH_SIZE: int = 256
# Decision 4 (Revised): AE_MAX_EPOCHS raised from 100 -> 150, based on
# EXP_AE_V1 diagnostic (300-epoch run) showing early stopping at epoch 138
# with best_epoch=133, under the same seed, architecture, monitor split,
# scaler, and training configuration. The 100-epoch run reached max_epochs
# while monitor MSE was still decreasing; convergence occurred at ep133.
AE_MAX_EPOCHS: int = 150
AE_PATIENCE: int = 5
AE_MONITOR_FRACTION: float = 0.10   # 10% of Normal TRAIN → monitor
AE_MONITOR_SEED: int = 42


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int = AE_SEED) -> None:
    """Set Python, NumPy, PyTorch, and CUDA seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MonitorSplit:
    ae_fit_indices: np.ndarray       # TRAIN row indices for AE training
    monitor_indices: np.ndarray      # TRAIN row indices for monitor
    split_seed: int = AE_MONITOR_SEED

    def to_dict(self) -> dict:
        return {
            "ae_fit_count": int(len(self.ae_fit_indices)),
            "monitor_count": int(len(self.monitor_indices)),
            "split_seed": self.split_seed,
            "ae_fit_fraction": round(len(self.ae_fit_indices) /
                                     (len(self.ae_fit_indices) + len(self.monitor_indices)), 4),
            "monitor_fraction": round(len(self.monitor_indices) /
                                      (len(self.ae_fit_indices) + len(self.monitor_indices)), 4),
        }


@dataclass
class EpochRecord:
    epoch: int
    ae_fit_mse: float
    monitor_mse: float


@dataclass
class AETrainResult:
    best_epoch: int
    final_epoch_count: int
    training_history: list[EpochRecord]
    scaler: StandardScaler
    model_state: dict                 # best-weights state_dict
    runtime_seconds: float
    device: str
    monitor_split: MonitorSplit
    ae_fit_rows: int
    monitor_rows: int

    def epoch_diagnostics(self) -> dict:
        return {
            "best_epoch": self.best_epoch,
            "final_epoch_count": self.final_epoch_count,
            "min_monitor_mse": min(r.monitor_mse for r in self.training_history),
            "final_ae_fit_mse": self.training_history[-1].ae_fit_mse if self.training_history else None,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "device": self.device,
            "ae_fit_rows": self.ae_fit_rows,
            "monitor_rows": self.monitor_rows,
        }


# ---------------------------------------------------------------------------
# Monitor split creation
# ---------------------------------------------------------------------------

def create_monitor_split(
    normal_train_indices: np.ndarray,
    monitor_fraction: float = AE_MONITOR_FRACTION,
    seed: int = AE_MONITOR_SEED,
) -> MonitorSplit:
    """
    Create a fixed 90/10 split of Normal TRAIN indices.

    Invariants (enforced by caller leakage tests):
        - All indices must be from Normal TRAIN (label == 0)
        - ae_fit ∩ monitor = empty
        - ae_fit ∪ monitor = all normal_train_indices

    Parameters
    ----------
    normal_train_indices : np.ndarray
        Row indices (into the full TRAIN DataFrame) of Normal rows only.
    """
    from sklearn.model_selection import train_test_split
    ae_fit_idx, monitor_idx = train_test_split(
        normal_train_indices,
        test_size=monitor_fraction,
        random_state=seed,
        shuffle=True,
    )
    return MonitorSplit(
        ae_fit_indices=np.array(ae_fit_idx),
        monitor_indices=np.array(monitor_idx),
        split_seed=seed,
    )


# ---------------------------------------------------------------------------
# Core trainer
# ---------------------------------------------------------------------------

class AETrainer:
    """
    Trains the Sprint 7 Autoencoder following the frozen protocol.

    Parameters
    ----------
    seed : int
        Master seed. Passed to set_all_seeds().
    device : str | None
        'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(self, seed: int = AE_SEED, device: Optional[str] = None) -> None:
        self.seed = seed
        self.device = (
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    # ------------------------------------------------------------------
    def _make_loader(
        self,
        X: np.ndarray,
        shuffle: bool,
        seed: Optional[int] = None,
    ) -> DataLoader:
        t = torch.tensor(X, dtype=torch.float32)
        ds = TensorDataset(t)
        gen = None
        if shuffle and seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
        return DataLoader(
            ds,
            batch_size=AE_BATCH_SIZE,
            shuffle=shuffle,
            generator=gen,
            drop_last=False,
        )

    # ------------------------------------------------------------------
    def _epoch_mse(self, model: Autoencoder, loader: DataLoader) -> float:
        """Compute mean MSE over all batches in loader (eval mode)."""
        model.eval()
        total, n = 0.0, 0
        criterion = nn.MSELoss(reduction="mean")
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                out = model(batch)
                total += criterion(out, batch).item() * len(batch)
                n += len(batch)
        return total / n if n > 0 else float("nan")

    # ------------------------------------------------------------------
    def fit(
        self,
        X_ae_fit: np.ndarray,
        X_monitor: np.ndarray,
        monitor_split: MonitorSplit,
    ) -> AETrainResult:
        """
        Phase 1 training: AE-fit subset with monitor-based early stopping.

        Parameters
        ----------
        X_ae_fit : np.ndarray  shape [40320, 75]  SCALED Normal AE-fit rows
        X_monitor : np.ndarray shape [4480, 75]   SCALED Normal monitor rows
        monitor_split : MonitorSplit              provenance record

        Returns
        -------
        AETrainResult  with best_epoch, model_state, scaler (passed in), etc.
        """
        assert X_ae_fit.shape[1] == 75, f"Expected 75 features, got {X_ae_fit.shape[1]}"
        assert X_monitor.shape[1] == 75

        set_all_seeds(self.seed)

        model = Autoencoder().to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=AE_LR, weight_decay=AE_WEIGHT_DECAY
        )
        criterion = nn.MSELoss(reduction="mean")

        fit_loader = self._make_loader(X_ae_fit, shuffle=True, seed=self.seed)
        mon_loader = self._make_loader(X_monitor, shuffle=False)

        best_monitor_mse = float("inf")
        best_epoch = 0
        best_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
        history: list[EpochRecord] = []

        t0 = time.perf_counter()
        for epoch in range(1, AE_MAX_EPOCHS + 1):
            # --- train ---
            model.train()
            for (batch,) in fit_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch)
                loss.backward()
                optimizer.step()

            # --- evaluate ---
            ae_fit_mse = self._epoch_mse(model, fit_loader)
            monitor_mse = self._epoch_mse(model, mon_loader)
            history.append(EpochRecord(epoch, ae_fit_mse, monitor_mse))

            logger.debug(
                "Epoch %3d | ae_fit_mse=%.6f | monitor_mse=%.6f",
                epoch, ae_fit_mse, monitor_mse,
            )

            # --- early stopping ---
            if monitor_mse < best_monitor_mse:
                best_monitor_mse = monitor_mse
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= AE_PATIENCE:
                    logger.info(
                        "Early stopping at epoch %d (best=%d, monitor_mse=%.6f)",
                        epoch, best_epoch, best_monitor_mse,
                    )
                    break

        runtime = time.perf_counter() - t0
        logger.info(
            "Phase-1 training complete | best_epoch=%d | monitor_mse=%.6f | %.1fs",
            best_epoch, best_monitor_mse, runtime,
        )

        return AETrainResult(
            best_epoch=best_epoch,
            final_epoch_count=best_epoch,
            training_history=history,
            scaler=None,           # scaler passed separately; set by caller
            model_state=best_weights,
            runtime_seconds=runtime,
            device=self.device,
            monitor_split=monitor_split,
            ae_fit_rows=len(X_ae_fit),
            monitor_rows=len(X_monitor),
        )

    # ------------------------------------------------------------------
    def final_refit(
        self,
        X_normal_train_all: np.ndarray,
        best_epoch: int,
    ) -> Autoencoder:
        """
        Phase 2 final refit on ALL 44,800 Normal TRAIN rows.

        - Trains for exactly best_epoch epochs
        - No early stopping
        - No VALIDATION data
        - Same seed = 42

        Parameters
        ----------
        X_normal_train_all : np.ndarray  shape [44800, 75]  SCALED Normal TRAIN
        best_epoch : int                 from Phase-1 result

        Returns
        -------
        Autoencoder  in eval() mode with final weights
        """
        assert X_normal_train_all.shape[1] == 75
        assert best_epoch >= 1

        set_all_seeds(self.seed)

        model = Autoencoder().to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=AE_LR, weight_decay=AE_WEIGHT_DECAY
        )
        criterion = nn.MSELoss(reduction="mean")
        loader = self._make_loader(X_normal_train_all, shuffle=True, seed=self.seed)

        t0 = time.perf_counter()
        for epoch in range(1, best_epoch + 1):
            model.train()
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch)
                loss.backward()
                optimizer.step()

        runtime = time.perf_counter() - t0
        model.eval()
        logger.info(
            "Final refit complete | epochs=%d | %.1fs | rows=%d",
            best_epoch, runtime, len(X_normal_train_all),
        )
        return model
