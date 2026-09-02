"""
src/models/autoencoder/ae_model.py
------------------------------------
Sprint 7 — EXP_AE_V1 Benign-Only Autoencoder.

Architecture (FINAL — frozen by design):
    Input:      75
    Encoder:    75 → 12   (Linear + ReLU)
                12 →  6   (Linear + ReLU)
    Bottleneck:  6
    Decoder:     6 → 12   (Linear + ReLU)
                12 → 75   (Linear — no activation)
    Output:     75

    No BatchNorm.  No Dropout.  No VAE.

Loss:
    MSELoss (mean over batch and features).

Per-sample reconstruction error:
    RE(x) = (1/75) × Σ_{j=1}^{75} (x_j − x̂_j)²
    = mean over features, one scalar per row.
    NOT sum.  NOT MAE.  NOT max.

Anomaly rule (conceptual):
    RE(x) > τ  →  ANOMALY
    RE(x) ≤ τ  →  NORMAL
    (τ chosen in Sprint 8; all candidates computed in Sprint 7)

Scaler-space limitation (MANDATORY):
    AE operates in a Normal-TRAIN-scaled feature space distinct from the
    full-TRAIN-scaled space used by DT/RF/SVM/NN. Sprint 8 fusion must
    account for this representation difference when combining the AE
    reconstruction error with the supervised branch outputs.

Single-seed limitation (MANDATORY):
    Sprint 7 uses a single AE training seed (42). No multi-seed stability
    estimate exists for AE reconstruction error or threshold values. This
    is an accepted scope limitation and not a null result.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Frozen architecture constants
# ---------------------------------------------------------------------------

AE_INPUT_DIM: int = 75
AE_ENCODER_SIZES: list[int] = [12, 6]
AE_DECODER_SIZES: list[int] = [12, 75]
AE_N_PARAMS: int = 2049  # (75*12+12)+(12*6+6)+(6*12+12)+(12*75+75)


class Autoencoder(nn.Module):
    """
    Benign-only Autoencoder for Sprint 7 (EXP_AE_V1).

    Fixed architecture: 75 → 12 → 6 → 12 → 75
    Hidden activation:  ReLU
    Output activation:  Linear (none)
    BatchNorm:          NO
    Dropout:            NO

    Parameters
    ----------
    input_dim : int
        Must be 75 (frozen by EXP_MI_V1_1 feature contract).
    """

    def __init__(self, input_dim: int = AE_INPUT_DIM) -> None:
        super().__init__()
        if input_dim != AE_INPUT_DIM:
            raise ValueError(
                f"AE input_dim must be {AE_INPUT_DIM} (frozen EXP_MI_V1_1 "
                f"feature count), got {input_dim}."
            )

        # Encoder: 75 → 12 → 6
        self.encoder = nn.Sequential(
            nn.Linear(75, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
        )

        # Decoder: 6 → 12 → 75
        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 75),
            # No output activation — Linear
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode. Output shape == input shape."""
        return self.decoder(self.encoder(x))

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return bottleneck representation (shape: [N, 6])."""
        return self.encoder(x)

    # ------------------------------------------------------------------
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample MSE reconstruction error.

        RE(x) = (1/75) × Σ_j (x_j − x̂_j)²

        Returns
        -------
        torch.Tensor
            Shape [N] — one scalar per sample. Mean over features, NOT sum.
        """
        x_hat = self.forward(x)
        # mean over feature dimension → shape [N]
        return ((x - x_hat) ** 2).mean(dim=1)

    # ------------------------------------------------------------------
    def architecture_dict(self) -> dict:
        """Return serialisable architecture description."""
        return {
            "input_dim": AE_INPUT_DIM,
            "encoder": [AE_INPUT_DIM] + AE_ENCODER_SIZES,
            "bottleneck": AE_ENCODER_SIZES[-1],
            "decoder": AE_ENCODER_SIZES[-1:] + AE_DECODER_SIZES,
            "hidden_activation": "ReLU",
            "output_activation": "Linear (none)",
            "batchnorm": False,
            "dropout": False,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters())
