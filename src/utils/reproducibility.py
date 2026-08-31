"""
src/utils/reproducibility.py
-----------------------------
Central seed-setting utility for the UNSW-NB15 IDS project.

All stochastic project components must call set_all_seeds() rather than
scattering ad-hoc random.seed() / np.random.seed() calls.

Primary experiment seeds (H1 three-seed protocol): 42, 123, 2024

Sprint 1 does not use random seeds (no model training); the function is
created here for use in later sprints.

The torch import is wrapped in a try/except so this module degrades
gracefully if torch is unavailable in a lightweight test environment.
"""

import random
import numpy as np


def set_all_seeds(seed: int) -> None:
    """
    Set all relevant random seeds for full reproducibility.

    Affects:
    - Python stdlib `random`
    - NumPy
    - PyTorch (CPU and CUDA), if available

    Parameters
    ----------
    seed : int
        The random seed to apply globally.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not available in this environment — acceptable for Sprint 1
