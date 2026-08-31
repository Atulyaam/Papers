"""
src/feature_selection/__init__.py
-----------------------------------
Public API for Sprint 4: Mutual Information feature selection.
"""

from src.feature_selection.mi_selector import (
    MIConfig,
    MIResult,
    MISelectorError,
    build_discrete_mask,
    compute_mi_scores,
    get_source_family,
    rank_features,
    select_top_k,
)
from src.feature_selection.k_selector import (
    InnerCVConfig,
    KSelectionResult,
    KSelectionSanity,
    run_k_selection_cv,
    select_best_k,
    check_selection_sanity,
)

__all__ = [
    # mi_selector
    "MIConfig",
    "MIResult",
    "MISelectorError",
    "build_discrete_mask",
    "compute_mi_scores",
    "get_source_family",
    "rank_features",
    "select_top_k",
    # k_selector
    "InnerCVConfig",
    "KSelectionResult",
    "KSelectionSanity",
    "run_k_selection_cv",
    "select_best_k",
    "check_selection_sanity",
]
