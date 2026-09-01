"""
src/models/stacking/__init__.py
--------------------------------
Public API for Sprint 6 OOF stacking module.
"""

from src.models.stacking.oof_runner import (
    OOF_SEED,
    OOF_N_SPLITS,
    OOF_FIXED_EPOCH_COUNT,
    OOF_POS_WEIGHT,
    SCALING_LIMITATION_TEXT,
    make_oof_folds,
    run_oof_seed,
    set_all_seeds,
)
from src.models.stacking.meta_learner import (
    META_CONFIG,
    META_EVALUATION_LIMITATION_TEXT,
    SPRINT5_RF_REFERENCE,
    SPRINT5_RF_REFERENCE_LABEL,
    train_meta_learner,
    predict_meta,
    compute_oof_metrics,
    compute_h1_summary,
)
from src.models.stacking.artifacts import (
    save_oof_predictions,
    save_fold_assignments,
    save_seed_metrics,
    save_seed_metadata,
    save_meta_learner_checkpoint,
    save_meta_learner_metadata,
    load_meta_learner_checkpoint,
)

__all__ = [
    # Constants
    "OOF_SEED",
    "OOF_N_SPLITS",
    "OOF_FIXED_EPOCH_COUNT",
    "OOF_POS_WEIGHT",
    "SCALING_LIMITATION_TEXT",
    "META_CONFIG",
    "META_EVALUATION_LIMITATION_TEXT",
    "SPRINT5_RF_REFERENCE",
    "SPRINT5_RF_REFERENCE_LABEL",
    # OOF runner
    "set_all_seeds",
    "make_oof_folds",
    "run_oof_seed",
    # Meta-learner
    "train_meta_learner",
    "predict_meta",
    "compute_oof_metrics",
    "compute_h1_summary",
    # Artifacts
    "save_oof_predictions",
    "save_fold_assignments",
    "save_seed_metrics",
    "save_seed_metadata",
    "save_meta_learner_checkpoint",
    "save_meta_learner_metadata",
    "load_meta_learner_checkpoint",
]
