# src/preprocessing/__init__.py
"""
Public API for the UNSW-NB15 IDS preprocessing module.

Sprint 1 (FROZEN):
    loader, attack_cat_canonicalization, schema_audit,
    schema_validator, protected_unseen_attack, withheld_candidate

Sprint 2 (FROZEN):
    cleaning, encoding, scaling, processed_dataset, preprocessing_pipeline,
    exceptions

Sprint 3:
    split_protocol
"""

# Sprint 2 public API
from original_split_benchmark.src.preprocessing.exceptions import (
    NonFiniteValueError,
    PreprocessingNotFittedError,
    PreprocessingSchemaError,
)
from original_split_benchmark.src.preprocessing.cleaning import (
    CleanedSplit,
    separate_target_and_features,
    validate_required_columns,
    detect_nonfinite,
    LABEL_COL,
    ATTACK_CAT_COL,
    CATEGORICAL_COLS,
    EXCLUDE_COLS,
)
from original_split_benchmark.src.preprocessing.encoding import (
    FittedEncoder,
    fit_encoder,
    transform_encoder,
    get_feature_names,
    get_encoder_metadata,
)
from original_split_benchmark.src.preprocessing.scaling import (
    FittedScaler,
    fit_scaler,
    transform_scaler,
    get_scaler_metadata,
)
from original_split_benchmark.src.preprocessing.processed_dataset import (
    ProcessedDataset,
    VIEW_SCALED,
    VIEW_UNSCALED,
)
from original_split_benchmark.src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

# Sprint 3 public API
from original_split_benchmark.src.preprocessing.split_protocol import (
    TrainValSplitResult,
    SplitIntegrityReport,
    create_train_val_split,
    verify_split_integrity,
    build_split_provenance,
    WITHHELD_ATTACK,
    NORMAL_CAT,
    NORMAL_TRAIN_FRAC,
    NORMAL_VAL_FRAC,
    ATTACK_CAT_COL,
    LABEL_COL,
)
