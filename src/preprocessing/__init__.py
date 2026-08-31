# src/preprocessing/__init__.py
"""
Public API for the UNSW-NB15 IDS preprocessing module.

Sprint 1 (FROZEN):
    loader, attack_cat_canonicalization, schema_audit,
    schema_validator, protected_unseen_attack, withheld_candidate

Sprint 2 (this sprint):
    cleaning, encoding, scaling, processed_dataset, preprocessing_pipeline
    exceptions
"""

# Sprint 2 public API
from src.preprocessing.exceptions import (
    NonFiniteValueError,
    PreprocessingNotFittedError,
    PreprocessingSchemaError,
)
from src.preprocessing.cleaning import (
    CleanedSplit,
    separate_target_and_features,
    validate_required_columns,
    detect_nonfinite,
    LABEL_COL,
    ATTACK_CAT_COL,
    CATEGORICAL_COLS,
    EXCLUDE_COLS,
)
from src.preprocessing.encoding import (
    FittedEncoder,
    fit_encoder,
    transform_encoder,
    get_feature_names,
    get_encoder_metadata,
)
from src.preprocessing.scaling import (
    FittedScaler,
    fit_scaler,
    transform_scaler,
    get_scaler_metadata,
)
from src.preprocessing.processed_dataset import (
    ProcessedDataset,
    VIEW_SCALED,
    VIEW_UNSCALED,
)
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
