"""
src/models/base_models/__init__.py
-----------------------------------
Public API for Sprint 5 base models.
"""

from original_split_benchmark.src.models.base_models.comparator import compare_model_configs
from original_split_benchmark.src.models.base_models.cv_utils import (
    CVSummary,
    FoldMetrics,
    aggregate_cv_results,
    compute_fold_metrics,
    make_model_skf,
)
from original_split_benchmark.src.models.base_models.preprocessing import (
    EXPECTED_FEATURE_COUNT,
    FEATURE_SET_ID,
    build_feature_matrix,
    fit_scaler,
    load_selected_features,
)
from original_split_benchmark.src.models.base_models.decision_tree import (
    DTConfig,
    DT_BASELINE_CONFIG,
    DT_TUNING_GRID,
    run_dt_baseline,
    run_dt_cv,
    run_dt_tuning,
    refit_dt,
)
from original_split_benchmark.src.models.base_models.random_forest import (
    RFConfig,
    RF_BASELINE_CONFIG,
    RF_TUNING_GRID,
    run_rf_baseline,
    run_rf_cv,
    run_rf_tuning,
    refit_rf,
)
from original_split_benchmark.src.models.base_models.linear_svc import (
    SVMConfig,
    SVM_BASELINE_CONFIG,
    SVM_TUNING_C_VALUES,
    run_svm_baseline,
    run_svm_cv,
    run_svm_tuning,
    refit_svm,
)
from original_split_benchmark.src.models.base_models.neural_network import (
    IDSNet,
    NNConfig,
    NNEpochDiagnostics,
    NN_BASELINE_CONFIG,
    NN_TUNING_GRID,
    TRAIN_POS_WEIGHT,
    TRAIN_N_NORMAL,
    TRAIN_N_ATTACK,
    compute_pos_weight,
    nn_predict,
    run_nn_baseline,
    run_nn_cv,
    run_nn_tuning,
    refit_nn,
)

__all__ = [
    # Preprocessing
    "load_selected_features",
    "build_feature_matrix",
    "fit_scaler",
    "EXPECTED_FEATURE_COUNT",
    "FEATURE_SET_ID",
    # CV
    "FoldMetrics",
    "CVSummary",
    "compute_fold_metrics",
    "aggregate_cv_results",
    "make_model_skf",
    # Comparator
    "compare_model_configs",
    # Decision Tree
    "DTConfig",
    "DT_BASELINE_CONFIG",
    "DT_TUNING_GRID",
    "run_dt_baseline",
    "run_dt_cv",
    "run_dt_tuning",
    "refit_dt",
    # Random Forest
    "RFConfig",
    "RF_BASELINE_CONFIG",
    "RF_TUNING_GRID",
    "run_rf_baseline",
    "run_rf_cv",
    "run_rf_tuning",
    "refit_rf",
    # LinearSVC
    "SVMConfig",
    "SVM_BASELINE_CONFIG",
    "SVM_TUNING_C_VALUES",
    "run_svm_baseline",
    "run_svm_cv",
    "run_svm_tuning",
    "refit_svm",
    # Neural Network
    "IDSNet",
    "NNConfig",
    "NNEpochDiagnostics",
    "NN_BASELINE_CONFIG",
    "NN_TUNING_GRID",
    "TRAIN_POS_WEIGHT",
    "TRAIN_N_NORMAL",
    "TRAIN_N_ATTACK",
    "compute_pos_weight",
    "nn_predict",
    "run_nn_baseline",
    "run_nn_cv",
    "run_nn_tuning",
    "refit_nn",
]
