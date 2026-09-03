"""
src/models/base_models/comparator.py
--------------------------------------
Single deterministic model-configuration comparator for Sprint 5.

Design
------
One function, ``compare_model_configs``, handles all four model types using
the frozen tie-breaking hierarchy:

    1. Higher mean Macro-F1 (tolerance 1e-6)
    2. Lower std Macro-F1 (tolerance 1e-6)
    3. Model-specific simplicity ordering (see below)
    4. Deterministic serialised config ordering (json.dumps, sort_keys=True)

Model-specific simplicity
--------------------------
DT:   max_depth (5<10<20<None)  →  min_samples_leaf  →  criterion (gini<entropy)
RF:   n_estimators  →  max_depth (10<20<None)  →  min_samples_leaf  →  max_features (sqrt<0.3)
SVM:  C (0.01<0.1<1.0<10.0)
NN:   param_count ([128,64]<[256,128])  →  weight_decay (desc)  →  learning_rate (asc)

None (max_depth) is treated as infinity (largest/deepest) for DT/RF.

Returns
-------
+1 : result_a is preferred (better)
 0 : exactly equal under all criteria
-1 : result_b is preferred (better)

This function is unit-tested exhaustively.  Do NOT duplicate this logic
in individual model modules.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# Comparison tolerance for floating-point metric equality
_F1_TOLERANCE = 1e-6

# Canonical ordering for model types
_VALID_MODEL_TYPES = {"dt", "rf", "svm", "nn"}

# DT max_depth ordering:  5 < 10 < 20 < None (None = ∞)
_DT_DEPTH_ORDER = {5: 0, 10: 1, 20: 2, None: 3}

# RF max_depth ordering:  10 < 20 < None (None = ∞)
_RF_DEPTH_ORDER = {10: 0, 20: 1, None: 2}

# RF max_features ordering:  sqrt < 0.3
_RF_FEATURES_ORDER = {"sqrt": 0, 0.3: 1}

# DT criterion ordering:  gini < entropy
_DT_CRITERION_ORDER = {"gini": 0, "entropy": 1}

# NN hidden_sizes param count
_NN_PARAM_COUNT = {
    (128, 64): 75 * 128 + 128 + 128 * 64 + 64 + 64 * 1 + 1,
    (256, 128): 75 * 256 + 256 + 256 * 128 + 128 + 128 * 1 + 1,
}


def compare_model_configs(
    result_a: Any,
    result_b: Any,
    model_type: str,
) -> int:
    """
    Compare two CV results using the frozen tie-breaking hierarchy.

    Parameters
    ----------
    result_a, result_b : CVSummary or object with attributes
        ``mean_macro_f1``, ``std_macro_f1``, ``config`` (dict).
    model_type : str
        One of {"dt", "rf", "svm", "nn"}.

    Returns
    -------
    int
        +1 if result_a is preferred,
         0 if identical under all criteria,
        -1 if result_b is preferred.

    Raises
    ------
    ValueError
        If model_type is not in the valid set.
    """
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Must be one of {sorted(_VALID_MODEL_TYPES)}."
        )

    # ------------------------------------------------------------------ #
    # Step 1: Higher mean Macro-F1 wins                                   #
    # ------------------------------------------------------------------ #
    f1_a = float(result_a.mean_macro_f1)
    f1_b = float(result_b.mean_macro_f1)

    if f1_a > f1_b + _F1_TOLERANCE:
        logger.debug("Comparator step 1: a wins by mean_macro_f1 (%.8f > %.8f)", f1_a, f1_b)
        return 1
    if f1_b > f1_a + _F1_TOLERANCE:
        logger.debug("Comparator step 1: b wins by mean_macro_f1 (%.8f > %.8f)", f1_b, f1_a)
        return -1

    # ------------------------------------------------------------------ #
    # Step 2: Lower std Macro-F1 wins                                     #
    # ------------------------------------------------------------------ #
    std_a = float(result_a.std_macro_f1)
    std_b = float(result_b.std_macro_f1)

    if std_a < std_b - _F1_TOLERANCE:
        logger.debug("Comparator step 2: a wins by std_macro_f1 (%.8f < %.8f)", std_a, std_b)
        return 1
    if std_b < std_a - _F1_TOLERANCE:
        logger.debug("Comparator step 2: b wins by std_macro_f1 (%.8f < %.8f)", std_b, std_a)
        return -1

    # ------------------------------------------------------------------ #
    # Step 3: Model-specific simplicity ordering                          #
    # ------------------------------------------------------------------ #
    cfg_a: dict[str, Any] = dict(result_a.config)
    cfg_b: dict[str, Any] = dict(result_b.config)

    simplicity = _simplicity_compare(cfg_a, cfg_b, model_type)
    if simplicity != 0:
        logger.debug("Comparator step 3: %s simplicity -> %+d", model_type, simplicity)
        return simplicity

    # ------------------------------------------------------------------ #
    # Step 4: Deterministic serialised config ordering                    #
    # ------------------------------------------------------------------ #
    serial_a = json.dumps(cfg_a, sort_keys=True, default=str)
    serial_b = json.dumps(cfg_b, sort_keys=True, default=str)

    if serial_a < serial_b:
        logger.debug("Comparator step 4: a wins by serialised config ordering")
        return 1
    if serial_a > serial_b:
        logger.debug("Comparator step 4: b wins by serialised config ordering")
        return -1

    logger.debug("Comparator: completely equal")
    return 0


# ---------------------------------------------------------------------------
# Internal: model-specific simplicity
# ---------------------------------------------------------------------------


def _simplicity_compare(
    cfg_a: dict[str, Any],
    cfg_b: dict[str, Any],
    model_type: str,
) -> int:
    """
    Model-specific simplicity ordering.

    Returns +1 (a simpler), -1 (b simpler), 0 (equal).
    """
    if model_type == "dt":
        return _dt_simplicity(cfg_a, cfg_b)
    if model_type == "rf":
        return _rf_simplicity(cfg_a, cfg_b)
    if model_type == "svm":
        return _svm_simplicity(cfg_a, cfg_b)
    if model_type == "nn":
        return _nn_simplicity(cfg_a, cfg_b)
    return 0  # unreachable — validated in compare_model_configs


def _cmp(a, b) -> int:
    """Simple three-way comparator: returns -1, 0, or +1 for a vs b."""
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


def _dt_simplicity(a: dict, b: dict) -> int:
    """
    DT simplicity: smaller max_depth, then smaller min_samples_leaf,
    then criterion (gini < entropy).
    None is treated as the deepest (largest) option.
    """
    # 1) max_depth: smaller is simpler
    d_a = _DT_DEPTH_ORDER.get(a.get("max_depth"), 3)
    d_b = _DT_DEPTH_ORDER.get(b.get("max_depth"), 3)
    r = _cmp(d_a, d_b)
    if r != 0:
        return -r  # lower rank = simpler = a preferred → return +1

    # 2) min_samples_leaf: smaller is simpler
    msl_a = a.get("min_samples_leaf", 1)
    msl_b = b.get("min_samples_leaf", 1)
    r = _cmp(msl_a, msl_b)
    if r != 0:
        return -r

    # 3) criterion: gini < entropy
    c_a = _DT_CRITERION_ORDER.get(a.get("criterion", "gini"), 0)
    c_b = _DT_CRITERION_ORDER.get(b.get("criterion", "gini"), 0)
    r = _cmp(c_a, c_b)
    if r != 0:
        return -r

    return 0


def _rf_simplicity(a: dict, b: dict) -> int:
    """
    RF simplicity: fewer n_estimators, then smaller max_depth (10<20<None),
    then smaller min_samples_leaf, then max_features (sqrt<0.3).
    """
    # 1) n_estimators
    n_a = a.get("n_estimators", 100)
    n_b = b.get("n_estimators", 100)
    r = _cmp(n_a, n_b)
    if r != 0:
        return -r

    # 2) max_depth
    d_a = _RF_DEPTH_ORDER.get(a.get("max_depth"), 2)
    d_b = _RF_DEPTH_ORDER.get(b.get("max_depth"), 2)
    r = _cmp(d_a, d_b)
    if r != 0:
        return -r

    # 3) min_samples_leaf
    msl_a = a.get("min_samples_leaf", 1)
    msl_b = b.get("min_samples_leaf", 1)
    r = _cmp(msl_a, msl_b)
    if r != 0:
        return -r

    # 4) max_features: sqrt < 0.3
    mf_a = _RF_FEATURES_ORDER.get(a.get("max_features", "sqrt"), 0)
    mf_b = _RF_FEATURES_ORDER.get(b.get("max_features", "sqrt"), 0)
    r = _cmp(mf_a, mf_b)
    if r != 0:
        return -r

    return 0


def _svm_simplicity(a: dict, b: dict) -> int:
    """
    SVM simplicity: smaller C is simpler.
    0.01 < 0.1 < 1.0 < 10.0
    """
    c_a = float(a.get("C", 1.0))
    c_b = float(b.get("C", 1.0))
    r = _cmp(c_a, c_b)
    if r != 0:
        return -r  # smaller C = simpler = preferred
    return 0


def _nn_simplicity(a: dict, b: dict) -> int:
    """
    NN simplicity:
    1. Fewer trainable parameters ([128,64] simpler than [256,128])
    2. Larger weight_decay (more regularisation = conceptually simpler fit)
    3. Smaller learning_rate
    """
    # 1) Param count
    hs_a = tuple(a.get("hidden_sizes", [128, 64]))
    hs_b = tuple(b.get("hidden_sizes", [128, 64]))
    pc_a = _NN_PARAM_COUNT.get(hs_a, _compute_nn_params(hs_a))
    pc_b = _NN_PARAM_COUNT.get(hs_b, _compute_nn_params(hs_b))
    r = _cmp(pc_a, pc_b)
    if r != 0:
        return -r  # fewer params = simpler

    # 2) Larger weight_decay is preferred (more regularisation)
    wd_a = float(a.get("weight_decay", 0.0001))
    wd_b = float(b.get("weight_decay", 0.0001))
    r = _cmp(wd_a, wd_b)
    if r != 0:
        return r  # larger wd = preferred → return +1 when wd_a > wd_b

    # 3) Smaller learning_rate is preferred
    lr_a = float(a.get("learning_rate", 0.001))
    lr_b = float(b.get("learning_rate", 0.001))
    r = _cmp(lr_a, lr_b)
    if r != 0:
        return -r  # smaller lr = simpler

    return 0


def _compute_nn_params(hidden_sizes: tuple[int, ...]) -> int:
    """Compute total trainable parameters for input_dim=75, output_dim=1."""
    input_dim = 75
    output_dim = 1
    prev = input_dim
    total = 0
    for h in hidden_sizes:
        total += prev * h + h  # weights + biases
        prev = h
    total += prev * output_dim + output_dim
    return total
