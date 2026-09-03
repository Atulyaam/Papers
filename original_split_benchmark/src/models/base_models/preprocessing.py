"""
src/models/base_models/preprocessing.py
-----------------------------------------
Sprint 5 model-training preprocessing utilities.

Responsibilities
----------------
1. Load the frozen 75 MI-selected feature names from EXP_MI_V1_1.
2. Build the feature matrix by selecting exactly those 75 columns from
   an encoded (unscaled) DataFrame.
3. Fit a fresh StandardScaler on a training portion for SVM/NN inner-CV
   folds and final TRAIN refits.

Design constraints (enforced)
------------------------------
- Feature set is FROZEN at 75 features from EXP_MI_V1_1.  Do not modify.
- No scaler is fitted on validation, test, or protected backdoor data.
- These utilities operate on already-encoded data (output of
  PreprocessingPipeline).  They do NOT own raw-data loading.
- The Sprint 2 smoke-test scaler is NEVER reused here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Canonical feature-set artifact path (relative to project root)
_DEFAULT_FEATURES_PATH = (
    "results/feature_selection/EXP_MI_V1_1/selected_features.json"
)

# Expected feature count — never change without a protocol amendment.
EXPECTED_FEATURE_COUNT = 75
FEATURE_SET_ID = "EXP_MI_V1_1"


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def load_selected_features(
    path: str | Path | None = None,
) -> list[str]:
    """
    Load the frozen 75 MI-selected feature names from EXP_MI_V1_1.

    Parameters
    ----------
    path : str | Path | None
        Path to ``selected_features.json``.  Defaults to the canonical
        project path ``results/feature_selection/EXP_MI_V1_1/selected_features.json``.

    Returns
    -------
    list[str]
        Ordered list of 75 feature names exactly as recorded in the artifact.

    Raises
    ------
    FileNotFoundError
        If the JSON file is not found.
    ValueError
        If the feature count is not exactly 75 or the key "features" is absent.
    """
    resolved = Path(path) if path is not None else Path(_DEFAULT_FEATURES_PATH)

    if not resolved.exists():
        raise FileNotFoundError(
            f"Selected features file not found: {resolved}. "
            "Run Sprint 4 (EXP_MI_V1_1) first or provide the correct path."
        )

    with open(resolved, encoding="utf-8") as fh:
        data = json.load(fh)

    if "features" not in data:
        raise ValueError(
            f"'features' key missing from {resolved}. "
            "This file may not be a valid EXP_MI_V1_1 artifact."
        )

    features: list[str] = data["features"]

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_FEATURE_COUNT} features in "
            f"{FEATURE_SET_ID} but got {len(features)}. "
            "Do not change K without a formal protocol amendment."
        )

    logger.debug(
        "Loaded %d selected features from %s | first5=%s",
        len(features),
        resolved,
        features[:5],
    )
    return features


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    """
    Select exactly the frozen 75 features from an encoded DataFrame.

    The DataFrame ``df`` must be the *unscaled* encoded output of
    ``PreprocessingPipeline`` (i.e. a DataFrame with feature columns).
    It may also be a DataFrame constructed directly for testing.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded (unscaled) DataFrame with feature columns.  Must contain
        all names in ``features``.
    features : list[str]
        Ordered list of feature names to select (must be the frozen 75).

    Returns
    -------
    np.ndarray
        Shape ``(n_rows, 75)``, dtype float64.

    Raises
    ------
    ValueError
        If any of the requested features are missing from ``df``,
        if duplicate feature names are present,
        or if the feature count is not exactly 75.
    """
    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"build_feature_matrix expects exactly {EXPECTED_FEATURE_COUNT} features, "
            f"got {len(features)}."
        )

    # Check for duplicates in the feature list
    if len(set(features)) != len(features):
        dupes = [f for f in features if features.count(f) > 1]
        raise ValueError(
            f"Duplicate feature names detected: {sorted(set(dupes))}. "
            "The frozen feature list must have unique names."
        )

    # Check all features are present
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"The following {len(missing)} feature(s) are missing from the DataFrame: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}. "
            "Ensure the DataFrame was produced by PreprocessingPipeline."
        )

    X = df[features].to_numpy(dtype=np.float64)

    # Validate no non-finite values
    if not np.isfinite(X).all():
        n_bad = (~np.isfinite(X)).sum()
        raise ValueError(
            f"Non-finite values detected in feature matrix ({n_bad} cells). "
            "Run cleaning before calling build_feature_matrix."
        )

    logger.debug(
        "Built feature matrix | shape=%s | features=%s...",
        X.shape,
        features[:3],
    )
    return X


# ---------------------------------------------------------------------------
# Scaler fitting (inner-fold and final-TRAIN)
# ---------------------------------------------------------------------------


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Fit a fresh StandardScaler on training data.

    This must be called ONLY on inner_train (during CV) or on the complete
    frozen TRAIN (for the final refit).  Never call this on validation, test,
    or protected data.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix, shape ``(n_train, 75)``.

    Returns
    -------
    StandardScaler
        Fitted scaler.  Use ``.transform()`` on other splits.

    Raises
    ------
    ValueError
        If X_train is empty or contains non-finite values.
    """
    if X_train.shape[0] == 0:
        raise ValueError("fit_scaler received an empty training matrix (0 rows).")

    if not np.isfinite(X_train).all():
        raise ValueError(
            "Non-finite values in X_train passed to fit_scaler. "
            "Clean the data before scaling."
        )

    scaler = StandardScaler()
    scaler.fit(X_train)

    logger.debug(
        "Scaler fitted | input_shape=%s | mean[:3]=%s",
        X_train.shape,
        scaler.mean_[:3],
    )
    return scaler
