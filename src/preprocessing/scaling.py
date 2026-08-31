"""
src/preprocessing/scaling.py
------------------------------
Feature scaling for the UNSW-NB15 IDS project.

Design decisions (frozen):
    - StandardScaler: fit on TRAIN only.
    - Scaler is applied to the FULL encoded feature matrix (OHE + numeric),
      not only to numeric columns. This ensures a consistent feature-name
      mapping between the unscaled and scaled views.
    - The unscaled view (for DT/RF) is the encoded matrix BEFORE this scaler.
    - The scaled view (for SVM/NN/AE) is the encoded matrix AFTER this scaler.
    - Both views share the same feature_names and row ordering.

API:
    fit_scaler(X_encoded_train)      -> FittedScaler
    transform_scaler(fs, X_encoded)  -> np.ndarray
    get_scaler_metadata(fs)          -> dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FittedScaler:
    """
    Wrapper around a fitted StandardScaler that records project metadata.

    Attributes
    ----------
    scaler : StandardScaler
        The fitted sklearn scaler.
    n_features : int
        Number of features the scaler was fit on.
    feature_names : list[str]
        Ordered feature names at fit time (same order as in the encoded matrix).
    train_mean : np.ndarray
        Per-feature mean from TRAIN (for provenance recording).
    train_scale : np.ndarray
        Per-feature std from TRAIN (for provenance recording).
    """

    scaler: StandardScaler
    n_features: int
    feature_names: list[str]
    train_mean: np.ndarray
    train_scale: np.ndarray


def fit_scaler(
    X_encoded_train: np.ndarray,
    feature_names: list[str],
) -> FittedScaler:
    """
    Fit a StandardScaler on the TRAIN encoded feature matrix.

    LEAKAGE RULE: This function must only ever be called with TRAIN data.
    Validation, test, and protected-unseen data must never be passed here.

    Parameters
    ----------
    X_encoded_train : np.ndarray
        Full encoded feature matrix from TRAIN (OHE + numeric concatenated).
        Shape: (n_train_rows, n_features).
    feature_names : list[str]
        Ordered feature names corresponding to columns of X_encoded_train.

    Returns
    -------
    FittedScaler
        Fitted scaler with recorded TRAIN statistics.

    Raises
    ------
    ValueError
        If X_encoded_train is empty or feature_names length does not match.
    """
    if X_encoded_train.shape[0] == 0:
        raise ValueError(
            "X_encoded_train has zero rows. Cannot fit scaler on empty data."
        )
    if X_encoded_train.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"X_encoded_train columns ({X_encoded_train.shape[1]})."
        )

    scaler = StandardScaler()
    scaler.fit(X_encoded_train)

    logger.info(
        "Scaler fitted on TRAIN | n_features=%d | mean_range=[%.4f, %.4f] | "
        "scale_range=[%.4f, %.4f]",
        X_encoded_train.shape[1],
        float(np.nanmin(scaler.mean_)),
        float(np.nanmax(scaler.mean_)),
        float(np.nanmin(scaler.scale_)),
        float(np.nanmax(scaler.scale_)),
    )

    return FittedScaler(
        scaler=scaler,
        n_features=X_encoded_train.shape[1],
        feature_names=feature_names,
        train_mean=scaler.mean_.copy(),
        train_scale=scaler.scale_.copy(),
    )


def transform_scaler(
    fitted_scaler: FittedScaler,
    X_encoded: np.ndarray,
) -> np.ndarray:
    """
    Apply the TRAIN-fitted scaler to any encoded feature matrix.

    The scaler is NEVER refit here — only the TRAIN-fitted parameters are used.

    Parameters
    ----------
    fitted_scaler : FittedScaler
        A FittedScaler returned by fit_scaler().
    X_encoded : np.ndarray
        Encoded feature matrix to scale. Must have the same number of columns
        as the TRAIN matrix used during fitting.

    Returns
    -------
    np.ndarray
        Scaled feature matrix, same shape as X_encoded.

    Raises
    ------
    ValueError
        If the column count does not match the fitted scaler.
    """
    if X_encoded.shape[1] != fitted_scaler.n_features:
        raise ValueError(
            f"X_encoded has {X_encoded.shape[1]} columns but scaler was fitted "
            f"on {fitted_scaler.n_features} features. Column count must match."
        )
    return fitted_scaler.scaler.transform(X_encoded).astype(np.float64)


def get_scaler_metadata(fitted_scaler: FittedScaler) -> dict[str, Any]:
    """
    Return a JSON-serialisable metadata dict capturing the fitted scaler state.

    NOTE: mean_ and scale_ are per-feature arrays — for large feature sets,
    only summary statistics are recorded (not the full arrays) to avoid
    storing huge raw-data blobs in metadata.
    """
    return {
        "scaler_type": "StandardScaler",
        "fitted_on": "TRAIN",
        "n_features": fitted_scaler.n_features,
        "mean_min": float(np.nanmin(fitted_scaler.train_mean)),
        "mean_max": float(np.nanmax(fitted_scaler.train_mean)),
        "scale_min": float(np.nanmin(fitted_scaler.train_scale)),
        "scale_max": float(np.nanmax(fitted_scaler.train_scale)),
        "note": (
            "Scaler fit on TRAIN only. mean_ and scale_ reflect TRAIN distribution. "
            "No validation/test/protected-unseen data influenced fitting."
        ),
    }
