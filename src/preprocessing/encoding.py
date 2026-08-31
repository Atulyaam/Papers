"""
src/preprocessing/encoding.py
-------------------------------
Categorical encoding for the UNSW-NB15 IDS project.

Design decisions (frozen):
    - OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    - Fit on TRAIN categorical columns only.
    - service="-" is treated as a real category, NOT as NaN.
    - No rare-value bucketing.
    - No ordinal / target encoding.
    - Fitted category ordering is preserved and recorded in metadata.
    - Unknown TEST/val categories are silently zeroed (not crashed).
    - The encoder handles the case where a TRAIN category is absent
      from a later split (the OHE column still exists with zeros).

API:
    fit_encoder(X_cat_train)      -> FittedEncoder
    transform_encoder(fe, X_cat)  -> np.ndarray  (encoded matrix)
    get_feature_names(fe)         -> list[str]
    get_encoder_metadata(fe)      -> dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


@dataclass
class FittedEncoder:
    """
    Wrapper around a fitted OneHotEncoder that preserves project metadata.

    Attributes
    ----------
    encoder : OneHotEncoder
        The fitted sklearn encoder.
    categorical_cols : list[str]
        Ordered categorical column names that were used during fit.
    categories : dict[str, list]
        Mapping from column name to the exact ordered list of fitted categories.
    feature_names : list[str]
        Flat ordered list of OHE output feature names (e.g., "proto_tcp").
    """

    encoder: OneHotEncoder
    categorical_cols: list[str]
    categories: dict[str, list] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)


def fit_encoder(
    X_cat_train: pd.DataFrame,
    categorical_cols: list[str],
) -> FittedEncoder:
    """
    Fit a OneHotEncoder on the TRAIN categorical columns.

    The encoder uses handle_unknown="ignore" so unseen categories in later
    splits produce all-zero rows rather than raising an error.

    service="-" is treated as a real category value (not NaN). It must be
    present as a string in the DataFrame — the caller must not silently
    replace it with NaN before this call.

    Parameters
    ----------
    X_cat_train : pd.DataFrame
        Subset of the TRAIN feature DataFrame containing only the categorical
        columns.
    categorical_cols : list[str]
        Ordered list of categorical column names to encode. Must match the
        columns of X_cat_train.

    Returns
    -------
    FittedEncoder
        Fitted encoder with preserved category metadata.

    Raises
    ------
    ValueError
        If X_cat_train is empty or categorical_cols is empty.
    """
    if len(categorical_cols) == 0:
        raise ValueError(
            "categorical_cols is empty. At least one categorical column must be "
            "specified for encoding."
        )
    if X_cat_train.empty:
        raise ValueError(
            "X_cat_train is empty (zero rows). Cannot fit encoder on empty data."
        )

    # Ensure column order matches the declared categorical_cols
    X_fit = X_cat_train[categorical_cols]

    # Verify no NaN in categorical columns (NaN would indicate a silent "-" conversion)
    for col in categorical_cols:
        nan_count = int(X_fit[col].isna().sum())
        if nan_count > 0:
            logger.warning(
                "Categorical column '%s' contains %d NaN values during encoder fit. "
                "Verify that the service value '-' was NOT silently converted to NaN.",
                col,
                nan_count,
            )

    enc = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )
    enc.fit(X_fit)

    # Build per-column category metadata
    categories: dict[str, list] = {
        col: list(cats)
        for col, cats in zip(categorical_cols, enc.categories_)
    }

    # Derive feature names from the encoder (canonical, stable)
    feature_names: list[str] = list(enc.get_feature_names_out(categorical_cols))

    logger.info(
        "Encoder fitted | cols=%s | total_features=%d | category_counts=%s",
        categorical_cols,
        len(feature_names),
        {col: len(cats) for col, cats in categories.items()},
    )

    return FittedEncoder(
        encoder=enc,
        categorical_cols=categorical_cols,
        categories=categories,
        feature_names=feature_names,
    )


def transform_encoder(
    fitted_encoder: FittedEncoder,
    X_cat: pd.DataFrame,
) -> np.ndarray:
    """
    Transform categorical columns using a previously fitted encoder.

    handle_unknown="ignore" ensures that categories absent from TRAIN are
    represented as all-zero rows. The output dimensionality is always equal
    to len(fitted_encoder.feature_names) regardless of what categories appear
    in X_cat.

    Parameters
    ----------
    fitted_encoder : FittedEncoder
        A FittedEncoder returned by fit_encoder().
    X_cat : pd.DataFrame
        DataFrame with the same categorical column names used during fitting.
        Column order must match fitted_encoder.categorical_cols.

    Returns
    -------
    np.ndarray
        Dense float64 array of shape (n_rows, n_ohe_features).
    """
    X_ordered = X_cat[fitted_encoder.categorical_cols]
    encoded = fitted_encoder.encoder.transform(X_ordered)
    return encoded.astype(np.float64)


def get_feature_names(fitted_encoder: FittedEncoder) -> list[str]:
    """Return the flat ordered list of OHE feature names."""
    return fitted_encoder.feature_names


def get_encoder_metadata(fitted_encoder: FittedEncoder) -> dict[str, Any]:
    """
    Return a JSON-serialisable metadata dict capturing the fitted encoder state.

    This is required for auditability and later category-drift analysis.
    """
    return {
        "encoder_type": "OneHotEncoder",
        "handle_unknown": "ignore",
        "sparse_output": False,
        "categorical_cols": fitted_encoder.categorical_cols,
        "categories": {
            col: list(cats) for col, cats in fitted_encoder.categories.items()
        },
        "n_output_features": len(fitted_encoder.feature_names),
        "feature_names_sample": fitted_encoder.feature_names[:10],
        "note": (
            "service='-' is treated as a real category. "
            "No rare-value bucketing applied. "
            "Unknown test categories produce all-zero rows."
        ),
    }
