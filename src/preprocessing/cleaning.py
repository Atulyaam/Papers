"""
src/preprocessing/cleaned.py
------------------------------
Cleaning and schema validation for raw DataFrames before encoding.

Responsibilities:
    1.  Validate required columns are present (label, attack_cat).
    2.  Validate excluded columns are removed before encoding.
    3.  Detect NaN / +inf / -inf in feature columns and raise loudly.
    4.  Separate target (y), metadata (attack_cat), and features (X_raw).
    5.  Identify categorical vs. numeric feature columns from the
        project schema contract — NOT solely from pandas dtype.

NO transform, fit, encode, or scale occurs here.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.preprocessing.exceptions import NonFiniteValueError, PreprocessingSchemaError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-fixed column names (from configs/data_schema.yaml)
# ---------------------------------------------------------------------------
LABEL_COL: str = "label"
ATTACK_CAT_COL: str = "attack_cat"

# Columns to exclude from model features (id is present in pre-split files;
# IP/port/time raw fields are absent from the pre-split layout — see data_schema.yaml).
EXCLUDE_COLS: frozenset[str] = frozenset({"id"})

# Candidate categorical columns per project data contract
# (confirmed present in pre-split files by Sprint 1 audit)
CATEGORICAL_COLS: tuple[str, ...] = ("proto", "service", "state")


# ---------------------------------------------------------------------------
# Public named tuple for the cleaned split
# ---------------------------------------------------------------------------
class CleanedSplit(NamedTuple):
    """Result of cleaning a raw DataFrame prior to encoding."""

    X_raw: pd.DataFrame
    """Feature DataFrame: excludes label, attack_cat, id; all other cols."""

    y: pd.Series
    """Binary target series (0/1)."""

    attack_cat: pd.Series
    """Attack category series (metadata — must NOT enter X)."""

    categorical_cols: list[str]
    """Ordered list of categorical columns present in X_raw."""

    numeric_cols: list[str]
    """Ordered list of numeric columns present in X_raw."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_required_columns(df: pd.DataFrame, split_name: str = "unknown") -> None:
    """
    Assert that label and attack_cat are both present in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame to validate.
    split_name : str
        Name of the split (for error messages).

    Raises
    ------
    PreprocessingSchemaError
        If either required column is missing.
    """
    missing = [c for c in (LABEL_COL, ATTACK_CAT_COL) if c not in df.columns]
    if missing:
        raise PreprocessingSchemaError(
            f"Split '{split_name}' is missing required columns: {missing}. "
            f"Expected both '{LABEL_COL}' and '{ATTACK_CAT_COL}' to be present.",
            missing_cols=missing,
        )


def detect_nonfinite(
    df: pd.DataFrame,
    split_name: str = "unknown",
    numeric_cols: list[str] | None = None,
) -> None:
    """
    Detect NaN, +inf, and -inf in numeric feature columns and raise loudly.

    Policy: Sprint 2 does NOT impute or drop. Any non-finite value raises
    NonFiniteValueError. The caller must handle data quality upstream.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check (feature columns only — targets already separated).
    split_name : str
        Split name for error reporting.
    numeric_cols : list[str] | None
        Explicit list of numeric columns to check. If None, all columns
        with numeric dtype are checked.

    Raises
    ------
    NonFiniteValueError
        If any NaN, +inf, or -inf is found.
    """
    cols_to_check = numeric_cols if numeric_cols is not None else list(
        df.select_dtypes(include=[np.number]).columns
    )

    nan_cols: dict[str, int] = {}
    pos_inf_cols: dict[str, int] = {}
    neg_inf_cols: dict[str, int] = {}

    for col in cols_to_check:
        if col not in df.columns:
            continue
        series = df[col]
        nan_count = int(series.isna().sum())
        pos_inf_count = int((series == np.inf).sum())
        neg_inf_count = int((series == -np.inf).sum())

        if nan_count:
            nan_cols[col] = nan_count
        if pos_inf_count:
            pos_inf_cols[col] = pos_inf_count
        if neg_inf_count:
            neg_inf_cols[col] = neg_inf_count

    if nan_cols or pos_inf_cols or neg_inf_cols:
        raise NonFiniteValueError(split_name, nan_cols, pos_inf_cols, neg_inf_cols)


def separate_target_and_features(
    df: pd.DataFrame,
    split_name: str = "unknown",
) -> CleanedSplit:
    """
    Validate, clean, and separate a raw DataFrame into features, target,
    and metadata.

    Steps:
        1. Validate required columns.
        2. Extract y (label) and attack_cat (metadata).
        3. Drop excluded columns (id, label, attack_cat) from features.
        4. Classify remaining columns as categorical or numeric using the
           project data contract (not purely by pandas dtype).
        5. Detect non-finite values in numeric features (fail loudly).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame (as loaded by loader.py — no prior transforms).
    split_name : str
        Split identifier (e.g. "train", "development_test").

    Returns
    -------
    CleanedSplit
        Named tuple with X_raw, y, attack_cat, categorical_cols, numeric_cols.

    Raises
    ------
    PreprocessingSchemaError
        If required columns are absent.
    NonFiniteValueError
        If non-finite values are present in numeric feature columns.
    """
    validate_required_columns(df, split_name)

    # --- Extract targets and metadata ---
    y = df[LABEL_COL].copy()
    attack_cat = df[ATTACK_CAT_COL].copy()

    # --- Build feature set: drop metadata and excluded columns ---
    always_drop = frozenset({LABEL_COL, ATTACK_CAT_COL}) | EXCLUDE_COLS
    feature_cols = [c for c in df.columns if c not in always_drop]
    X_raw = df[feature_cols].copy()

    # --- Classify columns from the data contract (not purely from dtype) ---
    # Categorical columns: use contract definition; only include those present
    categorical_cols = [c for c in CATEGORICAL_COLS if c in X_raw.columns]

    # Numeric columns: all remaining feature columns not in categorical set
    cat_set = set(categorical_cols)
    numeric_cols = [c for c in X_raw.columns if c not in cat_set]

    logger.debug(
        "CleanedSplit | split=%s | rows=%d | feat_cols=%d | cat=%d | num=%d",
        split_name,
        len(df),
        len(feature_cols),
        len(categorical_cols),
        len(numeric_cols),
    )

    # --- Non-finite validation on numeric features ---
    detect_nonfinite(X_raw, split_name, numeric_cols)

    return CleanedSplit(
        X_raw=X_raw,
        y=y,
        attack_cat=attack_cat,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )
