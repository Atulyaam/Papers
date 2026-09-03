"""
src/preprocessing/schema_audit.py
-----------------------------------
Read-only data quality audit functions for the UNSW-NB15 IDS project.

All functions are PURELY DIAGNOSTIC. They:
- Read data
- Compute statistics
- Return structured dicts

They do NOT:
- modify any DataFrame
- drop rows
- encode columns
- scale values
- fit any object

LEAKAGE POLICY: Zero leakage risk. No transforms applied.

The audit runs on both the training and testing splits separately.
Observations are recorded; modeling decisions are deferred to Sprint 2.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------


def audit_dataframe(df: pd.DataFrame, split_name: str) -> dict[str, Any]:
    """
    Produce a comprehensive structural audit of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The raw split DataFrame (train or test).
    split_name : str
        Label for logging ("train" or "test").

    Returns
    -------
    dict
        Structured audit result containing shape, dtypes, null counts,
        inf counts, duplicate count, and negative-value columns.
    """
    result: dict[str, Any] = {
        "split": split_name,
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "null_counts": {},
        "inf_counts": {},
        "duplicate_row_count": int(df.duplicated().sum()),
        "negative_value_columns": [],
        "unique_counts": {},
        "sample_values": {},
    }

    for col in df.columns:
        # Null / NaN counts
        null_cnt = int(df[col].isna().sum())
        result["null_counts"][col] = null_cnt

        # +inf / -inf counts (only for numeric columns)
        if pd.api.types.is_numeric_dtype(df[col]):
            pos_inf = int((df[col] == np.inf).sum())
            neg_inf = int((df[col] == -np.inf).sum())
            result["inf_counts"][col] = {"pos_inf": pos_inf, "neg_inf": neg_inf}

            # Negative value presence (informational)
            finite_vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite_vals) > 0 and (finite_vals < 0).any():
                result["negative_value_columns"].append(col)
        else:
            result["inf_counts"][col] = {"pos_inf": 0, "neg_inf": 0}

        # Unique count
        result["unique_counts"][col] = int(df[col].nunique(dropna=False))

        # Sample values (first 5 unique values, as strings for JSON safety)
        sample = df[col].dropna().unique()[:5].tolist()
        result["sample_values"][col] = [_json_safe(v) for v in sample]

    return result


def audit_label_distribution(
    df: pd.DataFrame, label_col: str
) -> dict[str, Any]:
    """
    Audit the binary label column distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Raw split DataFrame.
    label_col : str
        Name of the binary label column (expected values: 0, 1).

    Returns
    -------
    dict
        Label → count mapping, plus unexpected values.
    """
    if label_col not in df.columns:
        return {"error": f"Label column '{label_col}' not found in DataFrame."}

    counts = df[label_col].value_counts(dropna=False).to_dict()
    counts = {_json_safe(k): int(v) for k, v in counts.items()}

    expected_values = {0, 1}
    actual_values = set(df[label_col].dropna().unique())
    unexpected = [_json_safe(v) for v in (actual_values - expected_values)]

    return {
        "label_col": label_col,
        "counts": counts,
        "total": int(len(df)),
        "unexpected_values": unexpected,
    }


def audit_attack_cat_raw_strings(
    df: pd.DataFrame, cat_col: str
) -> list[str]:
    """
    Return the sorted list of unique raw attack_cat strings, preserving
    all capitalisation and whitespace exactly as stored.

    Parameters
    ----------
    df : pd.DataFrame
        Raw split DataFrame.
    cat_col : str
        Name of the attack-category column.

    Returns
    -------
    list[str]
        Sorted unique raw strings (NaN represented as the string "NaN").
    """
    if cat_col not in df.columns:
        return []

    raw_uniques = df[cat_col].astype(str).unique().tolist()
    return sorted(raw_uniques)


def audit_attack_cat_distribution(
    df: pd.DataFrame, cat_col: str
) -> dict[str, int]:
    """
    Return raw string → count mapping for attack_cat.

    Preserves exact original strings (spaces, mixed case, etc.).
    NaN values appear as the string "NaN".

    Parameters
    ----------
    df : pd.DataFrame
        Raw split DataFrame.
    cat_col : str
        Name of the attack-category column.

    Returns
    -------
    dict[str, int]
        Exact raw string → count.
    """
    if cat_col not in df.columns:
        return {}

    counts = df[cat_col].astype(str).value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def check_overlap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "train",
    label_b: str = "test",
) -> dict[str, Any]:
    """
    Exact row-level overlap check between two DataFrames.

    Uses an inner merge on all shared columns.

    Limitation: This detects only EXACT duplicate rows. Near-duplicate or
    semantically similar rows are not detected by this method.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        DataFrames to compare (e.g., train and test splits).
    label_a, label_b : str
        Human-readable labels for error reporting.

    Returns
    -------
    dict
        overlap_count, label_a, label_b, overlap_percentage, method.
    """
    # Use only columns that exist in both DataFrames
    shared_cols = list(set(df_a.columns) & set(df_b.columns))

    if not shared_cols:
        return {
            "overlap_count": 0,
            "label_a": label_a,
            "label_b": label_b,
            "overlap_percentage": 0.0,
            "method": "inner_merge_all_shared_columns",
            "note": "No shared columns found; cannot perform overlap check.",
        }

    merged = pd.merge(
        df_a[shared_cols].drop_duplicates(),
        df_b[shared_cols].drop_duplicates(),
        on=shared_cols,
        how="inner",
    )
    overlap_count = int(len(merged))
    total = int(len(df_a)) + int(len(df_b))
    pct = round(overlap_count / total * 100, 4) if total > 0 else 0.0

    return {
        "overlap_count": overlap_count,
        "label_a": label_a,
        "label_b": label_b,
        "train_row_count": int(len(df_a)),
        "test_row_count": int(len(df_b)),
        "overlap_percentage": pct,
        "method": "inner_merge_all_shared_columns",
        "limitation": (
            "Exact duplicate detection only. "
            "Near-duplicate or semantically similar rows are not detected."
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Convert a value to a JSON-serialisable type."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
