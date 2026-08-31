"""
src/feature_selection/mi_selector.py
--------------------------------------
Mutual Information feature ranking and selection for the UNSW-NB15 IDS project.

Design decisions (frozen Sprint 4):
    - Estimator: sklearn.feature_selection.mutual_info_classif
    - Target: binary label (0 = Normal, 1 = Attack)
    - n_neighbors: 3 (frozen, NOT tunable)
    - random_state: 42 (frozen)
    - Input representation: ENCODED + UNSCALED (OHE categorical + numeric)
    - Discrete mask: OHE-derived features = discrete; numeric = continuous
    - The mask is constructed from the feature_names list, NOT inferred from data

Leakage guarantees:
    - compute_mi_scores() operates on an externally supplied X, y.
    - The caller is responsible for ensuring X, y come ONLY from TRAIN
      (or the appropriate inner_train fold).
    - This module never reads any data file directly.
    - No sklearn .fit() calls occur here (MI is not a fitting step in the
      sklearn model sense — it is a scoring function).

Source families:
    - proto_*    (OHE from proto)
    - service_*  (OHE from service)
    - state_*    (OHE from state)
    - numeric    (raw numeric columns)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class MISelectorError(ValueError):
    """Raised for invalid inputs to MI feature selection functions."""


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MIConfig:
    """
    Frozen configuration for the MI estimator.

    All fields are fixed before observing any result.
    Do NOT change these values mid-experiment.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors for MI estimation (frozen = 3).
    random_state : int
        Random state for MI estimation reproducibility (frozen = 42).
    target_col : str
        Name of the binary target column (frozen = "label").
    """

    n_neighbors: int = 3
    random_state: int = 42
    target_col: str = "label"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MIResult:
    """
    Result of a MI scoring run.

    Attributes
    ----------
    feature_names : list[str]
        Ordered list of encoded feature names (aligns with mi_scores).
    mi_scores : np.ndarray
        Raw MI scores, one per feature, in the same order as feature_names.
    ranking_df : pd.DataFrame
        Full DataFrame with columns: rank, feature, mi_score, source_column,
        source_family, feature_type, selected.
    n_features : int
        Total number of encoded features scored.
    config : MIConfig
        The exact configuration used.
    discrete_mask : np.ndarray
        Boolean array of length n_features — True for discrete (OHE) features.
    """

    feature_names: list[str]
    mi_scores: np.ndarray
    ranking_df: pd.DataFrame
    n_features: int
    config: MIConfig
    discrete_mask: np.ndarray


# ---------------------------------------------------------------------------
# Source family utilities
# ---------------------------------------------------------------------------

# Categorical source columns whose OHE outputs are marked as discrete
_OHE_SOURCE_COLS = ("proto", "service", "state")


def get_source_family(feature_name: str) -> str:
    """
    Map an encoded feature name to its source family.

    OHE features follow the pattern "<source_col>_<value>".
    Numeric features do not contain an OHE prefix.

    Parameters
    ----------
    feature_name : str
        Encoded feature name (e.g. "proto_tcp", "dur", "service_-").

    Returns
    -------
    str
        One of: "proto", "service", "state", "numeric".
    """
    for src in _OHE_SOURCE_COLS:
        prefix = f"{src}_"
        if feature_name.startswith(prefix):
            return src
    return "numeric"


def build_discrete_mask(feature_names: list[str]) -> np.ndarray:
    """
    Build a boolean discrete-feature mask aligned with feature_names.

    OHE-derived features (proto_*, service_*, state_*) are discrete.
    Numeric features are continuous.

    The mask length MUST equal len(feature_names).
    This is verified by the caller's tests.

    Parameters
    ----------
    feature_names : list[str]
        Ordered list of encoded feature names.

    Returns
    -------
    np.ndarray
        Boolean array of shape (len(feature_names),).
        True = discrete (OHE). False = continuous (numeric).

    Raises
    ------
    MISelectorError
        If feature_names is empty.
    """
    if len(feature_names) == 0:
        raise MISelectorError(
            "feature_names is empty. Cannot build discrete mask on zero features."
        )

    mask = np.array(
        [get_source_family(f) != "numeric" for f in feature_names],
        dtype=bool,
    )
    logger.debug(
        "Discrete mask built | total=%d | discrete=%d | continuous=%d",
        len(mask),
        int(mask.sum()),
        int((~mask).sum()),
    )
    return mask


# ---------------------------------------------------------------------------
# Core MI scoring
# ---------------------------------------------------------------------------


def compute_mi_scores(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: MIConfig | None = None,
) -> MIResult:
    """
    Compute Mutual Information scores for all encoded features against the
    binary label.

    LEAKAGE CONTRACT: The caller MUST pass only TRAIN (or inner_train) data.
    This function does not enforce or verify the source of X, y — it relies
    entirely on the caller's data-access discipline.

    Parameters
    ----------
    X : np.ndarray
        Encoded + UNSCALED feature matrix of shape (n_samples, n_features).
        Must not contain NaN, +inf, or -inf.
    y : np.ndarray
        Binary target array of shape (n_samples,). Values must be in {0, 1}.
    feature_names : list[str]
        Ordered list of encoded feature names aligned with columns of X.
    config : MIConfig | None
        MI estimator configuration. If None, the frozen default is used.

    Returns
    -------
    MIResult
        Contains raw scores, ranking DataFrame, and discrete mask.

    Raises
    ------
    MISelectorError
        If inputs are invalid (empty, shape mismatch, NaN/inf, wrong target).
    """
    if config is None:
        config = MIConfig()

    # --- Input validation ---
    if X.shape[0] == 0:
        raise MISelectorError(
            "X has zero rows. Cannot compute MI on an empty feature matrix."
        )
    if X.shape[1] == 0:
        raise MISelectorError(
            "X has zero columns. Cannot compute MI on an empty feature matrix."
        )
    if len(feature_names) != X.shape[1]:
        raise MISelectorError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"X.shape[1] ({X.shape[1]}). They must be aligned."
        )
    if len(y) != X.shape[0]:
        raise MISelectorError(
            f"y length ({len(y)}) does not match X.shape[0] ({X.shape[0]})."
        )

    # NaN / inf check
    if not np.isfinite(X).all():
        raise MISelectorError(
            "X contains NaN or ±inf values. "
            "The preprocessing pipeline must produce a finite feature matrix."
        )
    if not np.isfinite(y).all():
        raise MISelectorError("y contains NaN or ±inf values.")

    # Binary target check
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        raise MISelectorError(
            f"y contains only one class: {unique_labels}. "
            "MI requires at least two classes in the target."
        )
    if not set(unique_labels.tolist()).issubset({0, 1}):
        raise MISelectorError(
            f"y contains non-binary values: {unique_labels}. "
            f"Expected only {{0, 1}}."
        )

    # Duplicate feature name check
    if len(feature_names) != len(set(feature_names)):
        from collections import Counter
        counts = Counter(feature_names)
        dups = [n for n, c in counts.items() if c > 1]
        raise MISelectorError(
            f"feature_names contains {len(dups)} duplicate name(s): {dups[:5]}..."
        )

    if config.n_neighbors < 1:
        raise MISelectorError(
            f"n_neighbors must be >= 1, got {config.n_neighbors}."
        )

    # --- Build discrete mask ---
    discrete_mask = build_discrete_mask(feature_names)

    logger.info(
        "Computing MI | n_samples=%d | n_features=%d | discrete=%d | continuous=%d "
        "| n_neighbors=%d | random_state=%d",
        X.shape[0],
        X.shape[1],
        int(discrete_mask.sum()),
        int((~discrete_mask).sum()),
        config.n_neighbors,
        config.random_state,
    )

    # --- Compute MI ---
    # Cast discrete feature columns to int32 before calling mutual_info_classif.
    # Reason: numpy concatenation produces float64 arrays even for OHE columns
    # whose values are exactly 0 or 1.  sklearn's internal mutual_info_score
    # (called for discrete features) checks dtype and emits a spurious
    # "Clustering metrics expects discrete values but received continuous values"
    # warning when it sees float64.  Casting to int32 silences the warning
    # without changing any MI values — OHE values are 0.0/1.0 exactly and
    # int32(0.0) == 0, int32(1.0) == 1.
    if discrete_mask.any():
        X_mi = X.copy()
        X_mi[:, discrete_mask] = X_mi[:, discrete_mask].astype(np.int32)
    else:
        X_mi = X

    mi_scores: np.ndarray = mutual_info_classif(
        X_mi,
        y,
        discrete_features=discrete_mask,
        n_neighbors=config.n_neighbors,
        random_state=config.random_state,
    )

    # --- Build ranking DataFrame ---
    ranking_df = rank_features(feature_names, mi_scores, discrete_mask)

    logger.info(
        "MI complete | top-5 features: %s",
        ranking_df.head(5)[["feature", "mi_score"]].to_dict("records"),
    )

    return MIResult(
        feature_names=list(feature_names),
        mi_scores=mi_scores,
        ranking_df=ranking_df,
        n_features=X.shape[1],
        config=config,
        discrete_mask=discrete_mask,
    )


# ---------------------------------------------------------------------------
# Feature ranking
# ---------------------------------------------------------------------------


def rank_features(
    feature_names: list[str],
    mi_scores: np.ndarray,
    discrete_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Build a ranked feature DataFrame from raw MI scores.

    Parameters
    ----------
    feature_names : list[str]
        Ordered feature names aligned with mi_scores.
    mi_scores : np.ndarray
        Raw MI scores (one per feature).
    discrete_mask : np.ndarray | None
        Boolean discrete mask. If None, inferred from feature_names.

    Returns
    -------
    pd.DataFrame
        Columns: rank, feature, mi_score, source_column, source_family,
        feature_type, selected.
        Sorted by mi_score descending (rank 1 = highest MI).
        'selected' is False for all rows — caller must call select_top_k()
        to fill it in.
    """
    if discrete_mask is None:
        discrete_mask = build_discrete_mask(feature_names)

    source_families = [get_source_family(f) for f in feature_names]
    source_columns = [
        f.split("_", 1)[0] if get_source_family(f) != "numeric" else f
        for f in feature_names
    ]
    feature_types = [
        "discrete" if d else "continuous" for d in discrete_mask
    ]

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "mi_score": mi_scores,
            "source_column": source_columns,
            "source_family": source_families,
            "feature_type": feature_types,
        }
    )

    # Sort by MI score descending; break ties by feature name for stability
    df = df.sort_values(
        by=["mi_score", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    df.insert(0, "rank", range(1, len(df) + 1))
    df["selected"] = False

    return df


# ---------------------------------------------------------------------------
# K selection
# ---------------------------------------------------------------------------


def select_top_k(
    ranking_df: pd.DataFrame,
    k: int,
    feature_names_reference: list[str],
) -> pd.DataFrame:
    """
    Mark the top-K features as selected in a copy of ranking_df.

    Parameters
    ----------
    ranking_df : pd.DataFrame
        Output from rank_features() or compute_mi_scores().
    k : int
        Number of top features to select.
    feature_names_reference : list[str]
        The complete ordered feature_names list (used to verify alignment).

    Returns
    -------
    pd.DataFrame
        Copy of ranking_df with 'selected' column updated.

    Raises
    ------
    MISelectorError
        If k <= 0, k > n_features, or alignment is violated.
    """
    n = len(ranking_df)
    if k <= 0:
        raise MISelectorError(f"k must be > 0, got k={k}.")
    if k > n:
        raise MISelectorError(
            f"k={k} exceeds the number of available features ({n}). "
            "Cannot select more features than exist."
        )

    result = ranking_df.copy()
    top_k_names = set(result.iloc[:k]["feature"].tolist())
    result["selected"] = result["feature"].isin(top_k_names)

    n_selected = int(result["selected"].sum())
    if n_selected != k:
        raise MISelectorError(
            f"select_top_k produced {n_selected} selected features but k={k}. "
            "Duplicate feature names or ranking inconsistency detected."
        )

    return result


# ---------------------------------------------------------------------------
# Source-family report
# ---------------------------------------------------------------------------


def build_family_report(
    ranking_df: pd.DataFrame,
    candidate_k: list[int] | None = None,
) -> dict[str, Any]:
    """
    Build a source-family report: candidate counts and selected counts.

    Parameters
    ----------
    ranking_df : pd.DataFrame
        Output of select_top_k() (with 'selected' column populated).
    candidate_k : list[int] | None
        The candidate K values for context in the report.

    Returns
    -------
    dict
        {
            "candidate": {"proto": N, "service": N, "state": N, "numeric": N},
            "selected":  {"proto": N, "service": N, "state": N, "numeric": N},
        }
    """
    families = ["proto", "service", "state", "numeric"]

    candidate_counts = {
        fam: int((ranking_df["source_family"] == fam).sum())
        for fam in families
    }
    selected_counts = {
        fam: int(((ranking_df["source_family"] == fam) & ranking_df["selected"]).sum())
        for fam in families
    }

    return {
        "candidate": candidate_counts,
        "selected": selected_counts,
        "note": "No quota applied. Natural MI result reported.",
    }
