"""
src/feature_selection/k_selector.py
--------------------------------------
K selection via training-only internal cross-validation.

Design (frozen Sprint 4):
    - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    - Stratification target: label
    - Candidate K values: [10, 20, 30, 40, 50]
    - Metric: macro-F1
    - Tie-break: smaller K
    - Evaluator: LogisticRegression(solver="liblinear", C=1.0, max_iter=1000,
                                    class_weight="balanced", random_state=42)
    - Evaluator scaling: StandardScaler fitted on inner_train ONLY
    - MI is refit inside every inner fold

Leakage guarantees (enforced in this module):
    - Encoder fitted on inner_train only (per fold).
    - MI computed on inner_train only (per fold).
    - Scaler fitted on inner_train selected features only (per fold).
    - LogisticRegression fitted on inner_train only (per fold).
    - inner_validation data NEVER influences encoder/MI/scaler statistics.
    - Outer VALIDATION, TEST, and protected data are NEVER accessed here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.feature_selection.mi_selector import (
    MIConfig,
    MISelectorError,
    compute_mi_scores,
)
from src.preprocessing.cleaning import (
    CATEGORICAL_COLS,
    separate_target_and_features,
)
from src.preprocessing.encoding import fit_encoder, transform_encoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InnerCVConfig:
    """
    Frozen configuration for inner K-selection cross-validation.

    All parameters are fixed before observing any result.
    """

    candidate_k: tuple[int, ...] = (10, 20, 30, 40, 50)
    n_splits: int = 5
    shuffle: bool = True
    cv_random_state: int = 42
    stratify_col: str = "label"

    # Evaluator (fixed — NOT a research model)
    evaluator_solver: str = "liblinear"
    evaluator_C: float = 1.0
    evaluator_max_iter: int = 1000
    evaluator_class_weight: str = "balanced"
    evaluator_random_state: int = 42

    # MI
    mi_n_neighbors: int = 3
    mi_random_state: int = 42


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class KFoldRecord:
    """Per-fold result for one (K, fold) pair."""

    k: int
    fold: int
    macro_f1: float


@dataclass
class KSelectionResult:
    """
    Aggregated K-selection result across all folds.

    Attributes
    ----------
    selected_k : int
        The chosen K* (highest mean macro-F1; smaller K on tie).
    fold_records : list[KFoldRecord]
        One record per (K, fold) combination.
    summary_df : pd.DataFrame
        Summary with columns: K, mean_macro_f1, std_macro_f1.
    config : InnerCVConfig
        The exact configuration used.
    """

    selected_k: int
    fold_records: list[KFoldRecord]
    summary_df: pd.DataFrame
    config: InnerCVConfig


@dataclass
class KSelectionSanity:
    """
    Result of the post-selection sanity check.

    Attributes
    ----------
    status : str
        "PASS" or "REVIEW_REQUIRED".
    reason : str
        Human-readable explanation of the check result.
    flat_range : float
        max(mean_macro_f1) - min(mean_macro_f1) across all K.
    flat_tolerance : float
        The tolerance used for the flat test (documented in config).
    is_monotonic : bool
        True if F1(K=10) <= F1(K=20) <= ... <= F1(K=50) (within tolerance).
    monotonic_tolerance : float
        Tolerance used for the monotonic test.
    """

    status: str
    reason: str
    flat_range: float
    flat_tolerance: float
    is_monotonic: bool
    monotonic_tolerance: float


# ---------------------------------------------------------------------------
# Inner-fold preprocessing (leakage-safe)
# ---------------------------------------------------------------------------


def _build_inner_encoded(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit encoder on inner_train, transform both inner_train and inner_val.

    The encoder is NEVER exposed to inner_val during fitting.
    This function isolates the leakage boundary for every inner fold.

    Parameters
    ----------
    train_df : pd.DataFrame
        Inner-fold TRAIN subset (feature columns only, no label/attack_cat).
    val_df : pd.DataFrame
        Inner-fold VALIDATION subset.
    categorical_cols : list[str]
        Categorical column names.

    Returns
    -------
    (X_train_ohe, X_val_ohe, X_train_num, X_val_num)
        OHE arrays and numeric arrays for train and val, ready to concatenate.
    """
    from src.preprocessing.cleaning import CATEGORICAL_COLS as CAT_COLS

    # Fit encoder on inner_train ONLY
    cat_cols_present = [c for c in categorical_cols if c in train_df.columns]
    fitted_enc = fit_encoder(train_df[cat_cols_present], cat_cols_present)

    # Transform both (inner_val sees NO encoder fitting)
    X_train_ohe = transform_encoder(fitted_enc, train_df[cat_cols_present])
    X_val_ohe   = transform_encoder(fitted_enc, val_df[cat_cols_present])

    # Numeric columns
    num_cols = [c for c in train_df.columns if c not in cat_cols_present]
    X_train_num = train_df[num_cols].to_numpy(dtype=np.float64)
    X_val_num   = val_df[num_cols].to_numpy(dtype=np.float64)

    # Build feature names for discrete mask
    from src.preprocessing.encoding import get_feature_names
    ohe_names = get_feature_names(fitted_enc)

    return X_train_ohe, X_val_ohe, X_train_num, X_val_num, ohe_names, num_cols


# ---------------------------------------------------------------------------
# Inner-fold evaluation for one K
# ---------------------------------------------------------------------------


def _evaluate_k_one_fold(
    X_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    X_inner_val: np.ndarray,
    y_inner_val: np.ndarray,
    feature_names: list[str],
    k: int,
    config: InnerCVConfig,
) -> float:
    """
    Run MI + Logistic Regression evaluation for one (K, fold) pair.

    Steps (all fitting on inner_train ONLY):
        1. Compute MI on inner_train.
        2. Select top-K features.
        3. Fit StandardScaler on inner_train selected columns.
        4. Fit LogisticRegression on scaled inner_train.
        5. Transform inner_val with SAME scaler (no refit).
        6. Predict inner_val.
        7. Return macro-F1.

    The inner_val data NEVER changes encoder/MI/scaler state.

    Parameters
    ----------
    X_inner_train, y_inner_train : arrays for inner fold TRAIN
    X_inner_val, y_inner_val     : arrays for inner fold VALIDATION
    feature_names : list[str]    : aligned with X_inner_train columns
    k : int                      : number of features to select
    config : InnerCVConfig

    Returns
    -------
    float
        Macro-F1 on inner_val.
    """
    mi_config = MIConfig(
        n_neighbors=config.mi_n_neighbors,
        random_state=config.mi_random_state,
    )

    # 1-2. MI + feature selection (inner_train ONLY)
    mi_result = compute_mi_scores(
        X_inner_train,
        y_inner_train,
        feature_names,
        config=mi_config,
    )

    # Rank: select top-K by score descending
    sorted_indices = np.argsort(mi_result.mi_scores)[::-1]
    top_k_indices = sorted_indices[:k]

    X_train_sel = X_inner_train[:, top_k_indices]
    X_val_sel   = X_inner_val[:, top_k_indices]

    # 3. Fit scaler on inner_train selected (NEVER on inner_val)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)

    # 4. Fit LR evaluator on scaled inner_train (NEVER on inner_val)
    lr = LogisticRegression(
        solver=config.evaluator_solver,
        C=config.evaluator_C,
        max_iter=config.evaluator_max_iter,
        class_weight=config.evaluator_class_weight,
        random_state=config.evaluator_random_state,
    )
    lr.fit(X_train_scaled, y_inner_train)

    # 5-6. Transform inner_val with SAME frozen scaler; predict
    X_val_scaled = scaler.transform(X_val_sel)
    y_pred = lr.predict(X_val_scaled)

    # 7. Macro-F1
    return float(f1_score(y_inner_val, y_pred, average="macro", zero_division=0))

def _evaluate_k_from_sorted_indices(
    X_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    X_inner_val: np.ndarray,
    y_inner_val: np.ndarray,
    sorted_indices: np.ndarray,
    k: int,
    config: InnerCVConfig,
) -> float:
    """
    Evaluate one K value using pre-computed MI sorted indices.

    This is called once per K after MI has been computed for the fold.
    MI is NOT recomputed here — sorted_indices is the result of
    np.argsort(mi_scores)[::-1] computed by the caller.

    Steps (all fitting on inner_train ONLY):
        1. Select top-K indices from sorted_indices.
        2. Fit StandardScaler on inner_train selected columns.
        3. Fit LogisticRegression on scaled inner_train.
        4. Transform inner_val with SAME scaler (no refit).
        5. Predict inner_val.
        6. Return macro-F1.

    Parameters
    ----------
    X_inner_train, y_inner_train : inner fold TRAIN arrays
    X_inner_val, y_inner_val     : inner fold VALIDATION arrays
    sorted_indices : np.ndarray  : indices sorted by MI score descending
    k : int                      : number of features to select (slice [:k])
    config : InnerCVConfig

    Returns
    -------
    float
        Macro-F1 on inner_val.
    """
    top_k_indices = sorted_indices[:k]

    X_train_sel = X_inner_train[:, top_k_indices]
    X_val_sel   = X_inner_val[:, top_k_indices]

    # Fit scaler on inner_train ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)

    # Fit LR evaluator on inner_train ONLY
    lr = LogisticRegression(
        solver=config.evaluator_solver,
        C=config.evaluator_C,
        max_iter=config.evaluator_max_iter,
        class_weight=config.evaluator_class_weight,
        random_state=config.evaluator_random_state,
    )
    lr.fit(X_train_scaled, y_inner_train)

    # Transform inner_val with frozen scaler; predict
    X_val_scaled = scaler.transform(X_val_sel)
    y_pred = lr.predict(X_val_scaled)

    return float(f1_score(y_inner_val, y_pred, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Full K selection (5 folds; MI computed once per fold)
# ---------------------------------------------------------------------------


def run_k_selection_cv(
    train_df: pd.DataFrame,
    config: InnerCVConfig | None = None,
) -> KSelectionResult:
    """
    Run training-only inner cross-validation to select the optimal K.

    LEAKAGE CONTRACT:
        train_df MUST be the frozen TRAIN split only.
        This function never reads VALIDATION, TEST, or protected data.
        Every fold fitting step (encoder, MI, scaler, LR) is isolated
        to that fold's inner_train rows.

    Parameters
    ----------
    train_df : pd.DataFrame
        Frozen TRAIN DataFrame with all raw columns including label and
        attack_cat.
    config : InnerCVConfig | None
        CV configuration. If None, the frozen default is used.

    Returns
    -------
    KSelectionResult
        Contains selected_k, all per-fold records, and summary DataFrame.
    """
    if config is None:
        config = InnerCVConfig()

    logger.info(
        "=== K-SELECTION CV START | candidate_k=%s | n_splits=%d ===",
        config.candidate_k,
        config.n_splits,
    )

    # Separate features and target from the full TRAIN DataFrame
    cleaned = separate_target_and_features(train_df, split_name="inner_cv_train")
    y_all = cleaned.y.to_numpy(dtype=np.int64)

    # Build feature DataFrame (no label, no attack_cat, no id)
    X_df_all = cleaned.X_raw

    categorical_cols = list(cleaned.categorical_cols)
    numeric_cols = list(cleaned.numeric_cols)

    skf = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=config.shuffle,
        random_state=config.cv_random_state,
    )

    fold_records: list[KFoldRecord] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_df_all, y_all)):
        fold_num = fold_idx + 1
        logger.info("Fold %d/%d | train_n=%d | val_n=%d",
                    fold_num, config.n_splits, len(train_idx), len(val_idx))

        X_inner_train_df = X_df_all.iloc[train_idx]
        X_inner_val_df   = X_df_all.iloc[val_idx]
        y_inner_train    = y_all[train_idx]
        y_inner_val      = y_all[val_idx]

        # --- Build encoded matrices (encoder fitted on inner_train ONLY) ---
        (X_tr_ohe, X_val_ohe,
         X_tr_num, X_val_num,
         ohe_names, num_col_names) = _build_inner_encoded(
            X_inner_train_df, X_inner_val_df, categorical_cols
        )

        X_inner_train = np.concatenate([X_tr_ohe, X_tr_num], axis=1)
        X_inner_val   = np.concatenate([X_val_ohe, X_val_num], axis=1)
        feature_names = ohe_names + list(num_col_names)

        # --- Compute MI ONCE per fold (K-agnostic: ranking doesn't depend on K) ---
        # This is the architecturally correct approach: MI ranks all features;
        # different K values simply slice different prefixes of the same ranking.
        # Computing MI 25 times (once per K per fold) would be wasteful and wrong.
        mi_config = MIConfig(
            n_neighbors=config.mi_n_neighbors,
            random_state=config.mi_random_state,
        )
        mi_result = compute_mi_scores(
            X_inner_train,
            y_inner_train,
            feature_names,
            config=mi_config,
        )
        # Pre-sort indices once: descending MI score
        sorted_indices = np.argsort(mi_result.mi_scores)[::-1]

        # --- Evaluate each K from the same pre-computed ranking ---
        for k in config.candidate_k:
            macro_f1 = _evaluate_k_from_sorted_indices(
                X_inner_train, y_inner_train,
                X_inner_val, y_inner_val,
                sorted_indices, k, config,
            )
            logger.info("  K=%d | fold=%d | macro_f1=%.6f", k, fold_num, macro_f1)
            fold_records.append(KFoldRecord(k=k, fold=fold_num, macro_f1=macro_f1))

    result = select_best_k(fold_records, config)
    logger.info(
        "=== K-SELECTION COMPLETE | selected_k=%d | mean_f1=%.6f ===",
        result.selected_k,
        result.summary_df.loc[
            result.summary_df["k"] == result.selected_k, "mean_macro_f1"
        ].iloc[0],
    )
    return result


# ---------------------------------------------------------------------------
# K selection rule
# ---------------------------------------------------------------------------


def select_best_k(
    fold_records: list[KFoldRecord],
    config: InnerCVConfig | None = None,
) -> KSelectionResult:
    """
    Apply the K selection rule to a list of fold records.

    Selection rule (frozen, applied before observing results):
        1. Compute mean macro-F1 and std macro-F1 per K.
        2. Select K with the highest mean macro-F1.
        3. Tie-break: smallest K.

    Parameters
    ----------
    fold_records : list[KFoldRecord]
    config : InnerCVConfig | None

    Returns
    -------
    KSelectionResult
    """
    if config is None:
        config = InnerCVConfig()

    records_df = pd.DataFrame(
        [{"k": r.k, "fold": r.fold, "macro_f1": r.macro_f1} for r in fold_records]
    )

    summary = (
        records_df.groupby("k")["macro_f1"]
        .agg(mean_macro_f1="mean", std_macro_f1="std")
        .reset_index()
        .sort_values(
            by=["mean_macro_f1", "k"],
            ascending=[False, True],       # highest F1, then smallest K
        )
        .reset_index(drop=True)
    )

    selected_k = int(summary.iloc[0]["k"])

    return KSelectionResult(
        selected_k=selected_k,
        fold_records=fold_records,
        summary_df=summary,
        config=config,
    )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


FLAT_TOLERANCE: float = 1e-3
MONOTONIC_TOLERANCE: float = 1e-4


def check_selection_sanity(
    summary_df: pd.DataFrame,
    flat_tolerance: float = FLAT_TOLERANCE,
    monotonic_tolerance: float = MONOTONIC_TOLERANCE,
) -> KSelectionSanity:
    """
    Post-selection sanity check on mean macro-F1 across candidate K values.

    Checks:
        A. Flat test: if max(mean_f1) - min(mean_f1) <= flat_tolerance,
           flag REVIEW_REQUIRED (K selection may be uninformative).

        B. Monotonic test: if F1 is monotonically increasing through all K,
           flag REVIEW_REQUIRED (grid may not have found a plateau).

    This is a SANITY FLAG only. It does NOT automatically:
        - change the K grid
        - change the selection rule
        - add new K values

    Tolerances are documented here and in config.yaml.

    Parameters
    ----------
    summary_df : pd.DataFrame
        From KSelectionResult.summary_df. Columns: k, mean_macro_f1, std_macro_f1.
    flat_tolerance : float
        Tolerance for the flat test (default: 1e-3 = 0.001 F1 units).
    monotonic_tolerance : float
        Tolerance for the monotonic test (default: 1e-4).

    Returns
    -------
    KSelectionSanity
    """
    # Sort by K ascending for monotonic check
    df_sorted = summary_df.sort_values("k").reset_index(drop=True)
    f1_vals = df_sorted["mean_macro_f1"].to_numpy()
    k_vals  = df_sorted["k"].to_numpy()

    flat_range = float(f1_vals.max() - f1_vals.min())

    # Flat test
    is_flat = flat_range <= flat_tolerance

    # Monotonic test: each step must be non-decreasing within tolerance
    # i.e., F1[i] <= F1[i+1] + monotonic_tolerance for all i
    is_monotonic = all(
        f1_vals[i] <= f1_vals[i + 1] + monotonic_tolerance
        for i in range(len(f1_vals) - 1)
    )

    if is_flat:
        status = "REVIEW_REQUIRED"
        reason = (
            f"FLAT: All candidate K mean macro-F1 values lie within a range of "
            f"{flat_range:.6f} (tolerance={flat_tolerance}). "
            "K selection may be uninformative — performance is essentially flat "
            "across all candidate K values."
        )
    elif is_monotonic:
        status = "REVIEW_REQUIRED"
        reason = (
            f"MONOTONIC: Mean macro-F1 is monotonically non-decreasing across "
            f"K={list(k_vals)} (within tolerance={monotonic_tolerance}). "
            "The selected K grid may not have reached a meaningful plateau. "
            "Consider whether a larger K grid would be appropriate, but do NOT "
            "change the K grid without human review."
        )
    else:
        status = "PASS"
        reason = (
            f"Flat range={flat_range:.6f} > tolerance={flat_tolerance}. "
            f"Not monotonically increasing. K selection appears informative."
        )

    logger.info(
        "Sanity check | status=%s | flat_range=%.6f | is_monotonic=%s | reason=%s",
        status, flat_range, is_monotonic, reason,
    )

    return KSelectionSanity(
        status=status,
        reason=reason,
        flat_range=flat_range,
        flat_tolerance=flat_tolerance,
        is_monotonic=is_monotonic,
        monotonic_tolerance=monotonic_tolerance,
    )
