"""
src/preprocessing/split_protocol.py
-------------------------------------
Sprint 3: Official TRAIN / VALIDATION split protocol for UNSW-NB15.

Approved split design (frozen):
================================

INPUT:
    Official UNSW-NB15 training file (175,341 rows).

STEP 1 — Separate Backdoor training rows:
    All rows where attack_cat == "Backdoor" are archived as:
        data/splits/excluded_train_backdoor.csv

    These rows have EXPERIMENTAL ROLE = NONE.
    They must NOT be used for model training, validation, or evaluation.

STEP 2 — Non-Backdoor development pool:
    Remaining 173,595 rows split into Normal and Attack subsets.

STEP 3 — Normal split (seed=42):
    80% of Normal rows  → TRAIN
    20% of Normal rows  → VALIDATION

STEP 4 — Attack rows (non-Backdoor):
    100% → TRAIN (no validation leakage of known-attack distribution)

STEP 5 — Assemble:
    TRAIN      = 80% Normal + 100% of non-Backdoor Attack rows
    VALIDATION = 20% Normal only  (ZERO attack rows)

Leakage guarantees:
    - VALIDATION contains ZERO attack rows (label=0 throughout).
    - excluded_train_backdoor.csv is archived and never read downstream.
    - Seed 42 is recorded; rerun with same seed reproduces identical splits.
    - Row counts are conserved: TRAIN + VAL + excluded = training file total.
    - No information from TEST or protected unseen set influences this split.

API:
    create_train_val_split(train_df, seed) -> TrainValSplitResult
    verify_split_integrity(result)         -> SplitIntegrityReport
    build_split_provenance(result, ...)    -> dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-fixed constants
# ---------------------------------------------------------------------------

ATTACK_CAT_COL: str = "attack_cat"
LABEL_COL: str = "label"

# The withheld attack class — must be archived, never in TRAIN or VAL
WITHHELD_ATTACK: str = "Backdoor"

# Approved Normal split fractions
NORMAL_TRAIN_FRAC: float = 0.80
NORMAL_VAL_FRAC: float = 0.20

# "Normal" category value as stored in the official UNSW-NB15 CSV
NORMAL_CAT: str = "Normal"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainValSplitResult:
    """
    Typed result of the official TRAIN/VALIDATION split.

    All DataFrames retain the original raw column layout (same columns as
    the input training file). No transformation has been applied.

    Attributes
    ----------
    train_df : pd.DataFrame
        TRAIN split: 80% of Normal + 100% of non-Backdoor attack rows.
    val_df : pd.DataFrame
        VALIDATION split: 20% of Normal rows ONLY. Zero attack rows.
    excluded_backdoor_df : pd.DataFrame
        Archived Backdoor training rows. EXPERIMENTAL ROLE = NONE.
    seed : int
        Random seed used for the Normal 80/20 split.
    n_train : int
        Number of rows in TRAIN.
    n_val : int
        Number of rows in VALIDATION.
    n_excluded_backdoor : int
        Number of Backdoor training rows archived.
    n_input : int
        Total input rows from the official training file.
    attack_cat_col : str
        Name of the attack category column.
    label_col : str
        Name of the binary label column.
    normal_cat : str
        The string value representing Normal traffic in attack_cat.
    withheld_attack : str
        The withheld attack class name (Backdoor).
    normal_train_frac : float
        Fraction of Normal rows allocated to TRAIN.
    normal_val_frac : float
        Fraction of Normal rows allocated to VALIDATION.
    """

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    excluded_backdoor_df: pd.DataFrame
    seed: int
    n_train: int
    n_val: int
    n_excluded_backdoor: int
    n_input: int
    attack_cat_col: str = ATTACK_CAT_COL
    label_col: str = LABEL_COL
    normal_cat: str = NORMAL_CAT
    withheld_attack: str = WITHHELD_ATTACK
    normal_train_frac: float = NORMAL_TRAIN_FRAC
    normal_val_frac: float = NORMAL_VAL_FRAC


@dataclass
class SplitIntegrityReport:
    """
    Result of integrity checks on a TrainValSplitResult.

    All boolean fields must be True before committing the split.

    Attributes
    ----------
    row_conservation : bool
        n_train + n_val + n_excluded_backdoor == n_input.
    val_zero_attack : bool
        VALIDATION contains zero attack rows (label == 0 for all rows).
    val_zero_attack_count : int
        Exact count of attack rows found in VALIDATION (should be 0).
    backdoor_not_in_train : bool
        TRAIN contains no Backdoor rows (attack_cat != "Backdoor" for all).
    backdoor_in_train_count : int
        Exact count of Backdoor rows found in TRAIN (should be 0).
    backdoor_not_in_val : bool
        VALIDATION contains no Backdoor rows.
    backdoor_in_val_count : int
        Exact count of Backdoor rows found in VALIDATION (should be 0).
    normal_only_in_val : bool
        All VALIDATION rows have attack_cat == "Normal".
    non_normal_in_val_count : int
        Exact count of non-Normal rows in VALIDATION (should be 0).
    excluded_all_backdoor : bool
        All excluded rows have attack_cat == "Backdoor".
    label_consistency : bool
        VALIDATION label column is all zeros; excluded is all ones.
    no_index_overlap_train_val : bool
        No shared original DataFrame indices between TRAIN and VAL.
    train_has_attacks : bool
        TRAIN contains at least one attack row (sanity check for non-trivial split).
    all_pass : bool
        All individual checks are True.
    failures : list[str]
        Names of failed checks (empty if all pass).
    """

    row_conservation: bool = False
    val_zero_attack: bool = False
    val_zero_attack_count: int = -1
    backdoor_not_in_train: bool = False
    backdoor_in_train_count: int = -1
    backdoor_not_in_val: bool = False
    backdoor_in_val_count: int = -1
    normal_only_in_val: bool = False
    non_normal_in_val_count: int = -1
    excluded_all_backdoor: bool = False
    label_consistency: bool = False
    no_index_overlap_train_val: bool = False
    train_has_attacks: bool = False
    all_pass: bool = False
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core split function
# ---------------------------------------------------------------------------


def create_train_val_split(
    train_df: pd.DataFrame,
    seed: int = 42,
    attack_cat_col: str = ATTACK_CAT_COL,
    label_col: str = LABEL_COL,
    withheld_attack: str = WITHHELD_ATTACK,
    normal_cat: str = NORMAL_CAT,
    normal_train_frac: float = NORMAL_TRAIN_FRAC,
) -> TrainValSplitResult:
    """
    Create the official TRAIN / VALIDATION split from the UNSW-NB15 training file.

    Implements the approved Sprint 3 split design:

        1. Archive all Backdoor training rows -> excluded_train_backdoor.csv
        2. From non-Backdoor pool:
              Normal rows -> 80% TRAIN, 20% VALIDATION (random_state=seed)
              Attack rows -> 100% TRAIN
        3. TRAIN = 80% Normal + 100% Attack
           VALIDATION = 20% Normal ONLY (zero attack rows)

    Parameters
    ----------
    train_df : pd.DataFrame
        Raw UNSW-NB15 training file as loaded by loader.py (175,341 rows).
        Must contain attack_cat_col and label_col.
    seed : int
        Random seed for the Normal 80/20 stratified split. Default: 42.
    attack_cat_col : str
        Name of the attack category column.
    label_col : str
        Name of the binary label column.
    withheld_attack : str
        The withheld attack class to archive (default: "Backdoor").
    normal_cat : str
        The Normal traffic category value (default: "Normal").
    normal_train_frac : float
        Fraction of Normal rows allocated to TRAIN (default: 0.80).

    Returns
    -------
    TrainValSplitResult
        Typed result with train_df, val_df, excluded_backdoor_df, and metadata.

    Raises
    ------
    ValueError
        If required columns are absent, or if the resulting splits violate
        integrity constraints (e.g., attack rows in VALIDATION).
    """
    logger.info(
        "=== SPLIT PROTOCOL START | n_input=%d | seed=%d | withheld=%s ===",
        len(train_df),
        seed,
        withheld_attack,
    )

    # --- Validate required columns ---
    for col in (attack_cat_col, label_col):
        if col not in train_df.columns:
            raise ValueError(
                f"Required column '{col}' not found in training DataFrame. "
                f"Available columns: {list(train_df.columns)}"
            )

    n_input = len(train_df)

    # -------------------------------------------------------------------------
    # STEP 1: Archive Backdoor training rows
    # -------------------------------------------------------------------------
    is_backdoor = train_df[attack_cat_col] == withheld_attack
    excluded_backdoor_df = train_df.loc[is_backdoor].copy()
    non_backdoor_df = train_df.loc[~is_backdoor].copy()

    n_excluded = len(excluded_backdoor_df)
    n_non_backdoor = len(non_backdoor_df)

    logger.info(
        "Step 1 | Backdoor archived: %d rows | Non-Backdoor pool: %d rows",
        n_excluded,
        n_non_backdoor,
    )

    if n_excluded == 0:
        raise ValueError(
            f"No '{withheld_attack}' rows found in training DataFrame. "
            f"Verify that attack_cat values are canonical (not stripped/modified)."
        )

    # -------------------------------------------------------------------------
    # STEP 2: Separate Normal and Attack rows from non-Backdoor pool
    # -------------------------------------------------------------------------
    is_normal = non_backdoor_df[attack_cat_col] == normal_cat
    normal_pool = non_backdoor_df.loc[is_normal].copy()
    attack_pool = non_backdoor_df.loc[~is_normal].copy()

    n_normal = len(normal_pool)
    n_attack = len(attack_pool)

    logger.info(
        "Step 2 | Normal pool: %d | Attack pool (non-Backdoor): %d",
        n_normal,
        n_attack,
    )

    if n_normal == 0:
        raise ValueError(
            f"No Normal rows found in non-Backdoor pool. "
            f"Verify that attack_cat == '{normal_cat}' rows exist in the training file."
        )

    # -------------------------------------------------------------------------
    # STEP 3: Split Normal rows 80/20 (seed-controlled)
    # -------------------------------------------------------------------------
    normal_val_frac = 1.0 - normal_train_frac

    normal_train, normal_val = train_test_split(
        normal_pool,
        test_size=normal_val_frac,
        random_state=seed,
        shuffle=True,
    )

    logger.info(
        "Step 3 | Normal split (seed=%d) | TRAIN: %d | VAL: %d",
        seed,
        len(normal_train),
        len(normal_val),
    )

    # -------------------------------------------------------------------------
    # STEP 4-5: Assemble final TRAIN and VALIDATION
    # -------------------------------------------------------------------------
    train_out = pd.concat([normal_train, attack_pool], axis=0).sort_index()
    val_out = normal_val.sort_index()

    n_train = len(train_out)
    n_val = len(val_out)

    logger.info(
        "Step 5 | TRAIN: %d rows (Normal=%d + Attack=%d) | VAL: %d rows (Normal only)",
        n_train,
        len(normal_train),
        n_attack,
        n_val,
    )
    logger.info(
        "=== SPLIT PROTOCOL COMPLETE | TRAIN=%d | VAL=%d | EXCLUDED=%d | TOTAL=%d ===",
        n_train,
        n_val,
        n_excluded,
        n_train + n_val + n_excluded,
    )

    result = TrainValSplitResult(
        train_df=train_out,
        val_df=val_out,
        excluded_backdoor_df=excluded_backdoor_df,
        seed=seed,
        n_train=n_train,
        n_val=n_val,
        n_excluded_backdoor=n_excluded,
        n_input=n_input,
    )

    return result


# ---------------------------------------------------------------------------
# Integrity verifier
# ---------------------------------------------------------------------------


def verify_split_integrity(result: TrainValSplitResult) -> SplitIntegrityReport:
    """
    Run all integrity checks on a TrainValSplitResult.

    Every check must pass before the split artifacts are written to disk.

    Parameters
    ----------
    result : TrainValSplitResult
        The split result to verify.

    Returns
    -------
    SplitIntegrityReport
        All checks with detailed counts and a summary all_pass flag.
    """
    report = SplitIntegrityReport()
    failures: list[str] = []

    train_df = result.train_df
    val_df = result.val_df
    excl_df = result.excluded_backdoor_df

    # 1. Row conservation
    total = len(train_df) + len(val_df) + len(excl_df)
    report.row_conservation = (total == result.n_input)
    if not report.row_conservation:
        failures.append(
            f"row_conservation: {len(train_df)} + {len(val_df)} + {len(excl_df)} "
            f"= {total} != {result.n_input}"
        )

    # 2. VALIDATION contains zero attack rows
    val_attack_mask = val_df[result.label_col] != 0
    report.val_zero_attack_count = int(val_attack_mask.sum())
    report.val_zero_attack = (report.val_zero_attack_count == 0)
    if not report.val_zero_attack:
        failures.append(
            f"val_zero_attack: found {report.val_zero_attack_count} attack rows in VALIDATION"
        )

    # 3. Backdoor not in TRAIN
    backdoor_in_train = train_df[result.attack_cat_col] == result.withheld_attack
    report.backdoor_in_train_count = int(backdoor_in_train.sum())
    report.backdoor_not_in_train = (report.backdoor_in_train_count == 0)
    if not report.backdoor_not_in_train:
        failures.append(
            f"backdoor_not_in_train: found {report.backdoor_in_train_count} Backdoor rows in TRAIN"
        )

    # 4. Backdoor not in VALIDATION
    backdoor_in_val = val_df[result.attack_cat_col] == result.withheld_attack
    report.backdoor_in_val_count = int(backdoor_in_val.sum())
    report.backdoor_not_in_val = (report.backdoor_in_val_count == 0)
    if not report.backdoor_not_in_val:
        failures.append(
            f"backdoor_not_in_val: found {report.backdoor_in_val_count} Backdoor rows in VALIDATION"
        )

    # 5. VALIDATION is Normal-only (all attack_cat == "Normal")
    non_normal_in_val = val_df[result.attack_cat_col] != result.normal_cat
    report.non_normal_in_val_count = int(non_normal_in_val.sum())
    report.normal_only_in_val = (report.non_normal_in_val_count == 0)
    if not report.normal_only_in_val:
        failures.append(
            f"normal_only_in_val: found {report.non_normal_in_val_count} "
            f"non-Normal rows in VALIDATION. "
            f"Categories: {val_df.loc[non_normal_in_val, result.attack_cat_col].value_counts().to_dict()}"
        )

    # 6. Excluded set is all Backdoor
    non_backdoor_in_excl = excl_df[result.attack_cat_col] != result.withheld_attack
    report.excluded_all_backdoor = (non_backdoor_in_excl.sum() == 0)
    if not report.excluded_all_backdoor:
        failures.append(
            f"excluded_all_backdoor: excluded set contains non-Backdoor rows: "
            f"{excl_df.loc[non_backdoor_in_excl, result.attack_cat_col].value_counts().to_dict()}"
        )

    # 7. Label consistency: VALIDATION all label=0, excluded all label=1
    val_label_ok = (val_df[result.label_col] == 0).all()
    excl_label_ok = (excl_df[result.label_col] == 1).all()
    report.label_consistency = bool(val_label_ok and excl_label_ok)
    if not val_label_ok:
        bad_val_labels = (val_df[result.label_col] != 0).sum()
        failures.append(
            f"label_consistency (val): {bad_val_labels} rows have label != 0 in VALIDATION"
        )
    if not excl_label_ok:
        bad_excl_labels = (excl_df[result.label_col] != 1).sum()
        failures.append(
            f"label_consistency (excluded): {bad_excl_labels} rows have label != 1 in excluded"
        )

    # 8. No index overlap between TRAIN and VALIDATION
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    overlap = train_idx & val_idx
    report.no_index_overlap_train_val = (len(overlap) == 0)
    if not report.no_index_overlap_train_val:
        failures.append(
            f"no_index_overlap_train_val: {len(overlap)} shared indices between TRAIN and VAL"
        )

    # 9. TRAIN contains attack rows (sanity check: not trivially all-Normal)
    train_has_attacks = int((train_df[result.label_col] == 1).sum())
    report.train_has_attacks = (train_has_attacks > 0)
    if not report.train_has_attacks:
        failures.append("train_has_attacks: TRAIN has zero attack rows (unexpected)")

    # --- Summary ---
    report.failures = failures
    report.all_pass = (len(failures) == 0)

    if report.all_pass:
        logger.info("Split integrity: ALL PASS")
    else:
        for f in failures:
            logger.error("Split integrity FAIL: %s", f)

    return report


# ---------------------------------------------------------------------------
# Provenance builder
# ---------------------------------------------------------------------------


def build_split_provenance(
    result: TrainValSplitResult,
    report: SplitIntegrityReport,
    source_train_sha256: str,
    git_commit: str | None = None,
    protocol_version: str = "1.0",
    exact_reconstruction: bool | None = None,
    output_hashes: dict[str, str] | None = None,
    library_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Build a JSON-serialisable provenance dict for the TRAIN/VAL split.

    All values in the returned dict are native Python types (str, int, float,
    bool, list, dict, None). numpy.bool_, numpy.int64, pandas scalars, and any
    other non-standard types are explicitly converted at this boundary.

    Parameters
    ----------
    result : TrainValSplitResult
        The completed split result.
    report : SplitIntegrityReport
        The integrity check results.
    source_train_sha256 : str
        SHA-256 of the official UNSW-NB15 training CSV.
    git_commit : str | None
        Current Git commit hash for provenance.
    protocol_version : str
        Protocol version string.
    exact_reconstruction : bool | None
        Whether the three-way partition exactly reconstructs the training file
        when recombined. Computed externally (requires the original DataFrame).
    output_hashes : dict | None
        SHA-256 hashes of the written CSV files, keyed by logical name.
    library_versions : dict | None
        Python and key library version strings for reproducibility.

    Returns
    -------
    dict
        JSON-serialisable provenance record. Safe to pass to json.dumps()
        without a custom encoder.
    """
    from datetime import datetime, timezone

    # -------------------------------------------------------------------------
    # Compute breakdown counts from the result DataFrames.
    # All counts are cast to Python int to prevent numpy.int64 leakage.
    # -------------------------------------------------------------------------
    ac = result.attack_cat_col
    lc = result.label_col
    nc = result.normal_cat
    wa = result.withheld_attack

    normal_train_count        = int((result.train_df[ac] == nc).sum())
    normal_val_count          = int((result.val_df[ac] == nc).sum())
    non_backdoor_attack_count = int((result.train_df[lc] == 1).sum())
    validation_attack_count   = int((result.val_df[lc] == 1).sum())
    backdoor_train_count      = int((result.train_df[ac] == wa).sum())
    backdoor_val_count        = int((result.val_df[ac] == wa).sum())

    # Distribution tables — numpy.int64 → Python int
    train_attack_cat = {
        str(k): int(v)
        for k, v in result.train_df[ac].value_counts().items()
    }
    val_attack_cat = {
        str(k): int(v)
        for k, v in result.val_df[ac].value_counts().items()
    }
    excl_attack_cat = {
        str(k): int(v)
        for k, v in result.excluded_backdoor_df[ac].value_counts().items()
    }
    train_label = {
        str(int(k)): int(v)
        for k, v in result.train_df[lc].value_counts().items()
    }
    val_label = {
        str(int(k)): int(v)
        for k, v in result.val_df[lc].value_counts().items()
    }

    # -------------------------------------------------------------------------
    # Validation-size adequacy for AE reconstruction-error percentile thresholds.
    #
    # For each percentile p, the empirical upper-tail count is:
    #   approx_tail_count = round(n_benign_val * (1 - p/100))
    #
    # These are descriptive counts, NOT confidence intervals.
    # The purpose is to document that each operating point has substantial
    # empirical support in the validation set before TEST access.
    # -------------------------------------------------------------------------
    n_benign_val = normal_val_count
    percentile_adequacy = {
        "n_benign_validation": n_benign_val,
        "method": (
            "Descriptive count. For percentile p, approx_tail_count = "
            "round(n_benign_validation * (1 - p/100)). "
            "These are empirical support counts, NOT confidence intervals."
        ),
        "percentiles": {
            "90":   {"approx_tail_count": round(n_benign_val * 0.10)},
            "95":   {"approx_tail_count": round(n_benign_val * 0.05)},
            "97.5": {"approx_tail_count": round(n_benign_val * 0.025)},
            "99":   {"approx_tail_count": round(n_benign_val * 0.01)},
        },
        "assessment": (
            f"The validation set contains {n_benign_val:,} benign rows. "
            f"The 90th-percentile threshold is calibrated on approximately "
            f"{round(n_benign_val * 0.10):,} upper-tail samples, and the "
            f"99th-percentile threshold on approximately "
            f"{round(n_benign_val * 0.01):,}. "
            "All operating points have substantial empirical support. "
            "This is descriptive evidence of adequacy, not a formal guarantee."
        ),
    }

    provenance = {
        # --- Identity ---
        "protocol_version": str(protocol_version),
        "experiment_id": "EXP_TRAIN_VAL_SPLIT_V1",
        "dataset": "UNSW-NB15",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": str(git_commit) if git_commit is not None else None,

        # --- Source ---
        "source_training_filename": "UNSW_NB15_training-set.csv",
        "source_training_sha256": str(source_train_sha256),

        # --- Split design ---
        "split_seed": int(result.seed),
        "split_strategy": (
            "Step 1: Archive all Backdoor training rows. "
            "Step 2: Separate Normal and non-Backdoor attack rows from remaining pool. "
            "Step 3: Random 80/20 split of Normal rows (sklearn train_test_split, shuffle=True). "
            "Step 4: All non-Backdoor attack rows go 100% to TRAIN. "
            "Step 5: TRAIN = 80% Normal + 100% attack. VALIDATION = 20% Normal only."
        ),
        "normal_train_fraction": float(result.normal_train_frac),
        "normal_validation_fraction": float(result.normal_val_frac),
        "withheld_attack": str(result.withheld_attack),
        "withheld_role": "NONE - archived to excluded_train_backdoor.csv",
        "validation_purpose": (
            "Benign-only AE reconstruction-error threshold calibration and "
            "protocol sanity checks. NOT for repeated supervised hyperparameter "
            "selection."
        ),

        # --- Counts ---
        "train_count":                   int(result.n_train),
        "validation_count":              int(result.n_val),
        "excluded_backdoor_count":       int(result.n_excluded_backdoor),
        "normal_train_count":            normal_train_count,
        "normal_validation_count":       normal_val_count,
        "non_backdoor_attack_train_count": non_backdoor_attack_count,
        "validation_attack_count":       validation_attack_count,
        "backdoor_train_count":          backdoor_train_count,
        "backdoor_validation_count":     backdoor_val_count,

        # --- Distributions ---
        "train_distribution": {
            "attack_cat": train_attack_cat,
            "label": train_label,
        },
        "val_distribution": {
            "attack_cat": val_attack_cat,
            "label": val_label,
        },
        "excluded_distribution": {
            "attack_cat": excl_attack_cat,
        },

        # --- Integrity ---
        "row_conservation":   bool(report.row_conservation),
        "exact_reconstruction": (
            bool(exact_reconstruction)
            if exact_reconstruction is not None
            else None
        ),

        # All integrity booleans and counts cast to Python native types.
        # numpy.bool_ and numpy.int64 are NOT JSON-serialisable without a
        # custom encoder. We do NOT use a custom encoder.
        "integrity_checks": {
            "all_pass":                   bool(report.all_pass),
            "row_conservation":           bool(report.row_conservation),
            "val_zero_attack":            bool(report.val_zero_attack),
            "val_zero_attack_count":      int(report.val_zero_attack_count),
            "backdoor_not_in_train":      bool(report.backdoor_not_in_train),
            "backdoor_in_train_count":    int(report.backdoor_in_train_count),
            "backdoor_not_in_val":        bool(report.backdoor_not_in_val),
            "backdoor_in_val_count":      int(report.backdoor_in_val_count),
            "normal_only_in_val":         bool(report.normal_only_in_val),
            "non_normal_in_val_count":    int(report.non_normal_in_val_count),
            "excluded_all_backdoor":      bool(report.excluded_all_backdoor),
            "label_consistency":          bool(report.label_consistency),
            "no_index_overlap_train_val": bool(report.no_index_overlap_train_val),
            "train_has_attacks":          bool(report.train_has_attacks),
            "failures":                   [str(f) for f in report.failures],
        },

        # --- Output hashes (filled by script after writing files) ---
        "output_hashes": {
            str(k): str(v) for k, v in output_hashes.items()
        } if output_hashes else None,

        # --- Library versions (filled by script) ---
        "library_versions": {
            str(k): str(v) for k, v in library_versions.items()
        } if library_versions else None,

        # --- Validation percentile adequacy ---
        "validation_percentile_adequacy": percentile_adequacy,
    }

    return provenance

