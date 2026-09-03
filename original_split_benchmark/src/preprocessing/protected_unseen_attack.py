"""
src/preprocessing/protected_unseen_attack.py
----------------------------------------------
Deterministic protected unseen-attack reservation for UNSW-NB15 IDS project.

This module partitions the official TEST file into:
    1. protected_unseen_attack  — ALL rows whose canonical attack_cat = Backdoor
    2. development_test         — ALL remaining TEST rows (non-Backdoor)

KEY DESIGN DECISIONS:
- ALL Backdoor rows from TEST are used (no random sampling).
- The raw source file is NEVER modified.
- The partition is fully deterministic (no random operations).
- The original row count is conserved exactly.
- Both output DataFrames are validated before writing.

LEAKAGE POLICY:
The protected unseen-attack set must NEVER be used for:
    - MI feature selection
    - scaler fitting
    - encoder fitting
    - model training
    - OOF stacking
    - meta-learner training
    - AE training
    - threshold calibration
    - hyperparameter tuning
    - model selection

It is used ONLY in the final evaluation phase of later sprints.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from original_split_benchmark.src.preprocessing.attack_cat_canonicalization import (
    CANONICALIZATION_VERSION,
    canonicalize_attack_cat,
)
from original_split_benchmark.src.utils.hashing import sha256_dataframe, sha256_file


_WITHHELD_TARGET = "Backdoor"
_ELIGIBILITY_THRESHOLD = 50


class ReservationError(RuntimeError):
    """Raised when any validation check in the reservation process fails."""


def reserve_protected_unseen_attack(
    df_test: pd.DataFrame,
    cat_col: str,
    withheld_target: str = _WITHHELD_TARGET,
    min_count: int = _ELIGIBILITY_THRESHOLD,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partition the test DataFrame into protected and development sets.

    Parameters
    ----------
    df_test : pd.DataFrame
        The raw, unmodified TEST DataFrame (loaded from raw CSV).
    cat_col : str
        Name of the attack-category column.
    withheld_target : str
        Canonical name of the withheld attack class (default "Backdoor").
    min_count : int
        Minimum acceptable row count for the protected set (default 50).
    logger : logging.Logger | None
        Optional logger.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (protected_unseen_attack, development_test)
        Both DataFrames have identical column order to df_test.
        Original index values are preserved (not reset).

    Raises
    ------
    ReservationError
        If any validation check fails (missing column, count below threshold,
        conservation mismatch, reconstruction failure, etc.).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    original_row_count = len(df_test)
    logger.info(
        "Starting reservation | original_test_rows=%d | target=%s",
        original_row_count,
        withheld_target,
    )

    # --- Step 1: Validate cat_col exists ---
    if cat_col not in df_test.columns:
        raise ReservationError(
            f"Attack-category column '{cat_col}' not found in test DataFrame. "
            f"Available columns: {list(df_test.columns)}"
        )

    # --- Step 2: Canonicalize attack_cat (new column, original preserved) ---
    logger.info("Applying canonicalization to attack_cat column.")
    canonical_series = canonicalize_attack_cat(df_test[cat_col], logger=logger)

    # Log raw vs canonical mapping
    raw_unique = df_test[cat_col].astype(str).unique().tolist()
    can_unique = canonical_series.astype(str).unique().tolist()
    logger.info("Raw attack_cat unique values: %s", sorted(raw_unique))
    logger.info("Canonical attack_cat unique values: %s", sorted(can_unique))

    # --- Step 3: Check that withheld_target exists in canonical values ---
    canonical_counts = canonical_series.value_counts(dropna=False).to_dict()
    canonical_counts_str = {str(k): int(v) for k, v in canonical_counts.items()}

    if withheld_target not in canonical_counts_str:
        raise ReservationError(
            f"Withheld target '{withheld_target}' not found in canonical attack_cat values. "
            f"Canonical values present: {sorted(canonical_counts_str.keys())}\n"
            f"STOP: Do not silently select a different target."
        )

    backdoor_count = canonical_counts_str[withheld_target]
    logger.info(
        "Backdoor count in TEST | canonical_name=%s | count=%d",
        withheld_target,
        backdoor_count,
    )

    # --- Step 4: Eligibility check ---
    if backdoor_count < min_count:
        raise ReservationError(
            f"Withheld target '{withheld_target}' has only {backdoor_count} TEST instances "
            f"(eligibility threshold: {min_count}). "
            f"STOP: Cannot proceed with withheld-attack protocol."
        )

    # --- Step 5: Partition ---
    backdoor_mask = canonical_series == withheld_target
    df_protected = df_test[backdoor_mask].copy()
    df_development = df_test[~backdoor_mask].copy()

    logger.info(
        "Partition complete | protected=%d | development=%d",
        len(df_protected),
        len(df_development),
    )

    # --- Step 6: Validation checks ---
    _validate_partition(
        df_test=df_test,
        df_protected=df_protected,
        df_development=df_development,
        canonical_series=canonical_series,
        cat_col=cat_col,
        withheld_target=withheld_target,
        original_row_count=original_row_count,
        logger=logger,
    )

    logger.info("All validation checks PASSED.")
    return df_protected, df_development


def _validate_partition(
    df_test: pd.DataFrame,
    df_protected: pd.DataFrame,
    df_development: pd.DataFrame,
    canonical_series: pd.Series,
    cat_col: str,
    withheld_target: str,
    original_row_count: int,
    logger: logging.Logger,
) -> None:
    """Run all post-partition validation checks. Raises ReservationError on failure."""

    # CHECK 1: Protected set is non-empty
    if len(df_protected) == 0:
        raise ReservationError("CHECK 1 FAILED: Protected set is empty.")
    logger.info("CHECK 1 PASS: Protected set is non-empty (%d rows).", len(df_protected))

    # CHECK 2: Protected set contains ONLY withheld_target (canonical)
    protected_canonical = canonicalize_attack_cat(df_protected[cat_col])
    non_backdoor_in_protected = protected_canonical[protected_canonical != withheld_target]
    if len(non_backdoor_in_protected) > 0:
        raise ReservationError(
            f"CHECK 2 FAILED: Protected set contains {len(non_backdoor_in_protected)} "
            f"non-Backdoor rows. Values found: "
            f"{non_backdoor_in_protected.unique().tolist()}"
        )
    logger.info("CHECK 2 PASS: Protected set contains only '%s'.", withheld_target)

    # CHECK 3: Development test contains ZERO withheld_target rows
    dev_canonical = canonicalize_attack_cat(df_development[cat_col])
    backdoor_in_dev = dev_canonical[dev_canonical == withheld_target]
    if len(backdoor_in_dev) > 0:
        raise ReservationError(
            f"CHECK 3 FAILED: Development test contains {len(backdoor_in_dev)} "
            f"'{withheld_target}' rows. These must all be in the protected set."
        )
    logger.info("CHECK 3 PASS: Development test contains zero '%s' rows.", withheld_target)

    # CHECK 4: Row count conservation
    total = len(df_protected) + len(df_development)
    if total != original_row_count:
        raise ReservationError(
            f"CHECK 4 FAILED: Row count not conserved. "
            f"protected={len(df_protected)} + development={len(df_development)} "
            f"= {total} != original={original_row_count}."
        )
    logger.info(
        "CHECK 4 PASS: Row conservation | %d + %d = %d.",
        len(df_protected),
        len(df_development),
        original_row_count,
    )

    # CHECK 5: Column order preserved
    if list(df_protected.columns) != list(df_test.columns):
        raise ReservationError("CHECK 5 FAILED: Protected set has different column order.")
    if list(df_development.columns) != list(df_test.columns):
        raise ReservationError("CHECK 5 FAILED: Development test has different column order.")
    logger.info("CHECK 5 PASS: Column order preserved in both output sets.")

    # CHECK 6: Exact reconstruction
    reconstructed = pd.concat([df_protected, df_development], ignore_index=False)
    reconstructed_sorted = reconstructed.sort_index()
    original_sorted = df_test.sort_index()

    # Compare by shape first (fast)
    if reconstructed_sorted.shape != original_sorted.shape:
        raise ReservationError(
            f"CHECK 6 FAILED: Reconstruction shape mismatch. "
            f"reconstructed={reconstructed_sorted.shape} vs original={original_sorted.shape}"
        )

    # Compare hash of both (authoritative)
    hash_original = sha256_dataframe(original_sorted.reset_index(drop=True))
    hash_reconstructed = sha256_dataframe(reconstructed_sorted.reset_index(drop=True))
    if hash_original != hash_reconstructed:
        raise ReservationError(
            "CHECK 6 FAILED: Reconstruction hash mismatch. "
            "The two partitions do not exactly reproduce the original TEST data."
        )
    logger.info("CHECK 6 PASS: Exact reconstruction verified (hash match).")


def build_split_metadata(
    source_test_path: Path,
    df_protected: pd.DataFrame,
    df_development: pd.DataFrame,
    original_row_count: int,
    source_sha256: str,
    withheld_target: str = _WITHHELD_TARGET,
    protocol_version: str = "1.0",
) -> dict[str, Any]:
    """
    Build the split_metadata.json content dict.

    Parameters
    ----------
    source_test_path : Path
        Path to the original raw test CSV.
    df_protected : pd.DataFrame
        Protected unseen-attack DataFrame.
    df_development : pd.DataFrame
        Development test DataFrame.
    original_row_count : int
        Row count of the original test file.
    source_sha256 : str
        SHA-256 of the original raw test file (pre-computed).
    withheld_target : str
        Canonical name of the withheld class.
    protocol_version : str
        Protocol version string.

    Returns
    -------
    dict
        Split metadata ready for JSON serialisation.
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    protected_hash = sha256_dataframe(df_protected.reset_index(drop=True))
    development_hash = sha256_dataframe(df_development.reset_index(drop=True))

    conservation_ok = (len(df_protected) + len(df_development)) == original_row_count

    return {
        "protocol_version": protocol_version,
        "dataset": "UNSW-NB15",
        "source_test_file": source_test_path.name,
        "source_test_sha256": source_sha256,
        "original_test_row_count": original_row_count,
        "withheld_target": withheld_target,
        "selection_rule": "fixed_named_target",
        "eligibility_threshold": _ELIGIBILITY_THRESHOLD,
        "protected_row_count": len(df_protected),
        "development_test_row_count": len(df_development),
        "raw_attack_cat_values": sorted(
            df_protected["attack_cat"].astype(str).unique().tolist()
            if "attack_cat" in df_protected.columns else []
        ),
        "protected_sha256": protected_hash,
        "development_test_sha256": development_hash,
        "row_conservation": "PASS" if conservation_ok else "FAIL",
        "reconstruction_verified": "PASS",
        "created_at": now_utc,
        "seed": None,
        "notes": (
            "No random sampling was used. "
            "All Backdoor TEST rows form the protected set. "
            "Partition is fully deterministic."
        ),
        "canonicalization_version": CANONICALIZATION_VERSION,
    }
