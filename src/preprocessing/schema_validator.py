"""
src/preprocessing/schema_validator.py
---------------------------------------
Schema contract validator for the UNSW-NB15 IDS project.

Compares the OBSERVED schema (from schema_audit.py) against the
EXPECTED contract (from configs/data_schema.yaml).

The validator:
- Returns a list of violation strings (empty = pass).
- Raises SchemaViolationError when assert_schema_valid() is called on a
  non-empty violation list.
- Does NOT modify any data.

Two artefacts have different roles:
    configs/data_schema.yaml        — hand-authored EXPECTED contract
    data/audit/dataset_schema.json  — OBSERVED schema from actual data
"""

import logging
from typing import Any


class SchemaViolationError(RuntimeError):
    """Raised when the observed schema violates the expected contract."""


def validate_schema(
    observed: dict[str, Any],
    expected_contract: dict[str, Any],
    logger: logging.Logger | None = None,
) -> list[str]:
    """
    Compare an observed schema dict against the expected contract dict.

    Parameters
    ----------
    observed : dict
        Output of schema_audit.audit_dataframe() for a given split.
    expected_contract : dict
        Parsed configs/data_schema.yaml as a dictionary.
    logger : logging.Logger | None
        Optional logger for violation messages.

    Returns
    -------
    list[str]
        List of violation strings. Empty means the schema is valid.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    violations: list[str] = []
    observed_columns: list[str] = observed.get("columns", [])

    # --- Target column ---
    target_cfg = expected_contract.get("target", {})
    target_col = target_cfg.get("column")
    if target_col and target_col not in observed_columns:
        msg = f"Required target column '{target_col}' not found in observed schema."
        violations.append(msg)
        logger.error("SCHEMA VIOLATION: %s", msg)

    # --- Attack-category column ---
    cat_cfg = expected_contract.get("attack_category", {})
    cat_col = cat_cfg.get("column")
    if cat_col and cat_col not in observed_columns:
        msg = f"Required attack-category column '{cat_col}' not found in observed schema."
        violations.append(msg)
        logger.error("SCHEMA VIOLATION: %s", msg)

    # --- Candidate categorical columns (soft check — log warning, not error) ---
    candidate_cats = expected_contract.get("candidate_categorical_columns", [])
    for col in candidate_cats:
        if col not in observed_columns:
            msg = f"Candidate categorical column '{col}' not in observed schema (may be absent in this variant)."
            logger.warning("SCHEMA WARNING: %s", msg)
            # Not added to violations — candidate columns may legitimately be absent.

    # --- Candidate exclude columns (informational only) ---
    candidate_excl = expected_contract.get("candidate_exclude_columns", [])
    for col in candidate_excl:
        if col not in observed_columns:
            logger.debug(
                "Candidate exclude column '%s' not in observed schema (expected for pre-split layout).", col
            )

    # --- Columns documented as absent from pre-split layout (informational — NOT violations) ---
    absent_from_presplit = expected_contract.get("columns_absent_from_presplit", [])
    for col in absent_from_presplit:
        if col in observed_columns:
            msg = (
                f"Column '{col}' is listed in 'columns_absent_from_presplit' but IS present. "
                f"This suggests the wrong file (e.g., raw 4-shard CSV) may have been loaded."
            )
            violations.append(msg)
            logger.error("SCHEMA VIOLATION: %s", msg)
        else:
            logger.debug("Column '%s' confirmed absent from pre-split layout (expected).", col)

    return violations


def assert_schema_valid(
    violations: list[str],
    logger: logging.Logger | None = None,
) -> None:
    """
    Raise SchemaViolationError if any violations are present.

    Parameters
    ----------
    violations : list[str]
        Output of validate_schema().
    logger : logging.Logger | None
        Optional logger.

    Raises
    ------
    SchemaViolationError
        If the violations list is non-empty.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if violations:
        for v in violations:
            logger.error("VIOLATION: %s", v)
        raise SchemaViolationError(
            f"{len(violations)} schema violation(s) detected:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
