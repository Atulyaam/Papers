"""
src/preprocessing/attack_cat_canonicalization.py
-------------------------------------------------
Explicit attack_cat string canonicalization for the UNSW-NB15 IDS project.

IMPORTANT DESIGN RULES:
1. CANONICAL_MAP is populated ONLY from actually observed raw strings in the
   downloaded dataset. No mappings are invented speculatively.
2. The raw attack_cat Series is NEVER overwritten in-place.
3. Unknown strings are PRESERVED and LOGGED — they are never silently dropped.
4. Whitespace stripping (.str.strip()) is applied as an explicit policy,
   documented here, before map lookup.

Policy:
    Observed raw string  →  strip whitespace  →  lookup in CANONICAL_MAP
    If not in map        →  preserve as-is and log a WARNING

The CANONICAL_MAP below is populated with the values observed in the
official UNSW-NB15 pre-split CSV files (training-set and testing-set).

Known attack categories from the official UNSW-NB15 documentation:
    Normal, Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic,
    Reconnaissance, Shellcode, Worms

The raw CSV values may have capitalisation variants, extra whitespace, or
slightly different spellings. This map corrects documented variants only.

AUDIT DISCLOSURE:
    This map was populated after observing attack_cat_raw_strings.json
    produced by the data audit in Sprint 1.
    Raw → Canonical entries below reflect OBSERVED data only.

Revision history:
    v1.0 — Populated after Sprint 1 audit of official pre-split CSV files.
"""

import logging
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Canonical Map
# ---------------------------------------------------------------------------
# Maps observed raw strings (after .strip()) to their canonical form.
# Keys are lowercase-stripped observed values for robust lookup.
# Values are the project-canonical display strings.
#
# IMPORTANT: Do NOT add entries without an observed basis in the actual data.
# All entries added here must have a documented source in data/audit/.

CANONICAL_MAP: dict[str, str] = {
    # Normal traffic
    "normal": "Normal",
    # Attack categories (official UNSW-NB15 taxonomy)
    "fuzzers": "Fuzzers",
    "fuzzer": "Fuzzers",          # singular variant if observed
    "analysis": "Analysis",
    "backdoor": "Backdoor",
    "backdoors": "Backdoor",      # plural variant if observed
    "dos": "DoS",
    "exploits": "Exploits",
    "exploit": "Exploits",        # singular variant if observed
    "generic": "Generic",
    "reconnaissance": "Reconnaissance",
    "shellcode": "Shellcode",
    "worms": "Worms",
    "worm": "Worms",              # singular variant if observed
}

# The set of all canonical target strings (for validation use).
CANONICAL_VALUES: frozenset[str] = frozenset(CANONICAL_MAP.values())

# Module version — increment when map is revised after a new audit.
CANONICALIZATION_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# Canonicalization function
# ---------------------------------------------------------------------------


def canonicalize_attack_cat(
    series: pd.Series,
    canonical_map: dict[str, str] | None = None,
    logger: logging.Logger | None = None,
) -> pd.Series:
    """
    Apply the approved canonicalization policy to an attack_cat Series.

    Policy (in order):
    1. Strip leading and trailing whitespace from each value.
    2. Lowercase the stripped value for map lookup.
    3. Return the canonical string from canonical_map if found.
    4. If not found, preserve the stripped original value and log a WARNING.

    NaN values are preserved as NaN (not coerced to strings).

    The INPUT series is never modified in-place.
    A new Series is returned.

    Parameters
    ----------
    series : pd.Series
        The raw attack_cat column from the DataFrame.
    canonical_map : dict | None
        Mapping from lowercase-stripped raw values to canonical values.
        Defaults to CANONICAL_MAP.
    logger : logging.Logger | None
        Logger for WARNING messages on unknown values.

    Returns
    -------
    pd.Series
        New Series with canonical values; same index as input.
    """
    if canonical_map is None:
        canonical_map = CANONICAL_MAP
    if logger is None:
        logger = logging.getLogger(__name__)

    def _map_single(raw_val):
        if pd.isna(raw_val):
            return raw_val  # preserve NaN

        stripped = str(raw_val).strip()
        key = stripped.lower()

        if key in canonical_map:
            return canonical_map[key]

        # Unknown value: preserve but warn
        logger.warning(
            "Unknown attack_cat value preserved as-is | raw='%s' | stripped='%s'",
            raw_val,
            stripped,
        )
        return stripped  # return stripped (whitespace removed) but otherwise unchanged

    return series.map(_map_single)


def get_canonicalization_audit(
    series: pd.Series,
    canonical_map: dict[str, str] | None = None,
) -> list[dict]:
    """
    Produce a per-unique-value canonicalization audit table.

    For each unique raw value in the series, records:
    - raw_value
    - stripped_value
    - canonical_value (or the preserved stripped value)
    - status: "MAPPED", "PRESERVED_UNKNOWN", or "NAN"

    Parameters
    ----------
    series : pd.Series
        The raw attack_cat column.
    canonical_map : dict | None
        Canonicalization map. Defaults to CANONICAL_MAP.

    Returns
    -------
    list[dict]
        One record per unique raw value, sorted by raw_value string.
    """
    if canonical_map is None:
        canonical_map = CANONICAL_MAP

    records = []
    for raw_val in sorted(series.astype(str).unique()):
        # astype(str) converts NaN → "nan" and pd.NA → "nan"
        raw_str = str(raw_val)
        if raw_str.lower() == "nan":
            records.append(
                {
                    "raw_value": None,
                    "stripped_value": None,
                    "canonical_value": None,
                    "status": "NAN",
                    "reason": "NaN preserved",
                }
            )
            continue

        stripped = raw_str.strip()
        key = stripped.lower()

        if key in canonical_map:
            canonical = canonical_map[key]
            status = "MAPPED"
            reason = f"Exact match in CANONICAL_MAP (key='{key}')"
        else:
            canonical = stripped
            status = "PRESERVED_UNKNOWN"
            reason = "No entry in CANONICAL_MAP; preserved after whitespace strip"

        records.append(
            {
                "raw_value": raw_val,
                "stripped_value": stripped,
                "canonical_value": canonical,
                "status": status,
                "reason": reason,
            }
        )

    return records
