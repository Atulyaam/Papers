"""
scripts/run_train_val_split.py
--------------------------------
Official Sprint 3 execution script: TRAIN / VALIDATION split protocol.

Creates (in data/splits/):
    train.csv
    validation.csv
    excluded_train_backdoor.csv
    train_val_split_metadata.json

And writes a human-readable audit report to:
    data/audit/train_val_split_audit.md

Run from project root:
    .venv\\Scripts\\python.exe scripts/run_train_val_split.py

LEAKAGE RULE:
    This script reads ONLY the official UNSW-NB15 training file.
    It does NOT read development_test.csv, protected_unseen_attack.csv,
    or the raw testing file. The split is determined entirely from the
    training file.

DATA ACCESS BOUNDARY:
    Input:  data/raw/UNSW_NB15_training-set.csv  (immutable)
    Output: data/splits/train.csv
            data/splits/validation.csv
            data/splits/excluded_train_backdoor.csv
            data/splits/train_val_split_metadata.json
            data/audit/train_val_split_audit.md

REPRODUCTION:
    Given the same commit, seed, and raw training file hash, this script
    produces byte-identical outputs.
"""

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

# --- Ensure project root is on sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.loader import load_raw_unswnb15
from src.preprocessing.split_protocol import (
    build_split_provenance,
    create_train_val_split,
    verify_split_integrity,
)
from src.utils.hashing import sha256_file
from src.utils.reproducibility import set_all_seeds

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "EXP_TRAIN_VAL_SPLIT_V1"
LOG_DIR = PROJECT_ROOT / "results" / "logs" / EXPERIMENT_ID
LOG_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR = PROJECT_ROOT / "data" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "run.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sprint 1 frozen hashes (immutable reference)
# ---------------------------------------------------------------------------
EXPECTED_HASHES = {
    "UNSW_NB15_training-set.csv": "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
    "UNSW_NB15_testing-set.csv":  "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
    "protected_unseen_attack.csv": "6ffd23479b575e438ad90678268f40f674a663c2b9507aaf65089623397a9d91",
    "development_test.csv":        "04725e85732ab2fc6d9eaaa6105418b22b083b5c651067e7b0785464f414e508",
}


def _row_fingerprint(df: pd.DataFrame) -> str:
    """
    Produce a deterministic SHA-256 fingerprint of a DataFrame's content.

    Used for exact-reconstruction and pairwise-disjointness verification.
    Rows are sorted by all columns before hashing to eliminate ordering
    dependence.
    """
    sorted_df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
    return hashlib.sha256(
        pd.util.hash_pandas_object(sorted_df, index=True).values.tobytes()
    ).hexdigest()


def _verify_source_integrity(config: dict) -> None:
    """Verify raw and Sprint 1 split file hashes. Fatal on mismatch."""
    raw_dir    = PROJECT_ROOT / config["data"]["raw_dir"]
    splits_dir = PROJECT_ROOT / config["splits"]["output_dir"]

    checks = [
        (raw_dir    / "UNSW_NB15_training-set.csv",   "UNSW_NB15_training-set.csv"),
        (raw_dir    / "UNSW_NB15_testing-set.csv",    "UNSW_NB15_testing-set.csv"),
        (splits_dir / "protected_unseen_attack.csv",  "protected_unseen_attack.csv"),
        (splits_dir / "development_test.csv",         "development_test.csv"),
    ]
    logger.info("=== SOURCE INTEGRITY VERIFICATION ===")
    all_ok = True
    for fpath, key in checks:
        actual = sha256_file(str(fpath))
        expected = EXPECTED_HASHES[key]
        ok = actual == expected
        if not ok:
            all_ok = False
        logger.info("  %s: %s", key, "MATCH" if ok else f"MISMATCH (got {actual})")
    if not all_ok:
        logger.error("FATAL: Source file integrity check failed.")
        sys.exit(1)
    logger.info("Source integrity: ALL MATCH")


def _get_library_versions() -> dict:
    """Collect Python and key library version strings."""
    import platform
    import numpy as np
    import sklearn
    return {
        "python":    platform.python_version(),
        "numpy":     np.__version__,
        "pandas":    pd.__version__,
        "sklearn":   sklearn.__version__,
    }


def _verify_no_fitting(script_path: Path) -> bool:
    """
    Scan the split_protocol.py source for forbidden fitting patterns.

    Returns True (no violations) or logs an error and returns False.

    Comments and docstrings are NOT excluded from the scan; a pattern
    found in a comment indicates a risk of accidental activation.
    """
    forbidden = [
        ".fit(",
        ".fit_transform(",
        "GridSearchCV",
        "RandomizedSearchCV",
        "optimizer.step",
        ".backward(",
        "model.train(",
    ]
    protocol_src = PROJECT_ROOT / "src" / "preprocessing" / "split_protocol.py"
    text = protocol_src.read_text(encoding="utf-8")

    violations = [p for p in forbidden if p in text]
    if violations:
        logger.error("NO-FITTING VIOLATION in split_protocol.py: %s", violations)
        return False
    logger.info("No-fitting scan: PASS (0 violations in split_protocol.py)")
    return True


# ---------------------------------------------------------------------------
# Exact reconstruction and pairwise disjointness
# ---------------------------------------------------------------------------


def _verify_exact_reconstruction(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    excl_df: pd.DataFrame,
) -> bool:
    """
    Verify the three-way partition exactly reconstructs the original DataFrame.

    Method: sort all columns in each DataFrame, hash, then compare the
    hash of the union against the hash of the original. This is equivalent
    to verifying set equality of rows without relying on index order.
    """
    # Reconstruct by concatenation then sort
    reconstructed = pd.concat([train_df, val_df, excl_df], axis=0)
    orig_fp  = _row_fingerprint(original_df)
    recon_fp = _row_fingerprint(reconstructed)
    ok = orig_fp == recon_fp
    logger.info("Exact reconstruction: %s", "PASS" if ok else f"FAIL (hash mismatch)")
    return ok


def _verify_pairwise_disjointness(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    excl_df: pd.DataFrame,
) -> dict:
    """
    Verify all three splits are pairwise index-disjoint.

    Returns a dict with check results.
    """
    train_idx = set(train_df.index)
    val_idx   = set(val_df.index)
    excl_idx  = set(excl_df.index)

    tv = train_idx & val_idx
    te = train_idx & excl_idx
    ve = val_idx   & excl_idx

    results = {
        "train_val_overlap":       len(tv),
        "train_excluded_overlap":  len(te),
        "val_excluded_overlap":    len(ve),
        "all_disjoint":            len(tv) == 0 and len(te) == 0 and len(ve) == 0,
    }
    logger.info(
        "Pairwise disjointness: TRAIN∩VAL=%d  TRAIN∩EXCL=%d  VAL∩EXCL=%d  all=%s",
        results["train_val_overlap"],
        results["train_excluded_overlap"],
        results["val_excluded_overlap"],
        "PASS" if results["all_disjoint"] else "FAIL",
    )
    return results


def _verify_count_identities(result, original_df: pd.DataFrame, config: dict) -> dict:
    """
    Verify the exact split-count identities defined in the spec.

    Returns dict with all identity checks.
    """
    ac  = "attack_cat"
    lc  = "label"
    nc  = "Normal"
    wa  = "Backdoor"

    # Original training file counts
    orig_backdoor    = (original_df[ac] == wa).sum()
    orig_non_bd      = (~(original_df[ac] == wa)).sum()
    orig_normal      = (original_df[ac] == nc).sum()
    orig_nb_attacks  = ((original_df[lc] == 1) & (original_df[ac] != wa)).sum()

    # Split counts
    train_normal   = (result.train_df[ac] == nc).sum()
    val_normal     = (result.val_df[ac] == nc).sum()
    train_attacks  = (result.train_df[lc] == 1).sum()
    val_attacks    = (result.val_df[lc] == 1).sum()
    train_backdoor = (result.train_df[ac] == wa).sum()
    val_backdoor   = (result.val_df[ac] == wa).sum()
    excl_backdoor  = (result.excluded_backdoor_df[ac] == wa).sum()

    ids = {
        # Normal TRAIN + Normal VAL == all non-Backdoor Normal rows
        "normal_train_plus_val_eq_all_normal": bool(
            int(train_normal) + int(val_normal) == int(orig_normal)
        ),
        # TRAIN attacks == all non-Backdoor attack rows
        "train_attacks_eq_all_nb_attacks": bool(
            int(train_attacks) == int(orig_nb_attacks)
        ),
        # VALIDATION attacks == 0
        "val_attacks_eq_zero": bool(int(val_attacks) == 0),
        # TRAIN Backdoor == 0
        "train_backdoor_eq_zero": bool(int(train_backdoor) == 0),
        # VALIDATION Backdoor == 0
        "val_backdoor_eq_zero": bool(int(val_backdoor) == 0),
        # Excluded Backdoor == all Backdoor rows in training file
        "excluded_backdoor_eq_all_train_backdoor": bool(
            int(excl_backdoor) == int(orig_backdoor)
        ),
        # Counts for reporting
        "counts": {
            "orig_backdoor":   int(orig_backdoor),
            "orig_normal":     int(orig_normal),
            "orig_nb_attacks": int(orig_nb_attacks),
            "train_normal":    int(train_normal),
            "val_normal":      int(val_normal),
            "train_attacks":   int(train_attacks),
            "val_attacks":     int(val_attacks),
            "train_backdoor":  int(train_backdoor),
            "val_backdoor":    int(val_backdoor),
            "excl_backdoor":   int(excl_backdoor),
        },
    }
    all_ok = all(v for k, v in ids.items() if k != "counts")
    ids["all_pass"] = bool(all_ok)
    logger.info("Count identities: %s", "ALL PASS" if all_ok else "FAIL")
    for k, v in ids.items():
        if k not in ("counts", "all_pass"):
            logger.info("  %s: %s", k, v)
    return ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    start = datetime.now(timezone.utc)
    logger.info("=== %s START | %s ===", EXPERIMENT_ID, start.isoformat())

    # --- Load config ---
    config_path = PROJECT_ROOT / "configs" / "project_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed: int = config.get("seed", 42)
    logger.info("Seed: %d", seed)
    set_all_seeds(seed)

    # --- Item 5/7: Source integrity + data access boundary ---
    _verify_source_integrity(config)

    # --- Item 8: No-fitting guarantee ---
    if not _verify_no_fitting(PROJECT_ROOT / "scripts" / "run_train_val_split.py"):
        sys.exit(1)

    # --- Load ONLY the raw training file ---
    logger.info("Loading raw UNSW-NB15 training file (only)...")
    splits_raw = load_raw_unswnb15(config, logger=logger)
    train_raw_df = splits_raw["train"]
    logger.info("Training file: %d rows x %d cols", *train_raw_df.shape)

    train_sha256 = sha256_file(
        str(PROJECT_ROOT / config["data"]["raw_dir"] / config["data"]["train_file"])
    )

    # --- Get Git commit ---
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        git_commit = "unknown"
    logger.info("Git commit: %s", git_commit)

    # --- Run split protocol ---
    logger.info("Running TRAIN/VALIDATION split (seed=%d)...", seed)
    result = create_train_val_split(train_raw_df, seed=seed)

    # --- Integrity check ---
    logger.info("Verifying split integrity...")
    report = verify_split_integrity(result)
    if not report.all_pass:
        for f in report.failures:
            logger.error("INTEGRITY FAIL: %s", f)
        sys.exit(1)
    logger.info("Split integrity: ALL PASS")

    # --- Exact reconstruction + pairwise disjointness (items 3/4) ---
    exact_ok = _verify_exact_reconstruction(
        train_raw_df, result.train_df, result.val_df, result.excluded_backdoor_df
    )
    disjoint = _verify_pairwise_disjointness(
        result.train_df, result.val_df, result.excluded_backdoor_df
    )
    if not exact_ok or not disjoint["all_disjoint"]:
        logger.error("FATAL: Reconstruction or disjointness check failed.")
        sys.exit(1)

    # --- Count identities (item 3) ---
    id_checks = _verify_count_identities(result, train_raw_df, config)
    if not id_checks["all_pass"]:
        logger.error("FATAL: Count identity checks failed.")
        sys.exit(1)

    # --- Write split files ---
    splits_dir = PROJECT_ROOT / config["splits"]["output_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    out_train    = splits_dir / "train.csv"
    out_val      = splits_dir / "validation.csv"
    out_excluded = splits_dir / "excluded_train_backdoor.csv"
    out_metadata = splits_dir / "train_val_split_metadata.json"

    logger.info("Writing TRAIN         -> %s (%d rows)", out_train.name,    result.n_train)
    result.train_df.to_csv(out_train, index=False)

    logger.info("Writing VALIDATION    -> %s (%d rows)", out_val.name,      result.n_val)
    result.val_df.to_csv(out_val, index=False)

    logger.info("Writing EXCLUDED      -> %s (%d rows)", out_excluded.name, result.n_excluded_backdoor)
    result.excluded_backdoor_df.to_csv(out_excluded, index=False)

    # --- Item 6: Determinism — verify re-running produces same hashes ---
    # (Done implicitly: same seed + same input SHA → same output. Hashes are
    #  recorded in metadata for audit. Caller should re-run and compare.)
    train_out_sha256 = sha256_file(str(out_train))
    val_out_sha256   = sha256_file(str(out_val))
    excl_out_sha256  = sha256_file(str(out_excluded))
    logger.info("train.csv SHA-256:               %s", train_out_sha256)
    logger.info("validation.csv SHA-256:          %s", val_out_sha256)
    logger.info("excluded_train_backdoor SHA-256: %s", excl_out_sha256)

    output_hashes = {
        "train_csv":                    train_out_sha256,
        "validation_csv":               val_out_sha256,
        "excluded_train_backdoor_csv":  excl_out_sha256,
    }
    lib_versions = _get_library_versions()

    # --- Build and write full provenance ---
    provenance = build_split_provenance(
        result=result,
        report=report,
        source_train_sha256=train_sha256,
        git_commit=git_commit,
        protocol_version="1.0",
        exact_reconstruction=exact_ok,
        output_hashes=output_hashes,
        library_versions=lib_versions,
    )

    # Augment with pairwise disjointness and count identity results
    provenance["pairwise_disjointness"] = {
        "train_val_overlap":       int(disjoint["train_val_overlap"]),
        "train_excluded_overlap":  int(disjoint["train_excluded_overlap"]),
        "val_excluded_overlap":    int(disjoint["val_excluded_overlap"]),
        "all_disjoint":            bool(disjoint["all_disjoint"]),
    }
    provenance["count_identity_checks"] = {
        k: (bool(v) if isinstance(v, (bool,)) else v)
        for k, v in id_checks.items()
    }

    with open(out_metadata, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    logger.info("Provenance -> %s", out_metadata.name)

    # --- Write human-readable audit report ---
    pct = provenance["validation_percentile_adequacy"]
    n_val = result.n_val
    n_normal_val = int((result.val_df["attack_cat"] == "Normal").sum())
    audit_md = f"""# Sprint 3 — TRAIN / VALIDATION Split Audit Report

**Experiment ID:** EXP_TRAIN_VAL_SPLIT_V1  
**Created:** {provenance["created_at"]}  
**Git commit:** {git_commit}  
**Seed:** {seed}

## Split Counts

| Split | Rows |
|---|---|
| TRAIN | {result.n_train:,} |
| VALIDATION | {result.n_val:,} |
| Excluded Backdoor | {result.n_excluded_backdoor:,} |
| **Input total** | **{result.n_input:,}** |
| Conservation check | {result.n_train + result.n_val + result.n_excluded_backdoor:,} == {result.n_input:,} |

## TRAIN Composition

| Component | Rows |
|---|---|
| Normal (80%) | {int((result.train_df["attack_cat"] == "Normal").sum()):,} |
| Non-Backdoor attacks (100%) | {int((result.train_df["label"] == 1).sum()):,} |
| Backdoor | 0 |

## VALIDATION Composition

| Component | Rows |
|---|---|
| Normal (20%) | {n_normal_val:,} |
| Attack rows | 0 |
| Backdoor | 0 |

## Excluded Backdoor

{result.n_excluded_backdoor:,} Backdoor training-file rows archived to `excluded_train_backdoor.csv`.  
**EXPERIMENTAL ROLE = NONE.** Must not be used for training, tuning, or evaluation.

## Integrity Checks

| Check | Result |
|---|---|
| Row conservation | {"PASS" if report.row_conservation else "FAIL"} |
| Exact reconstruction | {"PASS" if exact_ok else "FAIL"} |
| TRAIN ∩ VAL | {disjoint["train_val_overlap"]} (expected 0) |
| TRAIN ∩ EXCLUDED | {disjoint["train_excluded_overlap"]} (expected 0) |
| VAL ∩ EXCLUDED | {disjoint["val_excluded_overlap"]} (expected 0) |
| VAL attack rows | 0 |
| Backdoor in TRAIN | 0 |
| Backdoor in VAL | 0 |
| All integrity checks | {"ALL PASS" if report.all_pass else "FAIL"} |

## Source Integrity

| File | Status |
|---|---|
| UNSW_NB15_training-set.csv | MATCH |
| UNSW_NB15_testing-set.csv | MATCH |
| protected_unseen_attack.csv | MATCH |
| development_test.csv | MATCH |

## Validation Percentile Adequacy

The VALIDATION set contains **{n_normal_val:,} benign rows** for AE threshold calibration.

> These are descriptive empirical support counts, **not confidence intervals**.

| Threshold | Expected upper-tail count |
|---|---|
| 90th percentile | ~{pct["percentiles"]["90"]["approx_tail_count"]:,} rows |
| 95th percentile | ~{pct["percentiles"]["95"]["approx_tail_count"]:,} rows |
| 97.5th percentile | ~{pct["percentiles"]["97.5"]["approx_tail_count"]:,} rows |
| 99th percentile | ~{pct["percentiles"]["99"]["approx_tail_count"]:,} rows |

{pct["assessment"]}

## Library Versions

| Package | Version |
|---|---|
| Python | {lib_versions["python"]} |
| NumPy | {lib_versions["numpy"]} |
| pandas | {lib_versions["pandas"]} |
| scikit-learn | {lib_versions["sklearn"]} |

## Output File Hashes

| File | SHA-256 |
|---|---|
| train.csv | `{train_out_sha256}` |
| validation.csv | `{val_out_sha256}` |
| excluded_train_backdoor.csv | `{excl_out_sha256}` |
"""
    audit_path = AUDIT_DIR / "train_val_split_audit.md"
    audit_path.write_text(audit_md, encoding="utf-8")
    logger.info("Audit report -> %s", audit_path)

    # --- Final summary ---
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info("")
    logger.info("=== %s SUMMARY ===", EXPERIMENT_ID)
    logger.info("  TRAIN:             %d", result.n_train)
    logger.info("  VALIDATION:        %d", result.n_val)
    logger.info("  Excluded Backdoor: %d", result.n_excluded_backdoor)
    logger.info("  Conservation:      %d + %d + %d = %d (input=%d)",
        result.n_train, result.n_val, result.n_excluded_backdoor,
        result.n_train + result.n_val + result.n_excluded_backdoor,
        result.n_input,
    )
    logger.info("  Exact recon:       %s", "PASS" if exact_ok else "FAIL")
    logger.info("  Pairwise disjoint: %s", "PASS" if disjoint["all_disjoint"] else "FAIL")
    logger.info("  Val attack rows:   %d", report.val_zero_attack_count)
    logger.info("  VAL benign n:      %d", n_normal_val)
    logger.info("  Elapsed:           %.2fs", elapsed)
    logger.info("=== %s COMPLETE ===", EXPERIMENT_ID)

    print()
    print("STATUS: PASS")
    print(f"  TRAIN:             {result.n_train:,}")
    print(f"  VALIDATION:        {result.n_val:,}")
    print(f"  Excluded Backdoor: {result.n_excluded_backdoor:,}")
    print(f"  Conservation:      {result.n_train + result.n_val + result.n_excluded_backdoor:,} == {result.n_input:,}")
    print(f"  Exact recon:       {'PASS' if exact_ok else 'FAIL'}")
    print(f"  Pairwise disjoint: {'PASS' if disjoint['all_disjoint'] else 'FAIL'}")
    print(f"  Val attack rows:   {report.val_zero_attack_count}")
    print(f"  Val benign n:      {n_normal_val:,}")
    print(f"  Integrity:         ALL PASS")


if __name__ == "__main__":
    main()
