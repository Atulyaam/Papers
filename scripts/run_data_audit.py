"""
scripts/run_data_audit.py
--------------------------
Sprint 1 — UNSW-NB15 Data Acquisition and Audit

Orchestration script for EXP_DATA_ACQUISITION_AUDIT.

Phases:
    1.  Load project config
    2.  Initialise structured logger
    3.  Verify and hash raw files
    4.  Load raw DataFrames (no transforms)
    5.  Audit each split: shape, dtypes, nulls, infs, duplicates
    6.  Validate observed schema against expected contract
    7.  Audit raw attack_cat strings (preserve whitespace/case)
    8.  Apply canonicalization audit (log unknown values)
    9.  Audit label distribution (train + test)
    10. Audit attack-cat distribution (raw + canonical)
    11. Cross-split overlap check (train vs test)
    12. Withheld-candidate eligibility audit
    13. Write all JSON audit artifacts
    14. Generate audit_report.md
    15. Update experiment registry
    16. Log summary and timing

NO model training occurs in this script.
NO transforms are fitted in this script.
data/raw/ is treated as immutable after download.

Run from project root:
    python -m scripts.run_data_audit
    or
    python scripts/run_data_audit.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Allow running from project root without installing as a package ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.preprocessing.attack_cat_canonicalization import (
    CANONICAL_MAP,
    CANONICALIZATION_VERSION,
    canonicalize_attack_cat,
    get_canonicalization_audit,
)
from src.preprocessing.loader import load_raw_unswnb15
from src.preprocessing.schema_audit import (
    audit_attack_cat_distribution,
    audit_attack_cat_raw_strings,
    audit_dataframe,
    audit_label_distribution,
    check_overlap,
)
from src.preprocessing.schema_validator import SchemaViolationError, validate_schema
from src.preprocessing.withheld_candidate import (
    build_candidate_report,
    compute_eligible_candidates,
)
from src.utils.hashing import sha256_file
from src.utils.logging_utils import get_experiment_logger

EXPERIMENT_ID = "EXP_DATA_ACQUISITION_AUDIT"
CONFIG_PATH = Path("configs/project_config.yaml")
SCHEMA_PATH = Path("configs/data_schema.yaml")


def main() -> None:
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_dir = config["audit"]["log_dir"]
    audit_dir = Path(config["audit"]["output_dir"])
    audit_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Logger
    # ------------------------------------------------------------------
    logger = get_experiment_logger(EXPERIMENT_ID, log_dir)
    now_utc = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 70)
    logger.info("EXPERIMENT START | id=%s | timestamp=%s", EXPERIMENT_ID, now_utc)
    logger.info("Config: %s", CONFIG_PATH)
    logger.info("Protocol version: %s", config.get("protocol_version", "1.0"))
    logger.info("Seed: %s (not used in Sprint 1 — no model training)", config.get("seed"))

    # ------------------------------------------------------------------
    # 3. Hash raw files
    # ------------------------------------------------------------------
    raw_dir = Path(config["data"]["raw_dir"])
    train_path = raw_dir / config["data"]["train_file"]
    test_path = raw_dir / config["data"]["test_file"]

    file_hashes: dict = {}
    for label, fpath in [("train", train_path), ("test", test_path)]:
        if not fpath.exists():
            logger.error("Raw file MISSING: %s", fpath)
            logger.error(
                "Place official UNSW-NB15 CSV files in %s before running this script.",
                raw_dir.resolve(),
            )
            sys.exit(1)

        fsize = fpath.stat().st_size
        fhash = sha256_file(fpath)
        logger.info("Hash | %s | file=%s | size=%d | sha256=%s", label, fpath.name, fsize, fhash)
        file_hashes[label] = {
            "filename": fpath.name,
            "path": str(fpath),
            "source": "Official UNSW-NB15 pre-split CSV (https://research.unsw.edu.au/projects/unsw-nb15-dataset)",
            "download_timestamp": now_utc,
            "size_bytes": fsize,
            "sha256": fhash,
        }

    _write_json(audit_dir / "file_hashes.json", file_hashes, logger)

    # ------------------------------------------------------------------
    # 4. Load raw DataFrames
    # ------------------------------------------------------------------
    splits = load_raw_unswnb15(config, logger=logger)
    df_train = splits["train"]
    df_test = splits["test"]

    # ------------------------------------------------------------------
    # 5. Structural audit
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("PHASE: Structural audit")
    schema_observed: dict = {}
    for split_name, df in [("train", df_train), ("test", df_test)]:
        audit = audit_dataframe(df, split_name)
        schema_observed[split_name] = audit
        logger.info(
            "Audit | split=%s | rows=%d | cols=%d | nulls_any=%d | dups=%d",
            split_name,
            audit["shape"]["rows"],
            audit["shape"]["cols"],
            sum(audit["null_counts"].values()),
            audit["duplicate_row_count"],
        )

    _write_json(audit_dir / "dataset_schema.json", schema_observed, logger)

    # ------------------------------------------------------------------
    # 6. Schema validation (soft-fail at pre-audit stage)
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("PHASE: Schema validation")
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        expected_contract = yaml.safe_load(f)

    all_violations = []
    for split_name, obs in schema_observed.items():
        violations = validate_schema(obs, expected_contract, logger=logger)
        if violations:
            all_violations.extend(violations)
            logger.warning("Schema violations in '%s': %d", split_name, len(violations))
        else:
            logger.info("Schema validation PASSED for split '%s'.", split_name)

    if all_violations:
        logger.warning(
            "Total schema violations: %d (soft-fail at pre-audit stage)",
            len(all_violations),
        )
    else:
        logger.info("Schema validation: ALL PASSED")

    # ------------------------------------------------------------------
    # 7. Raw attack_cat audit
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("PHASE: Raw attack_cat audit")
    cat_col = config["data"]["attack_cat_col"]

    raw_strings_train = audit_attack_cat_raw_strings(df_train, cat_col)
    raw_strings_test = audit_attack_cat_raw_strings(df_test, cat_col)
    all_raw_strings = sorted(set(raw_strings_train) | set(raw_strings_test))

    logger.info("Raw attack_cat unique values (combined): %s", all_raw_strings)

    raw_dist_train = audit_attack_cat_distribution(df_train, cat_col)
    raw_dist_test = audit_attack_cat_distribution(df_test, cat_col)

    raw_strings_artifact = {
        "train": raw_strings_train,
        "test": raw_strings_test,
        "combined": all_raw_strings,
    }
    _write_json(audit_dir / "attack_cat_raw_strings.json", raw_strings_artifact, logger)
    _write_json(
        audit_dir / "attack_cat_raw_distribution.json",
        {"train": raw_dist_train, "test": raw_dist_test},
        logger,
    )

    # ------------------------------------------------------------------
    # 8. Canonicalization audit
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("PHASE: Canonicalization audit")

    can_audit_train = get_canonicalization_audit(df_train[cat_col], CANONICAL_MAP)
    can_audit_test = get_canonicalization_audit(df_test[cat_col], CANONICAL_MAP)

    canonicalization_artifact = {
        "canonicalization_version": CANONICALIZATION_VERSION,
        "policy": "strip whitespace -> lowercase key -> lookup CANONICAL_MAP; unknown values preserved",
        "train": can_audit_train,
        "test": can_audit_test,
    }
    _write_json(audit_dir / "canonicalization_map.json", canonicalization_artifact, logger)

    # Compute canonical series
    canonical_train = canonicalize_attack_cat(df_train[cat_col], logger=logger)
    canonical_test = canonicalize_attack_cat(df_test[cat_col], logger=logger)

    # ------------------------------------------------------------------
    # 9. Label distribution
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("PHASE: Label distribution")
    label_col = config["data"]["label_col"]
    label_dist = {
        "train": audit_label_distribution(df_train, label_col),
        "test": audit_label_distribution(df_test, label_col),
    }
    logger.info("Label distribution train: %s", label_dist["train"]["counts"])
    logger.info("Label distribution test:  %s", label_dist["test"]["counts"])
    _write_json(audit_dir / "label_distribution.json", label_dist, logger)

    # ------------------------------------------------------------------
    # 10. Canonical attack-cat distribution
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("PHASE: Canonical attack-cat distribution")

    # Use canonical series (as a temporary DataFrame column for counting)
    import pandas as pd

    can_dist_train = canonical_train.value_counts(dropna=False).to_dict()
    can_dist_test = canonical_test.value_counts(dropna=False).to_dict()
    can_dist_train = {str(k): int(v) for k, v in can_dist_train.items()}
    can_dist_test = {str(k): int(v) for k, v in can_dist_test.items()}

    logger.info("Canonical train distribution: %s", can_dist_train)
    logger.info("Canonical test  distribution: %s", can_dist_test)

    attack_cat_dist_artifact = {
        "train": can_dist_train,
        "test": can_dist_test,
    }
    _write_json(audit_dir / "attack_cat_distribution.json", attack_cat_dist_artifact, logger)

    # ------------------------------------------------------------------
    # 11. Cross-split overlap
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("PHASE: Cross-split overlap check (train vs test)")
    overlap_report = check_overlap(df_train, df_test, label_a="train", label_b="test")
    logger.info(
        "Overlap | count=%d | pct=%.4f%%",
        overlap_report["overlap_count"],
        overlap_report["overlap_percentage"],
    )
    _write_json(audit_dir / "overlap_report.json", overlap_report, logger)

    # ------------------------------------------------------------------
    # 12. Withheld-candidate eligibility
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("PHASE: Withheld-candidate eligibility")

    # Use canonical TEST counts (exclude Normal/benign)
    candidate_report = build_candidate_report(
        canonical_test_counts=can_dist_test,
        min_count=config["withheld"]["min_test_instances"],
    )
    logger.info(
        "Eligible candidates: %s",
        list(candidate_report["eligible_candidates"].keys()),
    )

    # Check Backdoor specifically
    backdoor_count = can_dist_test.get("Backdoor", 0)
    if backdoor_count >= config["withheld"]["min_test_instances"]:
        logger.info(
            "Backdoor ELIGIBLE | count=%d >= threshold=%d",
            backdoor_count,
            config["withheld"]["min_test_instances"],
        )
    else:
        logger.error(
            "CRITICAL: Backdoor count=%d is BELOW threshold=%d. "
            "Cannot proceed with withheld-attack protocol without resolving this conflict.",
            backdoor_count,
            config["withheld"]["min_test_instances"],
        )

    _write_json(audit_dir / "withheld_candidates.json", candidate_report, logger)

    # ------------------------------------------------------------------
    # 13. Update pre-registration with actual counts
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("PHASE: Updating pre-registration with actual candidate counts")
    pre_reg_path = Path("experiments/pre_registration.json")
    with open(pre_reg_path, encoding="utf-8") as f:
        pre_reg = json.load(f)

    # Populate from actual audit
    eligible = candidate_report["eligible_candidates"]
    pre_reg["candidate_subclasses"] = sorted(eligible.keys())
    pre_reg["candidate_counts"] = eligible
    pre_reg["selected_subclass"] = "Backdoor"
    now_str = datetime.now(timezone.utc).isoformat()
    pre_reg["created_at"] = now_str
    pre_reg["selection_frozen_at"] = now_str

    with open(pre_reg_path, "w", encoding="utf-8") as f:
        json.dump(pre_reg, f, indent=2)
    logger.info("Pre-registration updated: %s", pre_reg_path)

    # ------------------------------------------------------------------
    # 14. Generate audit_report.md
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("PHASE: Generating audit_report.md")
    _generate_audit_report(
        audit_dir=audit_dir,
        file_hashes=file_hashes,
        schema_observed=schema_observed,
        all_raw_strings=all_raw_strings,
        can_audit_combined=(can_audit_train + can_audit_test),
        label_dist=label_dist,
        can_dist_train=can_dist_train,
        can_dist_test=can_dist_test,
        overlap_report=overlap_report,
        candidate_report=candidate_report,
        config=config,
        violations=all_violations,
    )

    # ------------------------------------------------------------------
    # 15. Final summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    logger.info("=" * 70)
    logger.info(
        "EXPERIMENT COMPLETE | id=%s | elapsed=%.2fs | status=SUCCESS",
        EXPERIMENT_ID,
        elapsed,
    )
    logger.info(
        "Audit artifacts written to: %s", audit_dir.resolve()
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict, logger) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Wrote: %s", path)


def _generate_audit_report(
    audit_dir,
    file_hashes,
    schema_observed,
    all_raw_strings,
    can_audit_combined,
    label_dist,
    can_dist_train,
    can_dist_test,
    overlap_report,
    candidate_report,
    config,
    violations,
) -> None:
    lines = [
        "# UNSW-NB15 Sprint 1 Data Audit Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "---",
        "",
        "## 1. Dataset Identity",
        "",
        "- **Dataset:** UNSW-NB15",
        "- **Layout:** Official pre-split CSV (Scenario B)",
        "- **Source:** https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "",
        "## 2. File Information",
        "",
    ]
    for split, info in file_hashes.items():
        lines += [
            f"### {split.upper()}",
            f"- **Filename:** `{info['filename']}`",
            f"- **Size:** {info['size_bytes']:,} bytes",
            f"- **SHA-256:** `{info['sha256']}`",
            f"- **Source:** {info['source']}",
            "",
        ]

    lines += [
        "## 3. Observed Schema",
        "",
        "> **OBSERVED** (from actual data) — not a design decision.",
        "",
    ]
    for split, obs in schema_observed.items():
        lines += [
            f"### {split.upper()}",
            f"- **Rows:** {obs['shape']['rows']:,}",
            f"- **Columns:** {obs['shape']['cols']}",
            f"- **Duplicate rows:** {obs['duplicate_row_count']}",
            f"- **Total nulls:** {sum(obs['null_counts'].values())}",
            "",
        ]

    lines += [
        "## 4. Raw `attack_cat` Values (OBSERVED)",
        "",
        "> Raw strings preserved exactly, including whitespace.",
        "",
        "```",
    ]
    for s in all_raw_strings:
        lines.append(f"  {repr(s)}")
    lines += ["```", ""]

    lines += [
        "## 5. Canonicalization Mapping",
        "",
        "| Raw Value | Stripped | Canonical | Status |",
        "|---|---|---|---|",
    ]
    seen = set()
    for rec in can_audit_combined:
        key = rec.get("raw_value")
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| `{rec['raw_value']}` | `{rec['stripped_value']}` | `{rec['canonical_value']}` | {rec['status']} |"
        )
    lines.append("")

    lines += [
        "## 6. Label Distribution",
        "",
        "### TRAIN",
        f"- Normal (0): {label_dist['train']['counts'].get(0, label_dist['train']['counts'].get('0', 'N/A')):,}",
        f"- Attack (1): {label_dist['train']['counts'].get(1, label_dist['train']['counts'].get('1', 'N/A')):,}",
        "",
        "### TEST",
        f"- Normal (0): {label_dist['test']['counts'].get(0, label_dist['test']['counts'].get('0', 'N/A')):,}",
        f"- Attack (1): {label_dist['test']['counts'].get(1, label_dist['test']['counts'].get('1', 'N/A')):,}",
        "",
    ]

    lines += [
        "## 7. Canonical Attack Category Distribution",
        "",
        "| Category | TRAIN | TEST |",
        "|---|---|---|",
    ]
    all_cats = sorted(set(list(can_dist_train.keys()) + list(can_dist_test.keys())))
    for cat in all_cats:
        lines.append(
            f"| {cat} | {can_dist_train.get(cat, 0):,} | {can_dist_test.get(cat, 0):,} |"
        )
    lines.append("")

    lines += [
        "## 8. Cross-Split Overlap",
        "",
        f"- **Method:** {overlap_report['method']}",
        f"- **Train rows:** {overlap_report['train_row_count']:,}",
        f"- **Test rows:** {overlap_report['test_row_count']:,}",
        f"- **Exact overlap:** {overlap_report['overlap_count']}",
        f"- **Overlap %:** {overlap_report['overlap_percentage']:.4f}%",
        f"- **Limitation:** {overlap_report['limitation']}",
        "",
    ]

    lines += [
        "## 9. Withheld-Attack Candidates",
        "",
        f"- **Eligibility threshold:** >= {config['withheld']['min_test_instances']} TEST instances",
        "",
        "| Candidate | TEST Count | Eligible |",
        "|---|---|---|",
    ]
    for cat, info in candidate_report["all_attack_subclasses"].items():
        lines.append(f"| {cat} | {info['count']:,} | {'Yes' if info['eligible'] else 'No'} |")
    lines.append("")

    lines += [
        "## 10. Selected Withheld Target",
        "",
        "- **Selection rule:** `fixed_named_target`",
        "- **Target:** `Backdoor`",
        "- **Timing:** Before any model training or evaluation",
        "- **Rationale:** See `experiments/pre_registration.json` revision_history block",
        "",
        "> **DECIDED** — not an observation from data alone.",
        "",
    ]

    lines += [
        "## 11. Schema Validation",
        "",
        f"- **Violations:** {len(violations)}",
    ]
    if violations:
        for v in violations:
            lines.append(f"  - {v}")
    else:
        lines.append("  - None")
    lines.append("")

    lines += [
        "## 12. Audit Limitations",
        "",
        "- Overlap check detects only EXACT duplicate rows. Near-duplicate or semantically similar rows are not detected.",
        "- Canonicalization map v1.0 may need revision if additional raw string variants are discovered in the full dataset.",
        "- `sport` and `dsport` treatment (numeric/categorical/exclude) is deferred to Sprint 2.",
        "- Identifier column treatment (`id`, `srcip`, `dstip`, `Stime`, `Ltime`) is deferred to Sprint 2.",
        "",
        "## 13. Sprint 1 Status",
        "",
        "- **EXP_DATA_ACQUISITION_AUDIT:** IMPLEMENTED / awaiting validation with real data",
        "- **EXP_UNSEEN_RESERVATION:** IMPLEMENTED / awaiting validation with real data",
        "",
    ]

    report_path = audit_dir / "audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Audit report written to: {report_path}")


if __name__ == "__main__":
    main()
