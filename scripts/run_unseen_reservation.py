"""
scripts/run_unseen_reservation.py
-----------------------------------
Sprint 1 — Protected Unseen-Attack Reservation

Orchestration script for EXP_UNSEEN_RESERVATION.

This script:
    1. Loads the official TEST CSV (immutable; never modified).
    2. Applies canonicalization to attack_cat.
    3. Validates Backdoor eligibility.
    4. Partitions into protected_unseen_attack and development_test.
    5. Runs 6 validation checks.
    6. Writes output CSVs to data/splits/.
    7. Writes split_metadata.json.
    8. Verifies the source file hash is unchanged after processing.
    9. Logs all steps to results/logs/EXP_UNSEEN_RESERVATION/run.log.

LEAKAGE GUARANTEE:
    This script performs NO model training.
    No sklearn fit(), no gradient updates, no threshold calibration.
    The protected set is final-evaluation-only (later sprints).

Run from project root:
    python scripts/run_unseen_reservation.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.preprocessing.loader import load_raw_unswnb15
from src.preprocessing.protected_unseen_attack import (
    build_split_metadata,
    reserve_protected_unseen_attack,
    ReservationError,
)
from src.utils.hashing import sha256_file
from src.utils.logging_utils import get_experiment_logger

EXPERIMENT_ID = "EXP_UNSEEN_RESERVATION"
CONFIG_PATH = Path("configs/project_config.yaml")


def main() -> None:
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_dir = config["audit"]["log_dir"]
    splits_dir = Path(config["splits"]["output_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Logger
    # ------------------------------------------------------------------
    logger = get_experiment_logger(EXPERIMENT_ID, log_dir)
    now_utc = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 70)
    logger.info("EXPERIMENT START | id=%s | timestamp=%s", EXPERIMENT_ID, now_utc)
    logger.info("Config: %s", CONFIG_PATH)

    # ------------------------------------------------------------------
    # 3. Hash source file BEFORE loading (baseline fingerprint)
    # ------------------------------------------------------------------
    raw_dir = Path(config["data"]["raw_dir"])
    test_path = raw_dir / config["data"]["test_file"]

    if not test_path.exists():
        logger.error("Test file MISSING: %s", test_path)
        sys.exit(1)

    pre_hash = sha256_file(test_path)
    logger.info("Source file pre-processing SHA-256: %s | file=%s", pre_hash, test_path.name)

    # ------------------------------------------------------------------
    # 4. Load test split
    # ------------------------------------------------------------------
    splits = load_raw_unswnb15(config, logger=logger)
    df_test = splits["test"]
    original_row_count = len(df_test)
    logger.info("Test set loaded | rows=%d | cols=%d", original_row_count, len(df_test.columns))

    # ------------------------------------------------------------------
    # 5. Reserve
    # ------------------------------------------------------------------
    cat_col = config["data"]["attack_cat_col"]
    withheld_target = config["withheld"]["target"]
    min_count = config["withheld"]["min_test_instances"]

    try:
        df_protected, df_development = reserve_protected_unseen_attack(
            df_test=df_test,
            cat_col=cat_col,
            withheld_target=withheld_target,
            min_count=min_count,
            logger=logger,
        )
    except ReservationError as e:
        logger.error("RESERVATION FAILED: %s", e)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Write outputs
    # ------------------------------------------------------------------
    protected_path = splits_dir / "protected_unseen_attack.csv"
    development_path = splits_dir / "development_test.csv"

    df_protected.to_csv(protected_path, index=False)
    df_development.to_csv(development_path, index=False)

    logger.info("Written: %s (%d rows)", protected_path, len(df_protected))
    logger.info("Written: %s (%d rows)", development_path, len(df_development))

    # ------------------------------------------------------------------
    # 7. Verify source file is unchanged
    # ------------------------------------------------------------------
    post_hash = sha256_file(test_path)
    if pre_hash != post_hash:
        logger.error(
            "CRITICAL: Source file hash changed after processing! "
            "pre=%s post=%s — raw file may have been modified.",
            pre_hash,
            post_hash,
        )
        sys.exit(1)
    logger.info("Source file hash UNCHANGED after processing. Raw file immutability confirmed.")

    # ------------------------------------------------------------------
    # 8. Build and write metadata
    # ------------------------------------------------------------------
    metadata = build_split_metadata(
        source_test_path=test_path,
        df_protected=df_protected,
        df_development=df_development,
        original_row_count=original_row_count,
        source_sha256=pre_hash,
        withheld_target=withheld_target,
        protocol_version=config.get("protocol_version", "1.0"),
    )

    metadata_path = splits_dir / "split_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Written: %s", metadata_path)

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("  Original test rows:       %d", original_row_count)
    logger.info("  Protected (Backdoor):     %d", len(df_protected))
    logger.info("  Development test:         %d", len(df_development))
    logger.info("  Row conservation:         %s", metadata["row_conservation"])
    logger.info("  Reconstruction verified:  %s", metadata["reconstruction_verified"])
    logger.info("  Source hash unchanged:    PASS")
    logger.info("  Elapsed: %.2fs", elapsed)
    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETE | id=%s | status=SUCCESS", EXPERIMENT_ID)


if __name__ == "__main__":
    main()
