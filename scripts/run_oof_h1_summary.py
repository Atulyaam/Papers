"""
scripts/run_oof_h1_summary.py
-------------------------------
Sprint 6: Aggregate three per-seed metrics.json files into h1_summary.json.

Reads:
    results/stacking/EXP_OOF_STACK_V1/seed_42/metrics.json
    results/stacking/EXP_OOF_STACK_V1/seed_123/metrics.json
    results/stacking/EXP_OOF_STACK_V1/seed_2024/metrics.json

Writes:
    results/stacking/EXP_OOF_STACK_V1/h1_summary.json

Must only be run after all three seeds have been completed.
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.stacking.meta_learner import (
    compute_h1_summary,
    META_EVALUATION_LIMITATION_TEXT,
    SPRINT5_RF_REFERENCE,
    SPRINT5_RF_REFERENCE_LABEL,
)
from src.models.stacking.artifacts import save_h1_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("h1_summary")

RESULTS_DIR = ROOT / "results/stacking/EXP_OOF_STACK_V1"
H1_SEEDS = [42, 123, 2024]


def main() -> None:
    logger.info("=== H1 SUMMARY START ===")

    seed_results = []
    for seed in H1_SEEDS:
        metrics_path = RESULTS_DIR / f"seed_{seed}" / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Metrics not found for seed {seed}: {metrics_path}\n"
                f"Run: python scripts/run_oof_stacking.py --seed {seed}"
            )
        with open(metrics_path, encoding="utf-8") as fh:
            m = json.load(fh)
        logger.info(
            "Seed %d | macro_f1=%.6f | in_sample=%s",
            seed, m["macro_f1"], m.get("in_sample_evaluation_warning", "N/A"),
        )
        seed_results.append(m)

    summary = compute_h1_summary(seed_results)
    save_h1_summary(summary, RESULTS_DIR)

    print("\n" + "=" * 55)
    print("EXP_OOF_STACK_V1 — H1 STACKING RESULTS (IN-SAMPLE OOF)")
    print("=" * 55)
    for seed in H1_SEEDS:
        f1 = summary["per_seed_macro_f1"][str(seed)]
        print(f"  Seed {seed:4d}: macro_f1 = {f1:.6f}")
    print("-" * 55)
    print(f"  Mean:          {summary['mean_macro_f1']:.6f}")
    print(f"  Std (ddof=1):  {summary['std_macro_f1']:.6f}")
    print("-" * 55)
    print(f"  Sprint 5 ref:  {SPRINT5_RF_REFERENCE:.6f}  [{SPRINT5_RF_REFERENCE_LABEL}]")
    print("=" * 55)
    print()
    print(f"LIMITATION: {META_EVALUATION_LIMITATION_TEXT}")
    print()
    print(summary["two_reporting_units_statement"])
    print()
    logger.info(
        "=== H1 SUMMARY DONE | mean=%.6f | std=%.6f ===",
        summary["mean_macro_f1"], summary["std_macro_f1"],
    )


if __name__ == "__main__":
    main()
