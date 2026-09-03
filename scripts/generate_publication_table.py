"""
scripts/generate_publication_table.py

Sprint 10 — EXP_ABLATION_V1 Publication Metric Table Generator
Reads results/ablation/EXP_ABLATION_V1/summary.json directly.
Generates:
  1. Authoritative Markdown Publication Table
  2. Per-Seed Detailed Breakdown Tables
  3. results/ablation/EXP_ABLATION_V1/publication_metrics.csv
"""

import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "results/ablation/EXP_ABLATION_V1"
SUMMARY_PATH = EXP / "summary.json"
CSV_PATH = EXP / "publication_metrics.csv"

CONFIG_ORDER = [
    "A0_RF",
    "A1_FULL_STACK",
    "A1b_SOFT_VOTE",
    "A2_NO_DT",
    "A3_NO_RF",
    "A4_NO_SVM",
    "A5_NO_NN",
    "A6_STACK_PLUS_AE",
]

SEEDS = ["42", "123", "2024"]

def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")

    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    configs_data = data["configs"]

    # 1. Generate publication_metrics.csv (Mean values across seeds)
    csv_rows = []
    for cid in CONFIG_ORDER:
        mean_metrics = configs_data[cid]["mean"]
        csv_rows.append({
            "config_id": cid,
            "macro_f1": f"{mean_metrics['macro_f1']:.6f}",
            "macro_precision": f"{mean_metrics['precision']:.6f}",
            "macro_recall": f"{mean_metrics['recall']:.6f}",
            "attack_f1": f"{mean_metrics['f1']:.6f}",
            "balanced_accuracy": f"{mean_metrics['balanced_accuracy']:.6f}",
            "fpr": f"{mean_metrics['fpr']:.6f}",
        })

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config_id",
                "macro_f1",
                "macro_precision",
                "macro_recall",
                "attack_f1",
                "balanced_accuracy",
                "fpr",
            ]
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    # 2. Output Markdown Table (Mean across 3 seeds on DEVELOPMENT_TEST)
    print("### Full Metrics Table (Mean across Seeds 42, 123, 2024 on DEVELOPMENT_TEST, N=81,749)\n")
    print("| Configuration | Macro-F1 | Macro Precision | Macro Recall | Attack F1 | Balanced Accuracy | FPR |")
    print("|---|---|---|---|---|---|---|")
    for cid in CONFIG_ORDER:
        m = configs_data[cid]["mean"]
        print(f"| {cid} | {m['macro_f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['balanced_accuracy']:.6f} | {m['fpr']:.6f} |")

    # 3. Output Per-Seed Metric Breakdown
    for seed in SEEDS:
        print(f"\n### Metric Breakdown — Seed {seed} (DEVELOPMENT_TEST, N=81,749)\n")
        print("| Configuration | Macro-F1 | Macro Precision | Macro Recall | Attack F1 | Balanced Accuracy | FPR |")
        print("|---|---|---|---|---|---|---|")
        for cid in CONFIG_ORDER:
            m = configs_data[cid]["per_seed"][seed]
            print(f"| {cid} | {m['macro_f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['balanced_accuracy']:.6f} | {m['fpr']:.6f} |")

if __name__ == "__main__":
    main()
