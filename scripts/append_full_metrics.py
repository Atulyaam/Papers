"""
scripts/append_full_metrics.py

Sprint 10 — Final Reporting Cleanup Pass:
1. Ground truth read from results/ablation/EXP_ABLATION_V1/summary.json
2. Programmatic cross-check against IMPORTANT VERIFIED VALUES block
3. Programmatic append of Full Metrics table and interpretation to quality_review.md
4. Programmatic post-append verification
"""

import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "results/ablation/EXP_ABLATION_V1"
SUMMARY_PATH = EXP / "summary.json"
CSV_PATH = EXP / "publication_metrics.csv"
QUALITY_REVIEW_PATH = EXP / "quality_review.md"

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

# CROSS-CHECK BLOCK FROM USER PROMPT (SANITY CHECK ONLY — NOT SOURCE OF TRUTH)
CROSS_CHECK_VALUES = {
    "A0_RF": {
        "Macro-F1": "0.881618",
        "Macro Precision": "0.904485",
        "Macro Recall": "0.875848",
        "Attack F1": "0.903881",
        "Balanced Accuracy": "0.875848",
        "FPR": "0.229189",
    },
    "A1_FULL_STACK": {
        "Macro-F1": "0.891977",
        "Macro Precision": "0.906642",
        "Macro Recall": "0.887181",
        "Attack F1": "0.909925",
        "Balanced Accuracy": "0.887181",
        "FPR": "0.194874",
    },
    "A1b_SOFT_VOTE": {
        "Macro-F1": "0.850642",
        "Macro Precision": "0.886852",
        "Macro Recall": "0.844651",
        "Attack F1": "0.883275",
        "Balanced Accuracy": "0.844651",
        "FPR": "0.293775",
    },
    "A2_NO_DT": {
        "Macro-F1": "0.892276",
        "Macro Precision": "0.906817",
        "Macro Recall": "0.887497",
        "Attack F1": "0.910133",
        "Balanced Accuracy": "0.887497",
        "FPR": "0.194144",
    },
    "A3_NO_RF": {
        "Macro-F1": "0.867496",
        "Macro Precision": "0.885319",
        "Macro Recall": "0.862615",
        "Attack F1": "0.890971",
        "Balanced Accuracy": "0.862615",
        "FPR": "0.232766",
    },
    "A4_NO_SVM": {
        "Macro-F1": "0.891022",
        "Macro Precision": "0.906944",
        "Macro Recall": "0.886033",
        "Attack F1": "0.909524",
        "Balanced Accuracy": "0.886033",
        "FPR": "0.199748",
    },
    "A5_NO_NN": {
        "Macro-F1": "0.891953",
        "Macro Precision": "0.906608",
        "Macro Recall": "0.887159",
        "Attack F1": "0.909902",
        "Balanced Accuracy": "0.887159",
        "FPR": "0.194874",
    },
    "A6_STACK_PLUS_AE": {
        "Macro-F1": "0.891807",
        "Macro Precision": "0.906522",
        "Macro Recall": "0.887005",
        "Attack F1": "0.909801",
        "Balanced Accuracy": "0.887005",
        "FPR": "0.195225",
    },
}

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== STEP 1: READ GROUND TRUTH FROM summary.json ===")
    summary_data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))["configs"]

    # Format values directly from summary.json ground truth
    ground_truth_table = {}
    for cid in CONFIG_ORDER:
        m = summary_data[cid]["mean"]
        ground_truth_table[cid] = {
            "Macro-F1": f"{m['macro_f1']:.6f}",
            "Macro Precision": f"{m['precision']:.6f}",
            "Macro Recall": f"{m['recall']:.6f}",
            "Attack F1": f"{m['f1']:.6f}",
            "Balanced Accuracy": f"{m['balanced_accuracy']:.6f}",
            "FPR": f"{m['fpr']:.6f}",
        }

    print("Ground truth successfully read for all 8 configurations.")

    print("\n=== STEP 2: CROSS-CHECK GROUND TRUTH AGAINST CROSS-CHECK BLOCK ===")
    cross_check_diffs = []
    for cid in CONFIG_ORDER:
        for metric in ["Macro-F1", "Macro Precision", "Macro Recall", "Attack F1", "Balanced Accuracy", "FPR"]:
            actual = ground_truth_table[cid][metric]
            expected = CROSS_CHECK_VALUES[cid][metric]
            if actual != expected:
                cross_check_diffs.append(f"{cid} {metric}: ground_truth={actual} vs expected={expected}")

    if cross_check_diffs:
        print("STOP: Mismatch detected in cross-check!")
        for diff in cross_check_diffs:
            print(f"  {diff}")
        sys.exit(1)
    else:
        print("Cross-check PASS: All 48 cells match the expected verified values exactly.")

    print("\n=== STEP 3: PRE-MODIFICATION HASHES ===")
    pre_hashes = {
        "summary.json": sha256_file(SUMMARY_PATH),
        "ablation_table.csv": sha256_file(EXP / "ablation_table.csv"),
        "paired_deltas.csv": sha256_file(EXP / "paired_deltas.csv"),
    }
    for k, v in pre_hashes.items():
        print(f"  {k}: {v}")

    print("\n=== STEP 4: CONSTRUCT CLEANUP SECTION ===")
    table_lines = [
        "\n## Full Metrics Across All Configurations\n",
        "Metrics below are computed on DEVELOPMENT_TEST (N=81,749) and aggregated as the arithmetic mean across seeds 42, 123, and 2024.\n\n",
        "| Configuration | Macro-F1 | Macro Precision | Macro Recall | Attack F1 | Balanced Accuracy | FPR |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for cid in CONFIG_ORDER:
        row = ground_truth_table[cid]
        table_lines.append(
            f"| {cid} | {row['Macro-F1']} | {row['Macro Precision']} | {row['Macro Recall']} | "
            f"{row['Attack F1']} | {row['Balanced Accuracy']} | {row['FPR']} |\n"
        )

    table_lines.append("\n### A1 vs A6 Performance & Recall Interpretation\n")
    table_lines.append(
        "- **Attack Recall Preservation:** Unsupervised Autoencoder OR-fusion preserves Attack Recall (Sensitivity) identically between A1 and A6 (0.969236 across both models, delta = +0.000000). Not a single true positive attack detection is lost.\n"
        "- **Benign Specificity & FPR:** The Autoencoder introduces 13 additional benign false positives across the 37,000 benign samples on Development-Test, slightly increasing False Positive Rate (FPR) from 0.194874 to 0.195225 (mean delta = +0.000351).\n"
        "- **Macro Recall Distinction:** The slight drop in Macro Recall from 0.887181 to 0.887005 (delta = -0.000176) occurs strictly because Macro Recall is an unweighted arithmetic average of Benign Specificity and Attack Sensitivity. This is NOT a decrease in attack sensitivity.\n"
    )

    table_lines.append("\n### History Note (Audit 9 & 10 Reconciliation)\n")
    table_lines.append(
        "The previously cited values 0.168784 and 0.895055 were not present in repository artifacts. "
        "They appeared only in an earlier assistant chat response, typed without reading any source file. "
        "They are NOT experiment results and must NOT be used in publication material. "
        "The repository's stored values, as regenerated by this audit's script directly from summary.json, are authoritative.\n"
    )

    append_text = "".join(table_lines)

    # Read original quality_review.md content
    original_qr = QUALITY_REVIEW_PATH.read_text(encoding="utf-8")

    # If section already exists (e.g. rerun), replace from section header, else append
    if "## Full Metrics Across All Configurations" in original_qr:
        idx = original_qr.index("## Full Metrics Across All Configurations")
        updated_qr = original_qr[:idx].rstrip() + "\n" + append_text
    else:
        updated_qr = original_qr.rstrip() + "\n" + append_text

    QUALITY_REVIEW_PATH.write_text(updated_qr, encoding="utf-8")
    print(f"quality_review.md updated ({len(updated_qr)} bytes).")

    print("\n=== STEP 5: POST-MODIFICATION VERIFICATION ===")
    post_hashes = {
        "summary.json": sha256_file(SUMMARY_PATH),
        "ablation_table.csv": sha256_file(EXP / "ablation_table.csv"),
        "paired_deltas.csv": sha256_file(EXP / "paired_deltas.csv"),
    }
    for k, v in post_hashes.items():
        match = "MATCH" if v == pre_hashes[k] else "MISMATCH"
        print(f"  {k}: {v} [{match}]")
        assert v == pre_hashes[k], f"Hash mismatch on {k}!"

    # Programmatic parse of appended quality_review.md table
    re_read_qr = QUALITY_REVIEW_PATH.read_text(encoding="utf-8")
    assert "## Full Metrics Across All Configurations" in re_read_qr
    assert "Metrics below are computed on DEVELOPMENT_TEST (N=81,749)" in re_read_qr

    parsed_rows = {}
    in_table = False
    headers = []
    for line in re_read_qr.splitlines():
        if line.startswith("| Configuration |"):
            in_table = True
            headers = [c.strip() for c in line.split("|")[1:-1]]
            continue
        if in_table:
            if line.startswith("|---|"):
                continue
            if line.startswith("|"):
                parts = [p.strip() for p in line.split("|")[1:-1]]
                cid = parts[0]
                parsed_rows[cid] = dict(zip(headers[1:], parts[1:]))
            else:
                in_table = False

    print(f"\nParsed {len(parsed_rows)} rows from updated quality_review.md.")
    assert len(parsed_rows) == 8, f"Expected 8 rows, got {len(parsed_rows)}"
    assert list(parsed_rows.keys()) == CONFIG_ORDER, "Config order mismatch!"

    # Verify each parsed cell against ground_truth_table
    cell_mismatches = []
    for cid in CONFIG_ORDER:
        for metric in headers[1:]:
            val_in_file = parsed_rows[cid][metric]
            val_ground_truth = ground_truth_table[cid][metric]
            if val_in_file != val_ground_truth:
                cell_mismatches.append(f"{cid} {metric}: file={val_in_file} vs truth={val_ground_truth}")

    if cell_mismatches:
        print("STOP: Cell mismatches in updated file!")
        for m in cell_mismatches:
            print(f"  {m}")
        sys.exit(1)
    else:
        print("Verification PASS: All 48 table cells in quality_review.md match ground truth identically.")

if __name__ == "__main__":
    main()
