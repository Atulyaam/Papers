"""
scripts/audit10_verification.py

Audit 10 Verification Suite:
1. Programmatic diff against quality_review.md
2. Publication-safe terminology check (grep/search on generated table)
3. Source-of-truth check for A1 seed 42
4. Dataset / Headline split check (N=81,749)
5. SHA-256 integrity hashes check
"""

import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EXP = ROOT / "results/ablation/EXP_ABLATION_V1"
QUALITY_REVIEW_PATH = EXP / "quality_review.md"
PUBLICATION_CSV_PATH = EXP / "publication_metrics.csv"
SUMMARY_PATH = EXP / "summary.json"

def test_quality_review_comparison():
    print("=== 1. QUALITY REVIEW COMPARISON ===")
    content = QUALITY_REVIEW_PATH.read_text(encoding="utf-8")
    has_full_metrics = "Full Metrics" in content or "Macro Precision" in content or "Attack F1" in content
    print(f"quality_review.md has 'Full Metrics' table: {has_full_metrics}")

    sections = [line for line in content.splitlines() if line.startswith("## ")]
    print(f"Existing sections in quality_review.md: {sections}")

    # Check Macro-F1 values in quality_review.md against summary.json
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    configs_data = data["configs"]

    mismatches = []
    for line in content.splitlines():
        for cid in configs_data.keys():
            if f"| {cid} |" in line:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                # parts: [Config, Seed 42, Seed 123, Seed 2024, Mean, Std]
                stored_mean = f"{configs_data[cid]['mean']['macro_f1']:.6f}"
                if parts[4] != stored_mean:
                    mismatches.append(f"{cid} mean mismatch: file={parts[4]} vs summary={stored_mean}")
    print(f"Macro-F1 table diffs against summary.json: {mismatches if mismatches else '0 mismatches (exact match)'}")

def test_terminology():
    print("\n=== 2. PUBLICATION-SAFE TERMINOLOGY CHECK ===")
    import subprocess
    cmd_out = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_publication_table.py")],
        capture_output=True, text=True
    ).stdout

    lines = cmd_out.splitlines()
    header_lines = [l for l in lines if "Configuration" in l]
    print(f"Table headers found: {len(header_lines)}")
    for h in header_lines:
        print(f"  Header: {h}")

    # Check that "Recall" without "Macro" does NOT exist in table header
    # Check that "Precision" without "Macro" does NOT exist in table header
    errors = []
    for h in header_lines:
        columns = [c.strip() for c in h.split("|")[1:-1]]
        if "Recall" in columns:
            errors.append("FAIL: Naked 'Recall' found in columns")
        if "Precision" in columns:
            errors.append("FAIL: Naked 'Precision' found in columns")
        if "Macro Recall" not in columns:
            errors.append("FAIL: 'Macro Recall' missing")
        if "Macro Precision" not in columns:
            errors.append("FAIL: 'Macro Precision' missing")
        if "Attack F1" not in columns:
            errors.append("FAIL: 'Attack F1' missing")
        if "False Positive Rate (FPR)" not in columns and "FPR" not in columns:
            errors.append("FAIL: 'FPR' missing")

    print(f"Terminology check result: {'ALL PASS' if not errors else errors}")

def test_source_of_truth_a1_seed42():
    print("\n=== 3. SOURCE-OF-TRUTH CHECK (A1 Seed 42) ===")
    from scripts.run_ablation import build_meta_features, build_meta_lr, encode_split

    X_dev, y_dev = encode_split("development_test")
    assert len(y_dev) == 81749, f"N != 81749: {len(y_dev)}"

    meta_X_oof, y_oof = build_meta_features(["dt", "rf", "svm", "nn"], 42, "oof")
    meta_X_dev, _     = build_meta_features(["dt", "rf", "svm", "nn"], 42, "dev_test")

    lr = build_meta_lr(42)
    lr.fit(meta_X_oof, y_oof)
    preds = lr.predict(meta_X_dev)

    tp = int(np.sum((preds == 1) & (y_dev == 1)))
    fp = int(np.sum((preds == 1) & (y_dev == 0)))
    tn = int(np.sum((preds == 0) & (y_dev == 0)))
    fn = int(np.sum((preds == 0) & (y_dev == 1)))

    print(f"Raw Confusion Counts: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Total N = {tp + fp + tn + fn}")

    attack_recall = tp / (tp + fn)
    class0_recall = tn / (tn + fp)
    macro_recall = (class0_recall + attack_recall) / 2.0
    fpr = fp / (fp + tn)

    prec0 = tn / (tn + fn)
    prec1 = tp / (tp + fp)
    f1_0 = 2 * prec0 * class0_recall / (prec0 + class0_recall)
    f1_1 = 2 * prec1 * attack_recall / (prec1 + attack_recall)
    macro_f1 = (f1_0 + f1_1) / 2.0

    print(f"Computed Attack Recall (Sensitivity): {attack_recall:.6f}")
    print(f"Computed Benign Specificity:          {class0_recall:.6f}")
    print(f"Computed Macro Recall:                {macro_recall:.6f}")
    print(f"Computed FPR:                         {fpr:.6f}")
    print(f"Computed Macro-F1:                    {macro_f1:.6f}")

    # Read stored values in summary.json
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    s_a1_42 = data["configs"]["A1_FULL_STACK"]["per_seed"]["42"]

    print("\nComparison with stored summary.json A1 Seed 42:")
    print(f"  Macro-F1:    computed={macro_f1:.6f} vs stored={s_a1_42['macro_f1']:.6f} (diff={abs(macro_f1 - s_a1_42['macro_f1']):.1e})")
    print(f"  Macro Recall: computed={macro_recall:.6f} vs stored={s_a1_42['recall']:.6f} (diff={abs(macro_recall - s_a1_42['recall']):.1e})")
    print(f"  Balanced Acc: computed={macro_recall:.6f} vs stored={s_a1_42['balanced_accuracy']:.6f} (diff={abs(macro_recall - s_a1_42['balanced_accuracy']):.1e})")
    print(f"  FPR:          computed={fpr:.6f} vs stored={s_a1_42['fpr']:.6f} (diff={abs(fpr - s_a1_42['fpr']):.1e})")

def test_hashes():
    print("\n=== 4. SHA-256 INTEGRITY HASHES ===")
    files = [
        EXP / "summary.json",
        EXP / "ablation_table.csv",
        EXP / "paired_deltas.csv",
    ]
    def h(p):
        s = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1048576), b""):
                s.update(c)
        return s.hexdigest()

    expected = {
        "summary.json": "4440a755f2776871a89813d1936d7411b300c21922d9b614b156a9e061c375ce",
        "ablation_table.csv": "6405b019d6d28b1f28469ffdd2881cf053b86aeca7329a9ebe2d2df5edc4ce6a",
        "paired_deltas.csv": "01937e4e88ff5e74963c5bccf16b0c29ac4572fd48bee84ebaad3372bc2daad4",
    }

    for f in files:
        cur = h(f)
        exp = expected[f.name]
        status = "MATCH" if cur == exp else "MISMATCH"
        print(f"  {f.name}: {cur} [{status}]")
        assert cur == exp, f"Hash mismatch on {f.name}: {cur} != {exp}"

if __name__ == "__main__":
    test_quality_review_comparison()
    test_terminology()
    test_source_of_truth_a1_seed42()
    test_hashes()
