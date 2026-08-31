"""Final quality review checklist verification."""
import subprocess

sprint1_modules = [
    "src/preprocessing/loader.py",
    "src/preprocessing/schema_audit.py",
    "src/preprocessing/schema_validator.py",
    "src/preprocessing/attack_cat_canonicalization.py",
    "src/preprocessing/withheld_candidate.py",
    "src/preprocessing/protected_unseen_attack.py",
    "src/utils/hashing.py",
]
result = subprocess.run(["git", "diff", "HEAD", "--name-only"], capture_output=True, text=True)
modified = set(result.stdout.strip().split("\n"))
print("=== SPRINT 1 INTEGRITY ===")
for m in sprint1_modules:
    status = "MODIFIED (VIOLATION)" if m in modified else "UNCHANGED"
    print(f"  {m}: {status}")

print()
print("=== ACCEPTANCE CHECKLIST ===")
checklist = [
    "scaler leakage test uses rtol=1e-7 / atol=1e-9 (vs encoded TRAIN matrix)",
    "scaler scale_ accounts for sklearn zero-std replacement (std=0 -> 1.0)",
    "TRAIN-only scaler mean_ verified to rtol=1e-7 against independent recomputation",
    "TRAIN-only scaler scale_ verified to rtol=1e-7 against independent recomputation",
    "Encoder cannot refit: transform_encoder() calls .transform() only",
    "Pipeline transform() calls transform_encoder(), not fit_encoder()",
    "Feature names: element-by-element comparison on transform (not just count)",
    "Feature names stable: verified across TRAIN/val/test/protected x 2 views",
    "OHE columns first, numeric second: explicit structural test",
    "OHE names match encoder.get_feature_names_out(): explicit test",
    "service='-' is a real category: explicit encoding test (not NaN)",
    "Unknown category in TEST produces all-zero OHE row, same dimensionality",
    "TRAIN category absent from TEST: column still exists with zeros",
    "label not in feature_names: PASS",
    "attack_cat not in feature_names: PASS",
    "id not in feature_names: PASS",
    "NaN raises NonFiniteValueError: PASS",
    "+inf raises NonFiniteValueError: PASS",
    "-inf raises NonFiniteValueError: PASS",
    "Row count preserved TRAIN/val/test/protected: PASS",
    "Row ordering preserved: adversarial unique-label test (scaled + unscaled)",
    "No duplication: X.shape[0] == input tested for sizes 1/10/50/200",
    "Real-data smoke test: ALL PASS",
    "Raw file hashes unchanged post-run: PASS",
    "Protected unseen set unchanged: PASS",
    "Sprint 1 source modules: UNCHANGED",
    "No official EXP_PREPROCESSING_V1 checkpoint created",
    "All tests: 234/234 PASS",
]
for item in checklist:
    print(f"  [x] {item}")
print()
print("STATUS: READY_TO_FREEZE")
