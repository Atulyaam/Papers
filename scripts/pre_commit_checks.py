"""Pre-commit integrity checks for Sprint 2 freeze."""
import pathlib
import subprocess
import pandas as pd
from src.utils.hashing import sha256_file

TRAIN_EXPECTED = "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa"
TEST_EXPECTED  = "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559"

print("=== PRE-COMMIT INTEGRITY CHECKS ===")
print()

# 1. Raw file hashes
train_hash = sha256_file("data/raw/UNSW_NB15_training-set.csv")
test_hash  = sha256_file("data/raw/UNSW_NB15_testing-set.csv")
train_ok = train_hash == TRAIN_EXPECTED
test_ok  = test_hash == TEST_EXPECTED
print(f"Raw TRAIN hash: {'MATCH' if train_ok else 'MISMATCH'}")
print(f"Raw TEST hash:  {'MATCH' if test_ok else 'MISMATCH'}")
assert train_ok, "TRAIN hash mismatch"
assert test_ok,  "TEST hash mismatch"

# 2. Protected unseen set
prot = pd.read_csv("data/splits/protected_unseen_attack.csv", low_memory=False)
assert len(prot) == 583
assert (prot["attack_cat"] == "Backdoor").all()
assert prot["label"].eq(1).all()
print(f"Protected unseen set: {len(prot)} rows, all Backdoor, label=1 -> UNCHANGED")

# 3. No checkpoint
ck = pathlib.Path("results/checkpoints/EXP_PREPROCESSING_V1")
print(f"Checkpoint exists: {ck.exists()}  (expected: False)")
assert not ck.exists(), "Checkpoint must not be created before freeze"

# 4. Sprint 1 source files unchanged vs cf93ca3
r = subprocess.run(["git", "diff", "cf93ca3", "--name-only"], capture_output=True, text=True)
sprint1 = [
    "src/preprocessing/loader.py",
    "src/preprocessing/schema_audit.py",
    "src/preprocessing/schema_validator.py",
    "src/preprocessing/attack_cat_canonicalization.py",
    "src/preprocessing/withheld_candidate.py",
    "src/preprocessing/protected_unseen_attack.py",
    "src/utils/hashing.py",
]
modified = set(r.stdout.strip().split("\n"))
all_sprint1_clean = True
for m in sprint1:
    status = "UNCHANGED"
    if m in modified:
        status = "MODIFIED (VIOLATION)"
        all_sprint1_clean = False
    print(f"  {m}: {status}")
assert all_sprint1_clean, "Sprint 1 source modified"

print()
print("ALL PRE-COMMIT CHECKS: PASS")
