# Sprint 3 — TRAIN / VALIDATION Split Audit Report

**Experiment ID:** EXP_TRAIN_VAL_SPLIT_V1  
**Created:** 2026-08-31T17:10:50.882594+00:00  
**Git commit:** 59e9f872eee6573dc8fd1311ce96b9a3d827bffb  
**Seed:** 42

## Split Counts

| Split | Rows |
|---|---|
| TRAIN | 162,395 |
| VALIDATION | 11,200 |
| Excluded Backdoor | 1,746 |
| **Input total** | **175,341** |
| Conservation check | 175,341 == 175,341 |

## TRAIN Composition

| Component | Rows |
|---|---|
| Normal (80%) | 44,800 |
| Non-Backdoor attacks (100%) | 117,595 |
| Backdoor | 0 |

## VALIDATION Composition

| Component | Rows |
|---|---|
| Normal (20%) | 11,200 |
| Attack rows | 0 |
| Backdoor | 0 |

## Excluded Backdoor

1,746 Backdoor training-file rows archived to `excluded_train_backdoor.csv`.  
**EXPERIMENTAL ROLE = NONE.** Must not be used for training, tuning, or evaluation.

## Integrity Checks

| Check | Result |
|---|---|
| Row conservation | PASS |
| Exact reconstruction | PASS |
| TRAIN ∩ VAL | 0 (expected 0) |
| TRAIN ∩ EXCLUDED | 0 (expected 0) |
| VAL ∩ EXCLUDED | 0 (expected 0) |
| VAL attack rows | 0 |
| Backdoor in TRAIN | 0 |
| Backdoor in VAL | 0 |
| All integrity checks | ALL PASS |

## Source Integrity

| File | Status |
|---|---|
| UNSW_NB15_training-set.csv | MATCH |
| UNSW_NB15_testing-set.csv | MATCH |
| protected_unseen_attack.csv | MATCH |
| development_test.csv | MATCH |

## Validation Percentile Adequacy

The VALIDATION set contains **11,200 benign rows** for AE threshold calibration.

> These are descriptive empirical support counts, **not confidence intervals**.

| Threshold | Expected upper-tail count |
|---|---|
| 90th percentile | ~1,120 rows |
| 95th percentile | ~560 rows |
| 97.5th percentile | ~280 rows |
| 99th percentile | ~112 rows |

The validation set contains 11,200 benign rows. The 90th-percentile threshold is calibrated on approximately 1,120 upper-tail samples, and the 99th-percentile threshold on approximately 112. All operating points have substantial empirical support. This is descriptive evidence of adequacy, not a formal guarantee.

## Library Versions

| Package | Version |
|---|---|
| Python | 3.11.9 |
| NumPy | 2.4.6 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |

## Output File Hashes

| File | SHA-256 |
|---|---|
| train.csv | `4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c` |
| validation.csv | `13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180` |
| excluded_train_backdoor.csv | `b3f6e7e60c9815a53f40eb2d41df8b67d29f884b922a487c3fe83c02e0db0a02` |
