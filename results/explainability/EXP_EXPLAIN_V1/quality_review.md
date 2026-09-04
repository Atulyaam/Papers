# Sprint 11 — Quality Review: Explainability / SHAP (EXP_EXPLAIN_V1)
**Generated**: 2026-09-04T12:15:42Z

## 1. Scoping & Seed Locking
A0_RF, A1_FULL_STACK, and A6_STACK_PLUS_AE explainability in this sprint targets the frozen seed-42 instances of each system only. This is a scoping decision to avoid tripling SHAP compute cost without a corresponding research question; it does not imply seeds 123/2024 behave identically.

## 2. Pre-Verification Gates Summary
- **Total Pre-Verification Gates Evaluated**: 34
- **Pre-Verification Pass Rate**: 100% (ALL PASSED)

| Gate ID | Description | Status |
|:---|:---|:---|
| PV-01 | A0_RF artifact resolved | **PASS** |
| PV-02 | A1 DT artifact resolved | **PASS** |
| PV-03 | A1 RF artifact resolved | **PASS** |
| PV-04 | A1 SVM artifact resolved | **PASS** |
| PV-05 | A1 NN artifact resolved | **PASS** |
| PV-06 | A1 Logistic meta-learner resolved | **PASS** |
| PV-07 | A6 AE artifact resolved | **PASS** |
| PV-08 | Frozen preprocessing resolved | **PASS** |
| PV-09 | Exact 75-feature order confirmed | **PASS** |
| PV-10 | Class mapping confirmed (0=benign, 1=attack) | **PASS** |
| PV-11 | DEVELOPMENT_TEST confirmed | **PASS** |
| PV-11a | Global source_row_uid uniqueness confirmed across all four splits | **PASS** |
| PV-12 | Explanation-set generation reproducible (2,000 rows, independent RNG seed 42) | **PASS** |
| PV-13 | Direct overlap of canonical source-row identities between explanation set and TRAIN = zero | **PASS** |
| PV-14 | Direct overlap of canonical source-row identities between explanation set and OOF-fitting population, seed 42 = zero | **PASS** |
| PV-15 | Direct overlap of canonical source-row identities between explanation set and OOF-fitting population, seed 123 = zero | **PASS** |
| PV-16 | Direct overlap of canonical source-row identities between explanation set and OOF-fitting population, seed 2024 = zero | **PASS** |
| PV-17 | All other applicable fitting-population overlap = zero (PV-17a: VAL=0, PV-17b: PROT=0) | **PASS** |
| PV-18 | RF explainer compatibility verified (TreeExplainer, tree_path_dependent) | **PASS** |
| PV-19 | SVM explainer compatibility verified (LinearExplainer on decision_function) | **PASS** |
| PV-20 | Logistic explainer compatibility verified (LinearExplainer on 4 meta-features) | **PASS** |
| PV-21 | NN explainer compatibility verified (DeepExplainer on 2D probability output) | **PASS** |
| PV-22 | NN GPU determinism test executed | **PASS** |
| PV-23 | CPU fallback status | **PASS** |
| PV-24 | NN final determinism requirement passed (< 1e-10) | **PASS** |
| PV-25 | AE reconstruction-error formula confirmed (RE = 1/75 * sum((x_i - xhat_i)^2)) | **PASS** |
| PV-26 | A1 SVM meta-input representation confirmed (raw decision_function, svm_decision_score) | **PASS** |
| PV-27 | SHAP background/masker specification confirmed for every explainer | **PASS** |
| PV-28 | No retraining path required (all models strictly frozen) | **PASS** |
| PV-29 | No tuning path required (no hyperparameter/threshold tuning) | **PASS** |
| PV-30 | Sprint 9 artifacts unchanged (read-only verification) | **PASS** |
| PV-31 | Sprint 10 artifacts unchanged (read-only verification) | **PASS** |
| PV-32 | Report-generation provenance path verified | **PASS** |
| PV-33 | Per-target figure directory structure confirmed non-colliding | **PASS** |

## 3. Dataset Splitting & Row Count Provenance (Fix B & C)
### Raw Split Counts and Gap Origin (Fix B)
- `data/raw/UNSW_NB15_training-set.csv` (raw): 175,341 rows
- `data/splits/train.csv`: 162,395 rows
- `data/splits/validation.csv`: 11,200 rows
- Sum (train + validation): 173,595 rows
- Gap vs raw training-set: exactly 1,746 rows.

The exact originating step for this 1,746-row difference was NOT traced during Sprint 11 execution — no automated check in this pipeline located or verified a specific source file/experiment responsible for the gap. This is noted here for future investigation rather than asserted as fact, per this sprint's anti-hallucination protocol. It is unrelated to Sprint 11's leakage testing or explanation-set construction, both of which operate only on the four active splits (TRAIN/VALIDATION/DEVELOPMENT_TEST/PROTECTED_BACKDOOR) verified in PV-11a below.

### Pairwise Disjointness & Global Uniqueness (Fix C)
- Total rows across all four active splits: 255927
- Total unique canonical `source_row_uid` values: 255927 (0 collisions)
- **PV-11a status**: PASS
- **PV-17a**: Explanation set ∩ VALIDATION source_row_uids = **0**
- **PV-17b**: Explanation set ∩ PROTECTED_BACKDOOR source_row_uids = **0**

> **Disjointness Rationale**: This follows directly from PV-11a passing, which verifies that all UIDs across the four splits are globally unique (no collisions). Thus, any pair of splits are automatically disjoint.

## 4. Model Explanations & Global Importance

### A0_RF Top Features (TreeExplainer)
| Rank | Feature Name | Mean Absolute SHAP |
|:---|:---|:---|
| 1 | `sttl` | 0.159486 |
| 2 | `ct_state_ttl` | 0.069635 |
| 3 | `dttl` | 0.031746 |
| 4 | `sbytes` | 0.026620 |
| 5 | `dload` | 0.023754 |

### A1 Meta-Learner Importance (LinearExplainer)
| Rank | Meta-Feature | Coefficient | Mean Absolute SHAP |
|:---|:---|:---|:---|
| 1 | `svm_decision_score` | 1.004018 | 2.913711 |
| 2 | `rf_attack_probability` | 6.346733 | 2.462267 |
| 3 | `nn_attack_probability` | 0.684100 | 0.272296 |
| 4 | `dt_attack_probability` | 0.221841 | 0.096531 |

### A6 AE Reconstruction Error Top Features
| Rank | Feature Name | Mean Squared RE |
|:---|:---|:---|
| 1 | `sbytes` | 173.212211 |
| 2 | `service_pop3` | 100.780388 |
| 3 | `smean` | 38.590800 |
| 4 | `sloss` | 32.851453 |
| 5 | `dload` | 9.131884 |

### A6 AE-Decisive Population
- **Predicate**: `A1_pred == 0 AND AE_flag == 1`
- **Total AE-Decisive Cases on Full DEVELOPMENT_TEST (N=81,749)**: **13**
- Stored in `A6_STACK_PLUS_AE/ae_decisive_cases.csv` with complete reconstruction profiles.

## 5. Reproducibility & Environment
- **Python Version**: 3.11.9
- **NumPy Version**: 2.4.6
- **PyTorch Version**: 2.7.1+cu118
- **SHAP Version**: 0.51.0
- **NN Determinism Gate Difference**: 0.0
- **Final NN Computation Device**: cuda

## 6. Audit & Documentation Integrity
- All quantitative numbers are programmatically derived from generated CSVs and JSONs.
- No retraining, tuning, or modification of Sprint 9 / Sprint 10 artifacts occurred.