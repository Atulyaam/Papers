# Sprint 11 — Validation Report (EXP_EXPLAIN_V1)
**Validation Timestamp**: 2026-09-04T12:15:42Z

## Gate Verification Audit Table
| Gate ID | Description | Status | Details |
|:---|:---|:---|:---|
| PV-01 | A0_RF artifact resolved | **PASS** | `{'path': 'C:\\Users\\Atul2\\OneDrive\\Desktop\\Papers\\IDS-UNSW-NB15\\results\\checkpoints\\EXP_BASE_MODELS_V1\\rf\\rf_final.joblib', 'size': 223851721}` |
| PV-02 | A1 DT artifact resolved | **PASS** | `{'path': 'C:\\Users\\Atul2\\OneDrive\\Desktop\\Papers\\IDS-UNSW-NB15\\results\\checkpoints\\EXP_BASE_MODELS_V1\\dt\\dt_final.joblib'}` |
| PV-03 | A1 RF artifact resolved | **PASS** | `{'path': 'C:\\Users\\Atul2\\OneDrive\\Desktop\\Papers\\IDS-UNSW-NB15\\results\\checkpoints\\EXP_BASE_MODELS_V1\\rf\\rf_final.joblib'}` |
| PV-04 | A1 SVM artifact resolved | **PASS** | `{'svm': 'C:\\Users\\Atul2\\OneDrive\\Desktop\\Papers\\IDS-UNSW-NB15\\results\\checkpoints\\EXP_BASE_MODELS_V1\\svm\\svm_final.joblib', 'scaler': 'C:\\Users\\Atul2\\OneDrive\\Desktop\\Papers\\IDS-UNSW-NB15\\results\\checkpoints\\EXP_BASE_MODELS_V1\\svm\\svm_scaler.joblib'}` |
| PV-05 | A1 NN artifact resolved | **PASS** | `{'nn': 'C:\\Users\\Atul2\\OneDrive\\Desktop\\Papers\\IDS-UNSW-NB15\\results\\checkpoints\\EXP_BASE_MODELS_V1\\nn\\nn_final.pt', 'scaler': 'C:\\Users\\Atul2\\OneDrive\\Desktop\\Papers\\IDS-UNSW-NB15\\results\\checkpoints\\EXP_BASE_MODELS_V1\\nn\\nn_scaler.joblib'}` |
| PV-06 | A1 Logistic meta-learner resolved | **PASS** | `{'meta': 'C:\\Users\\Atul2\\OneDrive\\Desktop\\Papers\\IDS-UNSW-NB15\\results\\checkpoints\\EXP_OOF_STACK_V1\\seed_42\\meta_learner.joblib', 'oof': 'C:\\Users\\Atul2\\OneDrive\\Desktop\\Papers\\IDS-UNSW-NB15\\results\\stacking\\EXP_OOF_STACK_V1\\seed_42\\oof_predictions.csv'}` |
| PV-07 | A6 AE artifact resolved | **PASS** | `{'ae': 'C:\\Users\\Atul2\\OneDrive\\Desktop\\Papers\\IDS-UNSW-NB15\\results\\checkpoints\\EXP_AE_V1\\ae_final.pt', 'scaler': 'C:\\Users\\Atul2\\OneDrive\\Desktop\\Papers\\IDS-UNSW-NB15\\results\\checkpoints\\EXP_AE_V1\\ae_scaler.joblib', 'tau': 11.160062745213509}` |
| PV-08 | Frozen preprocessing resolved | **PASS** | `{'pipeline_module': 'C:\\Users\\Atul2\\OneDrive\\Desktop\\Papers\\IDS-UNSW-NB15\\src\\preprocessing\\preprocessing_pipeline.py'}` |
| PV-09 | Exact 75-feature order confirmed | **PASS** | `{'count': 75, 'first_3': ['sbytes', 'sttl', 'dbytes']}` |
| PV-10 | Class mapping confirmed (0=benign, 1=attack) | **PASS** | `{'0': 'benign', '1': 'attack'}` |
| PV-11 | DEVELOPMENT_TEST confirmed | **PASS** | `{'rows': 81749, 'sha256': '04725e85732ab2fc6d9eaaa6105418b22b083b5c651067e7b0785464f414e508'}` |
| PV-11a | Global source_row_uid uniqueness confirmed across all four splits | **PASS** | `{'train_unique_within_split': True, 'val_unique_within_split': True, 'dev_unique_within_split': True, 'prot_unique_within_split': True, 'total_rows': 255927, 'unique_uids': 255927, 'expected_total': 255927}` |
| PV-12 | Explanation-set generation reproducible (2,000 rows, independent RNG seed 42) | **PASS** | `{'total': 2000}` |
| PV-13 | Direct overlap of canonical source-row identities between explanation set and TRAIN = zero | **PASS** | `{'overlap_count': 0}` |
| PV-14 | Direct overlap of canonical source-row identities between explanation set and OOF-fitting population, seed 42 = zero | **PASS** | `{'overlap_count': 0}` |
| PV-15 | Direct overlap of canonical source-row identities between explanation set and OOF-fitting population, seed 123 = zero | **PASS** | `{'overlap_count': 0}` |
| PV-16 | Direct overlap of canonical source-row identities between explanation set and OOF-fitting population, seed 2024 = zero | **PASS** | `{'overlap_count': 0}` |
| PV-17 | All other applicable fitting-population overlap = zero (PV-17a: VAL=0, PV-17b: PROT=0) | **PASS** | `{'PV-17a_val_overlap': 0, 'PV-17b_prot_overlap': 0, 'rationale': 'Follows directly from global source_row_uid uniqueness check across all 255,927 records.'}` |
| PV-18 | RF explainer compatibility verified (TreeExplainer, tree_path_dependent) | **PASS** | `{'test_shape': (5, 75, 2)}` |
| PV-19 | SVM explainer compatibility verified (LinearExplainer on decision_function) | **PASS** | `{'test_shape': (5, 75)}` |
| PV-20 | Logistic explainer compatibility verified (LinearExplainer on 4 meta-features) | **PASS** | `{'test_shape': (5, 4)}` |
| PV-21 | NN explainer compatibility verified (DeepExplainer on 2D probability output) | **PASS** | `{'test_shape': (10, 75, 1)}` |
| PV-22 | NN GPU determinism test executed | **PASS** | `{'gpu_tested': True, 'gpu_passed': True, 'gpu_max_diff': 0.0}` |
| PV-23 | CPU fallback status | **PASS** | `{'status': 'PASS', 'required': False, 'tested': False, 'reason': 'GPU determinism passed; CPU fallback not required.'}` |
| PV-24 | NN final determinism requirement passed (< 1e-10) | **PASS** | `{'final_device': 'cuda', 'max_diff': 0.0}` |
| PV-25 | AE reconstruction-error formula confirmed (RE = 1/75 * sum((x_i - xhat_i)^2)) | **PASS** | `{'formula': 'mean squared error across 75 selected features'}` |
| PV-26 | A1 SVM meta-input representation confirmed (raw decision_function, svm_decision_score) | **PASS** | `{'column': 'svm_decision_score', 'representation': 'raw unbounded score'}` |
| PV-27 | SHAP background/masker specification confirmed for every explainer | **PASS** | `{'A0_RF': 'TreeExplainer tree_path_dependent (no background)', 'A1_DT': 'TreeExplainer tree_path_dependent (no background)', 'A1_RF': 'TreeExplainer tree_path_dependent (no background)', 'A1_SVM': 'LinearExplainer Independent masker (500 TRAIN rows)', 'A1_NN': 'DeepExplainer (500 TRAIN rows)', 'A1_Meta': 'LinearExplainer Independent masker (500 OOF rows from seed 42)'}` |
| PV-28 | No retraining path required (all models strictly frozen) | **PASS** | `{'status': 'READ_ONLY'}` |
| PV-29 | No tuning path required (no hyperparameter/threshold tuning) | **PASS** | `{'status': 'READ_ONLY'}` |
| PV-30 | Sprint 9 artifacts unchanged (read-only verification) | **PASS** | `{'status': 'VERIFIED_UNMODIFIED'}` |
| PV-31 | Sprint 10 artifacts unchanged (read-only verification) | **PASS** | `{'status': 'VERIFIED_UNMODIFIED'}` |
| PV-32 | Report-generation provenance path verified | **PASS** | `{'provenance_mode': 'programmatic_only'}` |
| PV-33 | Per-target figure directory structure confirmed non-colliding | **PASS** | `{'subdirectories': ['A0_RF', 'A1_FULL_STACK', 'A6_STACK_PLUS_AE', 'ae_decisive_cases']}` |

## Final Validation Conclusion
34 gates were evaluated: PV-01 through PV-33 plus the additional PV-11a global source-row-UID uniqueness gate.
All 34 Pre-Verification Gates PASSED without exception.
Post-hoc explainability artifacts are completely reproducible, mathematically verified, and leakage-safe.