# Sprint 12 — Validation Gates Report
**Experiment ID**: `EXP_FINAL_REPRO_V1`  
**Execution Timestamp**: `2026-09-04T13:45:52.398959+00:00`  

| Gate ID | Description | Status | Summary Details |
|:---|:---|:---|:---|
| **RV-01** | Repository baseline verified | **PASS** | `{'head_commit': '8eeece3bb5a8e4c05613e3e39aa2e98b4ef5eb39', 'tags_present': ['sp` |
| **RV-02** | Frozen reference package resolved | **PASS** | `{'sprint9_eval': True, 'sprint10_ablation': True, 'sprint11_explain': True, 'spr` |
| **RV-03** | Sprint 10 provenance package sufficient for FULL ablation reproduction | **NOT_REPRODUCED** | `{'status': 'FAIL_FOR_FULL_ABLATION', 'finding': 'Missing row-level predictions f` |
| **RV-04** | Single Sprint 12 environment captured | **PASS** | `{'python': '3.11.9', 'platform': 'Windows-10-10.0.26200-SP0', 'processor': 'Inte` |
| **RV-05** | Environment deviations documented | **PASS** | `{'hardware_policy': 'Locked tolerance (atol=1e-8, rtol=1e-8) applied to all comp` |
| **RV-06** | Dataset hashes verified | **PASS** | `{'train': {'path': 'data\\splits\\train.csv', 'sha256': '4a259324e604f013287a5de` |
| **RV-07** | Split identities verified | **PASS** | `{'train': 162395, 'validation': 11200, 'development_test': 81749, 'protected_bac` |
| **RV-08** | Canonical source-row identities verified | **PASS** | `{'total_uids': 255927, 'all_unique': True}` |
| **RV-09** | Feature-set hash verified | **PASS** | `{'sha256': '6a1816143a4fbe1141e406a820c5adbd0b1452b45172a9d7de8767a897db1024', '` |
| **RV-10** | Feature ordering verified | **PASS** | `{'first_5': ['sbytes', 'sttl', 'dbytes', 'ct_state_ttl', 'dttl'], 'last_5': ['pr` |
| **RV-11** | Model hashes verified | **PASS** | `{'dt_final': {'path': 'results\\checkpoints\\EXP_BASE_MODELS_V1\\dt\\dt_final.jo` |
| **RV-12** | Scaler hashes verified | **PASS** | `{'svm_scaler': {'path': 'results\\checkpoints\\EXP_BASE_MODELS_V1\\svm\\svm_scal` |
| **RV-13** | AE threshold verified | **PASS** | `{'tau': 11.160062745213509, 'rule': 'mean3sigma'}` |
| **RV-14** | Configuration hashes verified | **PASS** | `{'fusion_config': 'C06', 'epsilon': 0.005, 'epsilon_source': 'evaluate_sprint9.p` |
| **RV-15** | Required seeds verified | **PASS** | `{'seeds': [42, 123, 2024]}` |
| **RV-16** | Frozen checkpoint loading verified | **PASS** | `{'loaded_models': ['dt', 'rf', 'svm', 'nn', 'ae', 'stack_meta_42', 'stack_meta_1` |
| **RV-17** | Zero training operations executed | **PASS** | `{'training_operations_executed': 0, 'fit_calls': 0, 'optimizer_steps': 0, 'backw` |
| **RV-18** | Base-model inference reproduction completed | **PASS** | `{'dt_macro_f1': 0.849851877370434, 'rf_macro_f1': 0.8807332603329192, 'svm_macro` |
| **RV-19** | OOF stacking inference reproduction completed | **PASS** | `{'mean_macro_f1': 0.8929612508481996, 'seed_42_macro_f1': 0.8926091690431182, 's` |
| **RV-20** | AE inference reproduction completed | **PASS** | `{'tau': 11.160062745213509, 'dev_test_total': 81749, 'dev_test_flagged': 594, 'd` |
| **RV-21** | Fusion reproduction completed | **PASS** | `{'c06_macro_f1': 0.892439983171387, 'c06_fpr': 0.19224324324324324, 'c06_prot_de` |
| **RV-22** | H1/H2/H3 evaluation reproduction completed | **PASS** | `{'experiment_id': 'EXP_H123_V1', 'h1_verdict': 'SUPPORTED', 'h2_verdict': 'NOT_S` |
| **RV-23** | Ablation evaluation/reference handling completed | **PASS** | `{'A1b_status': 'REPRODUCED', 'A1b_macro_f1': 0.8506324370045575, 'unsupported_co` |
| **RV-24** | Prediction-level comparison completed | **PASS** | `{'fusion_mismatches': 0, 'stack_seed42_mismatches': 0}` |
| **RV-25** | Metric-level comparison completed | **PASS** | `{'total_metrics_compared': 12, 'all_passed': True, 'max_absolute_diff': 1.682861` |
| **RV-26** | Fixed numerical tolerance applied | **PASS** | `{'atol': 1e-08, 'rtol': 1e-08}` |
| **RV-27** | Protected Backdoor isolation verified | **PASS** | `{'access_type': 'Evaluation-only after model and threshold locking', 'influence_` |
| **RV-28** | Publication metrics generated programmatically | **PASS** | `{'rows_generated': 17}` |
| **RV-29** | Provenance manifest generated | **PASS** | `{'keys_recorded': ['experiment_id', 'protocol', 'timestamp_utc', 'git_commit', '` |
| **RV-30** | Reproducibility report generated | **PASS** | `{'final_verdict': 'PARTIALLY_REPRODUCED / SCOPED REPRODUCTION'}` |
| **RV-31** | Validation status generated | **PASS** | `{'total_gates': 37}` |
| **RV-32** | Sprint 9 unchanged | **PASS** | `{'status': 'read-only, intact'}` |
| **RV-33** | Sprint 10 unchanged | **PASS** | `{'status': 'read-only, intact'}` |
| **RV-34** | Sprint 11 unchanged | **PASS** | `{'status': 'read-only, intact'}` |
| **RV-35** | No frozen artifact overwritten | **PASS** | `{'namespace': 'results/final_reproducibility/EXP_FINAL_REPRO_V1/ isolated'}` |
| **RV-36** | Git diff reviewed | **PASS** | `{'working_tree_diff': 'clean'}` |
| **RV-37** | Final human handoff generated | **PASS** | `{'status': 'READY_FOR_HUMAN_FREEZE_REVIEW'}` |