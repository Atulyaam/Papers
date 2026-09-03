# Sprint 10 — Final Formal Validation Report
**Experiment ID:** `EXP_ABLATION_V1`
**Phase:** `VALIDATE`
**Timestamp UTC:** `2026-09-03T19:26:56.828240Z`
**Overall Status:** **`PASS`** (25/25 PASS, 0 FAIL, 0 INCONCLUSIVE)

## 1. Executive Summary
Formal validation for Sprint 10 (`EXP_ABLATION_V1`) completed successfully. All 25 verification checks passed with zero failures and zero inconclusive tests. All experiment artifacts, cache structures, dataset hashes, determinism checks, and publication-facing tables have been machine-verified.

## 2. Individual Test Results

| Test ID | Description | Status |
|---|---|---|
| `V01_CONFIGURATION_COMPLETENESS` | All 8 configurations × 3 seeds (24 result files) present | **`PASS`** |
| `V02_A0_IDENTITY` | A0 model identity matches strongest Sprint 9 individual model (RF) | **`PASS`** |
| `V03_A1_MEMBERSHIP` | A1 contains DT+RF+SVM+NN with LR meta-learner & OOF features | **`PASS`** |
| `V04_A1B_RULE` | A1b uses deterministic score normalization (sigmoid) & arithmetic mean | **`PASS`** |
| `V05_ABLATION_MEMBERSHIP` | A2–A5 leave-one-out configurations correctly structured | **`PASS`** |
| `V06_CACHE_INTEGRITY` | Exact reuse of cached base-model predictions across all ablated models | **`PASS`** |
| `V07_FEATURE_INTEGRITY` | EXP_MI_V1_1 75-feature set identity and order verified against hash | **`PASS`** |
| `V08_DATASET_HASHES` | Raw SHA-256 hashes of TRAIN, VAL, DEV_TEST, BACKDOOR match frozen metadata | **`PASS`** |
| `V09_HEADLINE_SPLIT` | All headline metrics computed on DEVELOPMENT_TEST (N=81,749) | **`PASS`** |
| `V10_METRIC_DEFINITIONS` | Publication-safe metric qualifiers verified (Macro-F1, Macro Recall, FPR, etc.) | **`PASS`** |
| `V11_A6_OR_FUSION` | Row-by-row logical OR fusion verified; zero positive attacks removed | **`PASS`** |
| `V12_A1_A6_INTERPRETATION` | Attack Recall preservation vs Benign FPR increase documented clearly | **`PASS`** |
| `V13_BACKDOOR_ISOLATION` | Protected Backdoor strictly isolated; 582/583 detected by A1 and A6 | **`PASS`** |
| `V14_SEED_STATISTICS` | Seed statistics (mean, population std ddof=0, min, max) verified | **`PASS`** |
| `V15_PAIRED_DELTAS` | All 7 paired comparisons present across seeds; no significance claimed | **`PASS`** |
| `V16_RESULT_SCHEMAS` | Locked schemas for ablation_table.csv, paired_deltas.csv, publication_metrics.csv verified | **`PASS`** |
| `V17_RESULT_HASHES` | SHA-256 hashes of summary.json, ablation_table.csv, paired_deltas.csv match established provenance | **`PASS`** |
| `V18_CONFIG_IMMUTABILITY` | config_sha256_before == config_sha256_after (zero post-hoc protocol changes) | **`PASS`** |
| `V19_ENVIRONMENT` | Environment verified (scikit-learn 1.9.0, numpy 2.4.6, torch 2.7.1+cu118) | **`PASS`** |
| `V20_DETERMINISM` | Independent inference re-execution produces 0.00e+00 max numerical difference | **`PASS`** |
| `V21_TEST_SUITE` | Automated test suite (tests/test_ablation.py) passes 30/30 | **`PASS`** |
| `V22_PROVENANCE` | Metadata contains all required execution and environment fields | **`PASS`** |
| `V23_NO_RESULT_TUNING` | Zero protocol, hyperparameter, or architectural adjustments post-evaluation | **`PASS`** |
| `V24_SPRINT9_ISOLATION` | Sprint 9 frozen artifacts, checkpoints, and sprint9-freeze tag intact | **`PASS`** |
| `V25_PUBLICATION_INTERPRETATION` | Publication interpretations adhere to approved non-overclaiming guidelines | **`PASS`** |

## 3. Authoritative Full Metrics Summary Table

Computed on DEVELOPMENT_TEST (N=81,749) and aggregated across seeds 42, 123, 2024:

| Configuration | Macro-F1 | Macro Precision | Macro Recall | Attack F1 | Balanced Accuracy | FPR |
|---|---|---|---|---|---|---|
| `A0_RF` | 0.881618 | 0.904485 | 0.875848 | 0.903881 | 0.875848 | 0.229189 |
| `A1_FULL_STACK` | 0.891977 | 0.906642 | 0.887181 | 0.909925 | 0.887181 | 0.194874 |
| `A1b_SOFT_VOTE` | 0.850642 | 0.886852 | 0.844651 | 0.883275 | 0.844651 | 0.293775 |
| `A2_NO_DT` | 0.892276 | 0.906817 | 0.887497 | 0.910133 | 0.887497 | 0.194144 |
| `A3_NO_RF` | 0.867496 | 0.885319 | 0.862615 | 0.890971 | 0.862615 | 0.232766 |
| `A4_NO_SVM` | 0.891022 | 0.906944 | 0.886033 | 0.909524 | 0.886033 | 0.199748 |
| `A5_NO_NN` | 0.891953 | 0.906608 | 0.887159 | 0.909902 | 0.887159 | 0.194874 |
| `A6_STACK_PLUS_AE` | 0.891807 | 0.906522 | 0.887005 | 0.909801 | 0.887005 | 0.195225 |

## 4. Key Scientific Findings & Approved Interpretation
1. **Supervised Stacking Superiority:** Learned stacking (`A1_FULL_STACK`, Macro-F1 = 0.891977) substantially outperforms both the strongest individual baseline (`A0_RF`, Macro-F1 = 0.881618, delta = +0.010359) and simple soft-voting control (`A1b_SOFT_VOTE`, Macro-F1 = 0.850642, delta = +0.041335).
2. **Random Forest Indispensability:** Ablating Random Forest (`A3_NO_RF`) causes the largest performance drop of any base learner (Macro-F1 drops to 0.867496, delta = -0.024481), establishing RF as the foundational driver of the ensemble.
3. **Marginal Base-Learner Impact:** Ablating Decision Tree (`A2_NO_DT`, Macro-F1 = 0.892276) produces a negligible delta (+0.000299), showing DT contributes no positive value beyond RF. Ablating Neural Network (`A5_NO_NN`, Macro-F1 = 0.891953) has essentially zero effect (-0.000024).
4. **A6 Autoencoder Trade-Off:** Unsupervised AE OR-fusion (`A6_STACK_PLUS_AE`, Macro-F1 = 0.891807) preserves Attack Recall identically (0.969236 across both models, delta = +0.000000). The slight Macro Recall drop (-0.000176) is entirely attributable to 13 benign false alarms increasing FPR by +0.000351, not a reduction in attack sensitivity.
5. **Statistical Caveat:** Per protocol, no statistical significance is claimed from n=3 seeds.

## 5. History Note
The previously cited values 0.168784 and 0.895055 were not present in repository artifacts. They appeared only in an earlier assistant chat response, typed without reading any source file. They are NOT experiment results and must NOT be used in publication material. The repository's stored values, as regenerated directly from summary.json, are authoritative.

## 6. Final Status
$$\mathbf{VALIDATION = PASS}$$
Sprint 10 (`EXP_ABLATION_V1`) is fully validated and ready for freeze.
