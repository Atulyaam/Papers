# Sprint 12 — Numerical & Prediction Comparison Report
**Experiment ID**: `EXP_FINAL_REPRO_V1`  
**Execution Timestamp**: `2026-09-04T13:45:52.398959+00:00`  
**Numerical Tolerance**: `atol=1e-08`, `rtol=1e-08`  

## 1. Prediction-Level Discrete Equality
| Target Pipeline | Population | Total Rows | Mismatch Count | Mismatch Rate | Verdict |
|:---|:---|:---|:---|:---|:---|
| C06 Fusion (OR) | DEV_TEST | 81,749 | 0 | 0.000% | **PASS (EXACT)** |
| C01 Supervised Stack (Seed 42) | DEV_TEST | 81,749 | 0 | 0.000% | **PASS (EXACT)** |

## 2. Metric-Level Floating-Point Comparison
| Component | Metric | Frozen Reference | Reproduced (Sprint 12) | Absolute Diff | Relative Diff | Status |
|:---|:---|:---|:---|:---|:---|:---|
| H1 | `stacking_mean_macro_f1` | 0.89296125 | 0.89296125 | 0.00e+00 | 0.00e+00 | **PASS** |
| H1 | `stacking_macro_f1_seed_42` | 0.89260917 | 0.89260917 | 0.00e+00 | 0.00e+00 | **PASS** |
| H1 | `stacking_macro_f1_seed_123` | 0.89261864 | 0.89261864 | 0.00e+00 | 0.00e+00 | **PASS** |
| H1 | `stacking_macro_f1_seed_2024` | 0.89365594 | 0.89365594 | 0.00e+00 | 0.00e+00 | **PASS** |
| H1 | `rf_dev_test_macro_f1` | 0.88073326 | 0.88073326 | 0.00e+00 | 0.00e+00 | **PASS** |
| H1 | `diff` | 0.01222799 | 0.01222799 | 0.00e+00 | 0.00e+00 | **PASS** |
| Fusion | `c06_dev_test_macro_f1` | 0.89244000 | 0.89243998 | 1.68e-08 | 1.89e-08 | **PASS** |
| Fusion | `c06_dev_test_fpr` | 0.19224324 | 0.19224324 | 0.00e+00 | 0.00e+00 | **PASS** |
| Fusion | `c06_prot_detected` | 582.00000000 | 582.00000000 | 0.00e+00 | 0.00e+00 | **PASS** |
| AE | `ae_val_fpr` | 0.00062500 | 0.00062500 | 0.00e+00 | 0.00e+00 | **PASS** |
| AE | `ae_prot_detected` | 0.00000000 | 0.00000000 | 0.00e+00 | 0.00e+00 | **PASS** |
| Ablation | `a1b_macro_f1` | 0.85063244 | 0.85063244 | 0.00e+00 | 0.00e+00 | **PASS** |