# COMPARISON REPORT — MAIN RESEARCH vs SUPPLEMENTARY ORIGINAL-SPLIT BENCHMARK

## Section 1 — Executive Summary

Main research protocol and supplementary original-split benchmark are distinct experimental protocols. 

The main research pipeline rigorously investigated hybrid Intrusion Detection System (IDS) capabilities using a controlled dataset split, out-of-fold (OOF) stacking, autoencoder-based anomaly fusion, and a protected unseen-attack backdoor evaluation. It achieved very high performance in individual models (e.g., Random Forest at 95.09% Macro-F1 on single-CV splits) and stacking (~94.72% OOF Macro-F1), though the selected late-fusion rule (C06) yielded a highly conservative Macro-F1 of 89.24% while retaining perfect supervised backdoor detection (582/583).

The supplementary original-split benchmark is a standard, independent evaluation using the canonical UNSW-NB15 training and testing files. It evaluated five fundamental models using the exact same 75-feature representation but omitted complex architecture (no stacking, no fusion). In this conventional setup, Random Forest achieved the highest Macro-F1 (88.94%).

These two results are NOT directly equivalent because their training distributions, dataset subsetting logic, threshold calibration populations, and evaluation scopes differ. Both are useful: the main project provides deep insights into fusion architectures on a protected subset, while the benchmark provides a standard baseline that other researchers can easily replicate on the original dataset split.

## Section 2 — Data / Protocol Comparison

| Property | Main Research | Original-Split Benchmark |
|---|---|---|
| Dataset | UNSW-NB15 | UNSW-NB15 |
| Feature set | EXP_MI_V1_1 | EXP_MI_V1_1 |
| Feature count | 75 | 75 |
| Data split | Controlled project protocol | Original UNSW train/test |
| Protected Backdoor | Yes | No special protected subset |
| OOF stacking | Yes | No |
| Fusion | Yes | No |
| C06 | Yes | No |
| Fresh training | According to sprint | Fresh benchmark |
| Purpose | Research hypothesis evaluation | Conventional benchmark reference |

## Section 3 — Five Model Comparison

| Model | Main Project Result | Original-Split Result | Metric | Notes |
|---|---|---|---|---|
| Decision Tree | 93.89% | 85.45% | Macro-F1 | Main: EXP_BASE_MODELS_V1 validation |
| Random Forest | 95.09% | 88.94% | Macro-F1 | Main: EXP_BASE_MODELS_V1 validation |
| SVM | 92.03% | 82.58% | Macro-F1 | Main: EXP_BASE_MODELS_V1 validation |
| Neural Network | 92.43% | 88.84% | Macro-F1 | Main: EXP_BASE_MODELS_V1 validation |
| Autoencoder | NOT RUN STANDALONE | 32.39% | Macro-F1 | Main: Evaluated via C06 Fusion only |

*(Note: The main-project and supplementary numbers were NOT generated under identical evaluation protocols and cannot be directly compared.)*

## Section 4 — Supervised Model Comparison

### Main Project (EXP_BASE_MODELS_V1 - Selected Models)
- **Decision Tree**: 93.89% Macro-F1
- **Random Forest**: 95.09% Macro-F1
- **SVM**: 92.03% Macro-F1
- **Neural Network**: 92.43% Macro-F1

*(Metrics such as Precision/Recall/F1 breakdowns are available in fold results but Macro-F1 was the primary selection metric).*

### Original-Split Benchmark (Original Test Set)
- **Decision Tree**: Accuracy=86.03%, Precision=81.65%, Recall=96.26%, F1=88.36%, Macro-F1=85.45%, Weighted-F1=85.74%, Balanced-Acc=84.88%
- **Random Forest**: Accuracy=89.31%, Precision=85.09%, Recall=97.70%, F1=90.96%, Macro-F1=88.94%, Weighted-F1=89.15%, Balanced-Acc=88.37%
- **SVM**: Accuracy=83.36%, Precision=79.08%, Recall=94.89%, F1=86.26%, Macro-F1=82.58%, Weighted-F1=82.95%, Balanced-Acc=82.06%
- **Neural Network**: Accuracy=89.12%, Precision=86.35%, Recall=95.31%, F1=90.61%, Macro-F1=88.84%, Weighted-F1=89.02%, Balanced-Acc=88.43%

**Highlights**:
- Strongest model in original split benchmark: **Random Forest** (88.94% Macro-F1)
- Strongest historical main-project individual model: **Random Forest** (95.09% Macro-F1)
- The main project protocol (single-CV controlled split) produced higher scores across the board due to the differing distributional properties of the controlled evaluation split compared to the original, highly imbalanced UNSW-NB15 test split.

## Section 5 — Stacking

### Main Project (EXP_OOF_STACK_V1)
- **Macro-F1**: 94.72% (OOF In-Sample Reference, mean across 3 seeds)
- **Seeds**: 42, 123, 2024
- **Limitations**: The Sprint 6 OOF ~94.72% reference reflects in-sample OOF behavior on the training partition rather than a fully held-out test distribution.

Original-split benchmark did not perform stacking.

## Section 6 — Autoencoder

| Property | Main Research AE | Original Benchmark AE |
|---|---|---|
| **Training Regime** | Benign-only training | Benign-only training |
| **Calibration** | 11,200 normal-only VAL rows | Internal 11,200 normal-only TRAIN split |
| **Threshold (τ)** | 11.160063 (mean+3σ) | 12.409052 (mean+3σ) |
| **Precision** | N/A (Embedded in C06) | 95.52% |
| **Recall** | N/A (Embedded in C06) | 1.27% |
| **F1** | N/A (Embedded in C06) | 2.51% |
| **Macro-F1** | N/A (Embedded in C06) | 32.39% |
| **Anomaly Count** | N/A | 603 (0.73% Anomaly Rate) |

*(Note: Thresholds cannot be directly compared as if they were intended to be identical, because their calibration protocols operate on entirely different datasets).*

**Interpretation**: 
The benchmark Autoencoder is highly conservative. At its calibrated threshold, it generated very few positive anomaly detections relative to the massive test attack population, resulting in a 95.52% Precision but a 1.27% Recall.

## Section 7 — Fusion

### Main Research Only
- **C01**: Supervised-only (No fusion)
- **C06**: Supervised OR AE (using mean+3σ threshold)

**C06 Validation Results (EXP_FUSION_V1)**:
- Macro-F1: 0.89244
- Recall: 0.967753
- FPR: 0.192243
- TP: 43306
- FP: 7113
- TN: 29887
- FN: 1443

**Protected Backdoor Results**:
- C01 (Supervised Only): 582/583
- C06 (Fusion): 582/583
- Detection Rate: 0.998285 (1 missed)

**Hypothesis Outcomes**:
- H-FUSION = FALSE (C06 failed to provide a statistically significant improvement over C01)
- H-PROT-BACKDOOR = FALSE (C06 failed to improve upon the 582/583 baseline)

The original-split benchmark contains no fusion stage and therefore has no C01/C06-equivalent result.

## Section 8 — Best Model By Protocol

**Original-split benchmark**:
Random Forest has the highest Macro-F1 = 0.8894 among the five independent benchmark models.

**Main project**:
Random Forest achieved 95.09% Macro-F1 during the single-CV baseline phase (EXP_BASE_MODELS_V1), representing the strongest historical main-project single-model value before stacking/fusion stages were introduced.

## Section 9 — Research Contribution

This side-by-side reporting reveals several key observations:
1. **RF is strongest** among the five independent original-split models, mirroring its strength in the main project.
2. **Neural Network is very close to RF** in Macro-F1 in the benchmark (88.84% vs 88.94%).
3. **Autoencoder alone is highly conservative** under its benchmark threshold, yielding high precision but capturing barely 1% of the attack space.
4. **Main project explores a different research question** using stacking, benign-only AE, fusion, and protected unseen-attack evaluation, separating it entirely from a standard model accuracy bake-off.
5. **C06 did not demonstrate measurable improvement** over C01 in the frozen Sprint 8 evidence, failing to prove the hybrid architecture hypothesis.

## Section 10 — Limitations

- **Different train/test protocols**: The main project uses controlled splits and a protected Backdoor evaluation space.
- **Supplementary benchmark**: Uses the original fixed UNSW-NB15 train/test split.
- **Historical limits**: Historical main-project baselines (e.g., Sprint 5 RF) may not be matched 3-seed comparisons and were primarily used for model tuning selection.
- **Calibration limits**: AE thresholds differ because calibration protocols operate on different validation distributions.
- **Comparability**: Some main-project and benchmark numbers are fundamentally not directly comparable. 

## Section 11 — Final Presentation Table

| Component | Main Research | Original-Split Benchmark |
|---|---:|---:|
| DT | 93.89% Macro-F1 | 85.45% Macro-F1 |
| RF | 95.09% Macro-F1 | 88.94% Macro-F1 |
| SVM | 92.03% Macro-F1 | 82.58% Macro-F1 |
| NN | 92.43% Macro-F1 | 88.84% Macro-F1 |
| Stacking | 94.72% Macro-F1 | NOT RUN |
| Autoencoder | NOT EVALUATED AS STANDALONE | 32.39% Macro-F1 |
| Fusion C06 | 89.244% Macro-F1 | NOT RUN |
| Protected Backdoor | 582/583 | NOT EVALUATED |
| H-FUSION | FALSE | NOT APPLICABLE |
