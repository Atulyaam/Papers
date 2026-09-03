# Data Provenance for Comparison Report

This document records the exact source file and experimental protocol for every quantitative claim made in the comparison report.

## Main Project Sources

| Metric | Value | Experiment ID | Source File | Protocol |
|--------|-------|---------------|-------------|----------|
| DT | 93.89% Macro-F1 | EXP_BASE_MODELS_V1 | `results/base_models/EXP_BASE_MODELS_V1/quality_review.md` | Sprint 5 (single-CV controlled split, baseline models) |
| RF | 95.09% Macro-F1 | EXP_BASE_MODELS_V1 | `results/base_models/EXP_BASE_MODELS_V1/quality_review.md` & `results/evaluation/EXP_H123_V1/h1_results.json` | Sprint 5 (single-CV controlled split, baseline models) |
| SVM | 92.03% Macro-F1 | EXP_BASE_MODELS_V1 | `results/base_models/EXP_BASE_MODELS_V1/quality_review.md` | Sprint 5 (single-CV controlled split, baseline models) |
| NN | 92.43% Macro-F1 | EXP_BASE_MODELS_V1 | `results/base_models/EXP_BASE_MODELS_V1/quality_review.md` | Sprint 5 (single-CV controlled split, baseline models) |
| OOF Stacking | ~94.72% Macro-F1 | EXP_OOF_STACK_V1 | `results/evaluation/EXP_H123_V1/h1_results.json` | Sprint 6 (mean across 3 seeds on OOF in-sample reference) |
| C06 Fusion | 89.244% Macro-F1 | EXP_FUSION_V1 | `results/fusion/EXP_FUSION_V1/quality_review.md` | Sprint 8 (OR fusion rule, threshold mean+3sigma) |
| Protected Backdoor | 582/583 | EXP_FUSION_V1 | `results/fusion/EXP_FUSION_V1/quality_review.md` | Sprint 8 (Protected Unseen-Attack evaluation) |

## Original-Split Benchmark Sources

| Metric | Value | Source File | Protocol |
|--------|-------|-------------|----------|
| DT | 85.45% Macro-F1 | `original_split_benchmark/artifacts/metrics/dt_metrics.json` | Benchmark (Original test split, no fusion) |
| RF | 88.94% Macro-F1 | `original_split_benchmark/artifacts/metrics/rf_metrics.json` | Benchmark (Original test split, no fusion) |
| SVM | 82.58% Macro-F1 | `original_split_benchmark/artifacts/metrics/svm_metrics.json` | Benchmark (Original test split, no fusion) |
| NN | 88.84% Macro-F1 | `original_split_benchmark/artifacts/metrics/nn_metrics.json` | Benchmark (Original test split, no fusion) |
| Autoencoder | 32.39% Macro-F1 | `original_split_benchmark/artifacts/metrics/ae_metrics.json` | Benchmark (Original test split, isolated thresholding) |
