# Supplementary Original-Split Benchmark Results

SUPPLEMENTARY ORIGINAL-SPLIT BENCHMARK. This experiment is independent of the main hybrid IDS research protocol and is intended only to provide a conventional reference using the original UNSW-NB15 train/test split.

No stacking, fusion, C01/C06 evaluation, or main-project hypothesis decision was performed in this benchmark.

## Final Comparison Table

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | Weighted-F1 | Balanced Accuracy | TP | FP | TN | FN |
|-------|----------|-----------|--------|----|----------|-------------|-------------------|----|----|----|----|
| Decision Tree | 0.8603 | 0.8165 | 0.9626 | 0.8836 | 0.8545 | 0.8574 | 0.8488 | 43637 | 9807 | 27193 | 1695 |
| Random Forest | 0.8931 | 0.8509 | 0.9770 | 0.9096 | 0.8894 | 0.8915 | 0.8837 | 44291 | 7759 | 29241 | 1041 |
| SVM | 0.8336 | 0.7908 | 0.9489 | 0.8626 | 0.8258 | 0.8295 | 0.8206 | 43014 | 11382 | 25618 | 2318 |
| Neural Network | 0.8912 | 0.8635 | 0.9531 | 0.9061 | 0.8884 | 0.8902 | 0.8843 | 43207 | 6831 | 30169 | 2125 |
| Autoencoder | 0.4561 | 0.9552 | 0.0127 | 0.0251 | 0.3239 | 0.2937 | 0.5060 | 576 | 27 | 36973 | 44756 |

## Autoencoder Threshold Information
- **Threshold (τ)**: 12.409052
- **Calibration Method**: Internal TRAIN-only split (80% fit, 20% cal). mean+3sigma on AE-CALIBRATION subset.
- **Anomaly Count**: 603 (0.73%)

*Note: The Autoencoder produces an anomaly decision (RE > τ) rather than a directly supervised class prediction.*
