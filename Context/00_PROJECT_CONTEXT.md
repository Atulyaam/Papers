# UNSW-NB15 Hybrid Stacked-Ensemble IDS
## Project Context and Authoritative Decisions

This document records the project decisions that are considered fixed unless explicitly revised.

## 1. Dataset

**Authoritative dataset:** UNSW-NB15.

The older proposal/master reference contains NSL-KDD-specific assumptions. Those assumptions must not be copied blindly into implementation. The methodology is retained, but dataset-specific values must be derived from the actual UNSW-NB15 files.

## 2. Core Research System

The final core system is:

1. UNSW-NB15 data pipeline
2. Leakage-controlled preprocessing
3. Mutual Information feature selection fit on TRAIN only
4. DT, RF, SVM, NN supervised base models
5. 5-fold out-of-fold stacking
6. Logistic-regression meta-learner
7. Benign-trained autoencoder anomaly branch
8. Validation-calibrated reconstruction threshold
9. Formal 2x2 fusion rule
10. Predefined withheld/unseen attack-subclass protocol
11. Two-level SHAP explainability
12. Controlled evaluation and ablation

## 3. Core Hypotheses

- H1: The stacked ensemble matches or exceeds the strongest individual classifier under identical conditions.
- H2: The benign-trained autoencoder flags a non-trivial proportion of the withheld subclass as anomalous.
- H3: Combining supervised output with the anomaly signal improves withheld-class detection relative to the ensemble alone, at a controlled false-positive rate.
- H4: SHAP attribution stability across repeated runs is optional/stretch only.

## 4. Protected Unseen-Attack Requirement

The unseen/withheld attack subclass must be selected using a rule fixed before model results are inspected.

The protected withheld subset must be stored separately and must never be used for:

- MI fitting
- scaling fitting
- encoder fitting
- base-model training
- OOF stacking
- meta-learner training
- autoencoder training
- anomaly-threshold calibration
- hyperparameter tuning
- model selection

It is used only for final evaluation.

## 5. Development Philosophy

Every feature follows:

PLAN -> DESIGN -> REVIEW -> IMPLEMENT -> TEST -> VALIDATE -> FREEZE

No large-batch implementation is allowed to bypass this lifecycle.

## 6. Source Documents

Primary methodology source: Final Revised Proposal v3.

Secondary reference: Major Project Master Document.

Where they conflict with the later UNSW-NB15 decision or the explicitly frozen project rules, the current project decisions take precedence.
