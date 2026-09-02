# Sprint Roadmap

## Sprint 0 — Architecture and Development Rules
Status: Designed

Deliverables:
- system architecture
- module boundaries
- leakage policy
- logging
- checkpoints
- provenance
- seed policy
- Git/FREEZE discipline
- feature lifecycle

## Sprint 1 — UNSW-NB15 Acquisition and Audit
Design first, then implement:

- acquire raw files
- verify file identity
- calculate SHA-256
- schema audit
- data quality audit
- class distribution
- attack-subclass distribution
- identify eligible withheld subclasses
- create schema contract
- create audit report

No model training.

## Sprint 2 — Preprocessing
Design:

- cleaning
- missing/non-finite policy
- categorical encoding
- unknown-category handling
- identifier exclusion
- train-fitted preprocessing
- tests

## Sprint 3 — Protected Unseen-Attack Reservation and Split Protocol
Design and freeze:

- eligibility threshold
- candidate list
- selection rule
- pre-registration
- protected set creation
- TRAIN/VALIDATION/TEST layout
- leakage tests

## Sprint 4 — Mutual Information Feature Selection
TRAIN-only MI fitting and selected-feature artifact.

## Sprint 5 — Individual Base Models
DT, RF, SVM, NN under identical preprocessing/evaluation.

## Sprint 6 — Leakage-Controlled OOF Stacking
5-fold OOF predictions + logistic-regression meta-learner.

## Sprint 7 — Autoencoder Branch
Benign-only training and validation-derived threshold.

## Sprint 8 — Formal Fusion
Implement and test the 2x2 deterministic fusion rule.

## Sprint 9 — H1/H2/H3 Evaluation
Run the must-have hypotheses under frozen protocol.

## Sprint 10 — Ablation
Populate the five-row ablation matrix.

## Sprint 11 — Explainability
Level-1 and Level-2 SHAP, bounded runtime.

## Sprint 12 — Reproducible Final Runs
Three-seed H1 runs, final artifacts, provenance.

## Sprint 13 — Results and Paper Integration
Only after numerical results are final.

## Stretch Goals

Only after all core sprints are stable:

- H4 SHAP stability
- KDDTest+
- all eligible withheld subclasses
- significance testing
- secondary meta-learner
