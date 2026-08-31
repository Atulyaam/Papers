# System Architecture

## 1. High-Level Architecture

```text
                         UNSW-NB15
                              |
                              v
                +---------------------------+
                | Data Acquisition & Audit |
                +-------------+-------------+
                              |
                              v
                +---------------------------+
                | Preprocessing              |
                | - cleaning                 |
                | - categorical encoding     |
                | - train-fitted transforms  |
                +-------------+-------------+
                              |
                              v
                +---------------------------+
                | TRAIN / VALIDATION / TEST |
                | + protected unseen attack |
                +-------------+-------------+
                              |
                              v
                +---------------------------+
                | Mutual Information        |
                | Feature Selection         |
                | TRAIN only                |
                +-------------+-------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
     +--------------------+          +--------------------+
     | Supervised Branch  |          | Anomaly Branch     |
     | DT                 |          | Benign-only AE     |
     | RF                 |          | N -> 12 -> 6      |
     | SVM                |          | -> 12 -> N        |
     | NN                 |          +---------+----------+
     +---------+----------+                    |
               |                               v
               v                       Reconstruction Error
        5-Fold OOF Stacking                    |
               |                               v
               v                       Validation Threshold
      Logistic Regression                      |
       Meta-Learner                            |
               |                               |
               +---------------+---------------+
                               |
                               v
                     +-------------------+
                     | Formal 2x2 Fusion |
                     +---------+---------+
                               |
                               v
                         Final Decision
                               |
                    +----------+----------+
                    |                     |
                    v                     v
                 Metrics                SHAP
```

## 2. Module Boundaries

### `src/preprocessing`
Owns raw-data loading, schema validation, cleaning, categorical encoding, splitting, and train-fitted preprocessing state.

### `src/feature_selection`
Owns Mutual Information ranking/selection. MI fitting occurs on TRAIN only.

### `src/models/base_models`
Owns DT, RF, SVM, and NN definitions and training interfaces.

### `src/models/stacking`
Owns fold generation, OOF prediction creation, meta-learner fitting, and inference-time stacking.

### `src/models/autoencoder`
Owns benign-only AE training, reconstruction scoring, threshold calibration, and anomaly flags.

### `src/fusion`
Owns the deterministic 2x2 decision function.

### `src/explainability`
Owns Level-1 meta-level SHAP, Level-2 full-pipeline SHAP, and lightweight AE reconstruction-error evidence.

### `src/evaluation`
Owns metrics, confusion matrices, ablation tables, threshold-sensitivity tables, and experiment reporting.

### `src/utils`
Owns reproducibility utilities, logging helpers, hashing/provenance helpers, and common infrastructure.

## 3. Dependency Direction

Lower-level modules must not depend on higher-level experiment orchestration.

Preferred direction:

```text
utils
  ^
preprocessing
  ^
feature_selection
  ^
models
  ^
fusion / explainability
  ^
evaluation
  ^
experiment orchestration
```

Circular imports are prohibited.

## 4. Notebooks

Notebooks are for:

- exploration
- visualization
- experiment inspection
- debugging
- result presentation

Core project logic must live in `src/`.
