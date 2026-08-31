# Experimental Protocol

## 1. Data Access Rules

### TRAIN
May be used to fit:

- encoders
- scalers
- MI selection
- base classifiers
- OOF base models
- meta-learner inputs through OOF predictions
- benign-only AE training

### VALIDATION
Used for:

- AE threshold calibration
- validation checks that were explicitly planned

Validation must not be used to fit the supervised models unless a later design explicitly states so.

### TEST
Used once for final evaluation.

TEST must not be used to fit:

- encoder
- scaler
- MI
- model parameters
- AE threshold

## 2. Fit/Transform Discipline

Any learned preprocessing object uses:

```text
TRAIN -> fit
VAL   -> transform
TEST  -> transform
```

The protected unseen attack set uses TRAIN-fitted transformations only.

## 3. OOF Stacking

Use K=5 folds.

For each fold:

1. train DT/RF/SVM/NN on the other four folds;
2. predict the held-out fold;
3. store OOF predictions.

After all folds:

```text
OOF predictions + true labels
        -> logistic-regression meta-learner
```

In-sample base-model predictions must not be used as training data for the meta-learner.

For final inference, retrain base models on the full permitted TRAIN data and pass their test predictions to the already-trained meta-learner.

## 4. Autoencoder

Train exclusively on benign TRAIN samples.

Target architecture:

```text
N -> 12 -> 6 -> 12 -> N
```

where N is the final MI-selected feature count determined from UNSW-NB15.

Fixed baseline settings:

- Adam
- learning rate 0.001
- batch size 32
- up to 50 epochs
- early stopping patience 5
- fixed seed

## 5. Threshold

Compute reconstruction errors on benign VALIDATION samples.

Primary operating threshold:

```text
tau = 95th percentile of benign validation reconstruction error
```

Also report a sensitivity sweep:

- 90th percentile
- 95th percentile
- 97.5th percentile
- 99th percentile

The 95th percentile remains the single operating point for the main experiment.

## 6. Formal 2x2 Fusion

| Ensemble says | AE says | Final output |
|---|---|---|
| Attack | Normal | Known attack |
| Attack | Anomaly | Known attack |
| Benign | Normal | Benign |
| Benign | Anomaly | Suspected unseen attack |

The fusion function is deterministic and contains no training.

## 7. Withheld-Attack Protocol

Eligibility threshold:

```text
at least 50 TEST instances
```

Selection rule must be frozen before results are inspected.

Recommended deterministic rule:

```text
filter eligible subclasses
sort alphabetically
select first eligible subclass
```

The final chosen subclass and candidate counts must be recorded in `experiments/pre_registration.json`.

## 8. Evaluation

Primary metrics:

### Withheld attack
- recall
- precision
- F1

### Benign
- false-positive rate
- specificity

### Overall
- macro-F1
- balanced accuracy
- 3-way confusion matrix:
  - benign
  - known attack
  - withheld attack

### H1 rigor
For the stacking-vs-best-individual comparison:

- seeds: 42, 123, 2024
- report mean ± standard deviation

## 9. Ablation Matrix

1. Best individual classifier
2. Majority/soft-vote baseline
3. OOF stacked ensemble
4. Autoencoder alone
5. Full system: stacking + AE fusion + explanation

## 10. Stretch Goals

Only after all must-haves work:

- H4 SHAP stability
- official KDDTest+ evaluation
- repeat withheld protocol across all eligible subclasses
- significance testing
- secondary decision-tree meta-learner
