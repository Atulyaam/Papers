# Leakage and Data Isolation Rules

## 1. Golden Rule

No information derived from TEST or the protected unseen-attack subset may influence model fitting, feature selection, preprocessing fitting, threshold tuning, hyperparameter selection, or experiment selection.

## 2. Raw Data

`data/raw/` is immutable.

Never edit the source CSVs.

## 3. Protected Unseen Attack

The selected withheld subclass is isolated into a protected artifact.

Conceptually:

```text
UNSW-NB15
   |
   +---- Protected unseen attack
   |
   +---- Development data
             |
             +---- TRAIN
             +---- VALIDATION
             +---- non-withheld TEST
```

The protected subset appears only for final evaluation.

## 4. Explicitly Forbidden Uses

Protected unseen samples may not be used for:

- feature engineering decisions
- MI ranking/selection
- scaler fitting
- encoder fitting
- imputation fitting
- hyperparameter tuning
- model training
- OOF generation
- meta-learner fitting
- AE training
- AE threshold calibration
- model selection

## 5. Test Leakage

TEST is evaluation-only.

Do not inspect performance and then change:

- model hyperparameters
- withheld subclass
- MI feature count
- threshold
- fusion rule
- preprocessing policy

and call the resulting evaluation the same experiment.

## 6. Unknown Categories

Categorical encoders must safely handle categories absent from TRAIN.

For one-hot encoding, the planned policy is:

```python
OneHotEncoder(handle_unknown="ignore")
```

This must be tested explicitly.

## 7. Scaling

Scalers are fit on TRAIN only.

Validation and test are transformed with the frozen TRAIN-fitted scaler.

## 8. MI

MI is fit on TRAIN only.

The selected feature list is saved as an artifact.

## 9. Threshold

The AE threshold is computed from benign VALIDATION reconstruction errors and frozen before TEST is accessed.

## 10. Leakage Tests

Automated tests should assert:

- withheld subclass count in TRAIN == 0
- withheld subclass count in validation == 0
- training arrays contain no protected rows
- scaler/encoder/MI are fit only on TRAIN
- threshold metadata points to validation
- meta-learner receives OOF predictions
