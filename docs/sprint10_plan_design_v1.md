# Sprint 10 — Ablation Study Plan and Design v1

## 1. Experiment ID

`EXP_ABLATION_V1`

## 2. Objective

Sprint 10 evaluates the contribution of the major components of the
research IDS architecture.

The study answers:

> Does each major supervised learner, learned stacking, and the
> benign-only AE/fusion component contribute measurable value beyond
> a strong individual baseline?

Sprint 9 remains frozen and is not modified.

## 3. Fixed Experimental Controls

All ablation configurations use:

- Dataset: UNSW-NB15
- Feature selection: `EXP_MI_V1_1`
- Feature count: exactly 75
- Existing research preprocessing pipeline
- Seeds: `[42, 123, 2024]`
- Primary metric: Macro-F1
- Secondary metrics:
  - Precision
  - Recall
  - F1
  - Balanced Accuracy
  - FPR

All headline model metrics are computed on the frozen
`DEVELOPMENT_TEST` split.

`VALIDATION` is used only where the already-established research
protocol explicitly requires it.

`PROTECTED_BACKDOOR` is evaluation-only and is never used for model
selection, tuning, seed selection, or configuration selection.

## 4. Configurations

### A0 — Strongest Individual Baseline

A0 is the strongest individual supervised learner from the frozen
Sprint 9 baseline results.

The implementation MUST verify this from the frozen Sprint 9 artifacts
before training.

If and only if those artifacts confirm RF is the strongest individual
model:

`A0 = RF`

A0 is freshly trained independently for seeds:

- 42
- 123
- 2024

The exact model hyperparameters MUST be taken from the established
Sprint 9/base-model configuration. No new tuning is allowed.

### A1 — Full Learned Stacking

Base learners:

- Decision Tree
- Random Forest
- SVM
- Neural Network

Their OOF predictions form the meta-feature matrix.

Meta-learner:

- Logistic Regression

### A1b — Simple Combination Control

Uses the same four base learners as A1.

No Logistic Regression meta-learner.

The established prediction-output contract from the existing project must
first be verified.

For binary prediction:

- use the established probability representation
- calculate the simple mean of the four model probabilities
- apply the established binary decision convention

For multiclass prediction, if the existing target is verified to be
multiclass:

- average corresponding class-probability vectors
- choose the class with maximum mean probability
- ties resolve to the lowest class index

No new probability calibration method is introduced solely for A1b.

### A2 — No DT

Retain:

- RF
- SVM
- NN

Use a newly fitted Logistic Regression meta-learner.

### A3 — No RF

Retain:

- DT
- SVM
- NN

Use a newly fitted Logistic Regression meta-learner.

### A4 — No SVM

Retain:

- DT
- RF
- NN

Use a newly fitted Logistic Regression meta-learner.

### A5 — No NN

Retain:

- DT
- RF
- SVM

Use a newly fitted Logistic Regression meta-learner.

### A6 — Full Stack + Frozen AE/Fusion

Start from A1 supervised stacking.

Add the exact frozen Sprint 9:

- Autoencoder
- AE preprocessing/scaler
- AE threshold
- fusion rule

A6 MUST NOT retrain or recalibrate the AE.

## 5. Base-Model Caching

Base-model OOF predictions MUST be generated once per:

`(base_model, seed)`

for:

- DT
- RF
- SVM
- NN

and:

- 42
- 123
- 2024

Both OOF predictions and DEVELOPMENT_TEST prediction outputs should
be cached per `(base_model, seed)`.

A2–A5 reuse the corresponding A1 cache slices.

Retained models MUST NOT be retrained merely because another learner
was removed from the meta-feature matrix.

## 6. Seed Policy

Every configuration is evaluated for:

- seed 42
- seed 123
- seed 2024

Seed identity is preserved throughout:

`(configuration, seed)`

Paired comparisons use the matching seed.

## 7. Deterministic Execution

For each seed explicitly control and record:

- Python random
- NumPy
- PyTorch
- CUDA, if applicable
- estimator `random_state`
- DataLoader randomness

Existing model-specific randomness must use the established project's
seed contract.

If GPU operations cannot be made deterministic reliably, use CPU.

## 8. A1b Output Semantics

Before implementation, verify:

- binary vs multiclass target
- label mapping
- model probability/output shapes
- existing prediction contracts

If verification is ambiguous or contradictory:

STOP before training.

Do not guess or silently adapt.

## 9. A0 Seed Policy

A0 is NOT the frozen Sprint 9 checkpoint reused three times.

The frozen Sprint 9 results are used to establish which individual model
is the strongest.

That strongest model is then freshly trained for seeds:

42, 123, 2024

using the established training configuration.

This permits matched per-seed comparisons:

`A1(seed) - A0(seed)`

## 10. Evaluation Split

Headline model metrics for A0–A6 MUST come from:

`DEVELOPMENT_TEST`

The Protected Backdoor is evaluated separately.

Protected Backdoor performance MUST NOT affect configuration selection.

## 11. Metrics

For every configuration and seed report:

- Macro-F1
- Precision
- Recall
- F1
- Balanced Accuracy
- FPR
- runtime_sec

Aggregate:

- mean
- population standard deviation (`ddof=0`)
- minimum
- maximum

No statistical-significance claim is made from the three seeds alone.

## 12. Paired Comparisons

Calculate per-seed paired Macro-F1 deltas:

- A1 − A0
- A1 − A1b
- A1 − A2
- A1 − A3
- A1 − A4
- A1 − A5
- A6 − A1

Also report the mean paired delta for each comparison.

## 13. A1 vs A6 Analysis

Explicitly compare:

- Macro-F1
- Recall
- FPR
- Protected Backdoor detection rate

The purpose is to determine whether AE/fusion causes a recall/FPR tradeoff
even when Macro-F1 does not improve.

## 14. Runtime

Record lightweight runtime for:

- base-model OOF generation
- meta-learner fitting
- A1b combination
- A6 AE/fusion inference

Runtime is descriptive only and is not a selection criterion.

## 15. Protected Backdoor

The Protected Backdoor is never used for:

- training
- feature selection
- hyperparameter tuning
- A0 selection
- A1b rule selection
- seed selection
- choosing the best ablation

It is evaluated only after the configuration definitions are fixed.

## 16. Smoke Test

Before launching the complete experiment matrix:

Run:

- A0 seed 42
- A1 seed 42

Verify:

- target handling
- preprocessing
- cache construction
- meta-feature construction
- metrics
- output schema
- provenance
- config immutability
- cache reuse

If smoke test fails:

STOP.

## 17. Resumability

Every successful `(configuration, seed)` result is written immediately.

On restart:

- if result exists and passes integrity checks → SKIP
- otherwise → EXECUTE

Do not delete partial artifacts automatically.

Do not silently restart completed work.

## 18. Execution Mode

Default execution mode is sequential.

One `(configuration, seed)` execution at a time.

Concurrent writes are not allowed unless explicit locking and atomic writes
are implemented and tested.

## 19. Run Timeout

Each `(configuration, seed)` execution must have a hard timeout.

The timeout must be derived from an available established runtime baseline
with a documented safety margin.

If no appropriate baseline exists, this must be reported during
pre-verification rather than inventing a value.

Timeout causes the affected run to be marked failed and preserves all
previous artifacts.

## 20. Output Root

All Sprint 10 outputs are isolated under:

`results/ablation/EXP_ABLATION_V1/`

Expected structure:

```text
results/ablation/EXP_ABLATION_V1/
├── config.yaml
├── metadata.json
├── environment.txt
├── cache/
├── A0_RF/
├── A1_FULL_STACK/
├── A1b_SOFT_VOTE/
├── A2_NO_DT/
├── A3_NO_RF/
├── A4_NO_SVM/
├── A5_NO_NN/
├── A6_STACK_PLUS_AE/
├── summary.json
├── ablation_table.csv
├── paired_deltas.csv
├── protected_backdoor_results.json
├── runtime_report.json
├── quality_review.md
└── provenance/
```
