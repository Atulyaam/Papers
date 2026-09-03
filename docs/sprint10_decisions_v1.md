# Sprint 10 — Locked Decisions v1

## Experiment

Experiment ID:

`EXP_ABLATION_V1`

Sprint 9 remains frozen.

---

## D1 — Ablation Set

The fixed configuration set is:

- A0 — strongest verified individual baseline
- A1 — full DT/RF/SVM/NN Logistic Regression stacking
- A1b — simple soft-vote control
- A2 — no DT
- A3 — no RF
- A4 — no SVM
- A5 — no NN
- A6 — A1 + frozen AE/fusion

No additional ablation configuration may be introduced after execution
begins.

---

## D2 — A0 Identity

A0 MUST be determined from the frozen Sprint 9 individual-model results.

The implementation must verify the strongest individual model before
training.

If Sprint 9 confirms RF is strongest:

`A0 = RF`

No assumption is permitted.

---

## D3 — A0 Seed Policy

A0 is freshly trained at:

- seed 42
- seed 123
- seed 2024

using the established Sprint 9/base-model configuration.

The frozen Sprint 9 checkpoint is used only to verify A0 identity and
historical baseline information.

---

## D4 — Base-Model Cache Identity

Cache identity is exactly:

`(base_model, seed)`

Never:

`base_model`

alone.

Every base-model/seed pair has its own OOF and DEVELOPMENT_TEST outputs.

---

## D5 — Base-Model Reuse

Base-model outputs are generated once per `(model, seed)` and reused.

A2–A5 select exact slices from the corresponding A1 cache.

Retained base models MUST NOT be retrained.

---

## D6 — Meta-Learner

A1 and A2–A5 each use a separately fitted Logistic Regression
meta-learner.

A meta-learner trained on one input set cannot be reused for a different
ablation input set.

---

## D7 — A1b Control

A1b is a control for:

`learned stacking vs simple combination`

No new calibration experiment is introduced solely for A1b.

The existing project's output contract is authoritative.

If the target is multiclass, class-probability vectors are averaged.

Tie-breaking is deterministic:

lowest class index wins.

If the target/output contract cannot be verified:

STOP before training.

---

## D8 — Target Definition

The binary/multiclass target type and exact label mapping MUST be verified
from the established project before training.

No new target definition is permitted.

---

## D9 — Pre-Verification Hard Stop

Before ANY training, verify:

1. target type
2. label mapping
3. prediction semantics
4. prediction shape
5. strongest individual model
6. exact model configurations
7. OOF generation semantics
8. preprocessing semantics
9. A1b output semantics
10. Sprint 9 environment identity
11. dataset identity
12. feature identity/order
13. frozen AE source
14. frozen fusion source

Any ambiguity, contradiction, or missing required evidence causes:

**STOP AND REPORT**

No guessing.
No substitution.
No silent adaptation.

---

## D10 — Seed Determinism

For every seed, set and log:

- Python random
- NumPy
- PyTorch
- CUDA if applicable
- estimator `random_state`
- DataLoader randomness

Use deterministic framework behavior where required.

No unseeded randomness is permitted.

The existing LinearSVC-based SVM implementation remains the project's
authoritative SVM implementation.

Do not introduce RBF SVC, `probability=True`, Platt scaling, or a new SVM
model merely for this study.

---

## D11 — Environment Lock

Capture `pip freeze`.

Compare against the recorded Sprint 9 environment.

Material version mismatch:

**STOP AND REPORT**

No silent environment substitution.

---

## D12 — Dataset Hash Lock

SHA-256 MUST match Sprint 9 metadata for:

- TRAIN
- VALIDATION
- DEVELOPMENT_TEST
- PROTECTED_BACKDOOR
- selected 75-feature list

Mismatch:

**STOP BEFORE TRAINING**

---

## D13 — Feature Lock

All configurations use:

`EXP_MI_V1_1`

with exactly:

`75 features`

Exact identity and order are mandatory.

---

## D14 — Run Timeout

Every `(configuration, seed)` run requires a hard timeout derived from an
established runtime baseline and documented safety margin.

No arbitrary timeout values.

If a valid runtime baseline cannot be derived during pre-verification:

**STOP AND REPORT**

---

## D15 — Sequential Execution

Execution is sequential.

Only one `(configuration, seed)` run writes at a time.

No concurrent matrix execution unless explicit tested file locking and
atomic writes exist.

---

## D16 — Headline Split

All A0–A6 headline metrics are computed on:

`DEVELOPMENT_TEST`

Not VALIDATION.

Not PROTECTED_BACKDOOR.

---

## D17 — Protected Backdoor Isolation

Protected Backdoor is evaluation-only.

It MUST NOT determine:

- configuration selection
- model selection
- hyperparameter values
- seed selection
- A1b rule
- reported winner

---

## D18 — Exact Result Schema

`ablation_table.csv` columns are exactly:

```text
config_id
seed
macro_f1
precision
recall
f1
balanced_accuracy
fpr
runtime_sec
```


runtime_sec

``text
comparison
seed
delta_macro_f1
``

## D19 — Config Immutability

Before execution, calculate SHA-256 of:

`config.yaml`

Store:

`config_sha256_before`

After execution:

`config_sha256_after`

Require exact equality.

No result-based tuning.

## D20 — Smoke Test

Before the full matrix:

- A0 seed 42
- A1 seed 42

Both must pass end-to-end.

Failure:

**STOP AND REPORT**

## D21 — Resumability

Completed valid `(configuration, seed)` results are skipped.

Invalid or missing results are executed.

Partial artifacts are preserved.

## D22 — Runtime

Runtime is descriptive only.

It cannot select a configuration.

## D23 — Per-Seed Statistics

Report:

- seed 42
- seed 123
- seed 2024
- mean
- population std (`ddof=0`)
- minimum
- maximum

## D24 — Paired Deltas

Report matched-seed deltas:

- A1-A0
- A1-A1b
- A1-A2
- A1-A3
- A1-A4
- A1-A5
- A6-A1

No significance claim from n=3.

## D25 — Frozen AE

A6 reuses the exact frozen Sprint 9:

- AE model
- AE preprocessing/scaler
- threshold
- fusion rule

No AE retraining.

No AE recalibration.

## D26 — Cache Integrity

For every seed:

A2/A3/A4/A5 retained OOF columns must exactly equal the corresponding
A1 cached columns.

The same exact-equality requirement applies to DEVELOPMENT_TEST
prediction columns.

## D27 — Independent Determinism

A deterministic verification run must independently load the required
inputs and checkpoints and execute the evaluation path.

It may not compare an artifact with itself or merely copy previously
computed values.

## D28 — Provenance

Record:

- configuration identity
- model identity
- dataset hashes
- feature identity
- environment
- seeds
- cache identity
- checkpoint identity
- A0 source
- A1b rule
- frozen AE source
- runtime
- config hash

## D29 — Failure Handling

If any execution fails:

- stop the affected run
- preserve artifacts
- report the exact error
- report completed pairs
- report remaining pairs
- report restart point

Do not silently restart the matrix.

## D30 — No Result-Based Tuning

After results exist, do not change:

- architecture
- features
- preprocessing
- hyperparameters
- seeds
- thresholds
- fusion rule
- A1b rule
- configuration definitions

## D31 — Sprint 9 Protection

No Sprint 9 file may be modified by Sprint 10.

The `sprint9-freeze` tag remains unchanged.

## D32 — Negative Results Valid

An ablation does not need to improve Macro-F1 to be a valid outcome.

Do not reinterpret a negative result as a failed experiment.

## D33 — Artifact Isolation

All generated Sprint 10 artifacts must remain under:

`results/ablation/EXP_ABLATION_V1/`

## D34 — Final Stop Condition

Any unresolved mandatory pre-verification item causes:

**STOP AND REPORT**

No implementation by assumption.

## Final Decision State

`PLAN/DESIGN: LOCKED FOR IMPLEMENTATION`

Implementation begins only after all mandatory pre-verification
requirements pass.


## D20 — Smoke Test

Before the full matrix:

- A0 seed 42
- A1 seed 42

Both must pass end-to-end.

Failure:

**STOP AND REPORT**

## D21 — Resumability

Completed valid `(configuration, seed)` results are skipped.

Invalid or missing results are executed.

Partial artifacts are preserved.

## D22 — Runtime

Runtime is descriptive only.

It cannot select a configuration.

## D23 — Per-Seed Statistics

Report:

- seed 42
- seed 123
- seed 2024
- mean
- population std (`ddof=0`)
- minimum
- maximum

## D24 — Paired Deltas

Report matched-seed deltas:

- A1-A0
- A1-A1b
- A1-A2
- A1-A3
- A1-A4
- A1-A5
- A6-A1

No significance claim from n=3.

## D25 — Frozen AE

A6 reuses the exact frozen Sprint 9:

- AE model
- AE preprocessing/scaler
- threshold
- fusion rule

No AE retraining.

No AE recalibration.

## D26 — Cache Integrity

For every seed:

A2/A3/A4/A5 retained OOF columns must exactly equal the corresponding
A1 cached columns.

The same exact-equality requirement applies to DEVELOPMENT_TEST
prediction columns.

## D27 — Independent Determinism

A deterministic verification run must independently load the required
inputs and checkpoints and execute the evaluation path.

It may not compare an artifact with itself or merely copy previously
computed values.

## D28 — Provenance

Record:

- configuration identity
- model identity
- dataset hashes
- feature identity
- environment
- seeds
- cache identity
- checkpoint identity
- A0 source
- A1b rule
- frozen AE source
- runtime
- config hash

## D29 — Failure Handling

If any execution fails:

- stop the affected run
- preserve artifacts
- report the exact error
- report completed pairs
- report remaining pairs
- report restart point

Do not silently restart the matrix.

## D30 — No Result-Based Tuning

After results exist, do not change:

- architecture
- features
- preprocessing
- hyperparameters
- seeds
- thresholds
- fusion rule
- A1b rule
- configuration definitions

## D31 — Sprint 9 Protection

No Sprint 9 file may be modified by Sprint 10.

The `sprint9-freeze` tag remains unchanged.

## D32 — Negative Results Valid

An ablation does not need to improve Macro-F1 to be a valid outcome.

Do not reinterpret a negative result as a failed experiment.

## D33 — Artifact Isolation

All generated Sprint 10 artifacts must remain under:

`results/ablation/EXP_ABLATION_V1/`

## D34 — Final Stop Condition

Any unresolved mandatory pre-verification item causes:

**STOP AND REPORT**

No implementation by assumption.

## Final Decision State

`PLAN/DESIGN: LOCKED FOR IMPLEMENTATION`

Implementation begins only after all mandatory pre-verification
requirements pass.
