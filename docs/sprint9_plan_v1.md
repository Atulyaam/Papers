# Sprint 9 — H1/H2/H3 Evaluation — Plan v1

## Status

PLAN: DRAFTED
DESIGN: DRAFTED — PENDING DISCUSSION
DISCUSSION: NOT STARTED
FINAL DESIGN: NOT STARTED
IMPLEMENTATION: NOT STARTED
TEST: NOT STARTED
VALIDATE: NOT STARTED
FREEZE: NOT STARTED

---

## 1. Objective

Evaluate the three must-have hypotheses under the frozen UNSW-NB15
project protocol:

H1:
The stacked ensemble matches or exceeds the strongest individual
classifier under identical conditions.

H2:
The benign-trained Autoencoder flags a non-trivial proportion of the
pre-registered withheld attack subclass.

H3:
Combining the supervised stacking output with the anomaly signal
improves withheld-class detection relative to the ensemble alone at a
controlled false-positive rate.

Negative results are valid results and must be reported without
overclaiming.

---

## 2. Frozen Upstream Experiments

The following artifacts are frozen and MUST NOT be retrained,
re-tuned, modified, or replaced:

- EXP_BASE_MODELS_V1
- EXP_OOF_STACK_V1
- EXP_AE_V1
- EXP_FUSION_V1

Feature set remains exactly:

EXP_MI_V1_1
75 features

No MI rerun.
No feature change.
No feature reordering.

---

## 3. Evaluation Data

TRAIN:
162,395 rows.

VALIDATION:
11,200 rows, Normal-only.

Development TEST:
81,749 rows.

Protected Backdoor:
583 rows.

Excluded Backdoor:
FORBIDDEN.

The Protected Backdoor set is final-evaluation-only and must never be
used for training, threshold tuning, hyperparameter tuning, or model
selection.

Development TEST must not be used to tune or modify any frozen decision.

---

## 4. H1 — Stacking vs Strongest Individual Classifier

Evaluate the stacked ensemble against the strongest individual
classifier under the frozen Sprint 5 / Sprint 6 protocol.

Required seed set:

42
123
2024

Primary metrics:

- Macro-F1
- Weighted-F1
- Balanced Accuracy
- Accuracy

Report for each seed and as:

mean ± standard deviation

The exact H1 execution mechanism remains an explicit DESIGN decision:

A. reproduce/validate the existing Sprint 6 three-seed H1 evaluation
   using its frozen artifacts and recorded protocol

OR

B. perform a new Sprint 9 evaluation using existing frozen checkpoints
   without retraining

This choice MUST be resolved during DISCUSSION before implementation.

No new training is permitted unless explicitly approved by the final
Sprint 9 design.

---

## 5. H2 — Autoencoder Withheld-Attack Detection

Evaluate the frozen EXP_AE_V1 anomaly branch against the protected
withheld Backdoor subclass.

Use the already frozen Sprint 7 AE configuration.

No threshold retuning.

Primary metrics:

- detected count
- missed count
- detection rate
- n = 583

The Sprint 7 threshold provenance must be preserved.

Any detection-rate difference must be interpreted with the
583-row sample-size limitation.

---

## 6. H3 — Fusion vs Supervised-Only

Compare:

C01 = frozen supervised-only baseline

C06 = frozen EXP_FUSION_V1 configuration

C06 definition:

Supervised OR AE
threshold = mean+3σ
tau = 11.160062745213509

Primary comparison:

- withheld Backdoor detection rate
- detected count
- missed count
- false-positive behaviour under the frozen evaluation protocol

No configuration reselection.

No threshold modification.

No new fusion rule.

The Sprint 8 selected configuration remains authoritative.

---

## 7. Validation Usage

VALIDATION is Normal-only.

Therefore:

- FPR is computable.
- Attack recall is not computable.
- Macro-F1 is not computable.
- Weighted-F1 is not computable.
- Balanced Accuracy is not computable.

Sprint 9 must not reinterpret VALIDATION as an attack-evaluation set.

The Sprint 7/Sprint 8 Validation reuse limitation must remain documented.

---

## 8. Evaluation Ordering

The exact execution order must preserve the frozen evaluation discipline.

No Development TEST or Protected Backdoor result may influence
selection, hypothesis definition, threshold choice, or protocol changes.

Primary evaluation results must be generated before any optional
exploratory analysis.

No backward leakage is permitted.

---

## 9. H1/H2/H3 Decision Criteria

Before implementation, each hypothesis must have an explicit,
measurable decision criterion.

The criteria must distinguish:

- hypothesis supported
- hypothesis not supported
- inconclusive / insufficient evidence

No conclusion may be selected after observing results unless that
decision rule was pre-registered.

---

## 10. Required Artifacts

Proposed artifact root:

results/evaluation/EXP_H123_V1/

Expected artifacts:

- config.yaml
- metadata.json
- quality_review.md
- h1_results.json
- h2_results.json
- h3_results.json
- summary.json
- runtime_report.json
- provenance information

Exact artifact schema is pending DESIGN.

---

## 11. Tests

Sprint 9 must include dedicated evaluation-isolation tests covering:

- frozen model protection
- 75-feature invariance
- protected-data isolation
- Development TEST isolation
- no threshold modification
- no fusion reselection
- exact seed set
- deterministic evaluation
- metric correctness
- provenance completeness

Exact test names and counts are pending DESIGN.

---

## 12. Non-Goals

- retraining frozen models
- changing the 75-feature set
- changing Sprint 8 C06
- changing the protected Backdoor selection
- threshold tuning after Sprint 8
- new fusion rules
- learned fusion
- SHAP
- ablation
- deployment

---

## 13. Sprint 9 Lifecycle

PLAN
↓
DESIGN
↓
DISCUSSION
↓
FINAL DESIGN
↓
IMPLEMENT
↓
TEST
↓
VALIDATE
↓
FREEZE