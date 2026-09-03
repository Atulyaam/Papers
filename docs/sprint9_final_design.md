# Sprint 9 — H1/H2/H3 Evaluation — Final Design

## Status

```
PLAN:           READY
DESIGN:         READY
DISCUSSION:     COMPLETE — DD-1 THROUGH DD-8 APPROVED
FINAL DESIGN:   APPROVED — 2026-09-03 (see §21)
IMPLEMENTATION: COMPLETE
TEST:           COMPLETE — 16/16 PASS
VALIDATE:       COMPLETE
FREEZE:         NOT STARTED
```

**Authority**: All design decisions DD-1 through DD-8 are human-approved
and LOCKED. This document is the authoritative implementation specification
for Sprint 9. No decision herein may be altered without reopening Discussion.

---

## 1. Purpose and Scope

Sprint 9 is the formal H1/H2/H3 evaluation stage of the IDS-UNSW-NB15
project. It evaluates three must-have pre-registered hypotheses using only
frozen upstream artifacts and held-out evaluation data. No model training,
threshold tuning, or configuration selection occurs in Sprint 9.

**Sprint 9 is NOT a fusion sprint.** Fusion was Sprint 8 and is frozen.

**Hypotheses**:

**H1**: The stacked ensemble matches or exceeds the strongest individual
classifier under identical held-out evaluation conditions.

**H2**: The benign-trained Autoencoder flags a non-trivial proportion of
the pre-registered withheld Backdoor subclass.

**H3**: Combining the supervised stacking output with the anomaly signal
improves withheld-class detection relative to the ensemble alone at a
controlled false-positive rate.

Negative results are valid and must be reported without overclaiming.

---

## 2. Frozen Upstream Artifacts

The following experiments are FROZEN and must NOT be retrained, re-tuned,
modified, or replaced during Sprint 9 or any downstream sprint.

| Experiment | Role | Frozen checkpoints |
|------------|------|--------------------|
| EXP_MI_V1_1 | 75-feature selection | Feature list artifact |
| EXP_BASE_MODELS_V1 | DT/RF/SVM/NN checkpoints | `checkpoints/EXP_BASE_MODELS_V1/` |
| EXP_OOF_STACK_V1 | Per-seed meta-learners | `checkpoints/EXP_OOF_STACK_V1/seed_{42,123,2024}/` |
| EXP_AE_V1 | AE weights + scaler + threshold calibration | `checkpoints/EXP_AE_V1/` |
| EXP_FUSION_V1 | Frozen H3 evidence; C06 selected | `results/fusion/EXP_FUSION_V1/` |

**Verified checkpoint inventory** (from DD-8 infrastructure audit):

```
results/checkpoints/EXP_BASE_MODELS_V1/dt/dt_final.joblib        (single)
results/checkpoints/EXP_BASE_MODELS_V1/rf/rf_final.joblib         (single, 224 MB)
results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_final.joblib       (single)
results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_scaler.joblib      (single)
results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_final.pt             (single)
results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_scaler.joblib        (single)
results/checkpoints/EXP_OOF_STACK_V1/seed_42/meta_learner.joblib  (per-seed)
results/checkpoints/EXP_OOF_STACK_V1/seed_123/meta_learner.joblib (per-seed)
results/checkpoints/EXP_OOF_STACK_V1/seed_2024/meta_learner.joblib(per-seed)
results/checkpoints/EXP_AE_V1/ae_final.pt
results/checkpoints/EXP_AE_V1/ae_scaler.joblib
results/autoencoder/EXP_AE_V1/threshold/threshold_calibration.json
```

> [!IMPORTANT]
> No per-seed base-model (RF/DT/SVM/NN) checkpoints exist. The seed
> variation in H1 stacking inference comes exclusively from the three
> per-seed meta-learner checkpoints. No retraining is required or permitted.

---

## 3. Locked Decisions (DD-1 through DD-8)

These decisions are HUMAN-APPROVED and must be carried unchanged into
implementation. They must not be reinterpreted, weakened, or silently altered.

| DD | Decision | Status |
|----|----------|--------|
| DD-1 | H1 = Option B: inference-only on Development TEST with frozen checkpoints | LOCKED |
| DD-2 | H1 baseline = single frozen RF checkpoint on Dev TEST; Sprint 5 number = supplementary | LOCKED |
| DD-3 | H2 threshold = mean+3sigma, tau=11.160062745213509 | LOCKED |
| DD-4 | H2 criterion: ≥2 detected=SUPPORTED; 1=INCONCLUSIVE; 0=NOT SUPPORTED | LOCKED |
| DD-5 | H3 = re-presentation of frozen Sprint 8 evidence; mandatory wording required | LOCKED |
| DD-6 | H3 FPR_delta ≤ 2 pp, with required disclosures L6 and L7 | LOCKED |
| DD-7 | Exact H1/H2/H3 verdict functions as specified in §10 | LOCKED |
| DD-8 | Checkpoint audit confirmed; no retraining required or permitted | LOCKED |

---

## 4. Exact H1 Protocol

### 4.1 Execution Mode (DD-1: LOCKED)

H1 uses inference-only evaluation on the held-out Development TEST set
using frozen checkpoints. No model training occurs.

**Characterization**: "matched evaluation data and preprocessing, but
unmatched replication depth."
- RF: one deterministic inference run (single checkpoint)
- Stacking: three per-seed inference runs (seeds 42, 123, 2024)

**Must NOT be described as** a fully matched 3-seed paired comparison.

### 4.2 Baseline (DD-2: LOCKED)

The H1 individual-classifier baseline is the Random Forest from
`EXP_BASE_MODELS_V1/rf/rf_final.joblib`.

- RF produces one deterministic prediction set on Dev TEST.
- The frozen Sprint 5 RF Macro-F1 = 0.9508532447968256 is retained as
  supplementary historical context only, with this verbatim limitation
  (h1_summary.json; must not be paraphrased):

> "Two reporting units are used: (a) three-seed H1 stacking mean±std;
> (b) frozen Sprint 5 single-CV base-model reference. These are not
> statistically matched quantities."

### 4.3 Stacking Inference

For each seed in {42, 123, 2024}:
1. Run all four frozen base models (DT, RF, SVM, NN) on Dev TEST using
   their frozen checkpoints and scalers → four per-base-model prediction
   vectors.
2. Assemble the four-column meta-feature matrix.
3. Run the per-seed frozen meta-learner on the meta-feature matrix →
   per-seed stacking prediction.
4. Compute per-seed Macro-F1, Weighted-F1, Balanced Accuracy, Accuracy
   against Dev TEST labels.

### 4.4 H1 Required Result Fields

The `h1_results.json` artifact MUST contain all of the following fields:

```json
{
  "stacking_macro_f1_seed_42":   <float>,
  "stacking_macro_f1_seed_123":  <float>,
  "stacking_macro_f1_seed_2024": <float>,
  "stacking_mean_macro_f1":      <float>,
  "stacking_std_macro_f1":       <float>,
  "rf_dev_test_macro_f1":        <float>,
  "diff":                        <float>,
  "h1_verdict":                  "SUPPORTED" | "NOT_SUPPORTED" | "INCONCLUSIVE",
  "epsilon":                     0.005,
  "stacking_weighted_f1_seed_42":   <float>,
  "stacking_weighted_f1_seed_123":  <float>,
  "stacking_weighted_f1_seed_2024": <float>,
  "stacking_mean_weighted_f1":   <float>,
  "stacking_balanced_acc_seed_42":  <float>,
  "stacking_balanced_acc_seed_123": <float>,
  "stacking_balanced_acc_seed_2024":<float>,
  "stacking_mean_balanced_acc":  <float>,
  "n_dev_test":                  81749,
  "seeds":                       [42, 123, 2024],
  "sprint6_oof_reference": {
    "mean_macro_f1": 0.9472415941099953,
    "std_macro_f1":  0.00026253378581352256,
    "label": "Sprint 6 OOF in-sample reference — NOT held-out Dev TEST"
  },
  "sprint5_rf_reference": {
    "macro_f1": 0.9508532447968256,
    "label": "Frozen Sprint 5 single-CV reference; not a matched 3-seed H1 baseline."
  },
  "limitations": [<L1>, <L2>]
}
```

> [!IMPORTANT]
> Sprint 9 Dev TEST seed variability (stacking_std_macro_f1 computed on
> Dev TEST) must NOT be used to modify or justify changing epsilon.
> Epsilon = 0.005 is pre-registered and locked.
>
> Sprint 6 OOF seed variability (std=0.000263) is in-sample historical
> context only and must be clearly labeled as such.

---

## 5. Exact H2 Protocol

### 5.1 AE-Only Branch (DD-3, DD-4: LOCKED)

H2 evaluates the Autoencoder anomaly branch **independently**. The fused
C06 OR-prediction is NOT used for H2.

**Threshold** (DD-3):
```
threshold_id = mean+3sigma
tau          = 11.160062745213509
source       = EXP_AE_V1/threshold/threshold_calibration.json
```

**AE anomaly flag rule**:
```python
ae_flag(r) = 1  if  RE(r) > 11.160062745213509   # strict greater-than
           = 0  otherwise
```

**AE scaler**: `results/checkpoints/EXP_AE_V1/ae_scaler.joblib`
(same frozen scaler used in Sprint 8; no new scaler is needed).

> [!CAUTION]
> The Sprint 8 EXP_FUSION_V1 result (582/583 detected by C06) is a
> **fused OR-prediction** and does NOT satisfy the H2 requirement.
> The supervised branch dominates C06. A new AE-only inference pass
> on the 583 Protected Backdoor rows is required.

### 5.2 AE Normal VALIDATION FPR (Provenance/Consistency Check)

Before accessing the Protected Backdoor, recompute the AE Normal VALIDATION
FPR from the frozen VALIDATION set:

```
AE_VAL_FPR_recomputed = count(ae_flag == 1 on 11,200 Normal VALIDATION rows) / 11200
```

From frozen calibration: expected = 7/11200 = 0.000625 (0.0625%).

This value must be recorded in `h2_results.json` **before** any Protected
Backdoor rows are accessed.

> [!IMPORTANT]
> **The Sprint 9 AE_VAL_FPR recomputation is a provenance/consistency check
> only and MUST NOT redefine the approved 2-row H2 criterion.**
>
> The H2 supported detection threshold is locked at **2 rows** by DD-4.
> It was derived from the frozen calibration record
> (7/11200 = 0.000625) and must not be recomputed dynamically.
>
> If the recomputed AE_VAL_FPR does NOT match 7/11200 = 0.000625 within
> the pre-registered numeric tolerance (T-AE-VAL-FPR-CONSISTENCY), Sprint 9
> must **HALT** and escalate. Do not recalculate or alter supported_threshold.

### 5.3 H2 Practical Criterion (DD-4: LOCKED)

> This is a **practical pre-registered criterion**, NOT a formal
> statistical significance test.

**The H2 supported detection threshold is a LITERAL LOCKED VALUE: 2 rows.**

Derivation (documentation only — must not be recomputed at runtime):

```
multiplier          = 3                     (DD-4, LOCKED)
frozen_ae_val_fpr   = 7 / 11200 = 0.000625  (from threshold_calibration.json)
derivation          = ceil(3 × 0.000625 × 583) = ceil(1.093) = 2
supported_threshold = 2                     (LOCKED)
```

Implementation must use the literal value `2`, not a runtime formula.
The Sprint 9 AE_VAL_FPR recomputation (§5.2) is a consistency check
only and does not feed into this threshold.

Required wording in `h2_results.json` and `quality_review.md`:

> "The H2 supported detection threshold is locked at 2 rows by DD-4.
> The Sprint 9 AE_VAL_FPR recomputation is a provenance/consistency
> check only and MUST NOT redefine the approved 2-row criterion."

Three-way verdict rule:
```
SUPPORTED:     ae_detected_count >= 2
INCONCLUSIVE:  ae_detected_count == 1
NOT SUPPORTED: ae_detected_count == 0
```

### 5.4 H2 Required Result Fields

```json
{
  "ae_val_fpr_recomputed":  <float>,
  "ae_val_fpr_frozen":      0.000625,
  "ae_val_flagged":         <int>,
  "ae_val_n":               11200,
  "tau":                    11.160062745213509,
  "threshold_id":           "mean+3sigma",
  "multiplier":             3,
  "supported_threshold":    2,
  "supported_threshold_derivation": "ceil(3 x 0.000625 x 583) = 2 (locked; not recomputed at runtime)",
  "supported_threshold_locked": true,
  "ae_detected_count":      <int>,
  "ae_missed_count":        <int>,
  "ae_detection_rate":      <float>,
  "n_prot":                 583,
  "pp_per_row":             0.1716,
  "h2_verdict":             "SUPPORTED" | "NOT_SUPPORTED" | "INCONCLUSIVE",
  "criterion_type":         "practical_preregistered_not_significance_test",
  "threshold_criterion_wording": "The H2 supported detection threshold is locked at 2 rows by DD-4. The Sprint 9 AE_VAL_FPR recomputation is a provenance/consistency check only and MUST NOT redefine the approved 2-row criterion.",
  "limitations":            [<L3>, <L4>, <L5>]
}
```

---

## 6. Exact H3 Protocol

### 6.1 Scope (DD-5: LOCKED)

H3 is a formalization and re-presentation of frozen Sprint 8 evidence
under pre-registered H3 decision criteria. No new configuration evaluation
is performed. Sprint 8 decisions are not reopened.

**Mandatory verbatim wording** in all H3 reporting and in `h3_results.json`:

> "Sprint 9 H3 formalizes the Sprint 8 H-FUSION/H-PROT-BACKDOOR findings
> under explicit pre-registered criteria and does not reopen those frozen
> decisions."

### 6.2 Evidentiary Sources

```
results/fusion/EXP_FUSION_V1/protected_backdoor/metrics.json
results/fusion/EXP_FUSION_V1/development_test/metrics.json
results/fusion/EXP_FUSION_V1/quality_review.md
```

These are immutable read-only inputs. The Sprint 8 Dev TEST FPR values
are inherited; this is exempt from T-DEV-TEST-ISOLATION because the values
predate Sprint 9 and are immutable (see §13).

### 6.3 Frozen Sprint 8 Evidence

| Metric | C01 | C06 |
|--------|-----|-----|
| Protected Backdoor detected | 582 | 582 |
| Missed | 1 | 1 |
| Dev TEST FPR | 19.1892% | 19.2243% |
| Dev TEST Macro-F1 | 0.892609 | 0.892440 |
| FPR_delta | — | +0.0351 pp |

### 6.4 FPR Control (DD-6: LOCKED — WITH REQUIRED DISCLOSURES)

```
FPR_delta = FPR_C06_dev_test - FPR_C01_dev_test
FPR_cap   = 0.02 (2 percentage points)
```

**Required disclosure L6** (must appear verbatim in h3_results.json and
quality_review.md):

> "The underlying Sprint 8 C01/C06 Development TEST FPR difference
> (+0.0351 pp) was already known when the 2-percentage-point H3 tolerance
> was proposed. Therefore this tolerance is a documented operational
> guardrail rather than a blind pre-registration made before any relevant
> FPR evidence existed."

**Required disclosure L7** (must appear verbatim):

> "For the frozen Sprint 8 result, the H3 FPR cap is not decision-
> determinative because C06 and C01 have identical Protected Backdoor
> detection counts (582/583). The primary H3 condition therefore already
> fails before the FPR cap can affect the verdict."

### 6.5 H3 Required Result Fields

```json
{
  "c01_detected":        582,
  "c01_missed":          1,
  "c01_dev_test_fpr":    0.191892,
  "c06_detected":        582,
  "c06_missed":          1,
  "c06_dev_test_fpr":    0.192243,
  "fpr_delta":           0.000351,
  "fpr_cap":             0.02,
  "n_prot":              583,
  "pp_per_row":          0.1716,
  "h3_verdict":          "NOT_SUPPORTED",
  "h3_verdict_reason":   "C06 detected_count == C01 detected_count; primary condition fails.",
  "fpr_cap_determinative": false,
  "evidence_source":     "EXP_FUSION_V1 frozen artifacts",
  "mandatory_wording":   "Sprint 9 H3 formalizes the Sprint 8 H-FUSION/H-PROT-BACKDOOR findings under explicit pre-registered criteria and does not reopen those frozen decisions.",
  "disclosures":         ["L6", "L7"],
  "sprint8_verdicts": {
    "H-FUSION":          "FALSE",
    "H-PROT-BACKDOOR":   "FALSE"
  }
}
```

---

## 7. Data-Access Boundaries

| Split | Rows | Composition | Sprint 9 permitted uses |
|-------|------|-------------|------------------------|
| TRAIN | 162,395 | Mixed | Read frozen preprocessing artifacts; no fitting |
| VALIDATION | 11,200 | 100% Normal | AE Normal FPR pre-computation (H2 step 5.2) only |
| Development TEST | 81,749 | Mixed | H1 inference; not accessed before criteria locked |
| Protected Backdoor | 583 | 100% Backdoor | H2 AE-only inference; H3 evidence read from frozen Sprint 8 |
| Excluded Backdoor | — | — | **FORBIDDEN** |

**Strict rules**:
- VALIDATION may not be used for attack metrics (it is Normal-only).
- Development TEST may not be accessed for H1 until all H1/H2/H3 criteria
  are registered in `config.yaml`.
- Protected Backdoor may not influence criteria, epsilon, tau, or any
  methodology decision.
- No result from any evaluation split may flow backward to modify criteria.

---

## 8. Evaluation Ordering

The following sequence is FIXED and must be strictly observed:

```
STEP 1  Load frozen artifacts; verify SHA-256 hashes.
STEP 2  Confirm H1/H2/H3 criteria locked in config.yaml.
STEP 3  Run T-CRITERION-PREREGISTERED.
STEP 4  Compute AE Normal VALIDATION FPR (H2 baseline; VALIDATION only).
STEP 5  Record AE_VAL_FPR in h2_results.json before any Protected Backdoor access.
STEP 6  H1: run RF and stacking inference on Development TEST.
STEP 7  H1: compute per-seed and aggregate metrics; apply H1 verdict function.
STEP 8  H2: run AE-only inference on Protected Backdoor.
STEP 9  H2: apply H2 verdict function.
STEP 10 H3: read frozen Sprint 8 evidence from EXP_FUSION_V1 artifacts.
STEP 11 H3: apply H3 verdict function.
STEP 12 Write h1_results.json, h2_results.json, h3_results.json, summary.json.
STEP 13 Write quality_review.md with all mandatory limitations and verdicts.
STEP 14 Run all required tests.
STEP 15 Optional exploratory analysis (INFORMATIONAL ONLY — labeled clearly).
STEP 16 Write runtime_report.json, metadata.json.
```

> [!CAUTION]
> No result from steps 6–11 may influence any criterion registered in
> step 2. Exploratory analysis in step 15 must not alter verdicts from
> steps 7/9/11.

---

## 9. Exact Metric Definitions

### H1 primary metrics

```
Macro-F1        = unweighted mean of per-class F1 scores
Weighted-F1     = class-support-weighted mean of per-class F1 scores
Balanced Acc    = unweighted mean of per-class recall
Accuracy        = correct_predictions / total_predictions
```

### H2/H3 detection metrics

```
detected_count  = sum(prediction == attack_label) on target rows
missed_count    = n - detected_count
detection_rate  = detected_count / n
pp_per_row      = 0.1716    (= 100 / 583 percentage points per row)
```

### FPR metrics (H3 control)

```
FPR      = FP / (FP + TN)   on Normal-label rows only
FPR_delta = FPR_C06 - FPR_C01
```

---

## 10. Exact Hypothesis Verdict Functions (DD-7: LOCKED)

### H1 Verdict Function

```python
def h1_verdict(stacking_mean_macro_f1, rf_dev_test_macro_f1, epsilon=0.005):
    diff = stacking_mean_macro_f1 - rf_dev_test_macro_f1
    if diff > epsilon:
        return "SUPPORTED", diff
    elif diff < -epsilon:
        return "NOT_SUPPORTED", diff
    else:  # abs(diff) <= epsilon
        return "INCONCLUSIVE", diff
```

`epsilon = 0.005` is LOCKED (DD-7). Must not be modified after evaluation
begins.

### H2 Verdict Function

```python
def h2_verdict(ae_detected_count):
    # Practical pre-registered criterion; NOT a significance test.
    if ae_detected_count >= 2:
        return "SUPPORTED"
    elif ae_detected_count == 1:
        return "INCONCLUSIVE"
    else:  # 0
        return "NOT_SUPPORTED"
```

### H3 Verdict Function

```python
def h3_verdict(c06_detected, c01_detected, fpr_delta, fpr_cap=0.02):
    if c06_detected > c01_detected and fpr_delta <= fpr_cap:
        return "SUPPORTED"
    elif c06_detected <= c01_detected:
        return "NOT_SUPPORTED"
    else:  # c06_detected > c01_detected AND fpr_delta > fpr_cap
        return "INCONCLUSIVE"
```

`fpr_cap = 0.02` is LOCKED (DD-6). Under frozen Sprint 8 evidence,
`c06_detected = c01_detected = 582` → verdict = `NOT_SUPPORTED`.

No McNemar or other statistical test may be introduced to alter a verdict
derived from these functions.

---

## 11. Reproducibility and Provenance

Each Sprint 9 evaluation run must record:

```yaml
# config.yaml — locked before evaluation
experiment_id: EXP_H123_V1
upstream_experiments:
  - EXP_MI_V1_1
  - EXP_BASE_MODELS_V1
  - EXP_OOF_STACK_V1
  - EXP_AE_V1
  - EXP_FUSION_V1
h1_seeds: [42, 123, 2024]
h1_epsilon: 0.005
h2_tau: 11.160062745213509
h2_threshold_id: mean+3sigma
h2_multiplier: 3
h3_fpr_cap: 0.02
n_features: 75
feature_set: EXP_MI_V1_1
n_dev_test: 81749
n_prot: 583
n_validation: 11200
```

```yaml
# metadata.json — recorded at evaluation time
git_commit_hash: <recorded>
git_tags: <all sprint freeze tags>
dataset_sha256:
  # Authoritative source: data/splits/train_val_split_metadata.json (output_hashes)
  #                       data/splits/split_metadata.json (protected_sha256, development_test_sha256)
  # The bec7dd... value is the raw source CSV hash (source_training_sha256), NOT the split hash.
  train:               "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c"
  validation:          "13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180"
  development_test:    "04725e85732ab2fc6d9eaaa6105418b22b083b5c651067e7b0785464f414e508"
  protected_backdoor:  "6ffd23479b575e438ad90678268f40f674a663c2b9507aaf65089623397a9d91"
checkpoint_sha256:
  # Hashes computed 2026-09-02 via Python hashlib.sha256 on exact frozen file bytes.
  # No checkpoint file was altered, normalized, or re-serialized before hashing.
  EXP_BASE_MODELS_V1:
    dt_final_joblib:       "748261c8106e5b12a93decb4de7df435e09dd587b03294dba3837e20c8a2e4a3"
    rf_final_joblib:       "f1f873ef4bd7f09c03885ffbbc4c9ec51306dc2aecc0f48e4584fddd7a97a68f"
    svm_final_joblib:      "f325d57525dda5bd92cc20c5393a38fa1b9ca055001b0c24fc9402bdbece990c"
    svm_scaler_joblib:     "a85eeeb74d34bed8cead09cc7506c4bbac6522bb1df0467d6904178996bdaa85"
    nn_final_pt:           "7f3dcdfa59cbd084fcd952645db3b14fa67554769500551f06737d42e5e058ae"
    nn_scaler_joblib:      "a85eeeb74d34bed8cead09cc7506c4bbac6522bb1df0467d6904178996bdaa85"
  EXP_OOF_STACK_V1:
    seed_42_meta_learner_joblib:   "e5b776680a99ffee3271624445f7f52593f8f94037d20ba56e9f4b54a848ef19"
    seed_123_meta_learner_joblib:  "f6517b59fac54864b82db07f3da35139f21f400e2a7664ef56ee29b09fcd6672"
    seed_2024_meta_learner_joblib: "f6139a79f3e7c96bb2c6610f22907184df117a06dd110ea74d6eb1897aeada74"
  EXP_AE_V1:
    ae_final_pt:               "4ab66af8d4a6e61212ef5d78360f30a8caa68aa85dac3d54042218e010f9a1d6"
    ae_scaler_joblib:          "c0128d42ed9ef5be695f261be75155e7de4ddf8e51b926e3ce516c4a88ad8211"
    threshold_calibration_json: "29bd47b8a0dd886383d312e1364320c9ada62d78989c4c5f847a96f8c1882971"
python_version: <recorded>
library_versions:
  sklearn: <recorded>
  torch: <recorded>
  numpy: <recorded>
  pandas: <recorded>
evaluation_timestamp: <recorded>
```

> These four dataset SHA-256 values are copied from the authoritative
> frozen project provenance and are not re-derived or altered by Sprint 9.
> (Sources: `data/splits/train_val_split_metadata.json` → `output_hashes`;
> `data/splits/split_metadata.json` → `development_test_sha256` and
> `protected_sha256`.)

> [!IMPORTANT]
> **Checkpoint SHA-256 hashes are a required implementation prerequisite.**
> Populate all `checkpoint_sha256` fields by computing hashes from the
> frozen checkpoint files before Sprint 9 evaluation begins. Do NOT
> begin implementation until all hash values are recorded in
> `metadata.json`. Do NOT fabricate any hash value.


---

## 12. Required Artifacts and Schemas

Artifact root: `results/evaluation/EXP_H123_V1/`

| File | Contents | Must contain |
|------|----------|-------------|
| `config.yaml` | All evaluation parameters | All fields from §11; locked before STEP 6 |
| `metadata.json` | Provenance, hashes, environment | All fields from §11 |
| `h1_results.json` | H1 per-seed and aggregate results | All fields from §4.4 |
| `h2_results.json` | H2 AE-only detection | All fields from §5.4 |
| `h3_results.json` | H3 C01 vs C06 comparison | All fields from §6.5 |
| `summary.json` | H1/H2/H3 verdicts + key metrics | h1_verdict, h2_verdict, h3_verdict, limitations list |
| `runtime_report.json` | Wall-clock times, environment | Per-step timings, library versions |
| `quality_review.md` | Full narrative review | All mandatory limitations (L1–L7), all verdicts, all disclosures |
| `provenance/` | SHA-256 hash confirmations | Checkpoint load confirmations |

---

## 13. Required Tests

All tests must pass before results are reported and before freeze.

> [!NOTE]
> **T-DEV-TEST-ISOLATION scope**: For NEW Sprint 9 computations (H1
> inference), Development TEST must be inaccessible until H1/H2/H3
> criteria are registered in `config.yaml`. Frozen Sprint 8 Dev TEST
> values used for H3 re-presentation are **exempt** because they predate
> Sprint 9 and are immutable read-only inputs.

| Test ID | Assertion | Scope |
|---------|-----------|-------|
| T-CRITERION-PREREGISTERED | `config.yaml` exists and contains all H1/H2/H3 criterion parameters before any evaluation data is accessed | All |
| T-NO-RESULT-BACKWARD | No field in `config.yaml` was modified after any evaluation step | All |
| T-FROZEN-UPSTREAM | SHA-256 hash of each loaded checkpoint matches the expected value recorded in `metadata.json` under `checkpoint_sha256`. Computed at STEP 1 using `hashlib.sha256`. Any mismatch HALTS Sprint 9 immediately. | All |
| T-NO-RETRAIN | No model training code executes during Sprint 9; all checkpoints are load-only | All |
| T-75-FEATURES | Feature matrix passed to all models has exactly 75 columns; column order matches EXP_MI_V1_1 | All |
| T-SEED-SET | Seeds used for stacking inference = {42, 123, 2024} exactly; no other seeds | H1 |
| T-TAU-PROVENANCE | `h2_results.json` tau field == 11.160062745213509; matches `threshold_calibration.json` | H2 |
| T-H2-AE-ONLY | H2 ae_detected_count computed from ae_flag alone; not from C06 OR-fusion prediction | H2 |
| T-H3-NO-RESELECT | H3 C06 parameters in `h3_results.json` match EXP_FUSION_V1 frozen record exactly | H3 |
| T-PROT-ISOLATION | Count of Protected Backdoor rows in TRAIN == 0; count in VALIDATION == 0 | All |
| T-DEV-TEST-ISOLATION | `config.yaml` timestamp precedes first Development TEST access timestamp | H1 |
| T-HASH-CONSISTENCY | All four dataset SHA-256 hashes in `metadata.json` match Sprint 8 provenance records | All |
| T-DETERMINISTIC | Re-running evaluation with identical config and checkpoints produces: (a) exact equality on all integer/count/categorical/verdict fields; (b) `np.allclose(rtol=1e-8, atol=1e-8)` on all floating-point fields; (c) exact equality on all derived binary prediction vectors. Deterministic evaluation requires identical derived predictions, counts, categorical outputs, and verdicts. Floating-point quantities are compared using the pre-registered numerical tolerance rather than requiring universal bit-identical floating-point serialization. Environment requirements: fixed seeds, deterministic PyTorch mode (`torch.use_deterministic_algorithms(True)` where available), controlled thread settings where practical. | All |
| T-PROVENANCE-COMPLETE | `metadata.json` contains all required fields from §11 with no null values | All |
| T-AE-VAL-FPR-CONSISTENCY | **Primary assertion (exact count)**: `ae_val_flagged == 7` on the 11,200-row Normal VALIDATION set. Consequently `ae_val_fpr_recomputed == 7/11200 == 0.000625`. Any mismatch (flagged count ≠ 7) HALTS Sprint 9. Do NOT recalculate or alter supported_threshold. Full wording: "ae_val_flagged MUST equal the frozen calibration count 7 on the 11,200-row Normal VALIDATION set. Consequently ae_val_fpr_recomputed must equal 7/11200 = 0.000625. Any mismatch HALTS Sprint 9." | H2 |
| T-RF-PREDICTION-REUSE | The RF prediction vector used as the H1 RF baseline and the RF prediction vector supplied to the stacking meta-feature matrix are the same computed array. Implementation must compute frozen RF predictions exactly once. If separate inference calls are used, their prediction arrays must satisfy exact equality. | H1 |

---

## 14. Leakage / Isolation Controls

| Control | Rule |
|---------|------|
| No threshold retuning | tau loaded from frozen `threshold_calibration.json`; not recomputed |
| No model retraining | All Sprint 9 inference uses frozen `.joblib` / `.pt` checkpoints |
| No post-hoc criterion change | epsilon, tau, multiplier, fpr_cap, supported_threshold are written to `config.yaml` before STEP 6 |
| No Dev TEST influence on criteria | Dev TEST not accessed before `config.yaml` is written (T-DEV-TEST-ISOLATION) |
| No Protected Backdoor influence on criteria | Backdoor not accessed before `config.yaml` and AE_VAL_FPR consistency check are complete |
| H2 threshold locked at 2 rows | supported_threshold = 2 is a literal locked value; AE_VAL_FPR recomputation is consistency check only (T-AE-VAL-FPR-CONSISTENCY) |
| Checkpoint cryptographic integrity | All checkpoint SHA-256 hashes verified at STEP 1 via T-FROZEN-UPSTREAM before any inference. Mismatch halts the sprint. |
| No backward leakage from exploratory to primary | Exploratory analysis (step 15) has no backward write path to verdict fields |
| VALIDATION attack metrics forbidden | VALIDATION is Normal-only; no F1/recall/balanced-accuracy computed on VALIDATION |
| Scaler discipline | All preprocessing uses TRAIN-fitted scalers from frozen checkpoints only |
| RF predictions computed once | RF inference run once; same prediction array used for both H1 RF baseline and stacking meta-feature construction (T-RF-PREDICTION-REUSE) |

---

## 15. Failure and Inconclusive Result Handling

- **NOT SUPPORTED** for any of H1/H2/H3 is a valid, expected result.
  Report it honestly without overclaiming. Do not introduce additional
  evaluation to attempt to reverse a NOT SUPPORTED verdict.

- **INCONCLUSIVE** must be reported with the explicit pre-registered reason
  (see §10 verdict functions). Inconclusive is not a license to re-evaluate.

- If a runtime error prevents evaluation of any hypothesis, record the
  error in `runtime_report.json` and report the hypothesis as NOT EVALUATED.
  Do not impute or infer a verdict.

- If infrastructure inspection during STEP 1 reveals a hash mismatch or
  missing checkpoint, halt and escalate before proceeding.

- Under no circumstances may Sprint 9 epsilon, tau, multiplier, or FPR_cap
  be adjusted because a preliminary result appears inconvenient.

### HALT Protocol (blocking integrity failures)

Any of the following conditions triggers an immediate HALT:
- T-FROZEN-UPSTREAM failure (checkpoint hash mismatch)
- T-AE-VAL-FPR-CONSISTENCY failure (ae_val_flagged ≠ 7)
- Any missing required checkpoint at STEP 1

On HALT:
1. Stop evaluation immediately at the current step.
2. Do NOT proceed to Protected Backdoor inference.
3. Do NOT write a partial summary.json or verdicts.
4. Write `results/evaluation/EXP_H123_V1/halt_report.json` with
   the following schema:

```json
{
  "status": "HALTED",
  "halt_reason": "<exact reason — e.g. T-AE-VAL-FPR-CONSISTENCY: ae_val_flagged=8 != 7>",
  "triggered_test": "<test ID — e.g. T-AE-VAL-FPR-CONSISTENCY>",
  "timestamp": "<ISO-8601 timestamp>",
  "last_completed_step": "<step number from §8 evaluation ordering>",
  "next_step_blocked": "<step number that was blocked>",
  "evaluation_data_access": {
    "validation_accessed":         true,
    "development_test_accessed":   false,
    "protected_backdoor_accessed": false
  },
  "artifacts_written_before_halt": [],
  "resolution_required": "Human project owner / methodology reviewer must review and resolve the integrity issue before a new Sprint 9 run may proceed.",
  "verdicts_final": false
}
```

5. `halt_report.json` is the ONLY output artifact written after a HALT.
   All upstream frozen artifacts remain untouched.
6. Escalate to: **Human project owner / methodology reviewer**.
7. A new Sprint 9 run may only proceed after the human resolves the
   blocking condition and explicitly authorizes restart.

`halt_report.json` is required ONLY if a blocking HALT condition occurs.
Do NOT write it for normal successful or inconclusive evaluation runs.

---

## 16. Reporting Requirements

### H1 reporting

- Report per-seed Macro-F1 on Dev TEST (seeds 42, 123, 2024).
- Report stacking mean ± std across the three seeds.
  **`stacking_std_macro_f1` is reported for transparency and robustness
  context but does not enter the locked H1 verdict function.** It must
  not be used to modify epsilon after evaluation begins.
- Report RF Dev TEST Macro-F1 (single run).
- Report diff and verdict.
- Clearly distinguish Sprint 6 OOF variability (in-sample, historical)
  from Sprint 9 Dev TEST variability (held-out, primary).
- Cite limitation L1 and L2 verbatim.

### H2 reporting

- Report AE Normal VALIDATION FPR.
- Report ae_detected_count, ae_missed_count, ae_detection_rate.
- Report 3× threshold and count-based verdict.
- State explicitly: "practical pre-registered criterion; not a formal
  statistical significance test."
- Cite limitations L3, L4, L5 verbatim.

### H3 reporting

- Report C01 and C06 Backdoor detection counts and FPR values.
- Report FPR_delta.
- Apply H3 verdict function.
- Include mandatory verbatim wording (DD-5).
- Include disclosures L6 and L7 verbatim.
- State that Sprint 8 H-FUSION=FALSE, H-PROT-BACKDOOR=FALSE are unchanged.

### Multiplicity

H1–H3 are evaluated as independently pre-registered engineering checks.
No multiple-comparisons adjustment is applied across the three hypotheses.
This must be stated in the reporting/limitations section of `quality_review.md`.

---

## 17. Mandatory Limitations

All of the following must appear verbatim in `quality_review.md` and in
the relevant results artifacts. They must not be paraphrased.

**L1 — Sprint 6 in-sample meta-learner limitation** (h1_summary.json):
> "H1 Macro-F1 is computed by evaluating the meta-learner on the same OOF
> matrix used to train it. No separate meta-learner holdout exists under the
> current data-isolation rules. This is in-sample evaluation at the
> meta-learner level and is NOT a fully held-out end-to-end generalisation
> estimate."

**L2 — Sprint 5 RF unmatched-baseline limitation** (h1_summary.json):
> "Two reporting units are used: (a) three-seed H1 stacking mean±std;
> (b) frozen Sprint 5 single-CV base-model reference. These are not
> statistically matched quantities."

**L3 — Sprint 7 single-seed AE limitation** (ae_model.py):
> "Sprint 7 uses a single AE training seed (42). No multi-seed stability
> estimate exists for AE reconstruction error or threshold values. This
> is an accepted scope limitation and not a null result."

**L4 — Validation reuse limitation** (sprint8_final_design.md):
> "VALIDATION is reused for Sprint 7 AE threshold calibration AND Sprint 8
> fusion-rule selection. Both are selection-stage uses, not final held-out
> evaluation. This reuse is within the frozen data-isolation rules but is
> an explicit limitation."

**L5 — n=583 sample-size caveat** (EXP_FUSION_V1):
> "1 row = 1/583 = 0.1716 percentage points; small differences not
> interpretable as strong generalisation evidence."

**L6 — DD-6 FPR cap post-evidence disclosure** (sprint9_discussion_v1.md):
> "The underlying Sprint 8 C01/C06 Development TEST FPR difference
> (+0.0351 pp) was already known when the 2-percentage-point H3 tolerance
> was proposed. Therefore this tolerance is a documented operational
> guardrail rather than a blind pre-registration made before any relevant
> FPR evidence existed."

**L7 — H3 FPR cap non-determinative disclosure** (sprint9_discussion_v1.md):
> "For the frozen Sprint 8 result, the H3 FPR cap is not decision-
> determinative because C06 and C01 have identical Protected Backdoor
> detection counts (582/583). The primary H3 condition therefore already
> fails before the FPR cap can affect the verdict."

---

## 18. Non-Goals

- Retraining any frozen model (EXP_BASE_MODELS_V1, EXP_OOF_STACK_V1,
  EXP_AE_V1, EXP_FUSION_V1)
- Changing the 75-feature set or MI ranking
- Changing Sprint 8 C06 configuration or tau
- Changing the protected Backdoor selection or the 583-row set
- Threshold tuning or selection after Sprint 8
- New fusion rules or learned fusion
- Reopening any Sprint 8 OD decision
- SHAP or explainability (deferred to Sprint 11)
- Ablation matrix (deferred to Sprint 10)
- Significance testing beyond the pre-registered criteria in §10
- McNemar or other statistical tests applied to reverse a NOT SUPPORTED verdict
- Deployment

---

## 19. Implementation Acceptance Criteria

Implementation may begin only after this Final Design is confirmed
internally self-consistent. Implementation is complete only when:

1. All required artifacts exist under `results/evaluation/EXP_H123_V1/`.
2. All 16 required tests (§13) pass with no failures.
3. `summary.json` contains h1_verdict, h2_verdict, h3_verdict.
4. `quality_review.md` contains all seven mandatory limitations verbatim.
5. `metadata.json` contains no null required fields.
6. `metadata.json` `checkpoint_sha256` section contains actual recorded hash values (not placeholder strings).
7. `h1_results.json` contains all eight required H1 fields.
8. `h2_results.json` contains the `criterion_type` field explicitly stating
   "practical_preregistered_not_significance_test".
9. `h2_results.json` contains `supported_threshold_locked: true`.
10. `h3_results.json` contains the `mandatory_wording` field verbatim.
11. `h3_results.json` contains both L6 and L7 disclosures.
12. No frozen Sprint 1–8 artifact has been modified (verified by SHA-256 via T-FROZEN-UPSTREAM).
13. `halt_report.json` does NOT exist (confirming no blocking HALT occurred).

---

## 20. Lifecycle and Freeze Criteria

### Lifecycle

```
PLAN:           READY
DESIGN:         READY
DISCUSSION:     COMPLETE — DD-1 THROUGH DD-8 APPROVED
FINAL DESIGN:   APPROVED — 2026-09-03 (see §21)
IMPLEMENTATION: COMPLETE
TEST:           COMPLETE — 16/16 PASS
VALIDATE:       COMPLETE
FREEZE:         NOT STARTED
```

### Freeze criteria

Sprint 9 freeze requires:
1. All implementation acceptance criteria (§19) satisfied.
2. All 16 required tests pass with no failures.
3. Human explicit approval: "proceed to freeze."
4. Git commit tagged `sprint9-freeze` with all EXP_H123_V1 artifacts included.
5. No frozen Sprint 1–8 artifact modified (SHA-256 cryptographic verification).
6. `checkpoint_sha256` fields in `metadata.json` are fully populated with
   actual hash values (no placeholder strings remain).
7. `halt_report.json` does not exist (no blocking HALT occurred).

### What must not happen before freeze

- No implementation change after any test result is observed.
- No verdict modification after Protected Backdoor or Dev TEST results
  are seen.
- No epsilon, tau, multiplier, or FPR_cap changes after STEP 6 begins.
- No checkpoint SHA-256 values fabricated or approximated.

---

## 21. Final Design Approval Record

| Field | Value |
|-------|-------|
| Approval event | Explicit human approval |
| Approver | Human project owner |
| Approval date | 2026-09-03 |
| Approval timestamp | 2026-09-03T20:59:24+05:30 (2026-09-03T15:29:24Z) |
| Approved document | docs/sprint9_final_design.md |
| Approval statement | "I explicitly approve docs/sprint9_final_design.md as the Final Design for Sprint 9." |
| Recorded by | Antigravity IDE agent |

### Approval Scope

This approval covers the Final Design document (`docs/sprint9_final_design.md`) as
the authoritative specification for Sprint 9 — EXP_H123_V1.

It is distinct from and subsequent to:
1. DD-1 through DD-8 Discussion approval (2026-09-02, docs/sprint9_discussion_v1.md §M/§N)
2. This Final Design document approval (2026-09-03, this section)

FREEZE remains NOT STARTED and requires a separate explicit "proceed to freeze" command.

### State at Approval

At the time of this approval the implementation was complete with the following verified results:

| Hypothesis | Verdict |
|------------|--------|
| H1 | SUPPORTED (diff=0.01222799051528034 > epsilon=0.005) |
| H2 | NOT_SUPPORTED (ae_detected_count=0 < supported_threshold=2) |
| H3 | NOT_SUPPORTED (c06_detected=582 ≤ c01_detected=582) |

All 16 §13 required tests: **PASS**.
T-DETERMINISTIC: empirically verified (17/17 fields, zero numerical difference).
