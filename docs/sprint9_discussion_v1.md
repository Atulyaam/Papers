# Sprint 9 — Formal Discussion Document v2

**Status**: DISCUSSION COMPLETE — DD-1 THROUGH DD-8 APPROVED
**Version**: 2 (Final Discussion Review incorporating critical cleanup pass + human approval)
**Date**: 2026-09-02
**References**:
- [`sprint9_plan_v1.md`](file:///c:/Users/Atul2/OneDrive/Desktop/Papers/IDS-UNSW-NB15/docs/sprint9_plan_v1.md)
- [`sprint9_design_v1.md`](file:///c:/Users/Atul2/OneDrive/Desktop/Papers/IDS-UNSW-NB15/docs/sprint9_design_v1.md)
- `EXP_OOF_STACK_V1/h1_summary.json`
- `EXP_AE_V1/threshold/threshold_calibration.json`
- `EXP_FUSION_V1/quality_review.md`

> [!IMPORTANT]
> No decision in this document is approved. Every value marked
> **PROPOSED — NOT APPROVED** requires explicit human confirmation
> before it may be used in FINAL DESIGN or implementation.

---

## A. Verified Correct

| # | Item | Source |
|---|------|--------|
| A1 | Seeds = 42, 123, 2024 | LOCKED — Context/07, h1_summary.json |
| A2 | Features = EXP_MI_V1_1, 75 features | LOCKED |
| A3 | All four upstream experiments frozen; no retraining | LOCKED |
| A4 | Protected Backdoor = Backdoor subclass, 583 rows | LOCKED |
| A5 | C06 = OR + mean+3sigma, tau=11.160062745213509 | LOCKED |
| A6 | H-FUSION=FALSE, H-PROT-BACKDOOR=FALSE | LOCKED — Sprint 8 |
| A7 | VALIDATION = Normal-only; attack metrics not computable there | LOCKED |
| A8 | AE Normal VALIDATION FPR at tau_mean+3sigma = 7/11200 = 0.0625% | CONFIRMED — threshold_calibration.json |
| A9 | No per-seed base-model checkpoints exist; only one RF/DT/SVM/NN set | CONFIRMED — checkpoint audit |
| A10 | Per-seed meta-learners exist for seeds 42/123/2024 | CONFIRMED — checkpoint audit |
| A11 | AE scaler is the same frozen artifact used in Sprint 8 | CONFIRMED |
| A12 | Option B (DD-1) requires no retraining | CONFIRMED — inference-only |
| A13 | Sprint 6 OOF std_macro_f1 = 0.000263 | CONFIRMED — h1_summary.json |
| A14 | Sprint 8 C01 vs C06 Protected Backdoor: both 582/583 | LOCKED — EXP_FUSION_V1 |

---

## B. Required Clarifications (All Addressed in This Document)

| # | Clarification | Status |
|---|--------------|--------|
| B1 | DD-6 2 pp cap was chosen with knowledge of 0.035 pp delta | Disclosed — see DD-6 section |
| B2 | DD-6 FPR cap is non-determinative for H3 (count difference is 0) | Explicit statement added — see DD-6 |
| B3 | T-DEV-TEST-ISOLATION must distinguish new H1 eval from inherited H3 evidence | Revised — see Tests section |
| B4 | H1 epsilon sanity check vs. Sprint 6 observed variability | Conducted — see K (epsilon sanity check) |

---

## C. DD-1: H1 Execution Mode

### Options

**Option A**: Reproduce Sprint 6 OOF in-sample result. Uses the same matrix
that trained the meta-learner. Not a held-out estimate.

**Option B**: Inference-only evaluation on Development TEST using frozen
checkpoints. Fully held-out.

### Grounds for rejecting Option A as primary

Verbatim from `EXP_OOF_STACK_V1/h1_summary.json` (must not be paraphrased):

> "H1 Macro-F1 is computed by evaluating the meta-learner on the same OOF
> matrix used to train it. No separate meta-learner holdout exists under the
> current data-isolation rules. This is in-sample evaluation at the
> meta-learner level and is NOT a fully held-out end-to-end generalisation
> estimate."

### Option B methodology (precise description)

- Same Development TEST split (81,749 rows, held-out)
- Same frozen preprocessing (TRAIN-fitted scalers, encoders)
- Same 75-feature representation (EXP_MI_V1_1)
- RF = one frozen full-TRAIN checkpoint → one deterministic prediction set
- Stacking = same RF predictions → three per-seed meta-learners → three
  per-seed stacking predictions
- No retraining

**Correct characterization**: "matched evaluation data and preprocessing,
but unmatched replication depth" (RF: one run; stacking: three seeds,
mean ± std).

**Must NOT be described as**: a fully matched 3-seed paired comparison.

### Recommendation

> **DD-1 PROPOSED DEFAULT: Option B**
> Option A retained as supplementary historical reference only,
> explicitly labeled as in-sample.
>
> PROPOSED — NOT APPROVED

### Forward-looking H1 reporting requirement (Cleanup 1)

When Option B is eventually implemented, the H1 results artifact MUST
report all of the following:

| Field | Description |
|-------|-------------|
| `stacking_macro_f1_seed_42` | Stacking Macro-F1 on Dev TEST, seed 42 |
| `stacking_macro_f1_seed_123` | Stacking Macro-F1 on Dev TEST, seed 123 |
| `stacking_macro_f1_seed_2024` | Stacking Macro-F1 on Dev TEST, seed 2024 |
| `stacking_mean_macro_f1` | Mean across the three seeds |
| `stacking_std_macro_f1` | Standard deviation across the three seeds |
| `rf_dev_test_macro_f1` | RF Dev TEST Macro-F1 (single frozen checkpoint) |
| `diff` | stacking_mean_macro_f1 − rf_dev_test_macro_f1 |
| `h1_verdict` | SUPPORTED / NOT SUPPORTED / INCONCLUSIVE per DD-7 |

The three per-seed held-out values provide direct evidence of
Development TEST seed-to-seed variability under Option B.

These must be explicitly distinguished from Sprint 6 OOF variability:

- **Sprint 6 OOF seed variability** (std=0.000263): historical in-sample
  sanity check. Not a held-out estimate.
- **Sprint 9 Dev TEST seed variability**: actual held-out variability.
  Computed from the three per-seed stacking predictions on Dev TEST.
  This is the primary H1 variability evidence.

> Do NOT use the observed Sprint 9 Dev TEST seed variability to redefine
> or adjust epsilon. Epsilon is pre-registered before evaluation begins
> and must remain fixed at the approved value.

---

## D. DD-2: Strongest Individual Classifier Identification Rule

### Infrastructure constraint (from checkpoint audit)

No per-seed RF/DT/SVM/NN checkpoints exist. Per-seed base-model
inference would require retraining — **this is forbidden**. DD-2b
(matched per-seed individual-model inference with separate per-seed
checkpoints) is not implementable.

### Viable approach

The single `rf_final.joblib` produces one deterministic RF prediction set
on Dev TEST. This can be compared with the three per-seed stacking results
on the same Dev TEST split.

The comparison is split-matched (same Dev TEST) but replication-asymmetric
(RF: 1 run; stacking: 3 seeds). This asymmetry must be documented, not
concealed.

Sprint 5 RF Macro-F1 = 0.9508532447968256 is retained as supplementary
historical context only, with this verbatim limitation (h1_summary.json):

> "Two reporting units are used: (a) three-seed H1 stacking mean±std;
> (b) frozen Sprint 5 single-CV base-model reference. These are not
> statistically matched quantities."

### Recommendation

> **DD-2 PROPOSED DEFAULT**: RF Dev TEST inference-only from single frozen
> checkpoint. Report as RF (1 run, Dev TEST) vs. stacking (3 seeds,
> mean ± std, Dev TEST). Sprint 5 historical number = supplementary only.
>
> PROPOSED — NOT APPROVED

---

## E. DD-3: H2 AE Threshold

### Provenance confirmed

From `EXP_AE_V1/threshold/threshold_calibration.json`:

| Threshold | tau | Normal VAL above threshold | FPR |
|-----------|-----|--------------------------|-----|
| mean+3sigma | **11.160062745213509** | 7/11200 | **0.0625%** |

`primary_threshold` = `"DEFERRED_TO_SPRINT_8"` in the Sprint 7 artifact.
Sprint 8 OD-4b selected mean+3sigma. That selection is frozen.

H2 AE-only rule:
```
ae_flag(r) = 1  if  RE(r) > 11.160062745213509   (strict greater-than)
           = 0  otherwise
```

The Sprint 8 C06 582/583 result is a **fused OR prediction** and is NOT
the H2 AE-only result. A new AE-only inference pass is required.

### Recommendation

> **DD-3 PROPOSED DEFAULT**: tau = mean+3sigma = 11.160062745213509
> Loaded from `EXP_AE_V1/threshold/threshold_calibration.json`.
> AE scaler = `EXP_AE_V1/ae_scaler.joblib` (confirmed same as Sprint 8).
>
> PROPOSED — NOT APPROVED

---

## F. DD-4: H2 "Non-Trivial" Quantitative Criterion

### Baseline (frozen)

AE Normal VALIDATION FPR at proposed tau: 7 / 11,200 = **0.0625%**

Expected AE detections on 583 Backdoor rows from Normal-rate noise alone:
583 × 0.000625 = **0.364 rows** (< 1 row)

### Proposed rule

> **DD-4 PROPOSED**: 3× multiplier criterion
>
> SUPPORTED if:
> ae_detected_count / 583 ≥ 3 × (7/11200)
> ↔ ae_detected_count ≥ ceil(3 × 0.001875 × 583) = ceil(1.09) = **2 rows**
>
> Three-way rule:
> - SUPPORTED:     ae_detected_count ≥ 2
> - INCONCLUSIVE:  ae_detected_count == 1
>   (above noise floor but below 3× bar; single detection may be noise)
> - NOT SUPPORTED: ae_detected_count == 0
>
> This is a **practical pre-registered criterion**, NOT a formal statistical
> significance test.
>
> PROPOSED — NOT APPROVED

**Why 3×**: Triple the baseline FPR is a minimum meaningful lift above the
AE's natural false-alarm rate. Below 3×, the detection is within the range
explainable by Normal-distribution noise at the given threshold.

---

## G. DD-5: H3 Scope

### Frozen Sprint 8 evidence (immutable)

| Metric | C01 | C06 |
|--------|-----|-----|
| Protected Backdoor detected | 582 | 582 |
| Missed | 1 | 1 |
| Dev TEST FPR | 19.1892% | 19.2243% |
| Dev TEST Macro-F1 | 0.892609 | 0.892440 |

Detection difference = **0**. H-PROT-BACKDOOR = FALSE (frozen).

### Recommendation

> **DD-5 PROPOSED DEFAULT**: H3 = formalization/re-presentation of frozen
> Sprint 8 evidence under pre-registered H3 decision criteria.
>
> Mandatory verbatim wording in all H3 reporting:
> "Sprint 9 H3 formalizes the Sprint 8 H-FUSION/H-PROT-BACKDOOR findings
> under explicit pre-registered criteria and does not reopen those frozen
> decisions."
>
> Evidentiary sources:
> - `EXP_FUSION_V1/protected_backdoor/metrics.json`
> - `EXP_FUSION_V1/development_test/metrics.json`
> - `EXP_FUSION_V1/quality_review.md`
>
> PROPOSED — NOT APPROVED

---

## H. DD-6: H3 FPR Control

### Frozen evidence

```
C01 Dev TEST FPR = 19.1892%
C06 Dev TEST FPR = 19.2243%
FPR_delta        = +0.0351 percentage points
```

### Proposed cap

> **DD-6 PROPOSED**: FPR_delta ≤ 2 percentage points
>
> FPR_delta = FPR_C06_dev_test − FPR_C01_dev_test
>
> PROPOSED — NOT APPROVED

### Required disclosures (critical cleanup B1, B2)

**Disclosure 1 — Cap is post-evidence**:

> "The underlying Sprint 8 C01/C06 Development TEST FPR difference
> (+0.0351 pp) was already known when the 2-percentage-point H3 tolerance
> was proposed. Therefore this tolerance is a **documented operational
> guardrail** rather than a blind pre-registration made before any relevant
> FPR evidence existed. This must be stated wherever the FPR cap is reported."

**Disclosure 2 — Cap is non-determinative for this result**:

> "For the frozen Sprint 8 result, the H3 FPR cap is not decision-
> determinative because C06 and C01 have identical Protected Backdoor
> detection counts (582/583). The primary H3 condition (count improvement)
> therefore already fails before the FPR cap can affect the verdict. The
> NOT SUPPORTED verdict is driven by the detection-count comparison, not
> by the FPR cap."

---

## I. DD-7: Exact H1/H2/H3 Verdict Rules

### H1 Verdict Rule

> **PROPOSED**: epsilon = 0.005 (0.5 percentage points Macro-F1)
>
> ```
> stacking_mean_f1 = mean(macro_f1 across seeds 42, 123, 2024) on Dev TEST
> baseline_f1      = RF Dev TEST Macro-F1 (single run, DD-2)
> diff             = stacking_mean_f1 − baseline_f1
>
> SUPPORTED:     diff > +0.005
> NOT SUPPORTED: diff < −0.005
> INCONCLUSIVE:  |diff| ≤ 0.005
> ```
>
> See K (epsilon sanity check) for calibration against historical variability.
>
> PROPOSED — NOT APPROVED

### H2 Verdict Rule

> **PROPOSED**:
>
> ```
> ae_detected_count = sum(ae_flag == 1) on 583 Protected Backdoor rows
>
> SUPPORTED:     ae_detected_count ≥ 2
> INCONCLUSIVE:  ae_detected_count == 1
> NOT SUPPORTED: ae_detected_count == 0
> ```
>
> PROPOSED — NOT APPROVED

### H3 Verdict Rule

> **PROPOSED**:
>
> ```
> FPR_delta = FPR_C06_dev_test − FPR_C01_dev_test  (from frozen Sprint 8)
>
> SUPPORTED:     C06_detected > C01_detected  AND  FPR_delta ≤ 0.02
> NOT SUPPORTED: C06_detected ≤ C01_detected
> INCONCLUSIVE:  C06_detected > C01_detected  AND  FPR_delta > 0.02
> ```
>
> Under frozen Sprint 8 evidence: C06_detected = C01_detected = 582
> → **verdict = NOT SUPPORTED** (primary condition fails; FPR cap not reached).
>
> No McNemar test is warranted for a zero detection-count difference.
>
> PROPOSED — NOT APPROVED

---

## J. DD-8: Checkpoint Infrastructure Confirmation

| Artifact | Path | Exists | Notes |
|----------|------|--------|-------|
| RF | `EXP_BASE_MODELS_V1/rf/rf_final.joblib` | YES (224 MB) | Single — not per-seed |
| DT | `EXP_BASE_MODELS_V1/dt/dt_final.joblib` | YES | Single |
| SVM | `EXP_BASE_MODELS_V1/svm/svm_final.joblib` | YES | Single |
| SVM scaler | `EXP_BASE_MODELS_V1/svm/svm_scaler.joblib` | YES | Single |
| NN | `EXP_BASE_MODELS_V1/nn/nn_final.pt` | YES | Single |
| NN scaler | `EXP_BASE_MODELS_V1/nn/nn_scaler.joblib` | YES | Single |
| Meta seed-42 | `EXP_OOF_STACK_V1/seed_42/meta_learner.joblib` | YES | Per-seed |
| Meta seed-123 | `EXP_OOF_STACK_V1/seed_123/meta_learner.joblib` | YES | Per-seed |
| Meta seed-2024 | `EXP_OOF_STACK_V1/seed_2024/meta_learner.joblib` | YES | Per-seed |
| AE weights | `EXP_AE_V1/ae_final.pt` | YES | seed-42 |
| AE scaler | `EXP_AE_V1/ae_scaler.joblib` | YES | Same as Sprint 8 |

**Conclusions**:
1. No per-seed base-model checkpoints → per-seed individual-baseline is infeasible without retraining.
2. No retraining required for Option B (DD-1).
3. No retraining required for H2 (DD-3) or H3 (DD-5).

> **DD-8**: PROPOSED CONFIRMED — NOT YET FORMALLY APPROVED

---

## K. H1 Epsilon Sanity Check (Against Frozen Sprint 6 Data)

**Purpose**: Evaluate whether the proposed epsilon=0.005 is calibrated
sensibly against the historical observed variability in stacking Macro-F1.

**Frozen Sprint 6 per-seed OOF stacking Macro-F1** (from h1_summary.json):

| Seed | OOF Macro-F1 |
|------|-------------|
| 42 | 0.946958 |
| 123 | 0.947290 |
| 2024 | 0.947476 |
| **Mean** | **0.947242** |
| **Std** | **0.000263** |

**Analysis**:

```
Historical seed-to-seed range = 0.947476 - 0.946958 = 0.000518 Macro-F1
Historical std                = 0.000263 Macro-F1

Proposed epsilon              = 0.005000 Macro-F1

Ratio: epsilon / std  = 0.005 / 0.000263 = 19×
Ratio: epsilon / range = 0.005 / 0.000518 = 9.7×
```

**Interpretation**:

The proposed epsilon (0.5 pp) is approximately **19 standard deviations**
of Sprint 6 seed-to-seed stacking variability and approximately **10×**
the seed-to-seed range. This means:

- The INCONCLUSIVE zone (|diff| ≤ 0.005) is very wide relative to the
  within-stacking variability observed in Sprint 6.
- Any difference between stacking and RF that falls within ±0.005 Macro-F1
  would be declared INCONCLUSIVE regardless of statistical consistency.
- The historical Sprint 6 OOF diff (stacking − Sprint5 RF reference)
  was −0.0037, which falls inside the INCONCLUSIVE zone under epsilon=0.005.

**Assessment**: epsilon=0.005 is a practical domain-relevance threshold
(0.5 pp is a standard "meaningful difference" benchmark in ML model
comparison), not a statistical significance threshold calibrated to
observed seed variance. The within-stacking variance is extremely small
(std=0.000263), so the meaningful question is whether stacking vs. RF
differs by more than 0.5 pp — a qualitative, not statistical, criterion.

**Key caveat**: The Sprint 6 variability is from in-sample OOF evaluation.
Sprint 9 Option B evaluates on held-out Dev TEST, where variability across
seeds may differ. No adjustment should be made based on this Sprint 6
sanity check — it is informational only.

> **epsilon=0.005 sanity-check verdict**: The value is methodologically
> coherent as a domain-relevance threshold. It is not a statistical
> significance threshold. It is wide relative to within-stacking variance,
> which is appropriate because the question is practical significance,
> not statistical significance.
>
> epsilon = 0.005 remains PROPOSED — NOT APPROVED.
> Do not modify epsilon based on Sprint 9 results after Dev TEST is accessed.

---

## L. Remaining Methodological Risks

| # | Risk | Severity | Mitigation |
|---|------|----------|-----------|
| L1 | H2 may return NOT SUPPORTED at tau_mean+3sigma (only 7/11,200 Normal rows exceed threshold; Backdoor RE may not be substantially elevated) | Medium | Pre-registered. NOT SUPPORTED is a valid result. The near-inert AE limitation is already documented in Sprint 8 quality_review.md. |
| L2 | H3 is NOT SUPPORTED by frozen evidence (count difference = 0) | Already known | Pre-registered. Sprint 8 H-PROT-BACKDOOR=FALSE. Sprint 9 H3 formalizes this. NOT SUPPORTED is the expected verdict. |
| L3 | H1 comparison is replication-asymmetric (RF: 1 run; stacking: 3 seeds) | Low | Disclosed in DD-2. Described precisely as "matched evaluation data, unmatched replication depth." |
| L4 | DD-6 FPR cap was chosen post-evidence | Low | Disclosed with exact wording. Cap is non-determinative for this H3 result. |
| L5 | epsilon for H1 may not capture the right effect size on Dev TEST | Low | Sanity-checked against Sprint 6 variability. Remains PROPOSED. Must not be adjusted after Dev TEST is accessed. |
| L6 | Sprint 9 may appear to duplicate Sprint 8 work | Low | Sprint 9 adds: held-out H1 (new), AE-only H2 (new), formal H3 verdict criteria (new framework over Sprint 8 evidence). These are distinct contributions. |
| L7 | Multiple-comparisons concern across H1/H2/H3 | Acknowledged — reporting guidance only | **Multiplicity note (Cleanup 2)**: H1–H3 are evaluated as independently pre-registered engineering checks and no multiple-comparisons adjustment is applied across the three hypotheses. This is downstream reporting guidance, not a new Sprint 9 selection rule. No hypothesis criterion is modified. |

---

## M. Human Approval Table

> [!IMPORTANT]
> The following decisions are **HUMAN-APPROVED** and are LOCKED for
> Sprint 9 FINAL DESIGN. They must be carried into FINAL DESIGN
> unchanged. They must not be reinterpreted, weakened, or silently
> altered during implementation.

| DD-ID | Approved Decision | Status |
|-------|------------------|---------|
| **DD-1** | Option B — H1 inference-only evaluation on Development TEST using frozen checkpoints. No retraining. | **APPROVED** |
| **DD-2** | RF single frozen checkpoint (EXP_BASE_MODELS_V1) on Dev TEST as H1 baseline. Sprint 5 RF value = supplementary only. Document replication asymmetry: RF=1 run; stacking=3 seeds. | **APPROVED** |
| **DD-3** | H2 AE threshold = mean+3sigma, tau=11.160062745213509. Loaded from EXP_AE_V1/threshold/threshold_calibration.json. | **APPROVED** |
| **DD-4** | H2 practical criterion: AE detected_count ≥ 2 = SUPPORTED; 1 = INCONCLUSIVE; 0 = NOT SUPPORTED. Explicitly a practical criterion, not a formal statistical significance test. | **APPROVED** |
| **DD-5** | H3 = formalization/re-presentation of frozen Sprint 8 evidence. Mandatory wording required. No reopening of Sprint 8. | **APPROVED** |
| **DD-6** | H3 FPR_delta ≤ 2 pp. APPROVED WITH DISCLOSURE: the +0.0351 pp Sprint 8 delta was already known when the 2 pp cap was proposed (operational guardrail, not blind pre-registration). FPR cap is non-determinative for this H3 result. | **APPROVED WITH DISCLOSURE** |
| **DD-7** | H1: epsilon=0.005 (SUPPORTED if diff>+0.005; NOT SUPPORTED if diff<−0.005; INCONCLUSIVE if ≤±0.005). H2: count rule as DD-4. H3: count-improvement + FPR_delta≤0.02. | **APPROVED** |
| **DD-8** | Checkpoint audit confirmed. Single base-model checkpoint set. Three per-seed meta-learners. No per-seed RF checkpoints. No retraining may be introduced. | **APPROVED** |

---

## N. Human Approval Summary

**Approval date**: 2026-09-02
**Approved by**: Human project owner (explicit approval)

All eight Sprint 9 Discussion decisions are now LOCKED. The following
complete decision record is authoritative for FINAL DESIGN.

### H1 — Locked protocol

```
Execution:     Option B — inference-only on Development TEST
Baseline:      RF, single frozen checkpoint from EXP_BASE_MODELS_V1
Seeds:         42, 123, 2024
Replication:   RF = 1 run (deterministic); stacking = 3 seeds (mean ± std)
Comparison:    Split-matched (same Dev TEST); replication-asymmetric
Epsilon:       0.005 Macro-F1

Verdict rules:
  SUPPORTED     if stacking_mean_macro_f1 - rf_dev_test_macro_f1 > +0.005
  NOT SUPPORTED if difference < -0.005
  INCONCLUSIVE  if |difference| ≤ 0.005

Required artifact fields:
  stacking_macro_f1_seed_42
  stacking_macro_f1_seed_123
  stacking_macro_f1_seed_2024
  stacking_mean_macro_f1
  stacking_std_macro_f1
  rf_dev_test_macro_f1
  diff
  h1_verdict

Note: Sprint 9 Dev TEST seed variability must NOT be used to modify epsilon.
```

### H2 — Locked protocol

```
Branch:        AE anomaly flag ONLY (not C06 OR-fusion)
Threshold:     mean+3sigma, tau = 11.160062745213509
Scaler:        EXP_AE_V1/ae_scaler.joblib
AE flag:       ae_flag = 1 if RE > tau, else 0
Baseline FPR:  7/11200 = 0.0625% (AE Normal VALIDATION at tau_mean+3sigma)
Multiplier:    3×

Verdict rules:
  SUPPORTED     if ae_detected_count ≥ 2
  INCONCLUSIVE  if ae_detected_count == 1
  NOT SUPPORTED if ae_detected_count == 0

Note: Practical pre-registered criterion; NOT a formal significance test.
Note: Sprint 8 C06 582/583 is a FUSED result and does NOT satisfy H2.
      A new AE-only inference pass is required.
```

### H3 — Locked protocol

```
Scope:         Formalization/re-presentation of frozen Sprint 8 evidence
Sources:       EXP_FUSION_V1/protected_backdoor/metrics.json
               EXP_FUSION_V1/development_test/metrics.json
               EXP_FUSION_V1/quality_review.md
FPR_delta:     FPR_C06_dev_test - FPR_C01_dev_test
FPR cap:       2 percentage points (with required post-evidence disclosure)

Verdict rules:
  SUPPORTED     if C06_detected > C01_detected AND FPR_delta ≤ 0.02
  NOT SUPPORTED if C06_detected ≤ C01_detected
  INCONCLUSIVE  if C06_detected > C01_detected AND FPR_delta > 0.02

Frozen Sprint 8 evidence:
  C01 detected = 582/583
  C06 detected = 582/583
  FPR_delta    = +0.0351 pp
  Verdict      = NOT SUPPORTED (count condition fails; FPR cap non-determinative)

Mandatory verbatim wording in all H3 reporting:
  "Sprint 9 H3 formalizes the Sprint 8 H-FUSION/H-PROT-BACKDOOR findings
  under explicit pre-registered criteria and does not reopen those frozen
  decisions."

Required disclosures:
  L6: DD-6 post-evidence disclosure (FPR cap not a blind pre-registration)
  L7: FPR cap non-determinative for this H3 result
```

### What remains forbidden

- Retraining any frozen model
- Modifying any Sprint 1–8 artifact
- Changing C06, tau, or Protected Backdoor selection
- Using Dev TEST for any new model selection or criterion modification
- Adjusting epsilon or multiplier after Sprint 9 evaluation begins
- Introducing McNemar solely to manufacture a p-value for a zero-difference result
- Silently altering any approved DD

---

## Required Tests (Revised)

> [!NOTE]
> T-DEV-TEST-ISOLATION has been revised (cleanup B3) to correctly
> distinguish newly computed H1 inference from inherited Sprint 8 H3 evidence.

| Test ID | Assertion | Scope |
|---------|-----------|-------|
| T-CRITERION-PREREGISTERED | H1/H2/H3 criteria locked in config.yaml before any Sprint 9 evaluation | All |
| T-NO-RESULT-BACKWARD | No criterion modified after Sprint 9 Dev TEST / Backdoor results are observed | All |
| T-FROZEN-UPSTREAM | All upstream experiment IDs match frozen records | All |
| T-NO-RETRAIN | No base model or meta-learner training occurs during Sprint 9 | All |
| T-75-FEATURES | Exactly 75 features in all Sprint 9 inference matrices | All |
| T-SEED-SET | Sprint 9 stacking seeds = {42, 123, 2024} exactly | H1 |
| T-TAU-PROVENANCE | tau loaded from `EXP_AE_V1/threshold/threshold_calibration.json`; value = 11.160062745213509 | H2 |
| T-H2-AE-ONLY | H2 verdict uses AE anomaly flag independently; not C06 OR-fusion prediction | H2 |
| T-H3-NO-RESELECT | H3 C06 parameters unchanged from EXP_FUSION_V1 frozen record | H3 |
| T-PROT-ISOLATION | Protected Backdoor row count in TRAIN == 0 and in VALIDATION == 0 | All |
| **T-DEV-TEST-ISOLATION** | **For NEW Sprint 9 computations (H1 inference): Development TEST is inaccessible until H1/H2/H3 criteria are registered. Frozen Sprint 8 Dev TEST evidence used for H3 re-presentation is exempt because it predates Sprint 9 and is immutable read-only input.** | H1 / H3 |
| T-HASH-CONSISTENCY | Dataset SHA-256 hashes match Sprint 8 provenance records | All |
| T-DETERMINISTIC | Re-running Sprint 9 evaluation with identical inputs produces identical outputs | All |
| T-PROVENANCE-COMPLETE | `metadata.json` contains all required provenance fields | All |

---

## Mandatory Limitation Inventory

Must appear verbatim in all Sprint 9 reporting. Do not paraphrase.

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

**L6 — DD-6 FPR cap post-evidence disclosure** (this document):
> "The underlying Sprint 8 C01/C06 Development TEST FPR difference
> (+0.0351 pp) was already known when the 2-percentage-point H3 tolerance
> was proposed. Therefore this tolerance is a documented operational
> guardrail rather than a blind pre-registration made before any relevant
> FPR evidence existed."

**L7 — H3 FPR cap non-determinative disclosure** (this document):
> "For the frozen Sprint 8 result, the H3 FPR cap is not decision-
> determinative because C06 and C01 have identical Protected Backdoor
> detection counts (582/583). The primary H3 condition therefore already
> fails before the FPR cap can affect the verdict."

---

## Lifecycle Status

```
PLAN:           READY
DESIGN:         READY
DISCUSSION:     COMPLETE — DD-1 THROUGH DD-8 APPROVED
FINAL DESIGN:   NOT STARTED
IMPLEMENTATION: NOT STARTED
TEST:           NOT STARTED
VALIDATE:       NOT STARTED
FREEZE:         NOT STARTED
```
