# Sprint 9 — H1/H2/H3 Evaluation — Design v1

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

Formally evaluate the three must-have hypotheses under the frozen
UNSW-NB15 project protocol:

**H1**: The stacked ensemble matches or exceeds the strongest individual
classifier under identical conditions.

**H2**: The benign-trained Autoencoder flags a non-trivial proportion of
the pre-registered withheld Backdoor subclass.

**H3**: Combining the supervised stacking output with the anomaly signal
improves withheld-class detection relative to the ensemble alone at a
controlled false-positive rate.

Negative results are valid results and must be reported without overclaiming.
Sprint 9 is the formal H1/H2/H3 evaluation stage — it is NOT a fusion
sprint. Fusion was Sprint 8 and is already frozen.

---

## 2. Frozen Upstream Artifacts

The following experiments are FROZEN and must NOT be retrained, re-tuned,
modified, or replaced:

| Experiment | Git Tag | Role |
|------------|---------|------|
| EXP_MI_V1_1 | — | 75 frozen features |
| EXP_BASE_MODELS_V1 | — | DT, RF, SVM, NN checkpoints |
| EXP_OOF_STACK_V1 | — | 3-seed meta-learner checkpoints (seeds 42, 123, 2024) |
| EXP_AE_V1 | — | Autoencoder checkpoint + AE scaler + threshold_config.json |
| EXP_FUSION_V1 | — | C06 frozen selection; H-FUSION=FALSE; H-PROT-BACKDOOR=FALSE |

Frozen checkpoint locations (verified present):

```
results/checkpoints/EXP_BASE_MODELS_V1/dt/dt_final.joblib
results/checkpoints/EXP_BASE_MODELS_V1/rf/rf_final.joblib
results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_final.joblib
results/checkpoints/EXP_BASE_MODELS_V1/svm/svm_scaler.joblib
results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_final.pt
results/checkpoints/EXP_BASE_MODELS_V1/nn/nn_scaler.joblib
results/checkpoints/EXP_OOF_STACK_V1/seed_42/meta_learner.joblib
results/checkpoints/EXP_OOF_STACK_V1/seed_123/meta_learner.joblib
results/checkpoints/EXP_OOF_STACK_V1/seed_2024/meta_learner.joblib
results/checkpoints/EXP_AE_V1/ae_final.pt
results/checkpoints/EXP_AE_V1/ae_scaler.joblib
```

> [!IMPORTANT]
> **Infrastructure gap (DD-8)**: EXP_BASE_MODELS_V1 has a single set of
> base-model checkpoints (not per-seed). EXP_OOF_STACK_V1 has per-seed
> meta-learner checkpoints. The OOF results per seed exist as
> `seed_*/oof_predictions.csv`. No per-seed base-model (DT/RF/SVM/NN)
> checkpoints exist separately from the single EXP_BASE_MODELS_V1 set.
>
> This means Option B inference uses the same base-model checkpoints for all
> three seeds, varying only the meta-learner seed (42/123/2024). This is
> consistent with the Sprint 6 protocol (seed governs CV fold assignment and
> meta-learner random_state; base models are the same trained-on-full-TRAIN
> instances). **This gap must be confirmed in DISCUSSION (DD-8).**

---

## 3. Data Access Boundaries

| Split | Rows | Label composition | Sprint 9 permitted uses |
|-------|------|-------------------|------------------------|
| TRAIN | 162,395 | Mixed | Read preprocessing artifacts; no refit |
| VALIDATION | 11,200 | 100% Normal | FPR metrics only; no attack metrics |
| Development TEST | 81,749 | Mixed | One-shot held-out evaluation; no tuning |
| Protected Backdoor | 583 | 100% Backdoor attack | Final evaluation only; no prior use |
| Excluded Backdoor | — | — | **FORBIDDEN** |

Data isolation rules (LOCKED — Context/04):
- No Protected Backdoor row may appear in TRAIN, VALIDATION, or Dev TEST.
- No Dev TEST or Protected Backdoor result may influence criteria, thresholds,
  or methodology defined in this document.
- VALIDATION may not be used to compute attack metrics.
- All preprocessing transformations use TRAIN-fitted objects only.

---

## 4. H1 Protocol

### Hypothesis Statement

> The stacked ensemble matches or exceeds the strongest individual classifier
> under identical held-out evaluation conditions.

### Proposed Execution Approach: Option B (PROPOSED DEFAULT — DD-1)

**Option A** (reproduce/validate Sprint 6 OOF in-sample result) is NOT
the preferred approach for the following reasons, inherited verbatim from
`EXP_OOF_STACK_V1/h1_summary.json`:

> "H1 Macro-F1 is computed by evaluating the meta-learner on the same OOF
> matrix used to train it. No separate meta-learner holdout exists under the
> current data-isolation rules. This is in-sample evaluation at the
> meta-learner level and is NOT a fully held-out end-to-end generalisation
> estimate."

**Option B (Proposed Default)**: Perform inference-only evaluation on
Development TEST using frozen checkpoints, with identical conditions for
stacking and the individual classifier baseline.

No retraining is permitted under either option.

**Infrastructure status under Option B** (from checkpoint audit above):
- Base-model inference on Dev TEST: supported by frozen EXP_BASE_MODELS_V1
  checkpoints (DT, RF, SVM, NN + scalers).
- Meta-learner inference on Dev TEST: supported by per-seed frozen meta-learner
  checkpoints (seed 42, 123, 2024).
- Stacking inference = base-model predictions on Dev TEST → per-seed
  meta-learner prediction.
- **No retraining required.** This is inference-only. ✅

**DD-1**: Approve Option B as the Sprint 9 H1 execution mechanism.

### Seeds

```
42, 123, 2024
```
LOCKED — Context/07, EXP_OOF_STACK_V1.

### Strongest Individual Classifier Identification (DD-2)

The comparison baseline must be identified **before** Sprint 9 Dev TEST
results are generated.

**Proposed default**: Use frozen Sprint 5 reference records to identify
the strongest individual model.

From `EXP_OOF_STACK_V1/h1_summary.json` (frozen):

```json
"sprint5_reference": {
  "model": "RF",
  "macro_f1": 0.9508532447968256,
  "label": "Frozen Sprint 5 single-CV reference; not a matched 3-seed H1 baseline."
}
```

Inherited limitation (verbatim, must not be paraphrased):

> "Two reporting units are used: (a) three-seed H1 stacking mean±std;
> (b) frozen Sprint 5 single-CV base-model reference. These are not
> statistically matched quantities."

The proposed rule is therefore: pre-register RF as the H1 comparison
baseline, using the Sprint 5 single-CV Macro-F1 as the reference,
while explicitly documenting the unmatched-baseline limitation.

**Alternatively**, Option B could run RF inference on Dev TEST under each
of the three seeds (same meta-learner seed → same fold assignment → same
base model), providing a properly matched per-seed comparison. This
approach avoids the unmatched-baseline problem entirely.

**DD-2**: Approve the strongest-classifier identification rule.
The two sub-options are:
- DD-2a: frozen Sprint 5 RF reference (with explicit unmatched limitation)
- DD-2b: new per-seed RF inference on Dev TEST under identical conditions
  (properly matched; requires one additional inference pass)

**No Training**: DD-2b requires only inference; the RF checkpoint exists.

### H1 Primary Metrics

Per seed and aggregated:

| Metric | Required |
|--------|---------|
| Macro-F1 | YES (primary) |
| Weighted-F1 | YES |
| Balanced Accuracy | YES |
| Accuracy | YES |

Report format: per-seed table + mean ± standard deviation.

### H1 Verdict Criterion (DD-7)

**OPEN — must be pre-registered before evaluation.**

Proposed structure (not yet approved):

- **SUPPORTED**: Stacking mean Macro-F1 ≥ baseline Macro-F1 (matched
  comparison) across all three seeds, or mean±std interval overlaps above
  baseline.
- **NOT SUPPORTED**: Stacking mean Macro-F1 < baseline Macro-F1 under the
  matched comparison, with std not bridging the gap.
- **INCONCLUSIVE**: Difference is within numerical tolerance but comparison
  is structurally unmatched (e.g., under DD-2a).

Exact numeric threshold must be locked in FINAL DESIGN.

---

## 5. H2 Protocol

### Hypothesis Statement

> The benign-trained Autoencoder flags a non-trivial proportion of the
> pre-registered withheld Backdoor subclass.

### AE-Only Evaluation (NOT Fusion)

H2 evaluates the **Autoencoder anomaly branch independently**.

> [!IMPORTANT]
> Sprint 8 EXP_FUSION_V1 reported 582/583 Protected Backdoor detected.
> That result is from the **C06 OR-fusion** prediction, not the AE alone.
> The C06 OR result inherits all Backdoor rows already detected by the
> supervised branch. H2 requires a separate AE-only inference pass.
>
> H2 is NOT answered by Sprint 8 results. A new AE-only computation is
> required.

AE anomaly flag definition (frozen OD-2a):
```
ae_flag(r) = 1 if RE(r) > tau else 0   (strict greater-than)
```

### Threshold (DD-3)

**Proposed default**: Use the frozen Sprint 8 selected threshold.

```
tau          = 11.160062745213509
threshold_id = mean+3sigma
```

Rationale:
- Already selected and frozen by Sprint 8 OD-4b.
- Avoids introducing any new threshold decision.
- Provides cross-sprint consistency; same AE operating point used in C06.

**DD-3**: Approve frozen C06 mean+3sigma τ=11.160062745213509 as the
H2 AE-only threshold.

Alternative: use the original Sprint 7 primary operating threshold (τ_p95,
Context/03 §5: "tau = 95th percentile"). This is the architecturally
motivated threshold. However, using τ_p95 would introduce a different
operating point than Sprint 8 selected, creating inconsistency. Unless
there is a methodological reason to prefer it, the frozen Sprint 8 threshold
is preferred.

### H2 Metrics

| Metric | Value |
|--------|-------|
| detected_count | sum(ae_flag == 1) on 583 Backdoor rows |
| missed_count | 583 − detected_count |
| detection_rate | detected_count / 583 |
| n | 583 (fixed) |

Mandatory caveat (inherited from Sprint 8):
**1 row = 1/583 = 0.1716 percentage points.**

### H2 "Non-Trivial" Criterion (DD-4)

**OPEN — exact quantitative definition required. Must be pre-registered.**

Proposed framework (not yet approved): compare AE Backdoor detection
rate to AE Normal VALIDATION false-positive rate.

```
AE_VALIDATION_FPR = sum(ae_flag == 1 on 11,200 Normal VALIDATION rows) / 11,200
H2_detection_rate = detected_count / 583
```

Candidate criterion (proposed, not approved):
- **SUPPORTED**: H2_detection_rate > AE_VALIDATION_FPR by at least X pp
  (X to be pre-registered)
- **NOT SUPPORTED**: H2_detection_rate ≤ AE_VALIDATION_FPR
- **INCONCLUSIVE**: H2_detection_rate > AE_VALIDATION_FPR but difference
  is within 0.1716 pp (< 1 row)

**DD-4**: Approve the exact quantitative H2 criterion and value of X
before evaluation.

### H2 AE Normal VALIDATION FPR

This quantity is computable from frozen artifacts (VALIDATION is Normal-only
and is a permitted input). Its computation must precede Protected Backdoor
evaluation and must not be modified after Protected Backdoor is observed.

---

## 6. H3 Protocol

### Hypothesis Statement

> Combining the supervised stacking output with the anomaly signal improves
> withheld-class detection relative to the ensemble alone at a controlled
> false-positive rate.

### Scope (DD-5)

**Proposed default**: Sprint 9 H3 is a **formalization and re-presentation
of the frozen Sprint 8 evidence** under explicit pre-registered H3 decision
criteria. It does NOT reopen Sprint 8 or perform a new configuration search.

Verbatim statement to include in the final design and all H3 reporting:

> "Sprint 9 H3 formalizes the Sprint 8 H-FUSION/H-PROT-BACKDOOR findings
> under explicit pre-registered criteria and does not reopen those frozen
> decisions."

**DD-5**: Approve H3 as re-presentation of Sprint 8 frozen evidence vs.
a new independent evaluation pass.

Frozen Sprint 8 results (DO NOT MODIFY):

| Configuration | Detected | Missed | Rate |
|---------------|---------|--------|------|
| C01 (supervised-only) | 582 | 1 | 99.83% |
| C06 (OR + mean+3sigma) | 582 | 1 | 99.83% |

Detection-count difference = **0**.

### H3 Verdict Criterion (DD-7)

**SUPPORTED** only if:
```
C06 Backdoor detected_count > C01 Backdoor detected_count
AND
FPR control condition (DD-6) is satisfied
```

Since detected counts are identical (both = 582), **H3 is NOT SUPPORTED
under the frozen evidence** unless the criterion explicitly provides for an
INCONCLUSIVE determination.

> [!NOTE]
> No post-hoc statistical test (e.g., McNemar) is warranted when the
> detection-count difference is exactly 0. Introducing McNemar to generate
> a p-value on a zero-difference result adds no information and should not
> be used to reverse a NOT SUPPORTED verdict.

**DD-7 (H3 component)**: Approve the exact three-way SUPPORTED /
NOT SUPPORTED / INCONCLUSIVE rule for H3 before evaluation.

### H3 FPR Control (DD-6)

**OPEN — exact numerical delta must be pre-registered.**

Proposed framework (not yet approved):

```
FPR_delta = FPR_C06 - FPR_C01  (Development TEST Normal rows)
```

From frozen Sprint 8:
- C01 Dev TEST FPR: 19.19%
- C06 Dev TEST FPR: 19.22%
- FPR_delta = +0.03% (C06 slightly higher)

Candidate FPR control rule: C06 FPR ≤ C01 FPR + max_delta, where max_delta
must be pre-registered (e.g., 1 pp, 0.5 pp).

**DD-6**: Approve the exact max_delta value before evaluation.

---

## 7. Metric Definitions

### Primary metrics (H1)

```
Macro-F1        = mean(per-class F1)
Weighted-F1     = weighted mean(per-class F1) by support
Balanced Acc    = mean(recall per class)
Accuracy        = correct / total
```

### Protected Backdoor metrics (H2, H3)

```
detected_count  = sum(final_prediction == 1) on Backdoor rows
missed_count    = 583 - detected_count
detection_rate  = detected_count / 583
pp_per_row      = 0.1716 percentage points
```

### FPR metrics (H3 control)

```
FPR = FP / (FP + TN)  on Normal-label rows only
```

---

## 8. Hypothesis Decision Criteria

All criteria must be pre-registered before any Sprint 9 evaluation data
is generated. Post-hoc modification is prohibited.

| Hypothesis | Primary rule | Status |
|------------|-------------|--------|
| H1 | Stacking mean Macro-F1 ≥ baseline per DD-2 | **DD-7 OPEN** |
| H2 | AE detection rate > AE VALIDATION FPR by ≥ X pp | **DD-4 OPEN** |
| H3 | C06 detected_count > C01 AND FPR_delta ≤ max_delta | **DD-6, DD-7 OPEN** |

All three criteria require explicit numeric thresholds locked in FINAL DESIGN.

---

## 9. Evaluation Ordering

The evaluation sequence is strictly fixed:

```
STEP 1: Load all frozen artifacts; verify hashes and provenance.
STEP 2: Pre-register all H1/H2/H3 decision criteria (DD-4, DD-6, DD-7).
STEP 3: Compute AE Normal VALIDATION FPR (permitted — no attack labels used).
STEP 4: H1 — stacking and baseline inference on Development TEST.
STEP 5: H2 — AE-only inference on Protected Backdoor.
STEP 6: H3 — C01 and C06 comparison on Protected Backdoor.
STEP 7: Apply pre-registered criteria; issue H1/H2/H3 verdicts.
STEP 8: Optional exploratory analysis (INFORMATIONAL ONLY, labeled as such).
STEP 9: Write quality_review.md; record all limitations.
```

> [!CAUTION]
> No result from steps 4–6 may influence any decision registered in step 2.
> Exploratory analysis in step 8 must not alter verdicts issued in step 7.

---

## 10. Leakage / Isolation Requirements

| Requirement | Rule |
|-------------|------|
| Protected Backdoor isolation | Zero Backdoor rows in TRAIN, VALIDATION, or Dev TEST |
| No threshold retuning | tau loaded from frozen EXP_AE_V1 + Sprint 8 OD-4 selection |
| No model retraining | All inference uses frozen checkpoints only |
| No post-hoc criterion change | H1/H2/H3 criteria locked in FINAL DESIGN before evaluation |
| No Dev TEST influence on criteria | Criteria defined before Dev TEST is accessed |
| No backward leakage | Exploratory analysis has no backward influence on primary verdicts |
| VALIDATION attack metrics forbidden | VALIDATION is Normal-only; no F1/recall/BA computed there |

---

## 11. Reproducibility Requirements

For each evaluation run, record:

```
upstream_experiment_ids: [EXP_MI_V1_1, EXP_BASE_MODELS_V1,
                          EXP_OOF_STACK_V1, EXP_AE_V1, EXP_FUSION_V1]
git_commit_hash: <hash at evaluation time>
git_tags: all sprint freeze tags
dataset_sha256:
  train.csv:                   4a259324e604f013...
  validation.csv:              13caf21a076a33f5...
  development_test.csv:        04725e85732ab2fc...
  protected_unseen_attack.csv: 6ffd23479b575e43...
feature_set: EXP_MI_V1_1, 75 features
seeds: [42, 123, 2024]
python_version: <recorded at runtime>
library_versions: sklearn, torch, numpy, pandas <recorded at runtime>
```

---

## 12. Required Artifacts

Artifact root: `results/evaluation/EXP_H123_V1/`

| File | Contents |
|------|---------|
| `config.yaml` | All evaluation parameters (thresholds, seeds, criteria) |
| `metadata.json` | Provenance, upstream IDs, git commit, dataset hashes |
| `h1_results.json` | Per-seed stacking and baseline metrics; mean ± std |
| `h2_results.json` | AE-only detected_count, missed_count, detection_rate, VALIDATION FPR |
| `h3_results.json` | C01 vs C06 comparison; FPR delta; verdict |
| `summary.json` | H1/H2/H3 verdicts, supporting metrics, limitations |
| `runtime_report.json` | Runtimes, environment, library versions |
| `quality_review.md` | Narrative review with all mandatory limitations |
| `provenance/` | SHA-256 hashes, checkpoint load confirmations |

Exact schema to be defined in FINAL DESIGN.

---

## 13. Required Tests

Sprint 9 must pass these isolation tests before results are reported:

| Test ID | Assertion |
|---------|-----------|
| T-FROZEN-UPSTREAM | All upstream experiment IDs match frozen records |
| T-75-FEATURES | Feature matrix has exactly 75 columns |
| T-SEED-SET | Seeds used = {42, 123, 2024} exactly |
| T-NO-RETRAIN | No base model or meta-learner training occurs during Sprint 9 |
| T-TAU-PROVENANCE | tau loaded from frozen EXP_AE_V1 + Sprint 8 OD-4 record |
| T-H2-AE-ONLY | H2 uses AE anomaly flag only, not fused C06 prediction |
| T-H3-NO-RESELECT | C06 parameters unchanged from EXP_FUSION_V1 frozen record |
| T-PROT-ISOLATION | Protected Backdoor count in TRAIN == 0, in VALIDATION == 0 |
| T-DEV-TEST-ISOLATION | Dev TEST not accessed before criteria are registered |
| T-CRITERION-PREREGISTERED | H1/H2/H3 decision criteria locked in config.yaml before eval |
| T-NO-RESULT-BACKWARD | No criterion modified after seeing Dev TEST or Backdoor result |
| T-HASH-CONSISTENCY | Dataset SHA-256 matches sprint8 provenance records |
| T-DETERMINISTIC | Re-running evaluation produces identical outputs |
| T-PROVENANCE-COMPLETE | metadata.json contains all required fields |

Exact test names and implementations pending FINAL DESIGN.

---

## 14. Non-Goals

- Retraining any frozen model
- Changing the 75-feature set or MI ranking
- Changing Sprint 8 C06 configuration
- Changing the protected Backdoor selection or the 583-row set
- Threshold tuning after Sprint 8
- New fusion rules or learned fusion
- SHAP / explainability (deferred to Sprint 11)
- Ablation (deferred to Sprint 10)
- Significance testing beyond pre-registered criteria
- Deployment
- New model architectures

---

## 15. Failure / Inconclusive Result Handling

- A NOT SUPPORTED verdict is a valid result. Report it without overclaiming.
- An INCONCLUSIVE verdict must be reported with the explicit reason
  (e.g., comparison is structurally unmatched per Sprint 6 limitation;
  sample size too small; criterion threshold not met).
- Sprint 9 does NOT permit re-running evaluation with modified criteria
  because a verdict is inconvenient.
- If infrastructure inspection reveals that Option B (DD-1) cannot be
  executed without unanticipated retraining, escalate to DISCUSSION before
  implementation.

---

## 16. Provenance Requirements

| Field | Source |
|-------|--------|
| Sprint 6 in-sample meta-learner limitation | Must be cited verbatim from h1_summary.json in any H1 reporting |
| Sprint 5 RF reference not a matched baseline | Must be cited verbatim from h1_summary.json |
| Sprint 7 single-seed AE limitation | Must be cited verbatim from ae_model.py |
| Sprint 7/8 Validation reuse limitation | Must be cited verbatim from sprint8_final_design.md |
| Sprint 8 n=583 caveat | 1 row = 0.1716 pp — must appear in H2 and H3 reporting |
| Sprint 8 verdicts | H-FUSION=FALSE, H-PROT-BACKDOOR=FALSE — must appear in H3 reporting |

No limitation may be paraphrased into a weaker or stronger claim.

---

## 17. Open Decisions for DISCUSSION

The following decisions are PROPOSED DEFAULTS until explicitly locked
by the human Discussion phase. None are approved.

| DD-ID | Decision | Proposed Default | Status |
|-------|----------|-----------------|--------|
| DD-1 | H1 execution: Option A (repro Sprint 6 OOF) vs. Option B (inference on Dev TEST) | **Option B** | PROPOSED — NOT APPROVED |
| DD-2 | Strongest individual classifier identification rule | **DD-2a**: frozen Sprint 5 RF (with unmatched limitation) — or — **DD-2b**: matched per-seed RF inference on Dev TEST | PROPOSED — NOT APPROVED |
| DD-3 | H2 AE threshold | **mean+3sigma, τ=11.160062745213509** (frozen Sprint 8 C06 threshold) | PROPOSED — NOT APPROVED |
| DD-4 | H2 "non-trivial" quantitative criterion | AE detection rate > AE Normal VALIDATION FPR by ≥ X pp; X = ? | **OPEN — X undefined** |
| DD-5 | H3 scope | **Re-presentation of frozen Sprint 8 evidence; no new evaluation** | PROPOSED — NOT APPROVED |
| DD-6 | H3 FPR control delta | Max allowed FPR increase C06 vs. C01 = ? pp | **OPEN — value undefined** |
| DD-7 | H1/H2/H3 exact SUPPORTED/NOT SUPPORTED/INCONCLUSIVE rules | Framework proposed above; exact thresholds undefined | **OPEN** |
| DD-8 | Sprint 9 inference source: base-model checkpoint per-seed availability | Single EXP_BASE_MODELS_V1 set; per-seed = meta-learner seed only; base models identical across seeds | **OPEN — requires confirmation** |

---

## 18. Lifecycle Status

```
PLAN:           DRAFTED
DESIGN:         DRAFTED — PENDING DISCUSSION
DISCUSSION:     NOT STARTED
FINAL DESIGN:   NOT STARTED
IMPLEMENTATION: NOT STARTED
TEST:           NOT STARTED
VALIDATE:       NOT STARTED
FREEZE:         NOT STARTED
```