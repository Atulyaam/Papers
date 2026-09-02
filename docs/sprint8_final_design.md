# Sprint 8 — EXP_FUSION_V1
## Final Design

```
PLAN:            DRAFTED
DESIGN:          DRAFTED
DISCUSSION:      COMPLETE
FINAL DESIGN:    APPROVED — READY FOR IMPLEMENTATION
IMPLEMENTATION:  NOT STARTED
TEST:            NOT STARTED
VALIDATE:        NOT STARTED
FREEZE:          NOT STARTED
```

---

## Lifecycle History

| Phase | Status | Date |
|-------|--------|------|
| PLAN | DRAFTED | 2026-09-02 |
| DESIGN v1 | DRAFTED | 2026-09-02 |
| DESIGN v2 | DRAFTED | 2026-09-02 |
| DISCUSSION | COMPLETE | 2026-09-02 |
| FINAL DESIGN | DRAFTED — READY FOR REVIEW | 2026-09-02 |
| FINAL DESIGN Amendment 1 | Exploratory all-11 analysis added | 2026-09-02 |
| **FINAL DESIGN** | **APPROVED** | **2026-09-02** |

---

## 1. Objective

> "Does combining the frozen supervised stacking branch with the frozen
> benign-only Autoencoder improve detection of difficult/unseen traffic
> while maintaining acceptable false-positive behavior on Normal traffic?"

Sprint 8 is the **fusion + final evaluation sprint**. No new model is trained.
All predictions come from frozen Sprint 6 and Sprint 7 artifacts.

### Formal Hypotheses (OD-10: APPROVED)

**H-FUSION**: The selected fusion configuration (from C01–C11, per the
approved OD-4/OD-5 selection protocol) achieves higher Development TEST
Macro-F1 than C01 (Supervised-only baseline) without exceeding the 5%
Normal-VALIDATION FPR gate.

> Note: C01 is the approved OD-5 fallback. H-FUSION is evaluable even if
> C01 is selected — the result is trivially H-FUSION=FALSE (no fusion
> improvement over baseline), which is a valid scientific outcome.

**H-PROT-BACKDOOR**: The selected fusion configuration detects a higher
proportion of Protected Backdoor rows than C01.

> Sprint 6 H1 terminology is kept entirely separate. Sprint 8 results are
> never called "H1" and do not belong to the Sprint 6 H1 framework.

---

## 2. Frozen Upstream Artifacts

| Sprint | Experiment | Freeze commit | Tag |
|--------|-----------|--------------|-----|
| 5 | EXP_BASE_MODELS_V1 | `36b6ef1` | `EXP_BASE_MODELS_V1` |
| 6 | EXP_OOF_STACK_V1 | `c946fa6` | `EXP_OOF_STACK_V1` |
| **7** | **EXP_AE_V1** | frozen | **`EXP_AE_V1`** |

**None of the above may be retrained, re-tuned, or modified.**

### Sprint 6 Canonical Supervised Checkpoint (Locked)

Sprint 8 supervised inference uses **exactly**:

| Field | Value |
|-------|-------|
| Experiment | EXP_OOF_STACK_V1 |
| Seed | **42** |
| Checkpoint | frozen Logistic Regression meta-learner (seed-42 run) |
| Inference call | `LogisticRegression.predict()` at frozen 0.5 boundary |

> [!CAUTION]
> **Do NOT:**
> - Average predictions across Sprint 6 seeds (seed-42, seed-123, seed-2024)
> - Choose a seed using Sprint 8 data
> - Retune the supervised prediction threshold
> - Retrain the Sprint 6 meta-learner

### Sprint 7 AE Frozen Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | 75→12→6→12→75 |
| best_epoch | 133 |
| AE seed | 42 |
| Scaler | Normal AE-fit only (40,320 rows) |
| Threshold candidates | p95, p99, p99.9, mean+2σ, mean+3σ |

---

## 3. Approved Decisions (All Locked)

### OD-1 — Supervised Signal ✅ APPROVED: OD-1a

> **The frozen supervised prediction is the output of
> `sklearn.linear_model.LogisticRegression.predict()` applied to the
> EXP_OOF_STACK_V1 seed-42 frozen meta-learner checkpoint, using the
> model's default 0.5 posterior probability decision boundary as fitted
> during Sprint 6. No retuning of this boundary is permitted in Sprint 8.
> Sprint 8 must NOT average across Sprint 6 seeds and must NOT select a
> seed using Sprint 8 data.**

Applies to: C01 (baseline), C02–C06 (OR), C07–C11 (AND).

---

### OD-2 — AE Binary Signal ✅ APPROVED: OD-2a

> **AE anomaly = `RE > τ` (strict greater-than).
> AE normal = `RE ≤ τ`.
> The boundary value τ itself is classified as Normal.**

---

### OD-3 — Validation FPR Gate ✅ APPROVED: 5%

> **A configuration is accepted if and only if its Validation FPR ≤ 5%.**

Rationale (locked): The Sprint 7 p95 threshold produces approximately 5%
Normal-validation exceedance by construction. A 5% ceiling provides a
simple interpretable false-positive guardrail while still allowing tighter
thresholds to pass.

---

### OD-4 — Selection Rule ✅ APPROVED: Option A

> **Validation FPR is used only as a pass/fail gate (OD-3).
> Among passing configurations, FPR magnitude is NOT used for ranking.
> Rule priority: OR > AND > Supervised-only.**

Rationale (locked): Sprint 8's stated research question requires OR to
have a path to selection. "Lower FPR wins" is a mathematical identity that
permanently excludes OR. Option A preserves OR's eligibility while respecting
the false-positive constraint.

---

### OD-4 sub — Within-Rule Threshold Priority ✅ APPROVED: OD-4b Conservative-first

> **Within each fusion rule, configurations are prioritized by descending τ
> (least aggressive AE signal first):**
>
> `mean+3σ > p99.9 > mean+2σ > p99 > p95`
> `(τ: 11.160 > 10.697 > 7.515 > 1.512 > 0.567)`

Rationale (locked): Preserves OR's ability to reach held-out evaluation while
avoiding automatic selection of the most aggressive AE threshold near the
full FPR ceiling. Under OD-4a (aggressive-first), the p95 OR configuration
would almost always be selected when OR passes — operating near the 5% ceiling
by construction. OD-4b selects the least aggressive passing threshold within
each rule instead.

---

### OD-5 — No-Candidate Fallback ✅ APPROVED: C01

> **If no configuration satisfies the 5% FPR gate, select C01
> (Supervised-only) and record the fallback explicitly in
> validation_selection.json.**

---

### OD-6 — Development TEST Metric Hierarchy ✅ APPROVED

Ordered primary to secondary:

1. **Macro-F1** ← primary
2. Weighted-F1
3. Balanced Accuracy
4. FPR (Normal class)
5. Recall (Attack class)
6. FNR (Attack class)
7. Confusion matrix

---

### OD-7 — Protected Backdoor Reporting ✅ APPROVED

Report **both**:
- Detected count and missed count (raw)
- Detection rate (percentage)

With mandatory caveat in all reports:

> **"Protected Backdoor evaluation contains 583 rows. One additional detected
> or missed row changes the detection rate by approximately 1/583 =
> 0.1716 percentage points. Differences of only a few rows must not be
> overinterpreted as strong evidence of generalization differences."**

---

### OD-8 — Normal Subgroup Analysis ✅ APPROVED

Perform per-configuration FPR analysis on the protected Sprint 7 Normal
subgroup (short/aborted TCP sessions: RST/FIN state, low bytes/packets).

Required outputs per configuration:
- `subgroup_fp_count`
- `subgroup_fpr` (subgroup false positives / subgroup size)
- `overall_fpr`
- `subgroup_fpr_vs_overall_fpr` ratio
- Comparison against C01 baseline

> **Do not alter, remove, filter, clip, or relabel any member of this subgroup.**

---

### OD-9 — Candidate Count ✅ APPROVED: 11 unique configurations

> **The candidate set contains exactly 11 unique configurations.**

Breakdown: 1 Supervised-only + 5 OR + 5 AND.

The conceptual 5×3 cross-product produces 1 + (5×2) = 11 unique
configurations because Supervised-only has no AE-threshold dependency.
All "15 configurations" wording is permanently retired.

---

### OD-10 — Hypothesis Naming ✅ APPROVED

> **Retain Sprint-8-specific hypothesis names: H-FUSION and H-PROT-BACKDOOR.**
> Sprint 6 H1 terminology remains separate and is never reused for Sprint 8.

---

## 4. Canonical Frozen τ Values

| Threshold | τ | τ rank (ascending) | Caution |
|-----------|---|-------------------|---------|
| p95 | **0.567386** | 1 (most aggressive) | — |
| p99 | 1.512164 | 2 | — |
| mean+2σ | 7.515109 | 3 | ⚠️ outlier-inflated |
| p99.9 | 10.696876 | 4 | — |
| mean+3σ | **11.160063** | 5 (least aggressive) | ⚠️ outlier-inflated |

⚠️ mean+2σ and mean+3σ are influenced by Sprint 7 extreme Normal VALIDATION
rows (row_id 10737 RE≈269.09, row_id 10731 RE≈269.03 — short/aborted TCP
sessions, RST/FIN, label=Normal). Not filtered. If selected, must be flagged.

---

## 5. Canonical 11-Configuration Table

| Config ID | Rule | AE Threshold | τ | Outlier caution |
|-----------|------|-------------|---|----------------|
| **C01** | **Supervised-only** | **—** | **—** | — |
| C02 | Supervised OR AE | p95 | 0.567386 | — |
| C03 | Supervised OR AE | p99 | 1.512164 | — |
| C04 | Supervised OR AE | mean+2σ | 7.515109 | ⚠️ |
| C05 | Supervised OR AE | p99.9 | 10.696876 | — |
| C06 | Supervised OR AE | mean+3σ | 11.160063 | ⚠️ |
| C07 | Supervised AND AE | p95 | 0.567386 | — |
| C08 | Supervised AND AE | p99 | 1.512164 | — |
| C09 | Supervised AND AE | mean+2σ | 7.515109 | ⚠️ |
| C10 | Supervised AND AE | p99.9 | 10.696876 | — |
| C11 | Supervised AND AE | mean+3σ | 11.160063 | ⚠️ |

**C01 is the supervised-only reference baseline in all output artifacts.**

---

## 6. Canonical Selection Function (Fully Resolved)

All OD values are now locked. The selection function is fully determined:

```
INPUT:
  configs    = [C01, C02, ..., C11]          # 11 unique configurations
  fpr(c)     = Validation FPR for config c   # computed on 11,200 Normal rows
  gate       = 0.05                          # OD-3: 5% gate

STEP 1 — GATE:
  passing = {c : fpr(c) <= 0.05}
  if passing is empty:
    selected = C01
    fallback_triggered = true
    GOTO STEP 5

STEP 2 — RULE PRIORITY (OD-4, Option A):
  # Rule priority: OR > AND > Supervised-only
  or_passing   = {c in passing : rule(c) == "OR"}
  and_passing  = {c in passing : rule(c) == "AND"}
  sup_passing  = {C01} if C01 in passing else {}

STEP 3 — WITHIN-RULE PRIORITY (OD-4b, conservative-first):
  # Prioritize largest τ (least aggressive AE) within each rule
  # τ order: mean+3σ > p99.9 > mean+2σ > p99 > p95

  if or_passing is not empty:
    selected = argmax_τ(or_passing)    # largest τ in passing OR configs
  elif and_passing is not empty:
    selected = argmax_τ(and_passing)   # largest τ in passing AND configs
  else:
    selected = C01

STEP 4 — OUTLIER FLAG:
  if selected in {C04, C06, C09, C11}:
    outlier_influenced = true
  else:
    outlier_influenced = false

STEP 5 — RECORD:
  write validation_selection.json:
    n_candidates             = 11
    fpr_gate                 = 0.05
    n_passing                = len(passing)
    passing_configs          = [c.id for c in passing]
    fpr_values               = {c.id: fpr(c) for c in all_configs}
    selected_config          = selected.id
    selection_rule           = "OD-4 Option A + OD-4b conservative-first"
    outlier_influenced       = outlier_influenced
    fallback_triggered       = fallback_triggered
    baseline_config          = "C01"
    baseline_fpr             = fpr(C01)
    h_fusion_hypothesis      = "H-FUSION"
    h_prot_backdoor_hyp      = "H-PROT-BACKDOOR"

OUTPUT: ONE frozen configuration ID → proceed to Development TEST
```

### Pre-Registered Full Priority Table (OD-4 Option A + OD-4b)

The highest-priority **passing** configuration is selected:

| Priority | Config | Rule | Threshold | τ | ⚠️ |
|----------|--------|------|-----------|---|----|
| **1** | **C06** | **OR** | mean+3σ | 11.160 | ⚠️ |
| 2 | C05 | OR | p99.9 | 10.697 | — |
| 3 | C04 | OR | mean+2σ | 7.515 | ⚠️ |
| 4 | C03 | OR | p99 | 1.512 | — |
| 5 | C02 | OR | p95 | 0.567 | — |
| 6 | C11 | AND | mean+3σ | 11.160 | ⚠️ |
| 7 | C10 | AND | p99.9 | 10.697 | — |
| 8 | C09 | AND | mean+2σ | 7.515 | ⚠️ |
| 9 | C08 | AND | p99 | 1.512 | — |
| 10 | C07 | AND | p95 | 0.567 | — |
| 11 | C01 | Supervised-only | — | — | — |

**This is the only canonical priority table. No other ordering exists.**

---

## 7. Edge Cases (All Resolved)

| Scenario | Resolution |
|----------|-----------|
| All 11 pass the gate | C06 selected (highest priority: OR + mean+3σ) |
| All OR pass, all AND pass | C06 selected (first in priority table) |
| Only C01 passes | C01 selected; `only_baseline_passed = true` logged |
| Only AND configurations pass | Highest-τ AND that passes (C11 → C10 → C09...) |
| Two OR configs have identical FPR | Larger τ wins (OD-4b conservative-first) |
| No configuration passes gate | C01 selected; `fallback_triggered = true` logged |
| mean+3σ OR passes gate | C06 selected; `outlier_influenced = true` flagged |

---

## 8. Data Access (Confirmed)

| Split | Rows | Access | Computable |
|-------|------|--------|-----------|
| TRAIN | 162,395 | Inference only | — |
| **VALIDATION** | **11,200 (Normal)** | **FPR gate and selection** | **FPR only** |
| Development TEST | 81,749 | Single-shot after STEP 7 | Full metrics (OD-6) |
| Protected Backdoor | 583 | Single-shot after STEP 7 | Detection rate (OD-7) |
| Excluded Backdoor | — | **FORBIDDEN** | — |

### Validation FPR-Only Invariant

**Methodology invariant — not an OD.**

VALIDATION contains only Normal-class rows (label=0). Only FPR-type quantities
are computable. Macro-F1, recall, balanced accuracy, attack precision, and
attack detection rate are **not computable** on VALIDATION and must not appear
in any Validation-stage output.

### Validation Reuse Limitation

> [!WARNING]
> **VALIDATION is reused for Sprint 7 AE threshold calibration AND Sprint 8
> fusion-rule selection. Both are selection-stage uses, not final held-out
> evaluation. This reuse is within the frozen data-isolation rules but is
> an explicit limitation and must appear in metadata.json, quality_review.md,
> and the final report.**

### Fixed Evaluation Sequence

```
VALIDATION selection → freeze ONE config → Development TEST → Protected Backdoor
```

Development TEST and Protected Backdoor results must **never** feed backward
into configuration selection. This is a fixed invariant, not subject to override.

---

## 9. Fusion Rule Definitions (Fully Specified)

For each row r at evaluation time:

```python
# OD-1a: supervised prediction
sup_pred(r) = frozen_lr_meta.predict([meta_features(r)])[0]  # 0 or 1

# OD-2a: AE anomaly flag
ae_flag(r, τ) = 1 if reconstruction_error(r) > τ else 0     # strict >

# Fusion rules:
C01: final(r) = sup_pred(r)
COR: final(r) = int(sup_pred(r) == 1 or ae_flag(r, τ) == 1)
CAND: final(r) = int(sup_pred(r) == 1 and ae_flag(r, τ) == 1)
```

AE reconstruction error: `RE(r) = mean((r_scaled - AE(r_scaled))²)`
over 75 features, mean not sum (same definition as Sprint 7).

---

## 10. Metrics Specification

### Validation (Normal-only, 11,200 rows)

For each configuration c:
```
fpr(c) = sum(final(r, c) == 1 for r in VAL_NORMAL) / 11200
fp_count(c) = sum(final(r, c) == 1 for r in VAL_NORMAL)
subgroup_fpr(c) = fp_count on RST/FIN subgroup / subgroup_size
```

### Development TEST (81,749 rows — OD-6)

Computed once for the selected configuration only:
1. Macro-F1
2. Weighted-F1
3. Balanced Accuracy
4. FPR (Normal class)
5. Recall (Attack class)
6. FNR (Attack class)
7. Full confusion matrix

Also compute for C01 as reference.

### Protected Backdoor (583 rows — OD-7)

```
detected_count = sum(final(r) == 1 for r in PROT)
missed_count   = 583 - detected_count
detection_rate = detected_count / 583

# Mandatory caveat in all reports:
# "1 row = 1/583 = 0.1716 pp; small differences not interpretable as
#  strong generalization evidence."
```

### Normal Subgroup Analysis (OD-8)

Target subgroup: Normal VALIDATION rows matching Sprint 7 structural pattern
(state ∈ {RST, FIN}, low bytes/packets, proto=tcp).

For all 11 configurations:
```
subgroup_fp_count(c)
subgroup_fpr(c)
subgroup_fpr_vs_overall_fpr_ratio(c)
delta_vs_C01 = subgroup_fpr(c) - subgroup_fpr(C01)
```

---

## 11. Normal Subgroup Protection (Invariant)

> [!IMPORTANT]
> **Do not modify, remove, filter, clip, relabel, or exclude any member of
> the Sprint 7 short/aborted TCP Normal subgroup:**
> - State ∈ {RST, FIN}, proto=tcp, low bytes/packets
> - row_id 10737 (RE≈269.09) and row_id 10731 (RE≈269.03)
> - attack_cat=Normal, label=0

This subgroup must appear in:
- `fusion_candidate_results.csv` (subgroup FPR column)
- `validation_selection.json` (subgroup FPR for selected config)
- `quality_review.md` (subgroup FPR analysis section)
- `exploratory/all_11_development_test_metrics.csv` (subgroup FPR for all 11 configs)

---

## 12. Post-Selection Exploratory All-11 Analysis — INFORMATIONAL ONLY

> [!NOTE]
> **FINAL DESIGN AMENDMENT 1 — Added 2026-09-02**
> This section protects against a near-null primary result caused by
> OD-4 Option A + OD-4b conservative-first + the mean+3σ threshold (C06).
> **No approved OD decision is changed by this amendment.**

### Purpose

Under the approved selection protocol, C06 (OR + mean+3σ, τ=11.160) is the
highest-priority OR configuration under OD-4b. If C06 passes the 5% FPR gate
and is selected, its AE component is expected to activate very rarely — because
mean+3σ is the least aggressive threshold, influenced by Sprint 7 extreme Normal
rows. This may produce a "near-null fusion effect" where C06 behaves nearly
identically to C01 on Development TEST — not because the AE has no rescue
capability, but because the selected threshold is too conservative to express it.

To characterize this without contaminating the primary result, Sprint 8 computes
an exploratory all-11 analysis **only after** primary evaluations are frozen.

### Rules

1. The official primary selection protocol is **unchanged**.
2. The selected configuration is the **ONLY** configuration used for
   the primary H-FUSION and H-PROT-BACKDOOR verdicts.
3. **Timing**: Computed ONLY AFTER STEP 8 (Dev TEST) and STEP 9 (Protected
   Backdoor) are both complete and written to disk.
4. **Scope**: Dev TEST metrics + Protected Backdoor metrics for all 11 configs.
5. **No backward use**: Results MUST NOT affect selection, threshold choice,
   fusion-rule choice, H-FUSION verdict, H-PROT-BACKDOOR verdict, or any OD.
6. C01 is the reference baseline in the exploratory table.
7. All exploratory outputs go **only** to `exploratory/`:

```
results/fusion/EXP_FUSION_V1/exploratory/
├── all_11_development_test_metrics.csv
└── all_11_protected_backdoor_metrics.csv
```

### C06 Near-Null Interpretation Note

**Required in `quality_review.md` if C06 is the selected configuration:**

> "If the selected configuration is C06 (OR + mean+3σ), its AE component is
> expected to activate very rarely on Normal traffic because the frozen threshold
> is highly conservative and was influenced by the Sprint 7 extreme Normal rows.
> Therefore a near-null fusion effect must not automatically be interpreted as
> evidence that the AE branch has no useful rescue capability. Exploratory all-11
> results are provided only to characterize this behavior post-hoc and are not
> used to alter the frozen primary result."

**Expected C06 behavior** (document regardless of outcome):

> "C06 uses mean+3σ (τ=11.160), the least aggressive AE threshold. Outside
> high-reconstruction-error cases comparable to the Sprint 7 extreme Normal rows
> (RE≈269), C06's AE flag will be 0 for almost all rows, and C06 will behave
> effectively identically to C01. This is a known and expected property of the
> conservative threshold, not a model failure."

### Required Columns

`all_11_development_test_metrics.csv`:
`config_id`, `rule`, `threshold`, `tau`, `macro_f1`, `weighted_f1`,
`balanced_acc`, `fpr`, `recall`, `fnr`, `is_primary_selected`,
`outlier_influenced`, `informational_only=true`

`all_11_protected_backdoor_metrics.csv`:
`config_id`, `rule`, `threshold`, `tau`, `detected_count`, `missed_count`,
`detection_rate`, `n_prot=583`, `pp_per_row=0.1716`,
`is_primary_selected`, `informational_only=true`

---

## 13. Execution Order (Frozen)


```
STEP 0   Verify frozen Sprint 1–7 hashes, registry states, checkpoint integrity
         Confirm EXP_MI_V1_1, EXP_BASE_MODELS_V1, EXP_OOF_STACK_V1,
         EXP_AE_V1 all FROZEN

STEP 1   Build inference adapters (no retraining)
         - Supervised: LR.predict() on meta-features (DT/RF/SVM/NN probs)
         - AE: load frozen ae_final.pt + ae_scaler.joblib, compute RE

STEP 2   Run focused fusion tests (T-HASH, T-FROZEN-MODELS, T-75-FEATURES,
         T-NO-RETRAIN, T-VAL-NORMAL, T-VAL-FPR-ONLY, T-CANDIDATE-COUNT,
         T-CANDIDATE-DETERMINISTIC, T-NO-DEVTEST-PREFREEZE,
         T-NO-PROT-PREFREEZE, T-SUBGROUP-PRESENT, T-THRESHOLD-MATCH,
         T-SIGMA-CAUTION, T-ONE-CONFIG, T-NO-RESELECT)

STEP 3   Run full regression (878+ tests, zero failures)

STEP 4   Enumerate all 11 candidate configurations deterministically

STEP 5   Compute Validation FPR for all 11 configurations
         Compute subgroup FPR for all 11 configurations (OD-8)
         Write fusion_candidate_results.csv

STEP 6   Apply the canonical selection function:
         - OD-3 gate: FPR ≤ 5%
         - OD-4 Option A: rule priority OR > AND > Supervised-only
         - OD-4b: within-rule conservative-first (largest τ)
         - OD-5 fallback: C01 if no config passes
         - Flag outlier-influenced if mean±sigma selected
         Write validation_selection.json

STEP 7   Freeze ONE configuration
         Record in validation_selection.json

STEP 8   Run Development TEST exactly ONCE on frozen configuration
         Also compute C01 reference results on Development TEST
         Write predictions.csv, metrics.json, confusion_matrix.json
         DO NOT re-run after seeing results

STEP 9   Run Protected Backdoor evaluation exactly ONCE
         Apply n=583 caveat to all reported figures
         Write predictions.csv, metrics.json, confusion_matrix.json
         DO NOT re-run after seeing results

STEP 10  Generate comparison analyses (primary only):
         - supervised_vs_fusion.csv (C01 vs selected config, Dev TEST)
         - rf_vs_stack_vs_fusion.csv
           (Sprint 5 RF OOF / Sprint 6 Stack OOF / Sprint 8 Dev TEST)
         - Normal subgroup FPR analysis (all 11 configs, VALIDATION)
         NOTE: STEP 10 uses only the already-computed PRIMARY results for
         the single frozen selected configuration and the C01 reference.
         STEP 10 MUST NOT use, preview, or derive results from the
         STEP 11 exploratory all-11 analysis.

STEP 11  [EXPLORATORY — INFORMATIONAL ONLY]
         After primary single-shot evaluations (STEP 8 + STEP 9) are
         complete and frozen, compute Development TEST and Protected
         Backdoor results for ALL 11 candidate configurations.
         Write ONLY to exploratory/ directory.
         MUST NOT modify: validation_selection.json, selected_config,
         H-FUSION verdict, H-PROT-BACKDOOR verdict, or any OD decision.
         C01 serves as reference baseline in exploratory tables.

STEP 12  Generate quality_review.md
         Must include: H-FUSION verdict, H-PROT-BACKDOOR verdict,
         Validation reuse limitation, n=583 caveat,
         outlier_influenced flag if applicable,
         Normal subgroup analysis, all approved OD decisions,
         EXPLORATORY section with INFORMATIONAL ONLY header,
         C06 near-null interpretation note (if C06 is selected)

STEP 13  STOP — await human review and freeze approval
         Do NOT create freeze commit/tag until approved
```

---

## 14. Artifact Plan

```
results/fusion/EXP_FUSION_V1/
├── config.yaml
├── metadata.json
├── quality_review.md
├── validation_report.json
├── runtime_report.json
│
├── validation/
│   ├── fusion_candidate_results.csv      # all 11 configs: FPR + subgroup FPR
│   └── validation_selection.json        # selected config + full provenance
│
├── development_test/                    # PRIMARY — selected config only
│   ├── predictions.csv
│   ├── metrics.json
│   └── confusion_matrix.json
│
├── protected_backdoor/                  # PRIMARY — selected config only
│   ├── predictions.csv
│   ├── metrics.json
│   └── confusion_matrix.json
│
├── comparison/
│   ├── supervised_vs_fusion.csv
│   └── rf_vs_stack_vs_fusion.csv
│
└── exploratory/                         # INFORMATIONAL ONLY — post-hoc, all 11
    ├── all_11_development_test_metrics.csv
    └── all_11_protected_backdoor_metrics.csv
```

---

## 15. Required Metadata Fields

`metadata.json` must contain:

```yaml
experiment_id:               EXP_FUSION_V1
sprint:                      8
status:                      IMPLEMENTED   # when frozen
upstream_experiments:        [EXP_MI_V1_1, EXP_BASE_MODELS_V1,
                               EXP_OOF_STACK_V1, EXP_AE_V1]

# Approved decisions
od_1:                        OD-1a — LR.predict() at frozen 0.5 boundary,
                             EXP_OOF_STACK_V1 seed-42 checkpoint only
s6_canonical_checkpoint:     EXP_OOF_STACK_V1 / seed=42 / frozen LR meta-learner
s6_no_seed_averaging:        true  # do not average across Sprint 6 seeds
od_2:                        OD-2a — RE > tau (strict greater-than)
od_3:                        0.05  # 5% FPR gate
od_4:                        Option A — gate-only, OR > AND > Supervised-only
od_4_sub:                    OD-4b — conservative-first (largest tau)
od_5:                        C01 fallback
od_6:                        Macro-F1 primary
od_7:                        counts + rate + n=583 caveat
od_8:                        RST/FIN subgroup FPR per config
od_9:                        11 unique configurations
od_10:                       H-FUSION / H-PROT-BACKDOOR

n_candidates:                11
selected_config:             <from validation_selection.json>
fpr_gate:                    0.05
development_test_runs:       1
protected_backdoor_runs:     1
protected_backdoor_n:        583
protected_backdoor_caveat:   "1 row = 1/583 = 0.1716 percentage points"

# Exploratory analysis (Amendment 1)
exploratory_all_11_computed: true
exploratory_note:            "INFORMATIONAL ONLY — post-hoc, computed after
                              primary single-shot evaluations are frozen.
                              Not used for selection, not used for H-FUSION
                              or H-PROT-BACKDOOR verdicts, not used for
                              retraining or threshold revision."

# Inherited limitations
validation_reuse_limitation: "<required text — see §8>"
scaler_space_limitation:     "<from EXP_AE_V1>"
threshold_caution:           "<from EXP_AE_V1>"
outlier_note:                "<from EXP_AE_V1 — rows 10737/10731>"
```

---

## 16. Leakage / Isolation Test Plan

| Test ID | Assertion |
|---------|-----------|
| T-HASH | Sprint 1–7 SHA-256 hashes match frozen values |
| T-S6-CANONICAL-CHECKPOINT | Sprint 8 supervised inference uses the frozen EXP_OOF_STACK_V1 seed-42 meta-learner checkpoint; no other seed or averaged predictions permitted |
| T-FROZEN-MODELS | No gradient updates on DT/RF/SVM/NN/LR/AE |
| T-75-FEATURES | Input to all models is exactly 75 features |
| T-NO-RETRAIN | No `fit()` or `train()` called on any Sprint 1–7 model |
| T-VAL-NORMAL | VALIDATION subset contains only label=0 rows |
| T-VAL-FPR-ONLY | No attack-class metrics computed from VALIDATION |
| T-CANDIDATE-COUNT | `len(candidate_configs) == 11` |
| T-CANDIDATE-IDS | `set(ids) == {"C01",...,"C11"}` |
| T-CANDIDATE-DETERMINISTIC | Same seed → same config enumeration order |
| T-NO-DEVTEST-PREFREEZE | Development TEST not opened before STEP 7 |
| T-NO-PROT-PREFREEZE | Protected Backdoor not opened before STEP 7 |
| T-NO-DEVTEST-TUNE | Development TEST result does not modify selection |
| T-NO-PROT-TUNE | Protected Backdoor result does not modify selection |
| T-SUBGROUP-PRESENT | RST/FIN Normal subgroup rows not removed/filtered |
| T-THRESHOLD-MATCH | Sprint 8 τ values exactly match Sprint 7 frozen values |
| T-SIGMA-CAUTION | `outlier_influenced` flag set if mean±sigma config selected |
| T-ONE-CONFIG | Exactly one configuration frozen after STEP 7 |
| T-NO-RESELECT | No post-freeze re-selection occurs |
| T-NO-EXPLORATORY-RESELECT | Exploratory all-11 results cannot modify `validation_selection.json`, `selected_config`, H-FUSION verdict, or H-PROT-BACKDOOR verdict |
| T-EXPLORATORY-SEPARATION | Exploratory files written only to `exploratory/`; no exploratory file in `development_test/` or `protected_backdoor/` |
| T-EXPLORATORY-ORDER | Exploratory all-11 computed only AFTER STEP 8 + STEP 9 are both complete and frozen |
| T-RE-DEFINITION | RE uses mean (not sum) over 75 features |
| T-AE-STRICT | Boundary condition: RE == τ classified as Normal (not anomaly) |

---

## 17. Invariants (Cannot Be Changed During Implementation)

| Invariant | Rule |
|-----------|------|
| Validation = Normal-only = FPR only | No attack-class metrics from VALIDATION |
| Evaluation sequence | VAL → freeze → Dev TEST → Protected Backdoor → exploratory |
| No backward leakage | Dev TEST / Protected Backdoor results never affect selection |
| 11 configurations only | No additions, no removals |
| Single-shot PRIMARY evaluations | Dev TEST: 1 run; Protected Backdoor: 1 run — primary, selected config only |
| Exploratory is post-hoc | All-11 exploratory computed AFTER primary evaluations are frozen |
| Exploratory is INFORMATIONAL ONLY | Cannot alter selection, OD decisions, verdicts, or thresholds |
| Normal subgroup protection | No removal, filtering, relabeling |
| No retraining | No Sprint 1–7 model may be updated |
| No learned fusion | No new classifier introduced |
| No raw-score arithmetic | AE RE not arithmetically combined with LR probability |
| No SHAP | Out of scope for Sprint 8 |

---

## 18. Non-Goals

| ❌ Non-Goal |
|------------|
| Retrain DT/RF/SVM/NN |
| Retrain Sprint 6 LR meta-learner |
| Retrain Sprint 7 AE |
| Change the 75-feature set |
| Modify frozen Sprint 1–7 artifacts |
| Use Development TEST for tuning |
| Use Protected Backdoor for tuning |
| Remove/filter/relabel Normal subgroups |
| Introduce learned fusion |
| Introduce weighted raw-score fusion |
| Perform SHAP |
| Deployment / application work |

---

## 19. Reporting Separation

| Sprint | Unit | Result |
|--------|------|--------|
| 6 | 3-seed OOF on TRAIN | Macro-F1 mean=0.9472, std=0.0003 |
| 7 | Normal-VAL calibration | 5 threshold candidates |
| **8** | **Held-out evaluation** | **Dev TEST Macro-F1 + Backdoor detection** |

Sprint 8 evaluation is **not called H1** and does not belong to the Sprint 6
H1 framework.

Exploratory all-11 results are clearly labelled **INFORMATIONAL ONLY** in all
reporting contexts and are never presented as primary Sprint 8 findings.

---

```
PLAN:            DRAFTED
DESIGN:          DRAFTED
DISCUSSION:      COMPLETE
FINAL DESIGN:    DRAFTED — READY FOR REVIEW
IMPLEMENTATION:  NOT STARTED
TEST:            NOT STARTED
VALIDATE:        NOT STARTED
FREEZE:          NOT STARTED
```
