# Sprint 8 — EXP_FUSION_V1
## Fusion + Final Evaluation Plan & Design v2

```
Status:
  PLAN:            DRAFTED
  DESIGN:          DRAFTED — PENDING DISCUSSION
  DISCUSSION:      PENDING
  FINAL DESIGN:    NOT STARTED
  IMPLEMENTATION:  NOT STARTED
  TEST:            NOT STARTED
  VALIDATE:        NOT STARTED
  FREEZE:          NOT STARTED

Document:      Sprint 8 PLAN + DESIGN v2
Experiment:    EXP_FUSION_V1
Sprint:        8
Date:          2026-09-02
Preceding:     EXP_AE_V1 (Sprint 7) — FROZEN
```

---

## Changes from v1

| Issue | v1 Problem | v2 Resolution |
|-------|-----------|---------------|
| I-1 | Validation used for both AE calibration (Sprint 7) and fusion selection (Sprint 8) — reuse not documented | Explicit **Validation Reuse Limitation** section added |
| I-2 | Validation metrics incorrectly included attack recall / Macro-F1 | **Validation = FPR only**. Macro-F1 and recall removed from validation stage. |
| I-3 | FPR constraint (5%) treated as approved | OD-3 created. 5% explicitly **PROPOSED — NOT YET APPROVED** with full rationale and alternatives |
| I-4 | Tie-break rule was implicit or missing | OD-4 created. Priority order **PROPOSED — NOT YET APPROVED** |
| I-5 | No-candidate fallback not specified | OD-5 created with Options A/B/C, all **PROPOSED — NOT YET APPROVED** |
| I-6 | Score-level fusion (raw RE + probability) not explicitly blocked | Explicit **Score-Level Fusion Caution** section. Binary/decision-level only in v1. |
| I-7 | Sprint 6 H1 naming carried into Sprint 8 | Renamed to **H-FUSION / H-PROT-BACKDOOR**. Hypothesis retention itself is OD-10. |
| I-8 | Protected Backdoor n=583 caveat absent | Explicit n=583 caveat (0.1716 pp/row) in Metrics and Non-Goals. |
| + | AE threshold caution for mean+2σ/mean+3σ | Carried forward with **Validation selection flagging rule** |
| + | Normal subgroup protection not formalized | Explicit **Normal Subgroup Protection** invariant |
| + | 15-candidate framework unclear | Formally defined as 5 thresholds × 3 fusion rules = 15 |
| + | Development TEST ordering implicit | Formally defined as a **fixed methodology invariant** (not an OD) |
| + | OD list incomplete | OD-1 through OD-10 fully enumerated |

---

## 1. Objective

**Core question:**

> "Does combining the frozen supervised stacking branch with the frozen
> benign-only Autoencoder improve detection of difficult/unseen traffic
> while maintaining acceptable false-positive behavior on Normal traffic?"

Sprint 8 is the **fusion + final evaluation sprint**. It does not train any new
model. It evaluates whether the AE anomaly signal from Sprint 7 adds useful
information to the supervised stacking signal from Sprint 6, using a frozen
held-out evaluation protocol.

Sprint 8 does **NOT** produce a new metric for the supervised branch alone —
the Sprint 6 OOF Macro-F1 (mean=0.9472, std=0.0003) is the pre-existing
supervised baseline.

---

## 2. Frozen Upstream Artifacts

| Sprint | Experiment | Status | Key Artifact |
|--------|-----------|--------|-------------|
| 1 | Dataset & splits | FROZEN | TRAIN/VAL/TEST SHAs |
| 2 | Preprocessing | FROZEN | OHE mapping, encoders |
| 3 | Feature engineering | FROZEN | — |
| 4 | MI feature selection | FROZEN | EXP_MI_V1_1 |
| 5 | Base models | FROZEN | EXP_BASE_MODELS_V1 |
| 6 | OOF stacking | FROZEN | EXP_OOF_STACK_V1 |
| **7** | **Benign-only AE** | **FROZEN** | **EXP_AE_V1** |

**None of the above may be retrained, re-tuned, or modified.**

### Sprint 7 AE Frozen Parameters (Carried Forward)

| Parameter | Value |
|-----------|-------|
| Architecture | 75→12→6→12→75 |
| best_epoch | 133 |
| max_epochs | 150 |
| AE seed | 42 |
| Scaler | StandardScaler on AE-fit Normal TRAIN (40,320 rows) |
| Threshold candidates | p95, p99, p99.9, mean+2σ, mean+3σ |
| Primary threshold | **DEFERRED — Sprint 8 Discussion (OD-3)** |
| n=563 caveat | Inherited for mean+2σ/mean+3σ |

---

## 3. Data Access Boundaries

### TRAIN (`data/splits/train.csv` — 162,395 rows)
- **Allowed**: inference only (produce predictions using frozen models)
- **Forbidden**: retraining, model selection from TRAIN labels

### VALIDATION (`data/splits/validation.csv` — 11,200 rows, Normal only)
- **Allowed**: fusion candidate FPR diagnostics, AE threshold/rule selection, validation-stage filtering
- **Computable**: FPR only (Normal-class false positives)
- **NOT computable**: Macro-F1, weighted F1, balanced accuracy, attack recall, attack precision, attack detection rate
- **Forbidden**: Development TEST access during this stage

### Development TEST (`data/splits/development_test.csv` — 81,749 rows)
- **Role**: single-shot held-out evaluation AFTER fusion configuration is frozen
- **Computable**: full metric suite (Macro-F1, weighted F1, precision, recall, balanced accuracy, FPR, FNR, confusion matrix)
- **Forbidden**: using Development TEST results to choose, tune, or revise the fusion configuration

### Protected Backdoor (`data/splits/protected_unseen_attack.csv` — 583 rows)
- **Role**: final unseen-attack evaluation AFTER fusion configuration is frozen
- **Forbidden**: any use for selection, tuning, or threshold choice

### Excluded Backdoor (`data/splits/excluded_train_backdoor.csv`)
- **Forbidden**: do not access

---

## 4. Validation Reuse Limitation

> [!WARNING]
> **VALIDATION is being reused for two distinct model-selection purposes:
> Sprint 7 AE threshold calibration and Sprint 8 fusion-rule/threshold
> selection. Both uses are selection-stage activities, not final held-out
> evaluation. This reuse remains within the frozen data-isolation rules,
> but repeated selection reuse of VALIDATION is an explicit limitation and
> must not be treated as an independent final evaluation set.**

This limitation must appear in:
- Sprint 8 `metadata.json`
- `quality_review.md`
- final report / provenance

---

## 5. Validation FPR-Only Rule

**This is a methodology invariant — NOT an open decision.**

VALIDATION contains only Normal-class rows (11,200 rows, label=0).

At the Validation stage:
- Only FPR-type quantities are computable
- Macro-F1, weighted F1, recall, balanced accuracy, attack precision, attack detection rate are **NOT computable** on VALIDATION
- Any claim about attack recall or Macro-F1 at the Validation stage is **a methodology error**

This applies to all 15 candidate configurations.

---

## 6. Fusion Signal — PROPOSED (NOT YET APPROVED)

### Supervised Signal
- **Source**: frozen Sprint 6 Logistic Regression meta-learner
- **Candidate representations**: see **OD-1**
- No retraining permitted

### AE Signal
- **Source**: frozen Sprint 7 AE reconstruction error
- **Binary representation**: `RE > τ → anomaly; RE ≤ τ → normal`
- **Exact τ**: see **OD-2**
- **Exact binary mapping**: see **OD-2**

### Score-Level Fusion Caution

> [!CAUTION]
> **Do NOT directly add or arithmetically combine the raw AE reconstruction
> error with the supervised probability.**
>
> Sprint 7 established: AE operates in a Normal-TRAIN-scaled feature space
> distinct from the full-TRAIN-scaled space used by DT/RF/SVM/NN.
> Direct arithmetic combination of raw scores is not assumed valid.
>
> Sprint 8 v1 fusion must remain decision-level (binary) only.
> Score-level fusion requires explicit approval as a separate future extension.

---

## 7. Fusion Candidate Framework — PROPOSED (NOT YET APPROVED)

### Fusion Rules

| ID | Rule | Description |
|----|------|-------------|
| A | **Supervised-only** | Use only the frozen supervised meta-learner prediction |
| B | **Supervised OR AE** | Flag as attack if supervised OR AE signals attack |
| C | **Supervised AND AE** | Flag as attack only if both supervised AND AE signal attack |

### AE Threshold Candidates (Exact Sprint 7 Set)

| ID | Threshold | τ (official 150-ep) | FPR on Normal VAL | Caution |
|----|-----------|--------------------|--------------------|---------|
| T1 | p95 | 0.567386 | 5.00% | None |
| T2 | p99 | 1.512164 | 1.00% | None |
| T3 | p99.9 | 10.696876 | 0.11% | None |
| T4 | mean+2σ | 7.515109 | 0.21% | **Outlier-inflated** |
| T5 | mean+3σ | 11.160063 | 0.06% | **Outlier-inflated** |

Do not add or remove threshold candidates.
Primary threshold is **not predetermined** — see OD-3.

### 15 Fixed Candidate Configurations

| Config ID | Fusion Rule | AE Threshold |
|-----------|------------|-------------|
| C01 | Supervised-only | — |
| C02 | Supervised OR AE | p95 (T1) |
| C03 | Supervised OR AE | p99 (T2) |
| C04 | Supervised OR AE | p99.9 (T3) |
| C05 | Supervised OR AE | mean+2σ (T4) ⚠️ |
| C06 | Supervised OR AE | mean+3σ (T5) ⚠️ |
| C07 | Supervised AND AE | p95 (T1) |
| C08 | Supervised AND AE | p99 (T2) |
| C09 | Supervised AND AE | p99.9 (T3) |
| C10 | Supervised AND AE | mean+2σ (T4) ⚠️ |
| C11 | Supervised AND AE | mean+3σ (T5) ⚠️ |

> Note: Supervised-only (C01) produces the same prediction regardless of
> AE threshold, giving **1 + 5×2 = 11 unique configurations**, but the
> framework is enumerated as 15 for completeness where Rule A appears
> in all threshold slots.

> [!IMPORTANT]
> **OD-9** governs whether the candidate set is exactly
> {Supervised-only, OR, AND}. The 15-configuration table above is
> **PROPOSED — NOT YET APPROVED**.

⚠️ = mean+2σ and mean+3σ configurations inherit the Sprint 7 outlier-sensitivity caveat.

### Inherited Threshold Caution

> [!WARNING]
> **FPR computed for mean+2σ and mean+3σ configurations inherits the Sprint 7
> outlier-sensitivity caveat (row_id 10737 RE≈269.09, row_id 10731 RE≈269.03 —
> short/aborted TCP sessions, RST/FIN, Normal-labelled).**
>
> A configuration passing the Validation FPR filter with either of these
> thresholds must be explicitly flagged as outlier-influenced in
> `validation_selection.json`.

---

## 8. Validation Selection Protocol — PROPOSED (NOT YET APPROVED)

**Fixed sequence (not an OD):**

```
VALIDATION (Normal-only, 11,200 rows)
  ↓
Compute FPR for all 15 candidate configurations
  ↓
Apply pre-registered FPR acceptance constraint  ← OD-3
  ↓
Apply deterministic tie-break                   ← OD-4
  ↓
If no candidate passes → fallback rule          ← OD-5
  ↓
Freeze ONE fusion configuration
  ↓
Development TEST (single-shot)
  ↓
Protected Backdoor (single-shot)
```

Development TEST results must **NOT** feed backward into selection.
This ordering is a **fixed methodology invariant**.

### All-Tie / Multiple-Candidate Edge Cases

**OPEN — NOT YET APPROVED**

| Scenario | Proposed handling |
|----------|------------------|
| All 15 pass FPR constraint | Apply OD-4 tie-break → select single winner |
| Only Supervised-only passes | Select Supervised-only (safe fallback by construction) |
| Multiple configurations have identical FPR | Apply OD-4 fixed priority: Supervised-only > AND > OR |
| OR and AND have identical FPR | OD-4 priority: AND wins |
| No candidate passes | Apply OD-5 fallback rule |

These handling rules require **explicit approval** in FINAL DESIGN before use.

---

## 9. Normal Subgroup Protection

**This is a fixed methodology invariant — NOT an open decision.**

> [!IMPORTANT]
> **Do not modify, remove, filter, clip, relabel, or exclude any Normal
> subgroup to improve reported metrics.**

Specifically protected from Sprint 7:
- Short/aborted TCP sessions (RST/FIN state, low bytes/packets)
- Normal-labelled traffic (attack_cat=Normal, label=0)
- row_id 10737 (RE≈269.09), row_id 10731 (RE≈269.03)

Sprint 8 **must measure** whether fusion increases false positives on this subgroup.
This is part of **OD-8** (subgroup false-positive analysis).

---

## 10. Open Discussion Decisions (OD-1 through OD-10)

All decisions below are **OPEN — NOT YET APPROVED**.
No implementation may proceed on any OD without explicit FINAL DESIGN approval.

---

### OD-1 — Supervised Signal Representation

**Question**: What exact form of the Sprint 6 Logistic Regression output is used in fusion?

| Option | Description | Implication |
|--------|-------------|-------------|
| OD-1a | Frozen attack class **prediction** (binary, 0/1) | Simple; OR/AND logic is natural |
| OD-1b | Frozen attack **probability** (continuous [0,1]) with a threshold | Requires a probability threshold decision |

Notes:
- No retraining is permitted under either option
- If OD-1b is chosen, the probability threshold itself becomes an additional parameter (potential for leakage if tuned on Validation)
- OD-1a is simpler and avoids an additional threshold parameter

**PROPOSED (not approved)**: OD-1a (binary prediction)

---

### OD-2 — AE Binary Signal Representation

**Question**: What exact binary mapping is applied to the AE reconstruction error?

| Option | Description |
|--------|-------------|
| OD-2a | `RE > τ → 1 (anomaly); RE ≤ τ → 0 (normal)` |
| OD-2b | `RE ≥ τ → 1 (anomaly); RE < τ → 0 (normal)` |

Notes:
- Difference matters only at the exact boundary value τ
- OD-2a (strict greater-than) is proposed as default for reproducibility

**PROPOSED (not approved)**: OD-2a (strict greater-than)

---

### OD-3 — Validation FPR Acceptance Constraint

**Question**: What is the maximum allowable FPR on Normal VALIDATION for a configuration to be accepted?

| Proposed value | Rationale |
|---------------|-----------|
| **5% (proposed default)** | The Sprint 7 p95 threshold produces approximately 5% Normal-validation exceedance by construction. A 5% ceiling provides a simple interpretable false-positive guardrail while still allowing tighter thresholds (p99, p99.9) to pass when they do not introduce excessive Normal false positives. The 5% value is a **proposed selection constraint, not an automatically optimal operating point**. |
| 1% | Stricter guardrail; p95-based OR configuration would fail |
| 10% | Looser; allows more configurations through |

**Alternative discussion values**: 1%, 5%, 10%

**PROPOSED (not approved)**: 5%

---

### OD-4 — Deterministic Tie-Break Among FPR-Acceptable Configurations

**Question**: If multiple configurations pass OD-3, which is selected?

**Proposed priority (not approved)**:
1. **Lower Validation FPR wins** — the configuration with fewest Normal false positives
2. **If FPR is numerically tied**, apply fixed rule priority:

```
Supervised-only
    >
Supervised AND AE
    >
Supervised OR AE
```

**Rationale**: This ordering moves from most conservative (fewest additional anomaly
assumptions) toward most permissive. Supervised-only requires no AE-derived
false-positive mechanism. AND requires both signals to agree. OR adds the most AE-driven
false positives.

**PROPOSED (not approved)**

---

### OD-5 — No-Candidate Fallback Rule

**Question**: What happens if no configuration satisfies the OD-3 FPR constraint?

| Option | Description | Risk |
|--------|-------------|------|
| **Option A** | Relax the FPR constraint to a pre-registered next tier and rerun the Validation filter | Requires pre-registration of relaxation tiers |
| **Option B** | Default to Supervised-only as safe fallback (introduces no AE false-positive mechanism) | Ignores AE signal entirely |
| **Option C** | Do not select automatically; report all 15 configurations and escalate for explicit human decision | Requires human intervention |

**PROPOSED (not approved)**: Option B (Supervised-only safe fallback)

---

### OD-6 — Final Metric Hierarchy for Development TEST

**Question**: What is the primary metric hierarchy for Development TEST evaluation, and how are ties or ambiguous improvements reported?

Proposed hierarchy (not approved):
1. Macro-F1 (primary)
2. Weighted-F1
3. Balanced Accuracy
4. FPR (Normal class)
5. Attack class Recall
6. Confusion matrix (full)

---

### OD-7 — Protected Backdoor Reporting Format and n=583 Interpretation

**Question**: How are Protected Backdoor results reported and interpreted given n=583?

**Fixed caveat (not subject to OD — always present)**:
> "Protected Backdoor evaluation contains 583 rows. One additional detected
> or missed row changes the detection rate by approximately 1/583 =
> 0.1716 percentage points. Differences of only a few rows must therefore
> not be overinterpreted as strong evidence of generalization differences."

**OD-7 sub-questions**:
- Should raw counts (detected/missed) be the primary report, or rate (percentage)?
- What difference in detection rate is considered practically significant?
- How should near-identical detection rates across configurations be reported?

---

### OD-8 — Normal Subgroup False-Positive Analysis

**Question**: How is the Sprint 7 short/aborted TCP Normal subgroup analyzed in Sprint 8?

Proposed analysis (not approved):
- Identify the Normal-VALIDATION rows matching the Sprint 7 structural pattern (RST/FIN state, low bytes)
- Compute per-configuration FPR on this subgroup separately from overall FPR
- Report whether fusion increases false positives on this specific subgroup vs supervised-only
- Flag any configuration where the subgroup FPR materially exceeds the overall FPR

---

### OD-9 — Final Candidate Set Composition

**Question**: Is the initial v1 candidate set exactly {Supervised-only, Supervised OR AE, Supervised AND AE}?

- No additional fusion rules (learned, weighted, neural) are permitted in v1
- Candidate set proposed as {A=Supervised-only, B=OR, C=AND}
- Any expansion requires explicit discussion approval

**PROPOSED (not approved)**: exactly the 3 rules × 5 thresholds = 15 configurations

---

### OD-10 — Sprint 8-Specific Hypotheses: Retain or Drop

**Question**: Does Sprint 8 use formal Sprint 8-specific H-prefixed hypotheses (H-FUSION, H-PROT-BACKDOOR), or does it report without formal hypothesis framing?

> [!NOTE]
> It is acceptable for the Sprint 8 discussion to decide **not** to use formal
> H-prefixed hypotheses. If hypotheses are dropped, that decision must be recorded
> explicitly rather than left ambiguous.
>
> If hypotheses are retained, they **must** be named as Sprint 8-specific (e.g.,
> H-FUSION, H-PROT-BACKDOOR). **Do NOT reuse Sprint 6's H1 terminology.**

---

## 11. Metrics Specification

### Validation (Normal-only, 11,200 rows)

| Metric | Allowed |
|--------|---------|
| FPR (false-positive rate) | ✅ |
| False-positive count | ✅ |
| Threshold diagnostics | ✅ |
| Macro-F1 | ❌ |
| Weighted F1 | ❌ |
| Balanced Accuracy | ❌ |
| Attack recall | ❌ |
| Attack precision | ❌ |

### Development TEST (81,749 rows, full class distribution)

| Metric | Allowed |
|--------|---------|
| Macro-F1 | ✅ |
| Weighted F1 | ✅ |
| Precision (per class and macro) | ✅ |
| Recall (per class and macro) | ✅ |
| Balanced Accuracy | ✅ |
| FPR | ✅ |
| FNR | ✅ |
| Confusion matrix | ✅ |

### Protected Backdoor (583 rows, attack-only)

| Metric | Allowed |
|--------|---------|
| Backdoor detection rate / recall | ✅ |
| Detected count | ✅ |
| Missed count | ✅ |
| Confusion counts (where applicable) | ✅ |

> [!WARNING]
> **n=583 caveat**: One additional detected or missed row changes the detection
> rate by approximately **1/583 = 0.1716 percentage points**. Differences of
> only a few rows must not be overinterpreted as strong evidence of
> generalization differences.

---

## 12. Reporting Separation (Sprint 6 / 7 / 8)

| Sprint | Scope | Primary Result |
|--------|-------|---------------|
| Sprint 6 | 3-seed OOF Macro-F1 (TRAIN) | mean=0.9472, std=0.0003 |
| Sprint 7 | Benign-only AE, Normal-VAL threshold calibration | 5 candidate thresholds |
| **Sprint 8** | **Frozen-system held-out evaluation** | **Development TEST Macro-F1 + Protected Backdoor detection** |

**Sprint 8 evaluation is NOT called "H1"** and does not belong to the Sprint 6
H1 framework. It is a separate, final held-out evaluation.

---

## 13. Execution Order

```
STEP 0:  Verify frozen Sprint 1–7 artifacts, hashes, registry states,
         and checkpoint integrity.

STEP 1:  Build inference adapters only (no retraining).

STEP 2:  Run focused fusion tests.

STEP 3:  Run full regression (878+ tests, no failures allowed).

STEP 4:  Construct all 15 fixed candidate configurations deterministically.

STEP 5:  Run Validation-only FPR diagnostics for all 15 configurations.

STEP 6:  Apply the pre-registered Validation selection function:
         - OD-3 FPR filter
         - OD-4 tie-break
         - OD-5 fallback (if needed)
         - Flag mean±sigma configurations as outlier-influenced
           (if selected)

STEP 7:  Freeze ONE fusion configuration.
         Write validation_selection.json.

STEP 8:  Run Development TEST exactly once on the frozen configuration.
         Do not re-run after seeing results.

STEP 9:  Run Protected Backdoor evaluation exactly once.
         Apply n=583 caveat.

STEP 10: Generate comparison analyses:
         - Supervised-only vs Fusion (Macro-F1, FPR, FNR)
         - RF vs Stack vs Fusion comparison
         - Normal subgroup false-positive analysis (OD-8)

STEP 11: Generate quality_review.md.

STEP 12: STOP for human review and freeze approval.
```

---

## 14. Leakage / Isolation Test Plan

Tests to be written before implementation:

| Test | Description |
|------|-------------|
| T-HASH | Sprint 1–7 artifact SHA-256 hashes match frozen values |
| T-FROZEN-MODELS | Frozen model weights not refit during Sprint 8 |
| T-75-FEATURES | Input to all models is exactly 75 features from EXP_MI_V1_1 |
| T-NO-RETRAIN | No gradient updates applied to DT/RF/SVM/NN/LR-meta/AE |
| T-VAL-NORMAL | VALIDATION subset used contains only label=0 rows |
| T-VAL-FPR-ONLY | No attack-class metrics computed from VALIDATION |
| T-CANDIDATE-COUNT | Exactly 15 candidate configurations enumerated |
| T-CANDIDATE-DETERMINISTIC | Same seed → same candidate ordering |
| T-NO-DEVTEST-PREFREEZE | Development TEST not opened before STEP 7 complete |
| T-NO-PROT-PREFREEZE | Protected Backdoor not opened before STEP 7 complete |
| T-NO-DEVTEST-TUNE | Development TEST result does not modify selection |
| T-NO-PROT-TUNE | Protected Backdoor result does not modify selection |
| T-SUBGROUP-PRESENT | Normal subgroup (RST/FIN) rows not removed/filtered |
| T-THRESHOLD-MATCH | Sprint 8 thresholds exactly match Sprint 7 frozen values |
| T-SIGMA-CAUTION | mean±sigma caution flag present in validation_selection.json when applicable |
| T-ONE-CONFIG | Exactly one fusion configuration frozen |
| T-NO-RESELECT | Post-freeze re-selection does not occur |

---

## 15. Artifact Plan

```
results/fusion/EXP_FUSION_V1/
│
├── config.yaml
├── metadata.json
├── quality_review.md
├── validation_report.json
├── runtime_report.json
│
├── validation/
│   ├── fusion_candidate_results.csv     # FPR for all 15 configurations
│   └── validation_selection.json       # Selected config + rationale + OD outcomes
│                                        # (flags outlier-influenced configs if applicable)
│
├── development_test/
│   ├── predictions.csv
│   ├── metrics.json
│   └── confusion_matrix.json
│
├── protected_backdoor/
│   ├── predictions.csv
│   ├── metrics.json
│   └── confusion_matrix.json
│
└── comparison/
    ├── supervised_vs_fusion.csv         # Macro-F1, FPR, FNR: supervised vs fusion
    └── rf_vs_stack_vs_fusion.csv        # Sprint 5 RF / Sprint 6 Stack / Sprint 8 Fusion
```

**None of these files will be created until FINAL DESIGN is approved.**

---

## 16. Non-Goals

| Non-Goal | Explicitly Excluded |
|----------|-------------------|
| ❌ Retrain DT/RF/SVM/NN | No changes to Sprint 5 models |
| ❌ Retrain Sprint 6 LR meta-learner | No changes to Sprint 6 artifacts |
| ❌ Retrain Sprint 7 AE | No changes to Sprint 7 artifacts |
| ❌ Rerun MI feature selection | EXP_MI_V1_1 is frozen |
| ❌ Change the 75-feature set | Frozen |
| ❌ Modify Sprint 1–7 frozen artifacts | Frozen |
| ❌ Use Development TEST for tuning | Evaluation-only, single-shot |
| ❌ Use Protected Backdoor for tuning | Evaluation-only, single-shot |
| ❌ Remove/filter/relabel Normal subgroups | Fixed invariant |
| ❌ Modify short/aborted TCP Normal rows | Fixed invariant |
| ❌ Perform SHAP or feature attribution | Out of scope |
| ❌ Introduce learned fusion in v1 | Requires explicit future approval |
| ❌ Introduce weighted raw-score fusion in v1 | Blocked by scaler-space limitation |
| ❌ Deployment or application work | Out of scope |

---

## 17. Unresolved Questions for Discussion

The following must be resolved before FINAL DESIGN is written:

1. **OD-1**: Supervised signal = binary prediction or probability + threshold?
2. **OD-2**: Strict `>` or `≥` at the AE threshold boundary?
3. **OD-3**: FPR constraint = 1%, 5%, or 10%? (5% proposed)
4. **OD-4**: Approved priority ordering for tie-break?
5. **OD-5**: No-candidate fallback = Option A, B, or C?
6. **OD-6**: Approved metric hierarchy for Development TEST?
7. **OD-7**: Primary reporting unit for Protected Backdoor (count vs rate)?
8. **OD-8**: Exact scope and output format of Normal subgroup analysis?
9. **OD-9**: Candidate set confirmed as exactly {Supervised-only, OR, AND}?
10. **OD-10**: Retain H-FUSION / H-PROT-BACKDOOR hypotheses, or report without formal hypothesis framing?

---

## 18. Provenance / Metadata Fields (Proposed)

The following fields must appear in Sprint 8 metadata artifacts when generated:

```
experiment_id:             EXP_FUSION_V1
sprint:                    8
status:                    IMPLEMENTED (when frozen)
upstream_experiments:      [EXP_MI_V1_1, EXP_BASE_MODELS_V1,
                            EXP_OOF_STACK_V1, EXP_AE_V1]
validation_reuse_limitation: <required — see §4>
scaler_space_limitation:   <inherited from EXP_AE_V1>
threshold_caution:         <inherited from EXP_AE_V1>
outlier_note:              <inherited from EXP_AE_V1 — rows 10737/10731>
n_candidates:              15
selected_config:           <frozen after STEP 7>
selection_fpr_constraint:  <OD-3 approved value>
tie_break_rule:            <OD-4 approved value>
development_test_runs:     1 (single-shot)
protected_backdoor_runs:   1 (single-shot)
protected_backdoor_n:      583
protected_backdoor_caveat: "1 row = 0.1716 pp"
```

---

## Status

```
PLAN:            DRAFTED
DESIGN:          DRAFTED — PENDING DISCUSSION
DISCUSSION:      PENDING
FINAL DESIGN:    NOT STARTED
IMPLEMENTATION:  NOT STARTED
TEST:            NOT STARTED
VALIDATE:        NOT STARTED
FREEZE:          NOT STARTED
```
