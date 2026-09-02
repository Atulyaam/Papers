# Sprint 8 — EXP_FUSION_V1
## Selection Methodology Discussion

```
Document type:   DISCUSSION — not an implementation plan
Experiment:      EXP_FUSION_V1
Sprint:          8

PLAN:            DRAFTED
DESIGN:          DRAFTED — PENDING DISCUSSION
DISCUSSION:      IN PROGRESS
FINAL DESIGN:    NOT STARTED
IMPLEMENTATION:  NOT STARTED
TEST:            NOT STARTED
VALIDATE:        NOT STARTED
FREEZE:          NOT STARTED
```

> [!IMPORTANT]
> No decisions in this document are approved.
> No implementation has occurred.
> No files, commits, or tags have been created.

---

## Document Purpose

This document records the Sprint 8 discussion of the primary methodological
decision: **how to select one frozen fusion configuration from the 11 candidate
configurations using Normal VALIDATION FPR only.**

It supersedes the earlier "OD-4 lower FPR wins" proposal from Design v2, which
was identified as structurally biased against the OR fusion rule.

---

## Changes Applied in This Revision

| # | Correction | Location |
|---|-----------|---------|
| 1 | Stale priority table in §G marked SUPERSEDED; one canonical ordering defined | §G, §L |
| 2 | OD-4a (aggressive-first) and OD-4b (conservative-first) added as explicit sub-decision | §D, §K |
| 3 | "15 configurations" retired everywhere; 11 unique configs used throughout | All |
| 4 | Document renamed from `implementation_plan.md` to `sprint8_selection_methodology_discussion.md` | Header |
| 5 | OD-4 sub-question merged into one fork: aggressive-first vs conservative-first | §L |

---

## A. The Structural FPR Bias Problem

### Mathematical Identity

For any fixed AE threshold τ and any input row, the three fusion rules
produce predictions where:

```
flag_OR  = supervised_pred OR  ae_flag(τ)    ← always ≥ supervised_pred
flag_AND = supervised_pred AND ae_flag(τ)    ← always ≤ supervised_pred
```

On VALIDATION (all rows are Normal, label=0):

```
FPR(AND, τ)  ≤  FPR(Supervised-only)  ≤  FPR(OR, τ)
```

**This is a mathematical identity, not an approximation.** It holds for every τ
and every VALIDATION set.

### Consequence of "Lower FPR Wins" (Design v2 OD-4 Proposal)

If the primary selection criterion is "lower Validation FPR wins":

- AND is always ranked above Supervised-only
- Supervised-only is always ranked above OR
- **OR can never be selected** unless the AE flags zero Normal rows —
  which only occurs as τ → ∞, reducing OR to Supervised-only

**Conclusion**: "Lower FPR wins" as a primary ranking criterion structurally
eliminates OR before Development TEST is reached, defeating Sprint 8's stated
research question: *"Does the AE signal rescue supervised misses?"*

---

## B. Two Resolution Options

### Option A — FPR as Gate Only *(RECOMMENDED — NOT YET APPROVED)*

**Mechanism**:

1. Apply the OD-3 FPR value as a binary **pass/fail gate**.
   Any configuration where Validation FPR > gate is eliminated.
   FPR magnitude among passing configurations is **not used for ranking**.

2. Among passing configurations, apply a fixed pre-registered priority:
   - **Rule priority**: OR > AND > Supervised-only
   - **Within-rule threshold priority**: see OD-4a vs OD-4b (§D)

3. Select the highest-priority passing configuration.

**Rationale**: Sprint 8's stated question requires OR to have a path to
selection. Option A gives OR that path while respecting the FPR guardrail.
The final verdict still depends entirely on single-shot Development TEST and
Protected Backdoor evaluations, which are uncontaminated by the selection rule.

---

### Option B — Conservative Design (Lower FPR Wins)

**Mechanism**: Rank all passing configurations by Validation FPR ascending.
Ties broken by: Supervised-only > AND > OR.

**Consequence**: OR is never selected. The research question becomes:
*"Does AND with the AE reduce false positives while preserving attack detection?"*
OR results on Development TEST are informational only.

**Rationale**: Maximum conservatism. The only observable pre-Dev-TEST metric
is FPR. Minimizing it is a defensible Neyman-Pearson-style pre-specification.

---

## C. Trade-Off Summary

| Dimension | Option A (Gate Only) | Option B (Lower FPR Wins) |
|-----------|---------------------|--------------------------|
| OR ever selected? | Yes — if it passes the gate | **Never** (mathematical exclusion) |
| Primary research question answered? | **Yes** — AE rescue of missed attacks | Partially — AE as FP-reducer only |
| Conservative bias? | No | Maximum |
| Pre-registration transparency | Must declare "OR preferred" explicitly | FPR-minimization is self-evident |
| Risk of FPR ceiling consumption | Yes — aggressive OR may approach gate | No |
| Sprint 8 scientific value | Higher — tests both rescue and FP risk | Lower — tests FP-reduction only |

---

## D. OD-4 Sub-Decision — Within-Rule Threshold Priority

**This is a new sub-decision created in this discussion revision.**

### Canonical Frozen τ Values (from EXP_AE_V1, official 150-epoch run)

| Threshold | τ | Ordered position |
|-----------|---|-----------------|
| p95 | **0.567386** | Smallest τ — most aggressive AE |
| p99 | 1.512164 | |
| mean+2σ | 7.515109 | ⚠️ outlier-inflated |
| p99.9 | 10.696876 | |
| mean+3σ | **11.160063** | Largest τ — least aggressive AE |

Smaller τ = more rows flagged as anomaly = more aggressive AE signal.
Larger τ = fewer rows flagged = more conservative AE signal.

**Do not describe p99.9 as "stricter than mean+3σ" in any qualitative sense.**
The ordering is purely by τ value: p99.9 (10.70) < mean+3σ (11.16), so p99.9
is more aggressive than mean+3σ.

---

### OD-4a — Aggressive-First *(PROPOSED — NOT YET APPROVED)*

Within each fusion rule, prioritize the **smallest τ** (most aggressive AE) first:

```
p95 > p99 > mean+2σ > p99.9 > mean+3σ
(τ: 0.567 → 1.512 → 7.515 → 10.697 → 11.160)
```

**Effect**: maximizes the AE's rescue capability by selecting the most sensitive
AE threshold that passes the FPR gate.

**Consequence note**:

> "Under Option A with a 5% FPR gate and OD-4a (aggressive-first within-rule
> priority), the p95-threshold OR configuration (C02) will typically operate
> near the maximum permitted Validation false-positive rate. The p95 threshold
> produces approximately 5% Normal-validation FPR by construction. This is a
> deliberate consequence of prioritizing AE informativeness and should be
> weighed against OD-4b, which prefers the least aggressive passing AE threshold."

---

### OD-4b — Conservative-First *(PROPOSED — NOT YET APPROVED)*

Within each fusion rule, prioritize the **largest τ** (least aggressive AE) first:

```
mean+3σ > p99.9 > mean+2σ > p99 > p95
(τ: 11.160 → 10.697 → 7.515 → 1.512 → 0.567)
```

**Effect**: selects the most FPR-conservative AE threshold that still triggers
an AE signal — minimizing AE-induced false positives within the OR or AND rule.

**Consequence note**: Under OD-4b, the selected OR configuration (if OR is
selected) will have very few AE-triggered Normal false positives — but will also
have the weakest AE rescue signal. The AE flags only the most extreme
reconstruction errors.

---

### OD-4 Fork Summary

| Sub-option | Within-rule priority | AE behavior | FPR ceiling risk |
|------------|---------------------|-------------|-----------------|
| **OD-4a** | Aggressive-first (smallest τ) | Maximizes rescue | May approach FPR gate |
| **OD-4b** | Conservative-first (largest τ) | Minimizes AE-FP | Far below FPR gate |

**Both OD-4a and OD-4b are OPEN — NOT YET APPROVED.**

---

## E. 11 Unique Candidate Configurations

**11 unique configurations total**: 1 + 5 + 5.

> [!NOTE]
> The prior "15 configurations" language referred to a conceptual 5-threshold
> × 3-rule cross-product. Because Supervised-only has no AE threshold dependency,
> it contributes exactly **1 unique configuration** (not 5). The cross-product
> produces 1 + (5×2) = **11 unique configurations**. All "15" wording is retired.

| Config ID | Rule | AE Threshold | τ | FPR basis | Caution |
|-----------|------|-------------|---|-----------|---------|
| **C01** | **Supervised-only** | **—** | **—** | **Reference baseline** | — |
| C02 | Supervised OR AE | p95 | 0.567386 | Normal VAL FPR | — |
| C03 | Supervised OR AE | p99 | 1.512164 | Normal VAL FPR | — |
| C04 | Supervised OR AE | mean+2σ | 7.515109 | Normal VAL FPR | ⚠️ outlier-inflated |
| C05 | Supervised OR AE | p99.9 | 10.696876 | Normal VAL FPR | — |
| C06 | Supervised OR AE | mean+3σ | 11.160063 | Normal VAL FPR | ⚠️ outlier-inflated |
| C07 | Supervised AND AE | p95 | 0.567386 | Normal VAL FPR | — |
| C08 | Supervised AND AE | p99 | 1.512164 | Normal VAL FPR | — |
| C09 | Supervised AND AE | mean+2σ | 7.515109 | Normal VAL FPR | ⚠️ outlier-inflated |
| C10 | Supervised AND AE | p99.9 | 10.696876 | Normal VAL FPR | — |
| C11 | Supervised AND AE | mean+3σ | 11.160063 | Normal VAL FPR | ⚠️ outlier-inflated |

**C01 is the supervised-only reference baseline in all outputs.**

⚠️ C04, C06, C09, C11: FPR on Normal VALIDATION is influenced by Sprint 7
extreme Normal rows (row_id 10737 RE≈269, row_id 10731 RE≈269 — legitimate
short/aborted TCP sessions, not filtered). If any of these is selected,
`validation_selection.json` must set `outlier_influenced = true`.

### Priority Table Under Option A (both sub-options shown)

**OD-4a (Aggressive-first within-rule) — full pre-registered rank order:**

| Priority | Config | Rule | Threshold | τ |
|----------|--------|------|-----------|---|
| 1 | C02 | OR | p95 | 0.567 |
| 2 | C03 | OR | p99 | 1.512 |
| 3 | C04 | OR | mean+2σ ⚠️ | 7.515 |
| 4 | C05 | OR | p99.9 | 10.697 |
| 5 | C06 | OR | mean+3σ ⚠️ | 11.160 |
| 6 | C07 | AND | p95 | 0.567 |
| 7 | C08 | AND | p99 | 1.512 |
| 8 | C09 | AND | mean+2σ ⚠️ | 7.515 |
| 9 | C10 | AND | p99.9 | 10.697 |
| 10 | C11 | AND | mean+3σ ⚠️ | 11.160 |
| 11 | C01 | Supervised-only | — | — |

**OD-4b (Conservative-first within-rule) — full pre-registered rank order:**

| Priority | Config | Rule | Threshold | τ |
|----------|--------|------|-----------|---|
| 1 | C06 | OR | mean+3σ ⚠️ | 11.160 |
| 2 | C05 | OR | p99.9 | 10.697 |
| 3 | C04 | OR | mean+2σ ⚠️ | 7.515 |
| 4 | C03 | OR | p99 | 1.512 |
| 5 | C02 | OR | p95 | 0.567 |
| 6 | C11 | AND | mean+3σ ⚠️ | 11.160 |
| 7 | C10 | AND | p99.9 | 10.697 |
| 8 | C09 | AND | mean+2σ ⚠️ | 7.515 |
| 9 | C08 | AND | p99 | 1.512 |
| 10 | C07 | AND | p95 | 0.567 |
| 11 | C01 | Supervised-only | — | — |

> [!CAUTION]
> **The table in §G of the previous discussion draft is SUPERSEDED — DO NOT USE.**
> The tables above are the only canonical priority orderings.
> There is exactly ONE priority table per sub-option (OD-4a and OD-4b above).

---

## F. C01 Supervised-Only as Reference Baseline

C01 has no AE threshold. Its Validation FPR equals the Sprint 6 Logistic
Regression meta-learner's false-positive rate on the 11,200 Normal VALIDATION rows.

**Required treatment in all Sprint 8 outputs:**

1. C01 is the **first row** of `fusion_candidate_results.csv`
2. C01 is the `baseline_config` field in `validation_selection.json`
3. All comparison tables (Development TEST, Protected Backdoor) include C01
   as a reference column regardless of which configuration is selected
4. C01's Validation FPR is computed identically to all other configs

**OD-1a (Corrected Definition)**:

> "The frozen supervised prediction is the output of
> `sklearn.linear_model.LogisticRegression.predict()` applied to the
> Sprint 6 frozen meta-learner. This uses the model's default 0.5 posterior
> probability decision boundary as fitted during Sprint 6.
> **No retuning of this threshold is permitted in Sprint 8.**
> The 0.5 boundary is frozen."

---

## G. Proposed Selection Function

> **STATUS: PROPOSED — NOT YET APPROVED**
> Exact behavior depends on approval of OD-3, OD-4 (Option A/B), and OD-4a/4b.

```
INPUT:
  - 11 candidate configurations (C01–C11)
  - Validation FPR for each (computed on 11,200 Normal VALIDATION rows)
  - OD-3 FPR gate g  [proposed default: 5%]
  - OD-4 rule selection: Option A or Option B
  - OD-4 within-rule: OD-4a (aggressive-first) or OD-4b (conservative-first)

STEP 1 — GATE (OD-3):
  passing = {c ∈ {C01..C11} : FPR(c) ≤ g}
  if passing is empty → apply OD-5 fallback rule

STEP 2 — RANK:
  [Option A]
    Apply fixed pre-registered rule priority: OR > AND > Supervised-only
    Within same rule, apply OD-4a or OD-4b threshold ordering
  [Option B]
    Rank by Validation FPR ascending (lowest FPR = highest priority)
    Ties: Supervised-only > AND > OR

STEP 3 — SELECT:
  selected = highest-priority configuration in passing

STEP 4 — OUTLIER FLAG:
  if selected ∈ {C04, C06, C09, C11}:
    outlier_influenced = true   [mean±sigma selection]
  else:
    outlier_influenced = false

STEP 5 — RECORD (validation_selection.json):
  {
    "n_candidates": 11,
    "fpr_gate": g,
    "n_passing": len(passing),
    "passing_configs": [...],
    "selected_config": selected,
    "selection_option": "A" or "B",
    "within_rule_priority": "OD-4a" or "OD-4b",
    "outlier_influenced": true/false,
    "baseline_config": "C01",
    "baseline_fpr": FPR(C01),
    "fallback_triggered": false
  }

OUTPUT: ONE frozen configuration ID
```

### OD-5 — No-Candidate Fallback

If STEP 1 returns empty (no configuration passes the gate):

**Proposed**: Select C01 (Supervised-only). Record in validation_selection.json:
```json
{
  "fallback_triggered": true,
  "fallback_reason": "No configuration satisfied FPR <= <gate>",
  "selected_config": "C01",
  "selected_by": "OD-5 fallback"
}
```

---

## H. Edge Cases — All OPEN

| Scenario | Proposed handling |
|----------|------------------|
| All 11 pass the gate | Option A: highest OD-4a/4b priority = C02 or C06 |
| Only C01 passes | Select C01; log `only_supervised_passed = true` |
| Only AND configs pass | Apply AND priority (OD-4a or OD-4b sub-priority) |
| Two configs have identical FPR | Rule priority (OR > AND > Supervised-only) decides |
| OR and AND have identical FPR at same threshold | OR wins (rule priority) |
| No config passes | OD-5 fallback: C01 |

All edge-case handling is **OPEN — NOT YET APPROVED**.

---

## I. Preserved Invariants

All previously approved limitations are preserved exactly:

| Invariant | Source |
|-----------|--------|
| VALIDATION = Normal-only = FPR only; no attack metrics on VALIDATION | Design v2 §5 |
| Sequence: VALIDATION selection → freeze ONE config → Dev TEST → Protected Backdoor | Design v2 §8 |
| Dev TEST and Protected Backdoor never feed backward into selection | Design v2 §8 |
| VALIDATION reuse limitation (Sprint 7 AE calibration + Sprint 8 selection) | Design v2 §4 |
| AE scaler-space difference from supervised space | EXP_AE_V1 |
| mean±sigma outlier caveat (row_id 10737/10731, Normal, RST/FIN) | EXP_AE_V1 |
| Normal subgroup protection (RST/FIN rows must not be removed/filtered) | Design v2 §9 |
| Protected Backdoor n=583 → 0.1716 pp/row | Design v2 §11 |
| No learned fusion in v1 | Design v2 §6 |
| No weighted raw-score fusion in v1 (scaler-space limitation) | EXP_AE_V1, Design v2 §6 |
| No retraining of any Sprint 1–7 model | All sprints |
| No modification of frozen Sprint 1–7 artifacts | All sprints |
| No SHAP in Sprint 8 | Design v2 Non-Goals |
| `LogisticRegression.predict()` at frozen 0.5 boundary | OD-1a (above) |
| AE binary rule: `RE > τ` (strict, unless OD-2 overrides) | OD-2 |

---

## J. Metadata Fields Requiring Update After OD Approvals

When Sprint 8 is implemented, `metadata.json` must include:

```yaml
n_candidates: 11             # NOT 15
selection_option: "A" or "B" # OD-4
within_rule_priority: "OD-4a" or "OD-4b"
fpr_gate: <OD-3 approved value>
validation_reuse_limitation: <required text>
```

---

## K. Leakage / Isolation Tests — Updated

`n_candidates` assertions must use **11** (not 15):

```python
# T-CANDIDATE-COUNT
assert len(candidate_configs) == 11

# T-CANDIDATE-IDS
assert set(candidate_ids) == {"C01","C02","C03","C04","C05",
                               "C06","C07","C08","C09","C10","C11"}
```

All other test invariants from Design v2 §14 remain unchanged.

---

## L. Open Decisions — Updated List

All **OPEN — NOT YET APPROVED**.

| OD | Question | Proposed default | Changed in this revision? |
|----|----------|-----------------|--------------------------|
| **OD-1** | Supervised signal: binary prediction vs probability+threshold? | OD-1a: `LogisticRegression.predict()` at frozen 0.5 | Refined definition |
| **OD-2** | AE boundary: `RE > τ` (strict) vs `RE ≥ τ`? | OD-2a: strict `>` | Unchanged |
| **OD-3** | Validation FPR gate = 1%, 5%, or 10%? | 5% (with rationale) | Unchanged |
| **OD-4** | Selection rule: Option A (gate only) or Option B (lower FPR wins)? | **Option A recommended** | Unchanged |
| **OD-4 sub** | Within-rule threshold priority: OD-4a (aggressive-first) or OD-4b (conservative-first)? | **NEW** — no default; both presented | **New in this revision** |
| **OD-5** | No-candidate fallback? | C01 safe fallback | Unchanged |
| **OD-6** | Development TEST metric hierarchy? | Macro-F1 primary | Unchanged |
| **OD-7** | Protected Backdoor: count vs rate as primary report? | Both, with n=583 caveat | Unchanged |
| **OD-8** | Normal subgroup (RST/FIN) false-positive analysis scope? | Per-config subgroup FPR | Unchanged |
| **OD-9** | Confirm 11 unique configurations? | **11** (was "15") | **Updated in this revision** |
| **OD-10** | Retain H-FUSION/H-PROT-BACKDOOR hypotheses or drop? | Either, if explicit | Unchanged |

---

## M. Questions for Human Decision (Final List)

The following must be resolved before FINAL DESIGN:

1. **OD-4**: Option A (gate only, OR preferred) or Option B (lower FPR wins, OR excluded)?

2. **OD-4 sub**: Within-rule threshold priority:
   - **OD-4a** (aggressive-first): `p95 > p99 > mean+2σ > p99.9 > mean+3σ`
     → maximizes AE rescue capability; may approach FPR ceiling
   - **OD-4b** (conservative-first): `mean+3σ > p99.9 > mean+2σ > p99 > p95`
     → minimizes AE-induced FP risk; suppresses AE rescue signal

3. **OD-3**: FPR gate = 1%, 5%, or 10%?

4. **OD-1**: `LogisticRegression.predict()` at frozen 0.5 — approved as the
   supervised signal? Or probability with separately chosen threshold?

5. **OD-5**: No-candidate fallback = C01 (Option B), relax gate (Option A),
   or escalate (Option C)?

6. **OD-9**: Confirm the candidate set is exactly 11 unique configurations
   (C01–C11 as defined in §E)?

7. **OD-10**: Retain Sprint-8-specific hypothesis naming (H-FUSION /
   H-PROT-BACKDOOR), or report without formal hypothesis framing?

---

```
PLAN:            DRAFTED
DESIGN:          DRAFTED — PENDING DISCUSSION
DISCUSSION:      IN PROGRESS — awaiting human decisions on OD-1 through OD-10
FINAL DESIGN:    NOT STARTED
IMPLEMENTATION:  NOT STARTED
TEST:            NOT STARTED
VALIDATE:        NOT STARTED
FREEZE:          NOT STARTED
```
