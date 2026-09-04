# Sprint 13 — Zero-Day Simulation Final Report (EXP_ZERODAY_V1)
**Protocol Version**: V1.4 — FINAL OPERATOR, PREFLIGHT & STATISTICAL-PROVENANCE CORRECTIONS  
**Experiment ID**: `EXP_ZERODAY_V1`  
**Execution Timestamp**: `2026-09-04T15:45:09.520890+00:00`  
**Audit Status**: `READY_FOR_HUMAN_FREEZE_REVIEW`  

---
## 1. Executive Summary
- **Headline Generalization System**: `C06` (OR-logic fusion of Stacking C01 and AE anomaly flag)
- **Headline Generalization Verdict**: **`UNSEEN_CATEGORY_GENERALIZATION_SUPPORTED`**
  - Observed C06 ZDR: **0.9983** (582 / 583)
  - Two-sided 95% Wilson CI: **[0.9903, 0.9997]** (Threshold: $\ge 0.50$ and CI lower $> 0.50$)
- **Fusion Improvement Verdict**: **`FUSION_IMPROVEMENT_NOT_SUPPORTED`**
  - Observed Rescue Gain (Primary Estimand): **0.0000** (0.00 percentage points, $Q_3 = 0$)
  - Practical Threshold: $\text{RescueGain} \ge 0.05$ (Minimum integer $Q_3 \ge 30$) -> **NOT MET**
  - Exact One-Sided Binomial $p$-value: **1.0000e+00** against frozen baseline $p_0 = 0.000625$ ($H_0: p \le p_0$) -> **FAIL_TO_REJECT_H0**
- **Standalone AE-Only H2 Verdict**: **`NOT_SUPPORTED`**
  - Observed AE Detection on Protected Backdoor: **0 / 583** ($Q_1 + Q_3 = 0$)
  - Decision Rule DD-4: $\text{ae\_detected\_count} == 0 \implies \text{NOT\_SUPPORTED}$
  - Role: Standalone hypothesis evaluation, distinct from C01 vs C06 fusion improvement. Exact match with executed Sprint 9 H2 (`EXP_H123_V1`) and Sprint 10 AE Backdoor ablation (`EXP_ABLATION_V1`).
- **C01 Benign FPR Status**: **`C01_FPR_LINEAGE_DIFFERENCE_RECONCILED`**
  - Authoritative C01 value: $\text{FP} = 7,100 / 37,000 \implies \text{FPR} = 0.191892$ (exact match with Sprint 12 frozen seed-42 checkpoint).

---
## 2. Rescue Estimands Architecture
### Primary Rescue Rate
$$\text{all\_sample\_rescue\_rate} = \text{RescueGain} = \frac{Q_3}{583} = \frac{0}{583} = 0.000000$$
*Authoritative Label*: **AE rescue rate among all protected Backdoor samples**  
*Role*: Primary inferential estimand evaluated against the practical 5-percentage-point threshold and the exact binomial test.

### Secondary Conditional Rescue Rate
$$\text{conditional\_rescue\_rate} = \frac{Q_3}{Q_2 + Q_3} = \frac{0}{582} = 0.000000$$
*Authoritative Label*: **AE rescue rate conditional on C01 missing the sample**  
*Role*: Strictly descriptive secondary estimand; does not replace the primary all-sample rescue rate.

---
## 3. Quadrant Decomposition
For the 583 protected Backdoor rows:
- **$Q_1$ (Both Detected)**: 0 (0.00%)
- **$Q_2$ (C01 Detected Only)**: 582 (99.83%)
- **$Q_3$ (AE Rescue - C01 Missed, AE Detected)**: 0 (0.00%)
- **$Q_4$ (Both Missed)**: 1 (0.17%)
- **Total Check**: $Q_1 + Q_2 + Q_3 + Q_4 = 583 == 583$

---
## 4. Comprehensive Metrics Table (Combined Population: n = 37,583)
| System | ZDR (583) | Wilson 95% CI | Macro F1 | Attack F1 | Balanced Acc | FPR | Decision Scope |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **DT** | 0.9897 | [0.9777, 0.9953] | 0.4689 | 0.1003 | 0.8551 | 0.2795 | `DESCRIPTIVE_ONLY` |
| **RF** | 0.9983 | [0.9903, 0.9997] | 0.4946 | 0.1198 | 0.8836 | 0.2310 | `DESCRIPTIVE_ONLY` |
| **SVM** | 0.9897 | [0.9777, 0.9953] | 0.4537 | 0.0912 | 0.8396 | 0.3106 | `DESCRIPTIVE_ONLY` |
| **NN** | 0.9914 | [0.9801, 0.9963] | 0.5437 | 0.1700 | 0.9195 | 0.1524 | `DESCRIPTIVE_ONLY` |
| **Stacking** | 0.9983 | [0.9903, 0.9997] | 0.5173 | 0.1408 | 0.9032 | 0.1919 | `DESCRIPTIVE_ONLY` |
| **AE** | 0.0000 | [0.0000, 0.0065] | 0.4960 | 0.0000 | 0.4997 | 0.0005 | `DESCRIPTIVE_ONLY` |
| **C01** | 0.9983 | [0.9903, 0.9997] | 0.5173 | 0.1408 | 0.9032 | 0.1919 | `DESCRIPTIVE_ONLY` |
| **C06** | 0.9983 | [0.9903, 0.9997] | 0.5171 | 0.1406 | 0.9030 | 0.1922 | `FORMAL_GENERALIZATION_DECISION` |

### C01 Model Lineage Reconciliation
- **Historical Values Preserved**:
  - Sprint 10 Ablation A1 (Seed 42): $\text{FP} = 7,201 / 37,000 \implies \text{FPR} = 0.1946216216$
  - Sprint 12 Stacking (Seed 42): $\text{FP} = 7,100 / 37,000 \implies \text{FPR} = 0.1918918919$
  - Sprint 13 Authoritative Frozen C01: $\text{FP} = 7,100 / 37,000 \implies \text{FPR} = 0.1918918919$
- **Formal Reconciliation Statement**:
  > "The Sprint 10 A1 result (7201 FP) is a historical dynamically-fitted ablation result and is not computationally identical to the canonical frozen stacking system used by Sprint 12/13. Sprint 13 C01 (7100 FP) exactly matches the Sprint 12 frozen seed-42 checkpoint evaluation. Therefore, the 101-FP difference is a documented model-lineage difference, not an unresolved Sprint 13 reproduction error."
- **Status**: **`C01_FPR_LINEAGE_DIFFERENCE_RECONCILED`**

---
## 5. Statistical Methodology & Disclosures
### Practical Effect Criterion
- Threshold: $\text{RescueGain} \ge 0.05$ (at least 5 percentage points)
- Minimum integer $Q_3$: $\lceil 583 \times 0.05 \rceil = 30$ (Observed $Q_3 = 0$)

### Statistical Baseline
- Source: `EXP_AE_V1` frozen benign validation ($7 / 11,200$)
- $p_0 = 0.000625$
- Frozen anomaly threshold: $\tau = 11.160062745213509$

### Statistical Test
- Test: Exact one-sided binomial test against the frozen benign-validation AE alert-rate baseline
- $p$-value: 1.0000e+00 (Significance level: $\alpha = 0.05$)

### Statistical Assumption
- The exact binomial test treats the 583 protected evaluation rows as independent Bernoulli trials under the operational baseline.

### Binding-Criterion Disclosure
> "Because the frozen benign-validation baseline p0 = 0.000625 is very small, the statistical criterion is expected to be satisfied whenever the pre-registered practical RescueGain threshold is met. Consequently, the practical 5-percentage-point threshold is expected to be the binding criterion for SUPPORTED fusion-improvement verdicts in this design. The statistical test is retained as a formal consistency check against the frozen operational baseline, not as an independent second practical-effect threshold."

### Limitation: Network Flow Dependence
> "The exact binomial analysis treats protected rows as independent trials, but network-flow observations may be correlated by attack session, host, time, or behavioral similarity. The nominal p-value should therefore be interpreted under the independence assumption."

---
## 6. Locked Limitations
1. **Single Withheld Family**: This experiment evaluates a single controlled unseen-attack proxy: the Backdoor category. Therefore, its findings cannot be generalized to unseen attacks as a class.
2. **Flow Dependence**: Network-flow observations may exhibit correlation across time or sessions, making nominal binomial $p$-values anti-conservative relative to an independent population model.
3. **Baseline Interpretation**: The frozen AE benign-validation FPR is an operational reference baseline, not a random probability of flagging an unseen Backdoor sample.

---
## 7. Artifact Manifest Summary
- **Predictions**: `predictions/zero_day_backdoor_predictions.csv`, `predictions/benign_control_predictions.csv`, `predictions/combined_evaluation_predictions.csv`
- **Analysis**: `analysis/rescue_cases.csv`, `analysis/missed_cases.csv`, `analysis/detection_overlap.csv`
- **Metrics**: `metrics/zero_day_metrics.csv`, `metrics/c01_c06_statistical_test.csv`, `metrics/preregistered_decisions.json`
- **Plots**: `plots/zero_day_detection_rate.png`, `plots/benign_fpr.png`, `plots/detection_overlap.png`, `plots/c01_vs_c06_detection.png`, `plots/ae_reconstruction_error_benign_vs_backdoor.png`, `plots/stacking_score_benign_vs_backdoor.png`
- **Explainability**: `explainability/ae_rescue_feature_importance.csv`, `explainability/ae_feature_contributions_summary.csv`, `explainability/meta_learner_contributions_backdoor.csv`
- **Validation**: `validation_report.md`, `validation_results.json`
- **Audit**: `quality_review.md`, `metadata.json`, `config.yaml`
