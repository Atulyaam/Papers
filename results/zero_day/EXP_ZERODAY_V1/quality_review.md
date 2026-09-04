# Quality Review: Sprint 13 — Zero-Day Simulation (EXP_ZERODAY_V1)

**Protocol Version**: V1.4 — FINAL OPERATOR, PREFLIGHT & STATISTICAL-PROVENANCE CORRECTIONS  
**Execution Timestamp**: 2026-09-04T15:45:09.520890+00:00  
**Audit Status**: `READY_FOR_HUMAN_FREEZE_REVIEW`

---

## 1. Zero-Training & Methodological Integrity
- **Training Operations Executed**: 0 (strictly zero training or fitting).
- **Recalibration Operations Executed**: 0 (frozen thresholds $\tau = 11.160062745213509$ and $0.50$ maintained without alteration).
- **Zero-Day Data Isolation**: No Backdoor data was accessible during feature selection, model training, or threshold calibration.
- **Population Manifest**: 583 protected Backdoor rows; 37,000 benign control rows; 37,583 combined evaluation rows.

---

## 2. Standalone AE-Only H2 Verdict
- **Standalone AE-Only H2 Verdict**: **`NOT_SUPPORTED`**
- **Reason**: $\text{ae\_detected\_count} = Q_1 + Q_3 = 0$ / 583, satisfying locked Decision Rule DD-4 ($\text{ae\_detected\_count} == 0 \implies \text{NOT\_SUPPORTED}$).
- **Distinction**: This is a standalone AE-only hypothesis evaluation and is strictly distinct from the C01 vs C06 fusion-improvement verdict (`FUSION_IMPROVEMENT_NOT_SUPPORTED`).
- **Historical Cross-Check**: Exactly matches historical executed Sprint 9 H2 (`results/evaluation/EXP_H123_V1/h2_results.json`: `ae_detected_count = 0 / 583`) and Sprint 10 AE-only Backdoor evaluation (`results/ablation/EXP_ABLATION_V1/protected_backdoor_results.json`: `AE_detected = 0 / 583`).

---

## 3. Historical Protected-Backdoor Cross-Check (Sprint 10 vs Sprint 13)
- **Sprint 10 (`EXP_ABLATION_V1`)**:
  - A1 detected: 582 / 583 (99.83%)
  - A6 detected: 582 / 583 (99.83%)
  - AE-only detected: 0 / 583 (0.00%)
- **Sprint 13 (`EXP_ZERODAY_V1`)**:
  - C01 detected ($Q_1 + Q_2$): 582 / 583 (99.83%)
  - C06 detected ($Q_1 + Q_2 + Q_3$): 582 / 583 (99.83%)
  - AE detected ($Q_1 + Q_3$): 0 / 583 (0.00%)
- **Cross-Check Verdict**: **`EXACT_MATCH`**

---

## 4. C01 False Positive Rate (FPR) Lineage Reconciliation (Task 3)
- **Historical Values Preserved**:
  - **Sprint 10 Ablation A1 (Seed 42)**: $\text{FP} = 7,201 / 37,000 \implies \text{FPR} = 0.1946216216$
  - **Sprint 12 Reproduced Stacking (Seed 42)**: $\text{FP} = 7,100 / 37,000 \implies \text{FPR} = 0.1918918919$
  - **Sprint 13 Authoritative Frozen C01 (Seed 42)**: $\text{FP} = 7,100 / 37,000 \implies \text{FPR} = 0.1918918919$ (0 row differences vs Sprint 12)
- **Authoritative Identity**: Sprint 13's authoritative C01 value is strictly the frozen-checkpoint result ($\text{FP} = 7,100$, $\text{FPR} = 0.191892$).
- **Lineage Reconciliation Statement**:
  > "The Sprint 10 A1 result (7201 FP) is a historical dynamically-fitted ablation result and is not computationally identical to the canonical frozen stacking system used by Sprint 12/13. Sprint 13 C01 (7100 FP) exactly matches the Sprint 12 frozen seed-42 checkpoint evaluation. Therefore, the 101-FP difference is a documented model-lineage difference, not an unresolved Sprint 13 reproduction error."
- **Provenance Root Cause**: In Sprint 10 (`scripts/run_ablation.py`), `A1_FULL_STACK` dynamically fitted base models on full TRAIN and refitted a Logistic Regression meta-learner on the fly (`lr.fit()`), yielding 7,201 FPs. In Sprint 12 (`EXP_FINAL_REPRO_V1`), `A1_FULL_STACK` was classified as `NOT_REPRODUCED` because refitting was forbidden under zero-training rules. Sprint 12 reproduced the canonical frozen checkpoints (`EXP_OOF_STACK_V1/seed_42/meta_learner.joblib`), producing 7,100 FPs. Sprint 13 evaluates these exact same frozen checkpoints.
- **Reconciliation Status**: **`C01_FPR_LINEAGE_DIFFERENCE_RECONCILED`**

---

## 5. Statistical Calibration and Limitations
1. **Operational Reference Baseline**: $p_0 = 0.000625$ ($7 / 11,200$) is a frozen benign-validation operational baseline established in `EXP_AE_V1`.
2. **Non-Random Interpretation**: $p_0$ is not a chance probability or random rate of flagging an unseen Backdoor sample.
3. **Binding Criterion Disclosure**: "Because the frozen benign-validation baseline p0 = 0.000625 is very small, the statistical criterion is expected to be satisfied whenever the pre-registered practical RescueGain threshold is met. Consequently, the practical 5-percentage-point threshold is expected to be the binding criterion for SUPPORTED fusion-improvement verdicts in this design. The statistical test is retained as a formal consistency check against the frozen operational baseline, not as an independent second practical-effect threshold."
4. **Baseline Consistency Check**: The exact one-sided binomial test is retained as a formal baseline comparison.
5. **Primary Rescue Denominator**: Primary rescue uses prospective all-sample denominator $n = 583$ ($\text{RescueGain} = Q_3 / 583$).
6. **Secondary Descriptive Rescue Rate**: A conditional rescue rate is reported descriptively using the C01-missed subset ($Q_3 / (Q_2 + Q_3)$).
7. **Independence Assumption**: The exact binomial test treats protected rows as independent Bernoulli trials under the specified operational baseline.
8. **Flow/Session Dependence Limitation**: "The exact binomial analysis treats protected rows as independent trials, but network-flow observations may be correlated by attack session, host, time, or behavioral similarity. The nominal p-value should therefore be interpreted under the independence assumption."

---

## 6. Preflight and Validation Gate Audit
- **Preflight Gates**: 35/35 checks passed (including Sprint 12 freeze prerequisite gate `ZD-PREREQ-S12`, hard gate `ZD-PF-33`, and operator check `ZD-PF-34`).
- **Validation Gates**: 44/44 validation gates passed (`ZD-01` through `ZD-44`).
- **Operator Consistency**: AE flag operator verified strictly as `reconstruction_error > tau`.
