"""
scripts/report_syntheses.py
---------------------------
Generates analytical syntheses, problems & resolutions, consolidated tables A-F,
hypotheses matrix, major scientific findings, limitations, audit trail, timeline,
conclusion, appendices A-J, and the final checklist page.
"""

def get_why_it_happened_markdown() -> str:
    return """# 16. "Why Did This Result Happen?" Systematic Analytical Syntheses

To adhere to rigorous scientific standards, this section consolidates causal explanations, empirical evidence, and inferential boundaries for all major findings.

### 1. Supervised Stacking Outperformed Base Models (H1 Supported)
- **WHAT HAPPENED?**: Out-of-fold stacking achieved a mean Macro-F1 of $0.8930 \\pm 0.0005$, outperforming the best individual model (Random Forest, $0.8807$) by $+0.0122$.
- **WHY DID IT LIKELY HAPPEN?**: Base models exhibited decorrelated error distributions. Random Forest provided robust partition stability, while the Neural Network learned continuous representation spaces. The Logistic Regression meta-learner learned positive weights for RF (+2.15) and NN (+1.79) while actively penalizing noisy SVM probabilities (-0.18).
- **WHAT EVIDENCE SUPPORTS THAT?**: The ablation study confirmed that removing RF degraded performance by $-0.0245$, while soft voting (equal weighting) collapsed performance by $-0.0413$.
- **WHAT CANNOT BE CLAIMED?**: It cannot be claimed that stacking eliminates false positives; stacking incurred an FPR of $19.19\\%$.

### 2. Autoencoder Inertness on Protected Backdoors (H2 Rejected)
- **WHAT HAPPENED?**: The Autoencoder detected exactly $0 / 583$ Protected Backdoor samples ($0.00\\%$) at its frozen threshold ($\\tau = 11.16006$).
- **WHY DID IT LIKELY HAPPEN?**: Two legitimate benign validation flows with aborted TCP handshakes (RST/FIN flags) produced massive reconstruction errors ($\\approx 269$), inflating the validation standard deviation to $3.645$. The resulting `mean + 3*sigma` threshold ($11.16006$) was displaced far beyond the reconstruction error profile of Backdoors (which peaked at $\\sim 8.4$).
- **WHAT EVIDENCE SUPPORTS THAT?**: Forensic inspection of validation errors confirms the two extreme outliers. Histograms of Backdoor errors show they fall completely below $11.16$.
- **WHAT CANNOT BE CLAIMED?**: It cannot be claimed that deep autoencoders are inherently incapable of detecting network intrusions; rather, this specific global parametric threshold was displaced by benign protocol anomalies.

### 3. Supervised Stacking Generalized to Unseen Backdoors
- **WHAT HAPPENED?**: The frozen C01 stacking model detected $582 / 583$ ($99.83\\%$) Protected Backdoor samples despite having never been trained on Backdoor traffic.
- **WHY DID IT LIKELY HAPPEN?**: Backdoor attacks in UNSW-NB15 share underlying network flow characteristics with other attack categories (e.g., Exploits and Reconnaissance), such as abnormal TTL values (`sttl = 64` vs `254`), small packet window sizes, and asymmetric byte transfer ratios. The supervised models keyed into these shared intrusion primitives.
- **WHAT EVIDENCE SUPPORTS THAT?**: SHAP feature importance analysis identified `sttl`, `sbytes`, and `ct_state_ttl` as dominant predictive features across both known attacks and Backdoors.
- **WHAT CANNOT BE CLAIMED?**: It cannot be claimed that the system will detect all future zero-day attacks in operational networks, as real-world zero-days may employ novel, legitimate-looking protocol behaviors.

### 4. Zero-Day Fusion Improvement Failed (H3 Rejected)
- **WHAT HAPPENED?**: Hybrid fusion C06 achieved identical attack detection to C01 ($582/583$, RescueGain = $0.0$, exact binomial $p=1.0000$) while adding 13 false positives on benign traffic.
- **WHY DID IT LIKELY HAPPEN?**: Because the Autoencoder detected 0 Backdoor samples ($Q_1=0, Q_3=0$), there were zero opportunities for the Autoencoder to rescue attacks missed by C01. Meanwhile, the Autoencoder's false alarms on benign traffic ($19$ flows) resulted in $13$ novel false positives under the logical-OR operator.
- **WHAT EVIDENCE SUPPORTS THAT?**: The quadrant analysis yielded $Q_3 = 0$. Benign test evaluation confirmed an FPR increase from $0.191892$ to $0.192243$.
- **WHAT CANNOT BE CLAIMED?**: It cannot be claimed that hybrid fusion never works; under this dataset and threshold, the Autoencoder provided no complementary signal.

### 5. Sprint 10 vs Sprint 12/13 False Positive Lineage Difference
- **WHAT HAPPENED?**: Sprint 10 historical A1 reported 7,201 FP ($FPR = 0.194622$), whereas Sprint 12/13 C01 reported 7,100 FP ($FPR = 0.191892$) on the same 37,000 benign rows.
- **WHY DID IT LIKELY HAPPEN?**: Sprint 10 dynamically refitted the meta-learner during the ablation run, introducing minor solver convergence variations. Sprint 12/13 strictly evaluated the canonical frozen seed-42 checkpoint.
- **WHAT EVIDENCE SUPPORTS THAT?**: Pre-freeze cross-check audits confirmed exact bitwise matching between Sprint 12 and Sprint 13 when evaluating the frozen checkpoint.
- **WHAT CANNOT BE CLAIMED?**: It cannot be claimed that the discrepancy represents an error or bug; it represents two distinct, fully documented computational lineages.
"""

def get_problems_and_resolutions_markdown() -> str:
    return """# 17. Experimental Problems, Investigations, and Resolutions

Documenting experimental anomalies, investigative methodologies, and forensic resolutions is vital for scientific integrity.

### Issue 1: Autoencoder Class Redefinition and Provenance Discrepancy (Sprint 11)
- **Problem**: During Sprint 11 explainability, an ad-hoc local `TabularAutoencoder` class was implemented in `scripts/run_sprint11_explainability.py`. The class initially had wrong layer sizes and, in a secondary draft, omitted the ReLU activation on the second encoder layer (`encoder.2`).
- **Why It Mattered**: PyTorch loaded the weights with `strict=True` because activations lack parameter tensors. However, unrectified negative values entered the bottleneck layer, altering the mathematical reconstruction function.
- **Detection**: Detected during Phase A forensic auditing prior to sprint freezing.
- **Investigation**: Inspected raw state_dict tensor shapes in `results/checkpoints/EXP_AE_V1/ae_final.pt` and compared against training source `src/models/autoencoder/ae_model.py`.
- **Root Cause**: Human oversight during local script drafting; failure to import the authoritative class.
- **Resolution**:
  1. Contaminated explainability artifacts were moved to `results/explainability/EXP_EXPLAIN_V1/_quarantine_ae_provenance/`.
  2. `scripts/run_sprint11_explainability.py` was patched to strictly import `from src.models.autoencoder.ae_model import Autoencoder`.
  3. All AE explainability artifacts were completely regenerated and verified bitwise.
- **Validation**: Independent Python script verified `load_state_dict(strict=True)` succeeded with 0 missing/unexpected keys across all 2,049 parameters.

### Issue 2: C01 Benign False Positive Lineage Distinction (Sprint 10 vs Sprint 12/13)
- **Problem**: Sprint 10 reported 7,201 FP on benign test traffic, whereas Sprint 12 and 13 reported 7,100 FP (a 101-sample discrepancy).
- **Why It Mattered**: Could be misinterpreted as an uncontrolled drift or bug in the evaluation pipeline.
- **Detection**: Pre-freeze cross-check audit prior to Sprint 13 freeze.
- **Investigation**: Traced execution logs of `run_ablation.py` versus `run_sprint12_final_reproducibility.py`.
- **Root Cause**: Sprint 10 dynamically refitted the meta-learner on OOF predictions, yielding slightly different optimization weights, whereas Sprint 12/13 used the canonical frozen checkpoint.
- **Resolution**: Formally preserved both values in the scientific record, documenting Sprint 10 as a "dynamically fitted historical ablation" and Sprint 12/13 as "canonical frozen checkpoint evaluation".
- **Validation**: Sprint 13 verified 0 discrepancies against Sprint 12 canonical predictions.

### Issue 3: Validation Reconstruction Error Outlier Inflation (Sprint 7)
- **Problem**: The `mean + 3*sigma` Autoencoder threshold was displaced to $11.16006$, rendering the model inert during testing.
- **Why It Mattered**: Resulted in 0 detection on the zero-day Backdoor population.
- **Detection**: Inspection of threshold calibration artifacts (`threshold_calibration.json`).
- **Investigation**: Traced individual validation reconstruction errors; identified rows 10731 and 10737 with errors $\\approx 269$.
- **Root Cause**: Legitimate benign TCP connections terminated via RST/FIN flags with 0 payload bytes.
- **Resolution**: Preserved the flows to avoid artificial dataset curation; frozen threshold was locked at $11.16006$ to respect pre-registered protocol rules.

### Issue 4: Historical Ablation Non-Reproduction in Sprint 12
- **Problem**: In Sprint 12, ablation configurations A0 and A2–A6 were labeled `NOT_REPRODUCED`.
- **Why It Mattered**: Evaluators might assume ablation results were invalid or missing.
- **Detection**: Publication metrics audit (`final_metrics.csv`).
- **Investigation**: Reviewed Sprint 12 protocol constraints mandating `training_operations = 0`.
- **Root Cause**: Historical ablations required on-the-fly model refitting during Sprint 10; without refitting, they could not be rerun in a zero-training pipeline.
- **Resolution**: Formally documented that Sprint 12 verified frozen inference only; historical ablations remain valid historical records.
"""

def get_consolidated_results_markdown() -> str:
    return """# 18. Consolidated Final Results

To prevent misleading cross-population comparisons, final results are segregated into dedicated, methodologically distinct tables.

### Table A: Baseline Base Model Benchmark (Sprint 5 / Sprint 12 Reproduced)
*Evaluated on Held-Out DEVELOPMENT_TEST (81,749 rows: 37,000 Normal + 44,749 Known Attacks).*

| Model | Macro-F1 | Macro Precision | Macro Recall | Balanced Accuracy | FPR | Runtime (s) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Decision Tree** | 0.849852 | 0.878340 | 0.844352 | 0.844352 | 0.279541 | 9.42 | REPRODUCED |
| **Random Forest** | 0.880733 | 0.903932 | 0.874944 | 0.874944 | 0.231027 | 31.22 | REPRODUCED |
| **Linear SVM** | 0.823613 | 0.851945 | 0.818906 | 0.818906 | 0.310568 | 79.32 | REPRODUCED |
| **Neural Network**| **0.894293** | 0.898909 | **0.891850** | **0.891850** | **0.152432** | 284.21 | REPRODUCED |

---

### Table B: Development-Test Stacking & Fusion Performance (Sprint 9 / Sprint 12)
*Evaluated on Held-Out DEVELOPMENT_TEST (81,749 rows).*

| System / Pipeline | Macro-F1 | Macro Precision | Macro Recall | Balanced Accuracy | FPR | Benign FP (N=37,000) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RF Baseline** | 0.880733 | 0.903932 | 0.874944 | 0.874944 | 0.231027 | 8,548 | REPRODUCED |
| **Stacking (Seed 42)** | 0.892609 | 0.906552 | 0.887931 | 0.887931 | 0.191892 | 7,100 | REPRODUCED |
| **Stacking (Seed 123)**| 0.892619 | 0.906591 | 0.887935 | 0.887935 | 0.191973 | 7,103 | REPRODUCED |
| **Stacking (Seed 2024)**| 0.893656 | 0.907007 | 0.889071 | 0.889071 | 0.188784 | 6,985 | REPRODUCED |
| **Stacking (3-Seed Mean)**| **0.892961** | **0.906717** | **0.888312** | **0.888312** | **0.190883** | **7,063** | REPRODUCED |
| **Fusion C06 (Stack 42+AE)**| 0.892440 | 0.906432 | 0.887755 | 0.887755 | 0.192243 | 7,113 | REPRODUCED |
| **Ablation A1b (Soft Vote)**| 0.850632 | 0.886708 | 0.844649 | 0.844649 | 0.293541 | 10,861 | REPRODUCED |

---

### Table C: Systematic Ablation Study Findings (Sprint 10 Historical Dynamically-Fitted)
*Paired deltas evaluated across 3 random seeds relative to Full Stacking (A1).*

| Config ID | Description | Mean Macro-F1 | Delta vs A1 | Mean FPR | Backdoor Det (N=583) | Lineage Type |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **A0_RF** | Random Forest alone | 0.881618 | -0.010359 | 0.229189 | 582 | Historical Dynamically-Fitted |
| **A1_FULL_STACK**| Full 4-Model Stacking | **0.891977** | **0.000000** | **0.194874** | **582** | Historical Dynamically-Fitted |
| **A1b_SOFT_VOTE** | Equal Soft Voting | 0.850642 | -0.041335 | 0.293775 | 582 | Historical Dynamically-Fitted |
| **A2_NO_DT** | Stacking without DT | 0.892276 | +0.000299 | 0.194144 | 582 | Historical Dynamically-Fitted |
| **A3_NO_RF** | Stacking without RF | 0.867496 | **-0.024481** | 0.232766 | 578 | Historical Dynamically-Fitted |
| **A4_NO_SVM** | Stacking without SVM | 0.891022 | -0.000954 | 0.199748 | 582 | Historical Dynamically-Fitted |
| **A5_NO_NN** | Stacking without NN | 0.891953 | -0.000024 | 0.194874 | 582 | Historical Dynamically-Fitted |
| **A6_STACK_PLUS_AE**| Stacking + AE Fusion | 0.891807 | -0.000169 | 0.195225 | 582 | Historical Dynamically-Fitted |

---

### Table D: Zero-Day Simulation & Generalization Performance (Sprint 13 — EXP_ZERODAY_V1)
*Evaluated on Combined Zero-Day Population (583 Protected Backdoors + 37,000 Benign Controls = 37,583 rows).*

| System | TP (Backdoor) | FN (Backdoor) | ZDR | FP (Benign) | TN (Benign) | FPR | 95% Wilson CI | Decision Scope |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Decision Tree** | 577 | 6 | 0.989708 | 10,343 | 26,657 | 0.279541 | [0.9774, 0.9953] | Descriptive |
| **Random Forest** | 582 | 1 | 0.998285 | 8,548 | 28,452 | 0.231027 | [0.9903, 0.9997] | Descriptive |
| **Linear SVM** | 577 | 6 | 0.989708 | 11,491 | 25,509 | 0.310568 | [0.9774, 0.9953] | Descriptive |
| **Neural Network**| 578 | 5 | 0.991424 | 5,640 | 31,360 | 0.152432 | [0.9798, 0.9964] | Descriptive |
| **Stacking (C01)** | **582** | **1** | **0.998285** | **7,100** | **29,900** | **0.191892** | [0.9903, 0.9997] | Descriptive |
| **Autoencoder (AE)**| **0** | **583** | **0.000000** | **19** | **36,981** | **0.000514** | [0.0000, 0.0063] | Descriptive |
| **Fusion (C06)** | **582** | **1** | **0.998285** | **7,113** | **29,887** | **0.192243** | **[0.9903, 0.9997]** | **FORMAL DECISION** |

---

### Table E: Pre-Registered Hypothesis Testing Verdicts (EXP_H123_V1 & EXP_ZERODAY_V1)

| Hypothesis | Research Question Tested | Quantitative Evidence | Pre-Registered Decision Rule | Verdict |
|:---|:---|:---|:---|:---:|
| **H1** | Stacking superiority over best base model (RF) | $\\text{Mean}(\\text{Stack}) = 0.8930$ vs $\\text{RF} = 0.8807$ ($\\Delta = +0.0122$) | $\\Delta \\ge \\epsilon$ with $\\epsilon = 0.005$ | **SUPPORTED** |
| **H2** | Unsupervised AE standalone zero-day detection | Detected count $= 0 / 583$ ($0.0\\%$) at $\\tau = 11.16006$ | Detected count $== 0 \\rightarrow$ NOT_SUPPORTED (DD-4) | **NOT_SUPPORTED** |
| **H3** | Hybrid fusion rescue gain over supervised stacking | Detected $_{C06} = 582$, Detected $_{C01} = 582$ (Rescue $= 0$) | $\\text{Det}_{C06} > \\text{Det}_{C01}$ and $\\Delta \\text{FPR} \\le 0.02$ | **NOT_SUPPORTED** |
| **Generalization** | C06 unseen category zero-day generalization | C06 ZDR $= 0.9983$, Wilson 95% CI: $[0.9903, 0.9997]$ | $\\text{ZDR} \\ge 0.50$ and CI lower bound $> 0.50$ | **SUPPORTED** |
| **Fusion Gain** | Practical and statistical rescue superiority | $\\text{RescueGain} = 0.0$, Exact binomial $p = 1.0000$ | $\\text{Gain} \\ge 0.05$ and binomial $p < 0.05$ | **NOT_SUPPORTED** |

---

### Table F: Frozen Reproducibility Verification Summary (EXP_FINAL_REPRO_V1)

| Verified Pipeline | Parameter Count | Training Calls | Fit Transform Calls | Recalibrations | Max Float Residual | Verification Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Decision Tree** | N/A (Rules) | 0 | 0 | 0 | 0.000000 | **REPRODUCED** |
| **Random Forest** | 300 Trees | 0 | 0 | 0 | 0.000000 | **REPRODUCED** |
| **Linear SVM** | 76 | 0 | 0 | 0 | 0.000000 | **REPRODUCED** |
| **Neural Network**| 18,050 | 0 | 0 | 0 | 0.000000 | **REPRODUCED** |
| **Autoencoder** | 2,049 | 0 | 0 | 0 | 0.000000 | **REPRODUCED** |
| **Stacking Meta (42)**| 5 | 0 | 0 | 0 | 0.000000 | **REPRODUCED** |
| **Fusion C06 Engine**| Static Rules | 0 | 0 | 0 | $1.68 \\times 10^{-8}$ | **REPRODUCED** |
| **Zero-Training Audit**| **Total Ops = 0** | **0** | **0** | **0** | **Exact Match** | **AUDIT PASSED** |
"""

def get_findings_and_conclusion_markdown() -> str:
    return """# 19. Major Scientific Findings

This project establishes seven core scientific findings:

### Finding 1: Supervised Ensemble Stacking Demonstrates Genuine Generalization Superiority
- **What**: Out-of-fold Logistic Regression stacking achieved a statistically verified $+0.0122$ Macro-F1 improvement over the best individual model (Random Forest).
- **Why**: The meta-learner learned to leverage complementary representations between tree-based partitioning and neural continuous spaces while penalizing redundant linear classifiers.
- **Evidence**: 3-seed evaluation ($0.8930 \\pm 0.0005$ vs $0.8807$); H1 formally SUPPORTED.
- **Meaning**: Learned ensemble stacking provides genuine value in network intrusion detection.
- **Limitation**: Evaluated within the feature space of UNSW-NB15; does not eliminate false positives.

### Finding 2: Random Forest is the Dominant Supervised Pillar
- **What**: Ablation analysis revealed that removing Random Forest degraded Macro-F1 by $-0.0245$, by far the largest single drop.
- **Why**: Random Forest provides the highest individual precision and variance reduction.
- **Evidence**: Ablation config A3 dropped to $0.8675$; RF assigned the highest meta-weight (+2.15).
- **Meaning**: Ensemble IDS architectures must prioritize robust bagging trees as their backbone.
- **Limitation**: Random Forest has the highest model storage footprint.

### Finding 3: Soft Voting is Drastically Inferior to Learned Meta-Regression
- **What**: Simple soft voting collapsed performance by $-0.0413$ Macro-F1 relative to full stacking.
- **Why**: Equal weighting allows noisy base classifiers with high false-alarm rates (SVM, DT) to distort the decision boundary.
- **Evidence**: Ablation config A1b achieved only $0.8506$ Macro-F1 and inflated FPR to $29.38\\%$.
- **Meaning**: Practitioners should never use unweighted voting when meta-learning is feasible.
- **Limitation**: Meta-learning requires cross-validated OOF generation.

### Finding 4: Unsupervised Autoencoders Can Suffer Complete Operational Suppression
- **What**: The Autoencoder detected 0 out of 583 Protected Backdoor samples at its frozen threshold.
- **Why**: Benign connection-termination flows inflated validation variance, shifting $\\tau$ to $11.16006$.
- **Evidence**: H2 NOT_SUPPORTED; Backdoor reconstruction errors peaked at $\\sim 8.4$.
- **Meaning**: Anomaly detection thresholds are vulnerable to benign protocol diversity.
- **Limitation**: Investigated on a single bottleneck dimension (6); multi-scale architectures may behave differently.

### Finding 5: Hybrid OR-Fusion Failed to Deliver Rescue Value
- **What**: Logical-OR fusion C06 rescued exactly 0 Backdoor samples ($Q_3 = 0$) while adding 13 false positives.
- **Why**: AE was completely inert on Backdoor traffic.
- **Evidence**: Exact binomial test $p=1.0000$; H3 and Fusion Improvement NOT_SUPPORTED.
- **Meaning**: Hybrid fusion cannot compensate for an inert anomaly branch.
- **Limitation**: Validated on Backdoor proxy; other attack types with larger payload deviations might trigger rescue.

### Finding 6: Supervised Stacking Generalized Exceptionally to an Unseen Attack Proxy
- **What**: C01 stacking detected 582 out of 583 Protected Backdoors (99.83%).
- **Why**: Backdoor flows shared underlying traffic primitives (packet timing, byte ratios, TTL values) with other known attack classes.
- **Evidence**: Wilson 95% CI: $[0.9903, 0.9997]$; Unseen-Category Generalization formally SUPPORTED.
- **Meaning**: Supervised models can possess substantial latent zero-day detection capabilities.
- **Limitation**: Limited to attack proxies that share flow primitives with known attacks.

### Finding 7: Complete Reproducibility Requires Static Checkpoints and Zero-Training Audits
- **What**: Sprint 12 reproduced historical results with zero training operations and residuals $< 1.7 \\times 10^{-8}$.
- **Why**: Enforced static checkpoint loading, environment pinning, and AST scanning.
- **Evidence**: Dynamic monkeypatching confirmed 0 estimator fit calls across 1,276 lines of code.
- **Meaning**: Academic IDS benchmarks can achieve industrial-grade auditability.

---

# 20. Research Limitations & Threat Analysis

1. **Proxy Nature of Protected Backdoor**: While Backdoor traffic was completely withheld, it originates from the same network environment and testbed (IXIA PerfectStorm) as the training data. It does not represent arbitrary, real-world zero-day exploits.
2. **Tabular Flow Representation**: UNSW-NB15 aggregates network packets into bidirectional summary flows. This obscures raw byte payloads and packet-level timing jitter.
3. **Parametric Threshold Sensitivity**: The `mean + 3*sigma` threshold was heavily distorted by legitimate TCP RST/FIN aborts. Non-parametric quantile thresholds may offer better robustness.
4. **Binomial Independence Assumption**: The exact binomial test assumes independent trials, whereas network flows within the same session may exhibit temporal correlation.
5. **Class Imbalance in Evaluation**: The zero-day evaluation combined 583 Backdoor rows with 37,000 benign rows (1.55% base rate), causing small changes in false positive count to affect precision.

---

# 21. Research Timeline & Milestones

```
Sprint 1 (2026-08-31): Data Acquisition, Schema Validation & Backdoor Isolation [EXP_DATA_ACQUISITION_UNSEEN_RESERVATION_V1]
  │
Sprint 2 (2026-08-31): Deterministic Preprocessing & One-Hot Schemas [EXP_PREPROCESSING_V1]
  │
Sprint 3 (2026-08-31): Leakage-Free 4-Way Splitting (TRAIN, VAL, DEV_TEST, BACKDOOR) [EXP_TRAIN_VAL_SPLIT_V1]
  │
Sprint 4 (2026-09-01): Mutual Information Feature Selection (K=75 Selected) [EXP_MI_V1_1]
  │
Sprint 5 (2026-09-01): Base Classifier Optimization (DT, RF, SVM, NN) [EXP_BASE_MODELS_V1]
  │
Sprint 6 (2026-09-02): 5-Fold Out-of-Fold Stacking & Meta-Learning [EXP_OOF_STACK_V1]
  │
Sprint 7 (2026-09-02): Benign-Only Autoencoder & Threshold Calibration (tau=11.16006) [EXP_AE_V1]
  │
Sprint 8 (2026-09-02): Hybrid Logical Fusion Exploration & C06 Baseline Selection [EXP_FUSION_V1]
  │
Sprint 9 (2026-09-03): Formal Hypothesis Testing: H1 SUPPORTED, H2 NOT_SUPPORTED, H3 NOT_SUPPORTED [sprint9-freeze]
  │
Sprint 10 (2026-09-04): 8-Config Ablation Study: RF Dominance, Soft-Vote Failure, Lineage Identified [sprint10-freeze]
  │
Sprint 11 (2026-09-04): SHAP Explainability & Forensic Audit of AE Provenance Issue [sprint11-freeze]
  │
Sprint 12 (2026-09-04): Final Frozen Pipeline Reproducibility & Zero-Training Audit [sprint12-freeze]
  │
Sprint 13 (2026-09-04): Controlled Zero-Day Simulation (Protocol V1.4, 44 Gates Passed) [sprint13-freeze]
```

---

# 22. Final Scientific Conclusion

This publication-oriented research project provides an exhaustive, leakage-free empirical investigation into hybrid intrusion detection on the UNSW-NB15 dataset. By isolating the Backdoor attack family across the entire lifecycle, the study established a rigorous zero-day evaluation harness.

The experimental findings demonstrate a striking dichotomy:
1. **Supervised ensemble stacking (C01)** achieved outstanding discriminative performance on known attacks (Macro-F1 = 0.8930) and demonstrated remarkable latent generalization to the unseen Backdoor category, successfully detecting **582 out of 583 samples (99.83%)**. The learned meta-learner strategically prioritized Random Forest and Neural Network while suppressing weaker classifiers.
2. **Unsupervised Autoencoder anomaly detection (AE)** suffered complete operational suppression. At its frozen operational threshold ($\\tau = 11.16006$), calibrated on benign validation traffic to enforce low false alarms ($FPR = 0.000625$), the Autoencoder detected **0 out of 583 Backdoor samples**. Legitimate benign connection aborts (TCP RST/FIN) inflated validation variance, displacing the threshold far beyond the subtle profile of Backdoor intrusions.
3. Consequently, **hybrid logical-OR fusion (C06)** achieved zero attack rescue ($Q_3 = 0$), produced no statistical improvement over supervised stacking ($p = 1.0000$), and slightly inflated false alarms on benign traffic.

The central scientific conclusion is that while supervised ensemble stacking can achieve robust domain generalization against withheld attack categories that share flow primitives with known threats, unsupervised autoencoders require domain-aware feature subspace partitioning and adaptive thresholding to prevent total suppression by ambient protocol anomalies.
"""

def get_appendices_markdown() -> str:
    return """# Appendices

### Appendix A — Dataset and Split Manifest
- Source Training CSV: `UNSW_NB15_training-set.csv` (175,341 rows, SHA-256: `bec7dd5ec88dc2a0...`)
- Source Testing CSV: `UNSW_NB15_testing-set.csv` (82,332 rows, SHA-256: `734fe6642edf758f...`)
- Partition TRAIN: `data/splits/train.csv` (162,395 rows, SHA-256: `4a259324e604f013...`)
- Partition VALIDATION: `data/splits/validation.csv` (11,200 rows, SHA-256: `13caf21a076a33f5...`)
- Partition DEVELOPMENT_TEST: `data/splits/development_test.csv` (81,749 rows, SHA-256: `04725e85732ab2fc...`)
- Partition PROTECTED_BACKDOOR: `data/splits/protected_unseen_attack.csv` (583 rows, SHA-256: `6ffd23479b575e43...`)
- Partition EXCLUDED_BACKDOOR: `data/splits/excluded_train_backdoor.csv` (1,746 rows, SHA-256: `b3f6e7e60c9815a5...`)

### Appendix B — Feature Selection Summary (EXP_MI_V1_1)
- Method: Mutual Information (`mutual_info_classif`, $k=3$, seed 42)
- Input: 193 one-hot encoded features -> Output: 75 selected features
- Composition: 39 Continuous Numeric, 25 `proto`, 6 `service`, 5 `state`
- Top 5 Features: `sbytes`, `sttl`, `dbytes`, `ct_state_ttl`, `dttl`

### Appendix C — Base Model Configurations
- Decision Tree: `criterion='entropy', max_depth=None, min_samples_split=2, class_weight='balanced'`
- Random Forest: `n_estimators=300, max_features=0.3, criterion='gini', class_weight='balanced', seed=42`
- Linear SVM: `C=0.1, class_weight='balanced', max_iter=5000, seed=42`
- Neural Network (IDSNet): `MLP [75 -> 128 -> 64 -> 2], ReLU, Adam, lr=0.001, weight_decay=0.0001, epoch=18`

### Appendix D — Key Hyperparameters & Calibration Settings
- Stacking Meta-Learner: `LogisticRegression(solver='lbfgs', C=1.0, class_weight='balanced', max_iter=1000)`
- Autoencoder Topology: `75 -> 12 -> 6 -> 12 -> 75` (2,049 parameters)
- Autoencoder Training: Adam, lr = 0.001, weight decay = 0.0001, batch size = 256, best epoch = 133
- Frozen Threshold $\\tau$: `11.160062745213509` (`mean + 3*sigma` on Normal VALIDATION)
- Classification Operator: strictly `reconstruction_error > tau`

### Appendix E — Hypothesis Definitions and Decision Rules
- H1 Decision Rule: $\\text{Mean}(\\text{Macro-F1}_{\\text{Stack}}) - \\text{Macro-F1}_{\\text{RF}} \\ge 0.005 \\rightarrow$ SUPPORTED
- H2 Decision Rule (DD-4): $\\text{Detected Count}_{\\text{AE}} == 0 \\rightarrow$ NOT_SUPPORTED
- H3 Decision Rule: $\\text{Detected}_{C06} > \\text{Detected}_{C01}$ and $\\Delta \\text{FPR} \\le 0.02 \\rightarrow$ SUPPORTED
- Generalization Rule: $C06 \\text{ ZDR} \\ge 0.50$ and Wilson 95% CI lower bound $> 0.50 \\rightarrow$ SUPPORTED
- Fusion Gain Rule: $\\text{RescueGain} (Q_3 / 583) \\ge 0.05$ and exact one-sided binomial $p < 0.05 \\rightarrow$ SUPPORTED

### Appendix F — Important Checkpoint Hashes
- Autoencoder Weights: `results/checkpoints/EXP_AE_V1/ae_final.pt` (SHA: `4ab66af8d4a6e612...`)
- Autoencoder Scaler: `results/checkpoints/EXP_AE_V1/ae_scaler.joblib` (SHA: `c0128d42ed9ef5be...`)
- Stacking Checkpoint (Seed 42): `results/checkpoints/EXP_OOF_STACK_V1/meta_learner_seed_42.joblib`
- Neural Network Weights: `results/checkpoints/EXP_BASE_MODELS_V1/nn_final.pt`

### Appendix G — Experiment Timeline and Freeze Tags
- `EXP_DATA_ACQUISITION_UNSEEN_RESERVATION_V1` (Commit `cf93ca3`, 2026-08-31)
- `EXP_PREPROCESSING_V1` (Commit `59e9f87`, 2026-08-31)
- `EXP_TRAIN_VAL_SPLIT_V1` (Commit `3cb6349`, 2026-08-31)
- `EXP_MI_V1_1` (Commit `b67bc15`, 2026-09-01)
- `EXP_BASE_MODELS_V1` (Commit `2c8cdb6`, 2026-09-01)
- `EXP_OOF_STACK_V1` (Commit `c946fa6`, 2026-09-02)
- `EXP_AE_V1` (Commit `8dd9645`, 2026-09-02)
- `EXP_FUSION_V1` (Commit `1231dd6`, 2026-09-02)
- `sprint9-freeze` (Commit `fc57572`, 2026-09-03)
- `sprint10-freeze` (Commit `839a6ad`, 2026-09-04)
- `sprint11-freeze` (Commit `8eeece3`, 2026-09-04)
- `sprint12-freeze` (Commit `633ccf3`, 2026-09-04)
- `sprint13-freeze` (Commit `f694e19`, 2026-09-04)

### Appendix H — Validation Gate Summary
- Sprint 13 Validation Gates: Exactly 44 out of 44 passed (0 failures).
- Zero-Training Probe: 0 estimator fit calls, 0 optimizer steps, 0 backward passes.
- Cryptographic Integrity: 100% match across all dataset partitions and model checkpoints.

### Appendix I — Artifact Inventory
- Full Report Source Markdown: `UNSW_NB15_Complete_Research_Report.md`
- Master Publication PDF: `UNSW_NB15_Complete_Research_Report.pdf`
- Generated Figures: `report_assets/figures/fig01` to `fig15`
- Zero-Day Visualizations: `report_assets/figures/zd_*`
- Forensic Evidence: `results/explainability/EXP_EXPLAIN_V1/sprint11_ae_provenance_audit.md`

### Appendix J — Important Historical vs Frozen Lineage Distinctions
- **Sprint 10 A1 (Ablation Full Stack)**: Dynamically refitted on OOF predictions during the ablation run; evaluated with $FP = 7,201 / 37,000$ ($FPR = 0.194622$).
- **Sprint 12/13 C01 (Canonical Stacking)**: Static inference using the frozen seed-42 checkpoint; evaluated with $FP = 7,100 / 37,000$ ($FPR = 0.191892$).
- **Lineage Classification**: The 101-sample false positive difference represents dynamic solver convergence variations versus frozen static checkpoint execution. Both are preserved as authoritative records of their respective experimental contexts.
"""

def get_final_checklist_markdown() -> str:
    return """# Final Research Status

| Item | Specification | Observed Status | Audit Verdict |
|:---|:---|:---:|:---:|
| **Project** | UNSW-NB15 Intrusion Detection System | Publication-Oriented Research | **CONFIRMED** |
| **Documented Scope** | Sprint 7 through Sprint 13 | All 7 Sprints Fully Detailed | **COMPLETE** |
| **Final Frozen Experiment**| EXP_ZERODAY_V1 (Sprint 13) | Protocol V1.4 Executed & Frozen | **FROZEN** |
| **Final Git Status** | Commit: `f694e19e44a3dafb486ff216428f1be1f2ec9120` | Tag: `sprint13-freeze` | **FROZEN** |
| **Main C06 Result** | Zero-Day Detection on Protected Backdoor | **582 / 583 (99.8285%)** | **VERIFIED** |
| **AE Rescue Gain** | Attacks Missed by C01 but Detected by AE | **0 / 583 (0.0000%)** | **VERIFIED** |
| **Hypothesis H1** | Stacking Superiority over Best Single Model | **SUPPORTED (+0.0122 F1)** | **SUPPORTED** |
| **Hypothesis H2** | Standalone AE Zero-Day Anomaly Detection | **NOT_SUPPORTED (0/583 Det)** | **NOT_SUPPORTED** |
| **Hypothesis H3** | Hybrid Fusion Rescue without FPR Inflation | **NOT_SUPPORTED (Rescue=0)** | **NOT_SUPPORTED** |
| **Generalization Decision**| Formal Unseen-Category Generalization | **SUPPORTED (CI: [0.9903, 0.9997])** | **SUPPORTED** |
| **Fusion Decision** | Formal Hybrid Fusion Improvement | **NOT_SUPPORTED (p = 1.0000)** | **NOT_SUPPORTED** |
| **Training Operations** | Refitting / Training during S12 & S13 | **0 Operations Executed** | **AUDIT PASSED** |
| **Validation Gates** | Pre-registered Automated Verification Checks | **44 / 44 Passed (100%)** | **AUDIT PASSED** |

### Evidence-Based Final Conclusion
The comprehensive experimental investigation on the UNSW-NB15 dataset demonstrates that supervised ensemble stacking (C01) delivers superior discriminative accuracy on recognized network attacks and exhibits outstanding latent generalization to unseen attack categories, successfully detecting 99.83% (582/583) of the withheld Protected Backdoor population. Conversely, the unsupervised benign-only Autoencoder (AE), when calibrated conservatively ($\tau = 11.16006$) to prevent false alarms on benign traffic, suffers from total operational suppression by benign connection-termination outliers, yielding zero standalone zero-day detections and zero rescue gain within hybrid logical-OR fusion (C06). Future hybrid intrusion detection research must incorporate localized, domain-aware feature sub-spacing and adaptive thresholding to prevent benign protocol diversity from blinding unsupervised anomaly detectors.
"""

print("report_syntheses.py loaded successfully.")
