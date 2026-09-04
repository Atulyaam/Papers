"""
scripts/report_sprints.py
-------------------------
Detailed documentation of Sprints 7 through 13.
Each sprint strictly contains all 21 mandatory subsections,
authoritative metrics, Callout Boxes, and Expectation vs Reality tables.
"""

def get_sprint7_markdown() -> str:
    return """# 9. Sprint 7 — Unsupervised Autoencoder Development (EXP_AE_V1)

### 1. Sprint Objective
Develop, train, and calibrate an unsupervised deep Autoencoder trained exclusively on benign network traffic to establish an empirical baseline of normal protocol behavior, and calibrate a conservative reconstruction threshold for subsequent hybrid fusion.

### 2. Why This Sprint Was Needed
Supervised models are blind to attack categories absent from their training sets. Before hybrid fusion could be evaluated, an independent, unsupervised anomaly-detection model needed to be trained strictly on normal traffic without exposure to attack signatures.

### 3. Starting State
The 75-feature Mutual Information representation (`EXP_MI_V1_1`) was frozen. The official TRAIN (162,395 rows) and VALIDATION (11,200 benign rows) splits were partitioned and verified.

### 4. What Was Implemented
- PyTorch implementation of `Autoencoder` (`src/models/autoencoder/ae_model.py`).
- Training orchestration script (`scripts/run_ae_training.py`) using early stopping.
- Standardization pipeline (`StandardScaler`) fit strictly on the 40,320-row Normal AE-fit subset.
- Reconstruction error threshold calibration engine on Normal VALIDATION.

### 5. Methodology
The model was trained on 40,320 normal flows using MSE loss. A 4,480-flow monitor split evaluated validation loss. Training halted at epoch 138 with best epoch at 133. The frozen model was passed over the 11,200 benign flows of the independent VALIDATION set to compute sample-wise reconstruction errors.

### 6. Parameters / Configuration
- Architecture: `75 -> 12 -> 6 -> 12 -> 75` (2,049 parameters).
- Activations: Hidden ReLU, Output Linear.
- Optimizer: Adam, lr = 0.001, weight decay = 0.0001, batch size = 256.
- Frozen Threshold: $\\tau = 11.160062745213509$ (`mean + 3*sigma`).
- Operator: $\\text{RE}(\\mathbf{x}) > \\tau$.

### 7. Expected Result
The Autoencoder was expected to reconstruct normal flows with low error ($\le 1.0$), while anomalous or unseen network patterns would produce substantial reconstruction errors ($> 10.0$). Validation false alarm rate was expected to be $\le 0.1\%$.

### 8. Actual Result
- Normal Validation Mean RE: 0.2252, Std: 3.6450, Max: 269.16.
- Operational Threshold: $\\tau = 11.160063$.
- Validation False Positives: Exactly 7 out of 11,200 flows ($FPR = 0.000625$, 0.0625%).

### 9. Why Did This Result Occur?
The mean reconstruction error was very low (0.2252), confirming that typical benign traffic compresses well into 6 latent dimensions. However, two legitimate benign connection-termination flows (rows 10731 and 10737) yielded massive reconstruction errors (~269), heavily inflating the standard deviation (3.645) and shifting the `mean + 3*sigma` threshold to 11.16006.

### 10. Expectation vs Actual Table
| Aspect | Expected | Actual | Interpretation |
|:---|:---|:---|:---|
| Benign Validation Mean RE | $\le 0.50$ | 0.2252 | High compression fidelity on typical normal traffic |
| Validation Max RE | $\le 10.0$ | 269.16 | Extreme benign outliers present in TCP RST/FIN state |
| Operational Threshold $\\tau$ | $\approx 2.0 - 5.0$ | 11.160063 | Threshold pushed outward by legitimate connection aborts |
| Validation FPR | $\le 0.001$ | 0.000625 | Highly conservative operating point (7 false alarms / 11,200) |

### 11. Result Interpretation
The threshold calibration succeeded in bounding false alarms ($0.0625\%$), but created an ultra-conservative operating boundary that would require strong anomalous distortion to trigger.

> [!NOTE]
> **WHAT THIS RESULT MEANS**: The Autoencoder successfully learned a compact representation of normal network traffic with an exceptionally low false alarm rate on benign validation data.
>
> **WHAT THIS RESULT DOES NOT MEAN**: It does not guarantee that attack traffic will naturally exceed the $11.16$ threshold if the attack mimics standard benign byte distributions.

### 12. Problems / Anomalies / Issues
Extreme outlier flows in the benign validation set (RE ~ 269.09) distorted the parametric variance estimation.

### 13. Investigation
Inspection of raw packets revealed that rows 10731 and 10737 were short TCP sessions aborted via RST/FIN flags with 0 payload bytes.

### 14. Resolution
The outlier flows were verified as legitimate Normal instances (not mislabeled attacks). They were retained without data filtering to preserve strict protocol integrity.

### 15. What Changed After the Result?
The conservative threshold $\\tau = 11.160063$ was frozen, with explicit awareness that hybrid fusion would require substantial reconstruction deviation to trigger.

### 16. Validation / Audit
Validation gate checks confirmed zero attack contamination in VALIDATION and exact parameter count (2,049).

### 17. Graphs
Figure 4 displays the Validation Reconstruction Error distribution, showing the median, percentiles, and the outward position of $\\tau = 11.160063$.

### 18. Tables
Reconstruction error summary statistics: Mean = 0.2252, Std = 3.6450, P50 = 0.0659, P95 = 0.5674, P99 = 1.5122, P99.9 = 10.6969.

### 19. Final Scientific Outcome
The Autoencoder baseline was successfully trained and frozen. While achieving near-zero false alarms, the outward threshold displacement predisposed the model to high specificity and low sensitivity.

### 20. Status
COMPLETED and FROZEN under tag `EXP_AE_V1`.

### 21. Primary Evidence
- Checkpoint: `results/checkpoints/EXP_AE_V1/ae_final.pt` (SHA: `4ab66af8d4a6e612...`)
- Calibration: `results/autoencoder/EXP_AE_V1/threshold/threshold_calibration.json`
- Metadata: `results/autoencoder/EXP_AE_V1/metadata.json`
"""

def get_sprint8_markdown() -> str:
    return """# 10. Sprint 8 — Baseline & Evaluation Foundation (EXP_FUSION_V1)

### 1. Sprint Objective
Establish the baseline evaluation infrastructure for comparing supervised models, Out-of-Fold stacking (C01), and Autoencoder anomaly detection, and formulate the hybrid fusion rule (C06).

### 2. Why This Sprint Was Needed
To prepare for rigorous hypothesis testing, the project required a frozen evaluation harness that could simultaneously score supervised predictions, compute Autoencoder reconstruction errors, and evaluate 2x2 logical fusion without data leakage.

### 3. Starting State
Supervised base models (Sprint 5), OOF stacking (Sprint 6), and the benign-only Autoencoder (Sprint 7) were trained and frozen.

### 4. What Was Implemented
- Evaluation harness for scoring models on DEVELOPMENT_TEST (81,749 rows).
- 11 candidate fusion rules spanning OR, AND, and threshold variants.
- Selection of configuration `C06` ($C01 \\lor AE$ at frozen $\\tau$) as the canonical hybrid pipeline.

### 5. Methodology
The 11 candidate fusion combinations were evaluated against pre-registered False Positive Rate gates ($FPR \\le 0.05$ on validation). C06 was chosen as the most conservative logical-OR configuration that maximized sensitivity while bounding false alarms.

### 6. Parameters / Configuration
- Stacking Checkpoint: `EXP_OOF_STACK_V1` (seed 42).
- Autoencoder Threshold: $\\tau = 11.160062745213509$.
- Selection Rule: Gate-only conservative-first (OD-4b).

### 7. Expected Result
C06 was expected to match or exceed C01 performance on Development-Test, with the Autoencoder potentially rescuing boundary attack cases.

### 8. Actual Result
- C01 Development-Test: Macro-F1 = 0.892609, FPR = 0.191892 (7,100 FP).
- C06 Development-Test: Macro-F1 = 0.892440, FPR = 0.192243 (7,113 FP).
- Delta Macro-F1: $-0.000169$; Delta False Positives: $+13$.

### 9. Why Did This Result Occur?
On known attack families, the supervised stacking meta-learner already identified almost all detectable attack flows. The Autoencoder flagged 19 benign flows in the test set, 13 of which were correctly classified as benign by C01. Because the OR operator triggers if either model fires, those 13 samples became false alarms in C06, slightly depressing Macro-F1.

### 10. Expectation vs Actual Table
| Aspect | Expected | Actual | Interpretation |
|:---|:---|:---|:---|
| C06 vs C01 Macro-F1 | $\\Delta \\ge 0.0$ | $-0.000169$ | Marginal decrease due to false alarm additions |
| C06 False Positive Addition | $\\le 50$ | $+13$ flows | Bounded and well within protocol tolerance |
| Fusion Selection | Clean candidate | C06 selected | Formal hybrid baseline established |

### 11. Result Interpretation
Logical-OR fusion slightly degraded performance on known attack categories because supervised stacking was already near-optimal, while the Autoencoder added a small false-positive penalty.

> [!NOTE]
> **WHAT THIS RESULT MEANS**: On known attack categories where supervised models are well-trained, hybrid OR fusion provides no performance advantage and incurs a minor false-alarm penalty.
>
> **WHAT THIS RESULT DOES NOT MEAN**: It does not prove fusion is ineffective for unseen zero-day attacks, which motivated the subsequent Protected Backdoor evaluations.

### 12. Problems / Anomalies / Issues
Validation set reuse occurred for both Autoencoder threshold calibration (Sprint 7) and fusion candidate filtering (Sprint 8).

### 13. Investigation
Audit verified that both uses were selection-stage decisions, and neither had access to DEVELOPMENT_TEST or PROTECTED_BACKDOOR.

### 14. Resolution
Documented formally in metadata as a known methodological boundary.

### 15. What Changed After the Result?
C06 was locked as the sole hybrid pipeline for all subsequent sprints.

### 16. Validation / Audit
All 11 exploratory candidate metrics were archived with checksum verification.

### 17. Graphs
ROC and precision-recall trade-offs were generated for all baseline configurations.

### 18. Tables
Confusion matrix for C06 on Development-Test: TP = 43,306, FP = 7,113, TN = 29,887, FN = 1,443.

### 19. Final Scientific Outcome
The evaluation infrastructure was solidified, locking C01 (stacking) and C06 (fusion) as the formal comparative pipelines.

### 20. Status
COMPLETED and FROZEN under tag `EXP_FUSION_V1`.

### 21. Primary Evidence
- Configuration: `results/fusion/EXP_FUSION_V1/metadata.json`
- Metrics: `results/fusion/EXP_FUSION_V1/development_test/metrics.json`
"""

def get_sprint9_markdown() -> str:
    return """# 11. Sprint 9 — Formal Hypothesis Testing (EXP_H123_V1)

### 1. Sprint Objective
Execute the formal, pre-registered hypothesis tests for H1, H2, and H3 across multiple random seeds and locked decision boundaries.

### 2. Why This Sprint Was Needed
To prevent post-hoc rationalization, scientific claims regarding stacking superiority (H1), Autoencoder standalone detection (H2), and fusion rescue (H3) required strict evaluation against mathematical acceptance criteria.

### 3. Starting State
C01, AE, and C06 pipelines frozen. Protected Backdoor (583 rows) isolated.

### 4. What Was Implemented
- Automated multi-seed evaluation harness (`scripts/evaluate_sprint9.py`).
- Evaluation across seeds 42, 123, and 2024.
- Deterministic decision rule evaluators for H1, H2, and H3.

### 5. Methodology
- H1: Evaluated stacking across seeds 42, 123, 2024 on DEVELOPMENT_TEST against the RF baseline ($0.880733$). Required $\\Delta \\ge 0.005$.
- H2: Evaluated standalone AE detection on the 583 Protected Backdoor rows at $\\tau = 11.160063$.
- H3: Compared C06 vs C01 on Protected Backdoor detection count and benign FPR delta.

### 6. Parameters / Configuration
- Epsilon threshold for H1: $\\epsilon = 0.005$.
- Maximum permissible FPR inflation for H3: $0.02$ (2.0%).
- Decision Rule DD-4 for H2: `ae_detected_count == 0 -> NOT_SUPPORTED`.

### 7. Expected Result
- H1: Supported (Stacking outperforms RF).
- H2: Supported (AE detects unseen Backdoors).
- H3: Supported (Fusion rescues missed Backdoors).

### 8. Actual Result
- **H1 Result**: Stacking 3-seed mean Macro-F1 = $0.892961 \\pm 0.000491$. Difference from RF baseline = $+0.012228 > 0.005$. **H1 = SUPPORTED**.
- **H2 Result**: AE detected count on Protected Backdoor = **0 / 583** (0.0%). **H2 = NOT_SUPPORTED**.
- **H3 Result**: C01 detected = 582/583; C06 detected = 582/583; Rescued = 0. Primary rescue condition failed. **H3 = NOT_SUPPORTED**.

### 9. Why Did This Result Occur?
1. **H1 Supported**: Ensemble stacking effectively leveraged complementary strengths of Random Forest and Neural Network, producing a consistent $+0.0122$ Macro-F1 improvement across all three seeds.
2. **H2 Not Supported**: The Autoencoder threshold ($\tau = 11.16006$) was too conservative. Backdoor flows in UNSW-NB15 mimic legitimate protocol byte counts, generating reconstruction errors well below $11.16$.
3. **H3 Not Supported**: Because the AE detected 0 Backdoor samples, it could not rescue any samples missed by C01. C06 detection was identical to C01 (582/583).

### 10. Expectation vs Actual Table
| Hypothesis | Metric Tested | Expected | Actual | Decision |
|:---|:---|:---:|:---:|:---:|
| **H1** | Stacking Mean F1 vs RF | $\\Delta \\ge +0.005$ | $+0.012228$ | **SUPPORTED** |
| **H2** | AE Standalone Detection | Detected $> 0$ | 0 / 583 | **NOT_SUPPORTED** |
| **H3** | C06 Rescue over C01 | Detected $_{C06} >$ Detected $_{C01}$ | $582 = 582$ (Rescue = 0) | **NOT_SUPPORTED** |

### 11. Result Interpretation
Learned ensemble stacking delivers genuine, statistically robust gains on intrusion detection tasks. However, conservative Autoencoder thresholding completely suppresses zero-day anomaly detection capabilities.

> [!NOTE]
> **WHAT THIS RESULT MEANS**: Supervised stacking demonstrates statistically verified superiority over single models (H1). The Autoencoder at its frozen operating point provides zero standalone zero-day detection (H2) and zero rescue value (H3).
>
> **WHAT THIS RESULT DOES NOT MEAN**: It does not mean the Autoencoder is broken; rather, its operational threshold was calibrated too high to detect subtle Backdoor traffic.

### 12. Problems / Anomalies / Issues
Zero detection by the Autoencoder on the zero-day population represented a major negative experimental result.

### 13. Investigation
Verified that model weights loaded correctly and reconstruction errors were computed properly. The distribution of Backdoor reconstruction errors had a maximum of $\\sim 8.4$, entirely below $\\tau = 11.16006$.

### 14. Resolution
Accepted the negative finding in full accordance with scientific integrity rules. No thresholds were recalibrated.

### 15. What Changed After the Result?
The negative findings on H2 and H3 were permanently recorded and established as key research insights.

### 16. Validation / Audit
All test evaluations verified for zero leakage and exact seed reproducibility.

### 17. Graphs
Figure 7 displays the H1/H2/H3 hypothesis decision summary.

### 18. Tables
Per-seed stacking performance: Seed 42 = 0.892609, Seed 123 = 0.892619, Seed 2024 = 0.893656.

### 19. Final Scientific Outcome
Formal verification that supervised stacking works exceptionally well, while the conservative unsupervised Autoencoder failed to detect the zero-day proxy.

### 20. Status
COMPLETED and FROZEN under tag `sprint9-freeze`. Commit: `fc57572`.

### 21. Primary Evidence
- Summary: `results/evaluation/EXP_H123_V1/summary.json`
- H1/H2/H3 Reports: `results/evaluation/EXP_H123_V1/h1_results.json`, `h2_results.json`, `h3_results.json`
"""

def get_sprint10_markdown() -> str:
    return """# 12. Sprint 10 — Systematic Ablation Study (EXP_ABLATION_V1)

### 1. Sprint Objective
Quantify the precise contribution of each base classifier, compare learned stacking against rule-based soft voting, and evaluate Autoencoder fusion through systematic component ablation across 8 configurations.

### 2. Why This Sprint Was Needed
While Sprint 9 proved that stacking was superior to Random Forest alone, it did not establish which specific base models drove the ensemble's performance or whether learned meta-regression was superior to simple voting.

### 3. Starting State
Full 4-model ensemble, individual base models, and Autoencoder available.

### 4. What Was Implemented
Automated ablation suite (`scripts/run_ablation.py`) evaluating 8 configurations:
- `A0_RF`: Random Forest alone
- `A1_FULL_STACK`: Full 4-model stacking (dynamically fitted ablation)
- `A1b_SOFT_VOTE`: Equal-weight average of base model probabilities
- `A2_NO_DT`: Stacking without Decision Tree
- `A3_NO_RF`: Stacking without Random Forest
- `A4_NO_SVM`: Stacking without SVM
- `A5_NO_NN`: Stacking without Neural Network
- `A6_STACK_PLUS_AE`: Full stacking + Autoencoder fusion

### 5. Methodology
Each configuration was evaluated across seeds 42, 123, and 2024 on DEVELOPMENT_TEST and PROTECTED_BACKDOOR. Paired deltas were computed relative to `A1_FULL_STACK`.

### 6. Parameters / Configuration
- Meta-Learner: Logistic Regression ($C=1.0$, balanced, lbfgs).
- 8 configurations evaluated across 3 random seeds = 24 experimental runs.

### 7. Expected Result
Removing any model was expected to degrade Macro-F1. Soft voting was expected to perform comparably to learned stacking.

### 8. Actual Result
- **Full Stacking ($A1$)**: Mean Macro-F1 = $0.891977$.
- **Removing RF ($A3$)**: Mean Macro-F1 dropped to $0.867496$ ($\\Delta = -0.024481$).
- **Soft Voting ($A1b$)**: Mean Macro-F1 = $0.850642$ ($\\Delta = -0.041335$).
- **Removing DT ($A2$)**: Mean Macro-F1 = $0.892276$ ($\\Delta = +0.000299$).
- **Removing SVM ($A4$)**: Mean Macro-F1 = $0.891022$ ($\\Delta = -0.000954$).
- **Removing NN ($A5$)**: Mean Macro-F1 = $0.891953$ ($\\Delta = -0.000024$).
- **Adding AE ($A6$)**: Mean Macro-F1 = $0.891807$ ($\\Delta = -0.000169$).

### 9. Why Did This Result Occur?
1. **Dominance of Random Forest**: Removing RF caused the single largest collapse in performance ($-2.45\\%$ Macro-F1). RF provides the primary anchor of stability and precision.
2. **Failure of Soft Voting**: Soft voting equalizes weights across all models, allowing noisy models (SVM, DT) with high false positive rates to corrupt the final decision boundary.
3. **Redundancy of Decision Tree**: Stacking without DT slightly improved performance ($+0.03\\%$), indicating DT was largely redundant alongside RF.

### 10. Expectation vs Actual Table
| Configuration | Expected Impact | Observed Delta vs A1 | Key Takeaway |
|:---|:---|:---:|:---|
| A0 (RF alone) | Moderate drop | $-0.010359$ | Stacking provides +1.04% gain over RF |
| A1b (Soft Vote) | Near identical to A1 | **$-0.041335$** | **Learned weighting is drastically superior (+4.13%)** |
| A2 (No DT) | Moderate drop | $+0.000299$ | DT is functionally redundant in ensemble |
| A3 (No RF) | Significant drop | **$-0.024481$** | **RF is the single most critical base model** |
| A4 (No SVM) | Moderate drop | $-0.000954$ | SVM provides minor regularization value |
| A5 (No NN) | Significant drop | $-0.000024$ | NN signal captured in part by RF |
| A6 (Stack + AE) | Improvement | $-0.000169$ | AE adds false positive noise on known attacks |

### 11. Result Interpretation
The ablation study decisively proves that learned meta-regression is mandatory; naive probability averaging fails catastrophically. Random Forest is the essential supervised anchor.

> [!NOTE]
> **WHAT THIS RESULT MEANS**: Random Forest is the dominant supervised classifier. Learned Logistic Regression meta-learning is vastly superior to naive voting because it suppresses weak classifiers.
>
> **WHAT THIS RESULT DOES NOT MEAN**: It does not mean Neural Networks or SVMs are useless; they provide regularization and boundary smoothing to the meta-learner.

### 12. Problems / Anomalies / Issues — Computational Lineage Distinction
A subtle numerical difference was detected between Sprint 10 historical A1 and canonical frozen C01:
- **Sprint 10 A1 (dynamically refitted)**: False Positives = $7,201 / 37,000$ ($FPR = 0.194622$).
- **Sprint 12/13 C01 (canonical frozen checkpoint)**: False Positives = $7,100 / 37,000$ ($FPR = 0.191892$).

### 13. Investigation
Forensic audit confirmed that Sprint 10 dynamically refitted the meta-learner on the fly during the ablation suite, resulting in minor solver convergence differences, whereas C01 strictly evaluated the frozen seed-42 checkpoint.

### 14. Resolution
Both values were preserved and clearly labeled as distinct computational lineages: "dynamically refitted historical ablation" vs "canonical frozen checkpoint".

### 15. What Changed After the Result?
The distinction between dynamic ablation models and frozen canonical checkpoints was formally established.

### 16. Validation / Audit
All 24 ablation runs passed determinism checks.

### 17. Graphs
Figures 8 and 9 illustrate the Ablation Macro-F1 and paired deltas.

### 18. Tables
Ablation summary table: A0 (0.8816), A1 (0.8920), A1b (0.8506), A2 (0.8923), A3 (0.8675), A4 (0.8910), A5 (0.8920), A6 (0.8918).

### 19. Final Scientific Outcome
Demonstrated that Random Forest is the core model and learned meta-weighting is essential.

### 20. Status
COMPLETED and FROZEN under tag `sprint10-freeze`. Commit: `839a6ad`.

### 21. Primary Evidence
- Table: `results/ablation/EXP_ABLATION_V1/ablation_table.csv`
- Deltas: `results/ablation/EXP_ABLATION_V1/paired_deltas.csv`
"""

def get_sprint11_markdown() -> str:
    return """# 13. Sprint 11 — Model Explainability & Forensic Audit (EXP_EXPLAIN_V1)

### 1. Sprint Objective
Provide local and global feature attribution for the supervised ensemble using TreeSHAP and KernelSHAP, explain Autoencoder reconstruction errors, and ensure absolute provenance integrity.

### 2. Why This Sprint Was Needed
To satisfy academic interpretability requirements and verify whether models relied on genuine protocol semantics rather than spurious dataset artifacts.

### 3. Starting State
All models frozen. Sprint 10 ablation completed.

### 4. What Was Implemented
- SHAP attribution pipeline for Random Forest and Stacking meta-learner.
- Reconstruction error feature breakdown for the Autoencoder.
- Forensic audit of Autoencoder loading integrity.

### 5. Methodology
100 representative instances (balanced across normal, known attacks, and Backdoors) were sampled. TreeSHAP was applied to Random Forest; KernelSHAP explained the meta-learner.

### 6. Parameters / Configuration
Background samples: 100 k-means centroids of normal training data.

### 7. Expected Result
Models were expected to rely primarily on network payload and timing features (`sbytes`, `sttl`, `dbytes`).

### 8. Actual Result
SHAP attributions confirmed heavy reliance on `sttl` (Source Time-To-Live), `ct_state_ttl`, `sbytes`, and `dbytes`. Discrepancy between internal OS TTL values across synthetic and normal traffic served as a primary discriminator.

---

### CRITICAL EXPERIMENTAL ISSUE: Autoencoder Provenance Forensic Audit
#### Problem
During implementation of `scripts/run_sprint11_explainability.py`, an ad-hoc class `TabularAutoencoder` was locally drafted instead of importing the authoritative `Autoencoder` from `src.models.autoencoder.ae_model`. The local draft initially had mismatched dimensions (`75 -> 48 -> 32 -> 16...`), causing `load_state_dict` crashes. A subsequent draft corrected layer sizes to `75 -> 12 -> 6 -> 12 -> 75` but omitted the ReLU activation on the second encoder layer (`encoder.2`).

#### Investigation
Because PyTorch activation layers have no weights, `load_state_dict(strict=True)` succeeded, but the bottleneck layer evaluated negative linear outputs unrectified into the decoder. Forensic auditing of the raw state dictionary in `ae_final.pt` confirmed the true model required `nn.Sequential(Linear(75,12), ReLU(), Linear(12,6), ReLU())`.

#### Resolution
1. All contaminated Sprint 11 AE explainability artifacts were immediately moved into a secure forensic quarantine directory: `results/explainability/EXP_EXPLAIN_V1/_quarantine_ae_provenance/`.
2. `scripts/run_sprint11_explainability.py` was patched to strictly import the canonical `from src.models.autoencoder.ae_model import Autoencoder`.
3. Strict loading (`strict=True`) was enforced and verified across all 2,049 parameters.
4. All AE explainability artifacts and decisive cases were completely regenerated and verified.

---

### 9. Why Did This Result Occur?
TTL features (`sttl`, `ct_state_ttl`) dominated SHAP importance because IXIA PerfectStorm generated attack packets with standardized TTL values (e.g., 64, 254) differing from ambient normal background traffic.

### 10. Expectation vs Actual Table
| Aspect | Expected | Actual | Interpretation |
|:---|:---|:---|:---|
| Top Features | Payload bytes | `sttl`, `sbytes`, `ct_state_ttl` | Protocol headers and timing proved more discriminative |
| AE Architecture | Imported class | Initially redrafted locally | Caught and corrected via forensic audit |
| Checkpoint Match | 100% | 100% after patch | Absolute mathematical parity confirmed |

### 11. Result Interpretation
Explainability confirmed that the models operate on valid protocol features, while also revealing the ACCS synthetic generation artifact regarding TTL distributions.

### 12. Problems / Anomalies / Issues
Local redefinition of neural network architectures in evaluation scripts creates severe latent provenance risks.

### 13. Investigation
Comprehensive codebase scan for all `load_state_dict` calls across the entire repository.

### 14. Resolution
Enforced global rule: Never redefine model classes in test scripts; always import authoritative model classes.

### 15. What Changed After the Result?
Forensic quarantine procedure established; provenance audit document published.

### 16. Validation / Audit
All 34 pre-verification gates passed with bitwise parity.

### 17. Graphs
Figure 10 illustrates the learned meta-learner feature and model importance weights.

### 18. Tables
SHAP top-5 feature rankings: 1. `sttl`, 2. `sbytes`, 3. `ct_state_ttl`, 4. `dbytes`, 5. `sload`.

### 19. Final Scientific Outcome
Interpretability confirmed; full transparency and resolution of the Autoencoder provenance discrepancy achieved.

### 20. Status
COMPLETED and FROZEN under tag `sprint11-freeze`. Commit: `8eeece3`.

### 21. Primary Evidence
- Forensic Audit Document: `results/explainability/EXP_EXPLAIN_V1/sprint11_ae_provenance_audit.md`
- Quarantine Manifest: `results/explainability/EXP_EXPLAIN_V1/_quarantine_ae_provenance/quarantine_manifest.json`
"""

def get_sprint12_markdown() -> str:
    return """# 14. Sprint 12 — Final Reproducibility & Zero-Training Audit (EXP_FINAL_REPRO_V1)

### 1. Sprint Objective
Independently reproduce and audit the complete frozen evaluation pipeline under strict zero-training constraints.

### 2. Why This Sprint Was Needed
To guarantee that reported metrics reflect genuine frozen model inference rather than hidden refitting, hyperparameter leakage, or runtime recalibration.

### 3. Starting State
All models, checkpoints, splits, and evaluations frozen across Sprints 7–11.

### 4. What Was Implemented
- Standalone reproducibility runner (`scripts/run_sprint12_final_reproducibility.py`).
- Automated AST and dynamic execution audit (`scripts/audit_zero_training.py`).
- Verification engine computing bitwise differences against historical references.

### 5. Methodology
The runner loaded all frozen checkpoints, executed inference on DEVELOPMENT_TEST and PROTECTED_BACKDOOR, and verified zero training calls occurred via runtime monkeypatching and static AST parsing.

### 6. Parameters / Configuration
Locked numerical tolerance: Absolute tolerance $= 10^{-8}$, Relative tolerance $= 10^{-8}$.

### 7. Expected Result
Bitwise identical reproduction of all canonical models with zero training operations.

### 8. Actual Result
- **Zero-Training Audit**:
  - `training_operations_executed = 0`
  - `estimator_fit_calls = 0`
  - `partial_fit_calls = 0`
  - `optimizer_step_calls = 0`
  - `backward_passes = 0`
  - `threshold_recalibrations = 0`
- **Model Verification**:
  - Decision Tree: Historical F1 = 0.849852, Reproduced = 0.849852 (Diff: 0.0) — **REPRODUCED**
  - Random Forest: Historical F1 = 0.880733, Reproduced = 0.880733 (Diff: 0.0) — **REPRODUCED**
  - SVM: Historical F1 = 0.823613, Reproduced = 0.823613 (Diff: 0.0) — **REPRODUCED**
  - Neural Network: Historical F1 = 0.894293, Reproduced = 0.894293 (Diff: 0.0) — **REPRODUCED**
  - Stacking (Seed 42): Historical F1 = 0.892609, Reproduced = 0.892609 (Diff: 0.0) — **REPRODUCED**
  - Stacking (Seed 123): Historical F1 = 0.892619, Reproduced = 0.892619 (Diff: 0.0) — **REPRODUCED**
  - Stacking (Seed 2024): Historical F1 = 0.893656, Reproduced = 0.893656 (Diff: 0.0) — **REPRODUCED**
  - Fusion C06: Historical F1 = 0.892440, Reproduced = 0.892440 (Diff: $1.68 \\times 10^{-8}$) — **REPRODUCED**
  - Soft Vote (A1b): Historical F1 = 0.850632, Reproduced = 0.850632 (Diff: 0.0) — **REPRODUCED**

### 9. Why Did This Result Occur?
Deterministic seeding, strict checkpoint immutability, and explicit software environment pinning ensured exact floating-point reproducibility. The $1.68 \\times 10^{-8}$ difference in C06 was traced to JSON scalar serialization rounding (6 decimal places).

### 10. Expectation vs Actual Table
| Component | Historical Reference | Reproduced Result | Difference | Verdict |
|:---|:---:|:---:|:---:|:---:|
| DT (Base) | 0.849852 | 0.849852 | 0.000000 | **REPRODUCED** |
| RF (Base) | 0.880733 | 0.880733 | 0.000000 | **REPRODUCED** |
| SVM (Base) | 0.823613 | 0.823613 | 0.000000 | **REPRODUCED** |
| NN (Base) | 0.894293 | 0.894293 | 0.000000 | **REPRODUCED** |
| Stacking Seed 42 | 0.892609 | 0.892609 | 0.000000 | **REPRODUCED** |
| Stacking Seed 123 | 0.892619 | 0.892619 | 0.000000 | **REPRODUCED** |
| Stacking Seed 2024 | 0.893656 | 0.893656 | 0.000000 | **REPRODUCED** |
| Fusion C06 | 0.892440 | 0.892440 | $1.68 \\times 10^{-8}$ | **REPRODUCED** |
| Ablation A1b (Soft Vote) | 0.850632 | 0.850632 | 0.000000 | **REPRODUCED** |
| Historical Ablations (A0, A2-A6) | Refitted during S10 | N/A (Frozen only) | N/A | **NOT_REPRODUCED** |

### 11. Result Interpretation
Sprint 12 verified frozen inference reproducibility. The historical ablations (A0, A2–A6) were correctly flagged as `NOT_REPRODUCED` because they required model refitting, which was strictly banned in Sprint 12.

### 12. Problems / Anomalies / Issues
Scikit-learn version drift between Python environments.

### 13. Investigation
Confirmed that all estimators load and predict identically without deprecated parameter warnings.

### 14. Resolution
Environment locked with exact package versions (`scikit-learn==1.9.0`, `torch==2.7.1+cu118`).

### 15. What Changed After the Result?
The frozen pipeline was officially cleared and authorized for the final Sprint 13 zero-day study.

### 16. Validation / Audit
Passed all zero-training dynamic probes and static AST pattern checks.

### 17. Graphs
Figure 11 contrasts historical reference metrics against reproduced outputs.

### 18. Tables
Table 14 records all reproduced metrics, floating-point residuals, and verification statuses.

### 19. Final Scientific Outcome
100% verifiable proof of pipeline determinism and zero-training compliance.

### 20. Status
COMPLETED and FROZEN under tag `sprint12-freeze`. Commit: `633ccf3`.

### 21. Primary Evidence
- Verification: `results/final_reproducibility/EXP_FINAL_REPRO_V1/verification/freeze_verification.json`
- AST Audit: `results/final_reproducibility/EXP_FINAL_REPRO_V1/verification/ast_zero_training_audit.json`
"""

def get_sprint13_markdown() -> str:
    return """# 15. Sprint 13 — Controlled Zero-Day Simulation (EXP_ZERODAY_V1)

### 1. Sprint Objective
Evaluate true unseen-attack generalization on the isolated Protected Backdoor population (583 rows) under Protocol V1.4, test for Autoencoder rescue gain, and conduct pre-freeze cross-checks.

### 2. Why This Sprint Was Needed
This was the culminating experiment of the entire research journey. After ensuring zero leakage across 12 sprints, this experiment answered whether the hybrid IDS can detect a completely withheld attack category.

### 3. Starting State
Sprint 12 formally frozen. Protocol V1.4 locked. Protected Backdoor untouched.

### 4. What Was Implemented
- Pre-flight execution gate (`scripts/run_sprint13_preflight.py`).
- Zero-day evaluation runner (`scripts/run_sprint13_zero_day.py`).
- 44 automated validation gates verifying data hashes, thresholds, and statistical tests.

### 5. Methodology
- **Evaluated Populations**:
  1. Protected Backdoor (583 rows, zero-day proxy).
  2. Benign Control (37,000 normal testing rows).
  3. Attack Control (44,749 known testing attack rows).
  4. Combined Zero-Day Evaluation Set: $37,000 + 583 = 37,583$ rows.
- **Decision Rules**:
  - Unseen-Category Generalization: Supported if $C06 \\text{ ZDR} \\ge 0.50$ and Wilson 95% CI lower bound $> 0.50$.
  - Hybrid Rescue: Supported if $\\text{RescueGain} (Q_3 / 583) \\ge 0.05$ and exact one-sided binomial $p < 0.05$ against baseline $p_0 = 0.000625$.
  - Standalone AE (H2): Subject to rule DD-4 (`ae_detected == 0 -> NOT_SUPPORTED`).

### 6. Parameters / Configuration
- Autoencoder Threshold: $\\tau = 11.160062745213509$.
- Classification Operator: $\\text{RE} > \\tau$.
- C01: Canonical frozen seed-42 meta-learner checkpoint.

### 7. Expected Result
C01 was expected to miss a substantial portion of Backdoor attacks, which the Autoencoder would rescue, proving the value of hybrid fusion.

### 8. Actual Result
- **Quadrant Decomposition (583 Backdoor Rows)**:
  - $Q_1$ (Both Detect): **0 / 583**
  - $Q_2$ (C01 Detects, AE Misses): **582 / 583**
  - $Q_3$ (AE Rescues, C01 Misses): **0 / 583**
  - $Q_4$ (Both Miss): **1 / 583**
- **System Detection Rates on Protected Backdoor**:
  - C01 (Stacking): **582 / 583** ($99.8285\\%$)
  - AE (Autoencoder): **0 / 583** ($0.0000\\%$)
  - C06 (Hybrid Fusion): **582 / 583** ($99.8285\\%$)
- **Benign Control Performance (37,000 Benign Rows)**:
  - C01: False Positives = 7,100 ($FPR = 0.191892$)
  - AE: False Positives = 19 ($FPR = 0.000514$)
  - C06: False Positives = 7,113 ($FPR = 0.192243$)
- **Preregistered Statistical Decisions**:
  - **Unseen-Category Generalization**: **SUPPORTED** ($ZDR = 0.998285$, Wilson 95% CI: $[0.990349, 0.999697]$).
  - **Hybrid Fusion Improvement**: **NOT_SUPPORTED** ($\\text{RescueGain} = 0.0$, Exact binomial $p = 1.0000$).
  - **Standalone AE Detection (H2)**: **NOT_SUPPORTED** (Rule DD-4 satisfied, count = 0).

### 9. Why Did This Result Occur?
1. **Exceptional Stacking Generalization**: Despite never seeing Backdoors, the supervised base models (especially RF and NN) recognized flow-level characteristics (packet arrival windows, byte counts, TTL metrics) shared by Backdoor attacks and other attack classes, achieving $99.83\\%$ detection.
2. **Autoencoder Inertness**: Backdoor sessions generated low reconstruction errors (max $\\sim 8.4$), falling entirely below $\\tau = 11.16006$. Consequently, AE detected zero Backdoor samples ($Q_1=0, Q_3=0$).
3. **Generalization Attribution**: C06 achieved $99.83\\%$ detection, but this was driven 100% by C01 stacking. The Autoencoder contributed zero rescues and added 13 false positives.

### 10. Expectation vs Actual Table
| Aspect | Expected | Actual | Scientific Interpretation |
|:---|:---|:---|:---|
| C01 Backdoor Detection | $\\approx 60 - 80\\%$ | **582 / 583 (99.83%)** | Supervised ensemble generalized remarkably well |
| AE Backdoor Detection | $\\ge 20\\%$ | **0 / 583 (0.00%)** | AE completely inert at conservative threshold |
| AE Rescue ($Q_3$) | $\\ge 30$ samples | **0 / 583 (0.00%)** | Zero rescue gain achieved |
| C06 Generalization | Supported | **SUPPORTED** | Driven entirely by supervised stacking branch |
| Fusion Improvement | Supported | **NOT_SUPPORTED** | No empirical or statistical benefit from AE fusion |

### 11. Result Interpretation
Supervised stacking models can possess strong latent generalization to unseen attack categories sharing protocol-level characteristics. Unsupervised autoencoders cannot rescue attacks if their decision threshold is displaced by benign protocol anomalies.

> [!NOTE]
> **WHAT THIS RESULT MEANS**: The frozen C01/C06 system detected 99.83% of the protected unseen-category proxy samples in this evaluation, formally supporting unseen-category generalization.
>
> **WHAT THIS RESULT DOES NOT MEAN**: It does not prove performance on arbitrary real-world future zero-day attacks outside this protected Backdoor proxy, nor does it support claims that the Autoencoder contributed to this detection.

### 12. Problems / Anomalies / Issues — C01 Lineage Reconciliation
Before freezing Sprint 13, audit reconciled the benign false positive count:
- Sprint 10 historical A1: $FP = 7,201 / 37,000$ ($FPR = 0.194622$).
- Sprint 12/13 canonical frozen C01: $FP = 7,100 / 37,000$ ($FPR = 0.191892$).
- Result: Exactly 0 discrepancy between Sprint 12 and Sprint 13. The difference is 100% attributable to the historical dynamic refitting in Sprint 10 versus frozen checkpoint inference.

### 13. Investigation
Confirmed via pre-freeze cross-check audit (`pre_freeze_crosscheck_audit.json`).

### 14. Resolution
Fully documented and verified across all 44 validation gates.

### 15. What Changed After the Result?
The final experimental status of the repository was locked and frozen.

### 16. Validation / Audit
All 44 Sprint 13 validation gates passed without error.

### 17. Graphs
Figures 12, 13, and 14 present Zero-Day Detection Rates, Benign FPR, and the Quadrant Decomposition.

### 18. Tables
Zero-Day System Performance Table comparing DT, RF, SVM, NN, Stacking (C01), AE, and Fusion (C06).

### 19. Final Scientific Outcome
Unseen-category generalization was supported via supervised stacking; Autoencoder rescue and fusion improvement were decisively rejected.

### 20. Status
COMPLETED and FROZEN under tag `sprint13-freeze`. Commit: `f694e19e44a3dafb486ff216428f1be1f2ec9120`.

### 21. Primary Evidence
- Decisions: `results/zero_day/EXP_ZERODAY_V1/metrics/preregistered_decisions.json`
- Metrics: `results/zero_day/EXP_ZERODAY_V1/metrics/zero_day_metrics.csv`
- Pre-freeze Audit: `results/zero_day/EXP_ZERODAY_V1/pre_freeze_crosscheck_audit.json`
- Validation Report: `results/zero_day/EXP_ZERODAY_V1/validation_report.md`
"""

print("report_sprints.py loaded successfully.")
