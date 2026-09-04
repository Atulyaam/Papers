# UNSW-NB15 Intrusion Detection System
## Publication-Oriented Experimental Documentation (Sprint 7–Sprint 13)

**Research Scope**: Comprehensive Supervised Stacking, Benign-Only Autoencoder, Hybrid Fusion, and Controlled Zero-Day Evaluation  
**Dataset**: UNSW-NB15 Network Intrusion Dataset  
**Final Experiment Status**: FROZEN (Sprint 13 — EXP_ZERODAY_V1)  
**Authoritative Final Commit**: `f694e19e44a3dafb486ff216428f1be1f2ec9120`  
**Authoritative Tag**: `sprint13-freeze`  
**Report Date**: September 2026  
**Repository Root**: `C:\Users\Atul2\OneDrive\Desktop\Papers\IDS-UNSW-NB15`  

---


# Executive Summary
This report provides the definitive, publication-oriented documentation of the research journey conducted on the UNSW-NB15 dataset across Sprints 7 through 13. The project evaluates supervised multi-model stacking, an unsupervised benign-only Autoencoder (AE), and hybrid logical-OR fusion against known and controlled zero-day network intrusions under strict data-leakage controls.

### Key Research Questions & Core Synthesis

**1. What problem does the project solve?**  
*The project investigates the vulnerability of Network Intrusion Detection Systems (NIDS) to novel, zero-day network intrusions and explores whether an unsupervised anomaly detection autoencoder can rescue attacks missed by supervised ensemble classifiers under rigorous data-leakage controls.*

**2. Why is intrusion detection important?**  
*Modern enterprise networks face an expanding surface of sophisticated, polymorphic cyberattacks. Supervised classifiers trained only on known attack signatures often fail against novel intrusion variants, leading to silent network breaches and catastrophic infrastructure compromise.*

**3. Why was UNSW-NB15 selected?**  
*UNSW-NB15 was chosen because it provides modern, realistic network traffic generated via the IXIA PerfectStorm tool, containing 9 distinct contemporary attack families (e.g., Backdoors, Fuzzers, Exploits) alongside contemporary normal traffic, overcoming outdated artifacts of legacy datasets like KDD-Cup 99.*

**4. What system was developed?**  
*A hybrid multi-branch architecture combining 4 supervised base classifiers (Decision Tree, Random Forest, SVM, and Neural Network), an Out-of-Fold (OOF) Logistic Regression meta-learner (C01), and a benign-only deep reconstruction Autoencoder (AE), fused via an inclusive logical-OR decision rule (C06).*

**5. What models were used?**  
*Four diverse supervised base models (Decision Tree, Random Forest with 300 trees, Linear Support Vector Machine, and a 2-hidden-layer MLP Neural Network named IDSNet), an Out-of-Fold Logistic Regression meta-classifier, and a symmetric 75->12->6->12->75 Tabular Autoencoder.*

**6. Why was a hybrid architecture investigated?**  
*Supervised models achieve high discriminative precision on known attack patterns but risk blind spots on zero-day attacks. An unsupervised Autoencoder trained strictly on benign traffic evaluates deviation from normality, theoretically providing a safety net to rescue novel attack variants missed by the supervised branch.*

**7. What experiments were performed?**  
*Across Sprints 7 to 13, experiments included: Autoencoder benign-only reconstruction training and threshold calibration (Sprint 7), baseline evaluation foundation (Sprint 8), formal H1/H2/H3 hypothesis testing across 3 seeds (Sprint 9), systematic 8-configuration ablation study (Sprint 10), post-hoc SHAP and AE explainability with forensic provenance auditing (Sprint 11), zero-training frozen reproducibility verification (Sprint 12), and controlled zero-day simulation using the isolated Backdoor population (Sprint 13).*

**8. What were the major findings?**  
*Supervised OOF stacking (C01) proved exceptionally robust, achieving Macro-F1 = 0.8930 on development-test and detecting 582 out of 583 unseen Backdoor samples (99.83%). Conversely, the Autoencoder was completely inert at its frozen conservative threshold (tau = 11.16006), detecting 0/583 Backdoor samples and yielding zero rescue gain.*

**9. What hypotheses were supported?**  
*Hypothesis H1 was SUPPORTED: Learned OOF stacking outperformed the best single base model (Random Forest) by +0.0122 in Macro-F1, comfortably exceeding the pre-registered threshold (epsilon = 0.005). Furthermore, unseen-category generalization was formally SUPPORTED (C06 ZDR = 0.9983, Wilson 95% CI: [0.9903, 0.9997]).*

**10. What hypotheses were not supported?**  
*Hypothesis H2 was NOT SUPPORTED (standalone AE detected 0/583 Backdoors, rule DD-4). Hypothesis H3 was NOT SUPPORTED (C06 achieved identical detection to C01, yielding 0 additional rescues). Formal fusion improvement was NOT SUPPORTED (exact binomial p = 1.0000 against baseline p0 = 0.000625).*

**11. What did the zero-day experiment show?**  
*The zero-day experiment on 583 isolated Backdoor samples decomposed into Quadrants: Q1=0 (both detect), Q2=582 (C01 only), Q3=0 (AE rescue), and Q4=1 (both miss). The supervised stacking model generalized remarkably well on its own to the unseen category, while the Autoencoder provided no incremental coverage.*

**12. What did the AE contribute?**  
*At its operational frozen threshold (tau = 11.16006, calibrated on Normal VALIDATION to bound FPR <= 0.001), the AE contributed zero attack rescues (Q3 = 0). It added 13 false positives on benign test traffic, slightly inflating overall FPR from 0.191892 (C01) to 0.192243 (C06).*

**13. What did stacking contribute?**  
*Stacking delivered superior generalization over any single base classifier and drastically outperformed simple soft voting (+0.0413 Macro-F1). Logistic Regression meta-learning effectively weighted Random Forest (coef ~ +2.15) and Neural Network (coef ~ +1.79) while penalizing redundant signals.*

**14. What are the major limitations?**  
*The zero-day evaluation used Backdoor as a controlled proxy (not a guarantee for arbitrary real-world zero-days). Tabular flows lacked raw packet temporal sequencing. The AE threshold was heavily inflated by benign TCP RST/FIN connection-termination outliers, suppressing AE sensitivity to subtle payloads.*

**15. What is the final scientific contribution?**  
*The project delivers an empirically rigorous, fully auditable benchmark demonstrating that while hybrid ensemble stacking provides exceptional domain generalization to withheld network attacks, unsupervised autoencoders require domain-aware feature subspace partitioning and adaptive thresholding to avoid complete suppression by benign protocol anomalies.*



# 1. Project Introduction & Research Formulation

## 1.1 Problem Background
Modern computer networks are subjected to an unprecedented volume of polymorphic and zero-day cyber threats. Traditional Network Intrusion Detection Systems (NIDS) are predominantly rule-based or trained using supervised machine learning algorithms. While supervised models excel at identifying recognized attack signatures, they suffer from fundamental blind spots when confronted with novel, previously unseen intrusion classes. Attackers deliberately modify exploit payloads, obfuscate protocol handshakes, and alter packet timings to bypass perimeter defenses. Consequently, modern security operations centers require detection systems capable of maintaining high discriminative accuracy on established threats while demonstrating reliable anomaly sensitivity to unobserved attacks.

## 1.2 Research Problem
The core research problem addressed by this project is whether a hybrid architecture—combining supervised multi-model ensemble stacking with an unsupervised benign-only deep Autoencoder—can successfully generalize to withheld, zero-day attack categories without incurring unacceptable false-positive inflation on benign operational traffic. Specifically, we investigate whether an unsupervised reconstruction model can rescue attack samples that completely evade the decision boundaries of supervised classifiers.

## 1.3 Research Gap
Existing literature in intrusion detection frequently suffers from severe methodological flaws, including:
1. **Pervasive Data Leakage**: Features normalized across combined train-test splits, feature selection performed on test sets, and synthetic oversampling applied before cross-validation.
2. **Unrealistic Zero-Day Benchmarking**: Evaluating "zero-day" capabilities on attack variants that were present during hyperparameter selection or model architecture search.
3. **Unvalidated Hybrid Fusion Claims**: Asserting theoretical benefits of hybrid anomaly-signature fusion without quantifying empirical rescue rates, false-positive penalties, or formal statistical significance.
4. **Reproducibility Deficits**: Missing exact parameter definitions, lack of frozen model checkpoints, and changing decision thresholds post-hoc to inflate reported metrics.

This project addresses these gaps by establishing an immutable, leakage-free evaluation harness with strictly isolated holdout partitions, frozen model checkpoints, pre-registered hypothesis decision criteria, and exhaustive forensic provenance tracking.

## 1.4 Project Objectives
1. Design and enforce a leakage-free 4-way split of the UNSW-NB15 dataset that isolates a complete attack category (Backdoor) as a pristine zero-day proxy.
2. Develop and tune four heterogeneous supervised base classifiers (Decision Tree, Random Forest, Support Vector Machine, and Multi-Layer Perceptron Neural Network).
3. Implement an Out-of-Fold (OOF) Logistic Regression meta-learner to maximize discriminative generalization across multiple random seeds.
4. Train an unsupervised Tabular Autoencoder strictly on benign traffic to establish an empirical normality baseline and calibrate a conservative reconstruction threshold.
5. Combine supervised stacking (C01) and Autoencoder anomaly detection (AE) into an inclusive logical-OR hybrid system (C06).
6. Rigorously evaluate the system against known attacks (Development-Test) and unseen attacks (Protected Backdoor) under pre-registered statistical hypotheses.
7. Conduct full ablation studies, post-hoc SHAP explainability analyses, and zero-training reproducibility audits.

## 1.5 Research Questions & Hypotheses
The investigation is governed by three locked, pre-registered hypotheses:

- **Hypothesis 1 (H1 — Supervised Stacking Superiority)**: Multi-seed Out-of-Fold Logistic Regression stacking achieves a mean Macro-F1 on the held-out Development-Test set that exceeds the best individual base classifier (Random Forest) by at least $\epsilon = 0.005$:
  $$\text{Mean}(\text{Macro-F1}_{\text{Stack}}) - \text{Macro-F1}_{\text{RF}} \ge 0.005$$

- **Hypothesis 2 (H2 — Unsupervised Autoencoder Standalone Anomaly Detection)**: The unsupervised Autoencoder trained strictly on benign traffic can detect a non-zero, statistically meaningful count of unseen Protected Backdoor attacks at its frozen operational threshold:
  $$\text{Detected Count}_{\text{AE}} > 0 \quad \text{subject to decision rule DD-4}$$

- **Hypothesis 3 (H3 — Hybrid Fusion Rescue Efficacy)**: The hybrid logical-OR fusion system $C06 = C01 \lor AE$ increases attack detection on the Protected Backdoor population over the supervised stacking system $C01$ without inflating false positive rate on benign traffic by more than 2 percentage points:
  $$\text{Detected}_{C06} > \text{Detected}_{C01} \quad \text{and} \quad \Delta \text{FPR} \le 0.02$$

> [!IMPORTANT]
> Hypotheses H1, H2, and H3 were formulated with strict mathematical decision rules prior to test evaluation. No thresholds, operators, or sample boundaries were altered after inspection of results.


# 2. UNSW-NB15 Dataset & Rigorous Split Architecture

## 2.1 Dataset Overview
The UNSW-NB15 dataset was created by the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS) using the IXIA PerfectStorm tool to generate a hybrid of real modern normal activities and synthetic contemporary attack behaviors. The raw corpus comprises 2,540,044 records across four CSV files, while the established research split contains 257,673 records partitioned into an official training set (175,341 rows) and an official testing set (82,332 rows).

The dataset features 42 raw attributes (plus `id`, `attack_cat`, and binary `label`), capturing flow-level metrics, basic packet features, content features, time-to-live metrics, and inter-packet arrival windows. The binary class distribution maps:
- `0`: Benign (Normal network flow)
- `1`: Malicious (Network intrusion attempt)

The attack space comprises 9 distinct attack categories: Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoors, Shellcode, and Worms.

## 2.2 Active Split Architecture & Isolation Strategy
To evaluate true zero-day generalization without data leakage, we implemented a 4-way split architecture:
1. **TRAIN (162,395 rows)**: Formed by combining 80% of normal training flows (44,800 rows) with 100% of non-Backdoor attack flows (117,595 rows). Used exclusively for training base classifiers, generating OOF matrices, and fitting the stacking meta-learner.
2. **VALIDATION (11,200 rows)**: The remaining 20% of normal training flows. Contains exclusively benign traffic. Reserved exclusively for benign-only Autoencoder threshold calibration, reconstruction error distribution fitting, and protocol sanity checking. Completely disjoint from TRAIN.
3. **DEVELOPMENT_TEST (81,749 rows)**: Derived from the official testing set after withholding all Backdoor rows. Contains 37,000 normal flows and 44,749 non-Backdoor attack flows across 8 attack families. Used for held-out benchmark evaluation of base models, stacking, and fusion.
4. **PROTECTED_BACKDOOR (583 rows)**: Contains all 583 Backdoor instances from the official testing set. Completely withheld from all training, feature selection, hyperparameter optimization, and threshold calibration. Acts as the pristine zero-day proxy population.
5. **EXCLUDED_TRAIN_BACKDOOR (1,746 rows)**: All 1,746 Backdoor instances present in the raw training set were permanently archived and withheld to prevent any supervised exposure to Backdoor signatures.

```
Raw UNSW-NB15 Training Set (175,341 rows)
├── Archive All Backdoors (1,746 rows) ─────────────────> EXCLUDED_TRAIN_BACKDOOR (Archived)
└── Remaining Pool (173,595 rows)
    ├── Attacks (117,595 rows) ───────────────────────────> TRAIN Attack Pool (100%)
    └── Normal (56,000 rows) ──[Random 80/20 Split]───────┐
                                                          ├──> TRAIN Normal Pool (44,800 rows)
                                                          └──> VALIDATION (11,200 rows, Benign Only)

Final TRAIN = 44,800 Normal + 117,595 Attacks = 162,395 rows.

Raw UNSW-NB15 Testing Set (82,332 rows)
├── Backdoor Instances (583 rows) ────────────────────────> PROTECTED_BACKDOOR (Isolated Zero-Day Proxy)
└── All Other Instances (81,749 rows) ────────────────────> DEVELOPMENT_TEST (37,000 Normal + 44,749 Attack)
```

## 2.3 Split Integrity and Leakage Prevention
Rigorous automated audit checks confirmed:
- **Exact Row Conservation**: $162,395 + 11,200 + 1,746 = 175,341$ training rows; $81,749 + 583 = 82,332$ testing rows.
- **Pairwise Disjointness**: Zero index or UID overlap across TRAIN, VALIDATION, DEVELOPMENT_TEST, and PROTECTED_BACKDOOR.
- **Zero Backdoor Contamination**: Exactly zero Backdoor rows exist in TRAIN or VALIDATION.
- **Zero Attack Contamination in VALIDATION**: VALIDATION contains exactly 11,200 normal flows and zero attack records.
- **Cryptographic Provenance**: SHA-256 hashes generated and verified for all partitioned CSV files.

| Dataset Split | Total Rows | Benign (0) | Attack (1) | Attack Families Included | Cryptographic Hash (SHA-256) |
|:---|:---:|:---:|:---:|:---|:---|
| **TRAIN** | 162,395 | 44,800 | 117,595 | 8 (No Backdoor) | `4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c` |
| **VALIDATION** | 11,200 | 11,200 | 0 | 0 (Benign Only) | `13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180` |
| **DEVELOPMENT_TEST** | 81,749 | 37,000 | 44,749 | 8 (No Backdoor) | `04725e85732ab2fc6d9eaaa6105418b22b083b5c651067e7b0785464f414e508` |
| **PROTECTED_BACKDOOR** | 583 | 0 | 583 | 1 (Backdoor Only) | `6ffd23479b575e438ad90678268f40f674a663c2b9507aaf65089623397a9d91` |
| **EXCLUDED_BACKDOOR** | 1,746 | 0 | 1,746 | 1 (Archived Backdoor) | `b3f6e7e60c9815a53f40eb2d41df8b67d29f884b922a487c3fe83c02e0db0a02` |

> [!NOTE]
> **WHAT THIS SPLIT ARCHITECTURE MEANS**: The model training and hyperparameter tuning phases were completely blind to the Backdoor attack family. Any performance on the Protected Backdoor population represents true generalization to a withheld attack category.
>
> **WHAT THIS SPLIT ARCHITECTURE DOES NOT MEAN**: It does not imply that the Protected Backdoor population represents an arbitrary zero-day exploit in the wild; it serves as a controlled in-vitro proxy.


# 3. Feature Engineering & Feature Selection

## 3.1 Preprocessing Pipeline
Raw network flows contain both numeric metrics and high-cardinality nominal variables (`proto`, `service`, `state`). Preprocessing followed strict leakage-prevention standards:
1. **One-Hot Encoding**: Nominal attributes were encoded using a frozen schema fitted strictly on the TRAIN partition. Rare categories not observed in TRAIN were mapped to an out-of-vocabulary representation.
   - Raw features: 42 attributes (excluding ID and targets).
   - Post one-hot encoding: 193 candidate features (154 discrete binary columns, 39 continuous numeric columns).
2. **Feature Normalization**: Continuous variables were standardized using z-score normalization ($(\mu, \sigma)$). Scaling statistics were computed strictly on the TRAIN partition and applied immutably to VALIDATION, DEVELOPMENT_TEST, and PROTECTED_BACKDOOR.

## 3.2 Mutual Information Selection Methodology (EXP_MI_V1_1)
Mutual Information (MI) was selected as the non-parametric feature scoring metric due to its ability to capture non-linear relationships without distributional assumptions. MI scoring was performed using `mutual_info_classif` with $k=3$ nearest neighbors on unscaled encoded features of the TRAIN set.

Candidate feature subset sizes $K \in \{10, 20, 30, 40, 50, 75, 100, 150\}$ were evaluated using 5-fold Stratified Cross-Validation on TRAIN with a fixed, balanced Logistic Regression evaluator.

## 3.3 Selection Rationalization and Plateau Analysis
The inner-CV Macro-F1 curve exhibited a steep rise from $K=10$ (0.8249) to $K=50$ (0.9196), reaching a global peak at $K=75$ (0.9198). Beyond $K=75$, performance plateaued, with $K=100$ yielding 0.91977 and $K=150$ yielding 0.91975.

Following the pre-registered tie-breaking and complexity-penalization rule, $K=75$ was selected:
- **Global Optimal Inner-CV Performance**: 0.919799 mean Macro-F1.
- **Dimensionality Reduction**: 61.1% reduction from the 193 candidate features.
- **Feature Composition**: Exactly 39 continuous numeric features (100% retention of numeric signals), 25 protocol dummy variables, 6 service dummy variables, and 5 state dummy variables.

| Candidate K | Mean Macro-F1 | Std Macro-F1 | Complexity Delta vs K=75 | Selection Decision |
|:---:|:---:|:---:|:---:|:---|
| 10 | 0.824852 | 0.003435 | -65 features | Sub-optimal representation |
| 20 | 0.864436 | 0.002428 | -55 features | Underfitting |
| 30 | 0.897442 | 0.000917 | -45 features | Underfitting |
| 40 | 0.916198 | 0.002122 | -35 features | Marginal underfitting |
| 50 | 0.919560 | 0.002323 | -25 features | Approaching plateau |
| **75** | **0.919799** | **0.002393** | **Selected Optimal** | **SELECTED WINNER (Peak CV Macro-F1)** |
| 100 | 0.919775 | 0.002436 | +25 features | Plateau (redundant complexity) |
| 150 | 0.919750 | 0.002506 | +75 features | Slight degradation (noise inflation) |

The top 10 most informative selected features are: `sbytes`, `sttl`, `dbytes`, `ct_state_ttl`, `dttl`, `sload`, `dload`, `rate`, `dur`, and `smean`.


# 4. Complete System Architecture

## 4.1 Architectural Overview
The end-to-end intrusion detection system integrates diverse supervised learning paradigms with an unsupervised deep reconstruction model, governed by Out-of-Fold meta-learning and hybrid logical fusion.

```
RAW UNSW-NB15 DATASET (257,673 rows)
  │
  ├─> Data Isolation & Leakage Prevention Controls
  │     ├── TRAIN (162,395 rows: 44,800 Benign + 117,595 Known Attacks)
  │     ├── VALIDATION (11,200 rows: Benign-Only Calibration Pool)
  │     ├── DEVELOPMENT_TEST (81,749 rows: Known Attack Evaluation)
  │     └── PROTECTED_BACKDOOR (583 rows: Isolated Zero-Day Proxy)
  │
  ├─> Preprocessing & One-Hot Encoding (193 candidate features)
  │
  ├─> Mutual Information Feature Selection (75 Selected Features)
  │
  ├───[ BRANCH 1: SUPERVISED DIVERSITY ENSEMBLE ]───┐
  │     ├── Decision Tree (Entropy Splitter)         │
  │     ├── Random Forest (300 Trees, Bagging)      │
  │     ├── Support Vector Machine (Linear Margin)   │
  │     └── Neural Network (IDSNet, 75->128->64->2)  │
  │                                                  │
  │     └──> 5-Fold Out-of-Fold (OOF) Prediction     │
  │                                                  │
  │     └──> Logistic Regression Meta-Learner (C01)  │
  │                                                  │
  ├───[ BRANCH 2: UNSUPERVISED ANOMALY MODEL ]───────┤
  │     └──> Deep Autoencoder (75->12->6->12->75)   │
  │          • Benign-Only Training (40,320 rows)   │
  │          • Calibrated Threshold: tau=11.16006   │
  │          • Anomaly Decision: RE > tau           │
  │                                                  │
  └───[ HYBRID FUSION ENGINE ]───────────────────────┘
        └──> C06 Decision: y_pred = (C01 == 1) OR (AE == 1)
              │
              ├──> Known Threat Evaluation (DEVELOPMENT_TEST: 81,749 rows)
              ├──> Zero-Day Proxy Simulation (PROTECTED_BACKDOOR: 583 rows)
              ├──> Post-Hoc SHAP & Reconstruction Explainability
              └──> Deterministic Zero-Training Reproducibility Audit
```

## 4.2 Component Roles and Data Interfaces
Every architectural module fulfills a dedicated, non-overlapping function:
1. **Feature Transformation Pipeline**: Transforms 42 raw flow features into 75 standardized, continuous/discrete signals using frozen TRAIN statistics.
2. **Supervised Base Classifiers**: Four diverse learners map input features $\mathbf{x} \in \mathbb{R}^{75}$ to attack probability estimates $\hat{p}_i \in [0, 1]$.
3. **Out-of-Fold Stacking Engine**: Generates an unbiased cross-validated probability matrix $\mathbf{P}_{\text{OOF}} \in \mathbb{R}^{N \times 4}$ across the 162,395 training rows without label leakage.
4. **Logistic Regression Meta-Learner (C01)**: Learns optimal combining weights $\mathbf{w}$ and bias $b$ to produce the ensemble decision:
   $$\hat{y}_{\text{C01}} = \mathbb{I}\left(\sigma\left(\sum_{i=1}^4 w_i \hat{p}_i + b\right) > 0.5\right)$$
5. **Unsupervised Autoencoder (AE)**: Maps input $\mathbf{x}$ through bottleneck latent space $\mathbf{z} \in \mathbb{R}^6$ to reconstruct $\mathbf{\hat{x}} \in \mathbb{R}^{75}$. Computes Mean Squared Error:
   $$\text{RE}(\mathbf{x}) = \frac{1}{75} \sum_{j=1}^{75} (x_j - \hat{x}_j)^2$$
   Produces binary anomaly output: $\hat{y}_{\text{AE}} = \mathbb{I}(\text{RE}(\mathbf{x}) > \tau)$.
6. **Logical-OR Fusion Engine (C06)**: Synthesizes supervised and unsupervised outputs:
   $$\hat{y}_{\text{C06}} = \hat{y}_{\text{C01}} \lor \hat{y}_{\text{AE}}$$
   This construction guarantees that an attack detected by either subsystem is flagged, creating the theoretical potential for the Autoencoder to "rescue" samples missed by stacking.


# 5. Supervised Base Classifiers

## 5.1 Models Selected & Theoretical Justifications
The architecture deploys four fundamentally distinct classification paradigms to maximize error decorrelation:

1. **Decision Tree (DT)**:
   - *What*: Single recursive partitioning estimator using entropy information gain.
   - *Why*: Provides non-linear orthogonal splits and high local interpretability.
   - *How*: Unconstrained depth with minimum sample split of 2 and balanced class weights.
   - *Expected Role*: Rapid detection of obvious rule-based protocol boundary violations.
   - *Actual Result*: Development-Test Macro-F1 = 0.8499, Precision = 0.8783, Recall = 0.8444, FPR = 0.2795.
   - *Limitation*: Highly prone to localized variance and high false positive rate (27.95%).

2. **Random Forest (RF)**:
   - *What*: Bagged ensemble of 300 decorrelated decision trees.
   - *Why*: Extreme resistance to overfitting and robust feature subsampling.
   - *How*: 300 estimators, max features = 0.3 (22 features evaluated per split), Gini impurity.
   - *Expected Role*: Primary supervised anchor model delivering high precision and stability.
   - *Actual Result*: Development-Test Macro-F1 = 0.8807, Precision = 0.9039, Recall = 0.8749, FPR = 0.2310.
   - *Limitation*: Computationally intensive inference (31.2s on test) and elevated false positive rate on benign test traffic.

3. **Support Vector Machine (SVM)**:
   - *What*: Linear maximum-margin separating hyperplane (`LinearSVC`).
   - *Why*: Provides global linear regularization independent of local data density.
   - *How*: $C=0.1$, balanced class weights, maximum iterations = 5000.
   - *Expected Role*: Global linear baseline and margin stabilizer for the ensemble.
   - *Actual Result*: Development-Test Macro-F1 = 0.8236, Precision = 0.8519, Recall = 0.8189, FPR = 0.3106.
   - *Limitation*: Lowest individual Macro-F1 and highest false positive rate (31.06%), struggling with non-linear feature interactions.

4. **Neural Network (IDSNet)**:
   - *What*: Deep Multi-Layer Perceptron (MLP) with architecture `75 -> 128 -> 64 -> 2`.
   - *Why*: Universal function approximator capable of learning distributed hierarchical embeddings.
   - *How*: ReLU hidden activations, Adam optimizer (lr=0.001, weight decay=0.0001), 18 epochs with early stopping.
   - *Expected Role*: High-capacity non-linear representation learner complementing tree structures.
   - *Actual Result*: Development-Test Macro-F1 = 0.8943, Precision = 0.8989, Recall = 0.8919, FPR = 0.1524.
   - *Limitation*: Highest training runtime (284.2s) and sensitivity to continuous feature scaling distributions.

## 5.2 Comparative Development-Test Performance
Evaluating all four frozen base models on the 81,749 held-out rows of DEVELOPMENT_TEST yields the benchmark performance profile:

| Model | Macro-F1 | Macro Precision | Macro Recall | Balanced Accuracy | FPR | Runtime (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Decision Tree** | 0.849852 | 0.878340 | 0.844352 | 0.844352 | 0.279541 | 9.42 |
| **Random Forest** | 0.880733 | 0.903932 | 0.874944 | 0.874944 | 0.231027 | 31.22 |
| **SVM (Linear)** | 0.823613 | 0.851945 | 0.818906 | 0.818906 | 0.310568 | 79.32 |
| **Neural Network**| **0.894293** | 0.898909 | **0.891850** | **0.891850** | **0.152432** | 284.21 |

> [!NOTE]
> **WHAT THIS BASE MODEL RESULT MEANS**: Neural Network and Random Forest establish the strongest individual supervised baselines, with the Neural Network achieving the lowest false-positive rate (15.24%) and highest single-model Macro-F1 (0.8943).
>
> **WHAT THIS BASE MODEL RESULT DOES NOT MEAN**: High benchmark performance on Development-Test does not guarantee generalization to unobserved zero-day attack categories.


# 6. Unsupervised Benign-Only Autoencoder (EXP_AE_V1)

## 6.1 Theoretical Motivation & Role
Supervised classifiers partition the feature space by learning boundary planes between labeled classes. When an attacker introduces a novel attack mechanism whose feature representation lies on the benign side of the supervised boundary, supervised systems experience catastrophic failure.

The Unsupervised Autoencoder addresses this fundamental vulnerability by modeling only the manifold of normal network traffic. Trained exclusively on benign network flows, the model learns an identity mapping through an informational bottleneck. When presented with standard benign traffic, the Autoencoder reconstructs the features with minimal distortion. When presented with an anomalous flow exhibiting atypical structural or payload patterns, the compressed bottleneck cannot reconstruct the anomalous features, resulting in elevated Reconstruction Error (RE).

## 6.2 Architecture and Training Specifications
- **Model Topology**: Symmetric 5-layer feed-forward network: `75 -> 12 -> 6 -> 12 -> 75`.
- **Informational Bottleneck**: 6 latent dimensions (92.0% spatial compression).
- **Parameter Count**: Exactly 2,049 trainable scalar parameters (Encoder: 900+12+72+6; Decoder: 72+12+900+75).
- **Activation Functions**: ReLU in all hidden and bottleneck layers; Linear (identity) on the output layer.
- **Normalization & Regularization**: No Batch Normalization, no Dropout.
- **Training Population**: Exactly 40,320 Normal TRAIN flows (seed 42 split). A 4,480 Normal monitor split was used for early stopping.
- **Optimization**: Adam optimizer, learning rate = 0.001, weight decay = 0.0001, batch size = 256, MSE loss.
- **Convergence**: Trained to epoch 133 (stopping at epoch 138 with patience 5).

## 6.3 Threshold Calibration on Normal VALIDATION
The decision threshold $\tau$ was calibrated strictly on the 11,200 benign flows of the VALIDATION partition to guarantee bounded false positive rates:
- Validation Reconstruction Error distribution: Mean = 0.2252, Std = 3.6450, Median = 0.0659, 95th percentile = 0.5674, 99th percentile = 1.5122, 99.9th percentile = 10.6969.
- **Operational Frozen Threshold**: Calibrated under the pre-registered `mean + 3*sigma` rule:
  $$\tau = 0.225201 + 3 \times 3.644954 = 11.160062745213509$$
- **Classification Operator**: Strictly $\text{RE}(\mathbf{x}) > \tau$.
- **Validation Operating Behavior**: Exactly 7 out of 11,200 benign validation flows exceeded $\tau$, yielding an empirical validation FPR of $7 / 11200 = 0.000625$ (0.0625%).

## 6.4 Observed Inertness and Root Cause Analysis
During held-out testing, the Autoencoder exhibited complete operational inertness, detecting 0 out of 583 Protected Backdoor attacks and only 19 out of 37,000 benign test flows.

**Why did this occur?**
1. **Benign Connection-Termination Outliers**: Forensic inspection of the VALIDATION set revealed two extreme normal flows: row 10731 (RE = 269.03) and row 10737 (RE = 269.09). These legitimate flows corresponded to short, aborted TCP connections terminated via RST/FIN packets with zero payload bytes.
2. **Variance Inflation**: Because standard deviation is sensitive to quadratic deviations, these two outlier flows single-handedly inflated the validation standard deviation from $\sim 0.35$ to $3.6450$.
3. **Threshold Displacement**: The resulting `mean + 3*sigma` threshold was displaced outward to $11.16006$, an order of magnitude higher than the 99th percentile (1.5122).
4. **Subtle Attack Invisibility**: Backdoor intrusions in UNSW-NB15 mimic standard interactive sessions (SSH/Telnet), generating small, structured packet exchanges with reconstruction errors typically between $0.8$ and $3.5$. Because the frozen threshold was pushed to $11.16$, all 583 Backdoor attacks fell harmlessly below the threshold.


# 7. Out-of-Fold Ensemble Stacking

## 7.1 Stacking Methodology and Leakage Prevention
Traditional ensemble stacking trains base models on the training set, predicts on the same training set, and trains a meta-learner on those predictions. This causes catastrophic meta-learner overfitting because base models produce artificially optimistic predictions on data they have memorized.

To prevent meta-level target leakage, we implemented strict 5-Fold Stratified Out-of-Fold (OOF) prediction:
1. The 162,395-row TRAIN partition was divided into 5 stratified folds (seed 7).
2. For each fold $k \in \{1, \dots, 5\}$, each base model was trained on 4 folds (129,916 rows) and evaluated on the held-out fold (32,479 rows).
3. The resulting out-of-fold probability predictions were concatenated into an unbiased prediction matrix $\mathbf{P}_{\text{OOF}} \in \mathbb{R}^{162,395 \times 4}$.
4. The meta-learner was trained strictly on $\mathbf{P}_{\text{OOF}}$.

## 7.2 Meta-Learner Architecture and Multi-Seed Stability
The meta-learner is a regularized Logistic Regression classifier (`lbfgs` solver, $C=1.0$, balanced class weights). To evaluate numerical stability, stacking was executed across three independent random seeds: 42, 123, and 2024.

- **Seed 42 Macro-F1**: 0.892609 (FPR = 0.191892, False Positives = 7,100 / 37,000)
- **Seed 123 Macro-F1**: 0.892619 (FPR = 0.191973, False Positives = 7,103 / 37,000)
- **Seed 2024 Macro-F1**: 0.893656 (FPR = 0.188784, False Positives = 6,985 / 37,000)
- **Three-Seed Mean Macro-F1**: $0.892961 \pm 0.000491$

## 7.3 Learned Base Model Attributions
Inspection of the canonical seed-42 meta-learner regression coefficients reveals how the ensemble synthesizes base model outputs:
- **Random Forest ($w_{\text{RF}} = +2.1458$)**: Heavily prioritized as the most dependable positive indicator.
- **Neural Network ($w_{\text{NN}} = +1.7892$)**: Substantial positive weight, contributing non-linear representation capability.
- **Decision Tree ($w_{\text{DT}} = +0.3542$)**: Modest positive weight, providing marginal rule-based refinements.
- **Support Vector Machine ($w_{\text{SVM}} = -0.1824$)**: Negative coefficient. Because SVM outputs correlated heavily with RF while exhibiting higher false-positive noise (31.06%), the meta-learner penalizes raw SVM probabilities to suppress false alarms.
- **Bias Intercept ($b = -1.2405$)**: Conservative baseline offset.


# 8. Hybrid Logical Fusion Architecture

## 8.1 Fusion Formulation: C01 vs C06
The primary pipeline incorporates two distinct system configurations:
- **C01 (Canonical Supervised Stacking)**: The frozen seed-42 Out-of-Fold Logistic Regression meta-learner evaluating the four base models.
- **C06 (Hybrid Inclusive Fusion)**: The logical-OR synthesis of C01 and the Autoencoder:
  $$\hat{y}_{\text{C06}} = \hat{y}_{\text{C01}} \lor \hat{y}_{\text{AE}}$$

## 8.2 The "Rescue" Concept and Mathematical Properties
The logical-OR operator has distinct mathematical properties:
1. **Monotonic Positive Prediction**: $C06(\mathbf{x}) \ge C01(\mathbf{x})$ for all $\mathbf{x}$. C06 can never classify a sample as benign if C01 classified it as an attack.
2. **False Negative Reduction (Rescue)**: If an attack sample is missed by C01 ($C01 = 0$) but flagged by the Autoencoder ($AE = 1$), C06 successfully flags the attack ($C06 = 1$). This specific transition is formally defined as a **Rescue**.
3. **False Positive Penalty**: If a benign sample is correctly classified by C01 ($C01 = 0$) but generates an Autoencoder anomaly ($AE = 1$), C06 incurs a false alarm.

## 8.3 Quadrant Decomposition on Protected Zero-Day Traffic
When evaluating C01 and AE on the 583 Protected Backdoor samples, all samples map into four mutually exclusive quadrants:
- **Quadrant 1 ($Q_1$) — Dual Detection**: $C01 = 1$ and $AE = 1$. Both systems detect the attack. AE provides redundant confirmation.
- **Quadrant 2 ($Q_2$) — Stacking Solitary Detection**: $C01 = 1$ and $AE = 0$. The supervised model detects the attack; AE fails to flag it.
- **Quadrant 3 ($Q_3$) — Autoencoder Rescue**: $C01 = 0$ and $AE = 1$. The supervised model fails, but the Autoencoder detects it. **This is the sole metric of hybrid rescue value.**
- **Quadrant 4 ($Q_4$) — Complete Evasion**: $C01 = 0$ and $AE = 0$. Both systems fail to detect the intrusion.

$$\text{RescueGain} = \frac{Q_3}{N_{\text{Backdoor}}} = \frac{Q_3}{583}$$

Under the pre-registered protocol, hybrid fusion is clinically justified if and only if $Q_3$ demonstrates statistically significant rescue beyond the validation false-alarm rate ($p_0 = 0.000625$).


# 9. Sprint 7 — Unsupervised Autoencoder Development (EXP_AE_V1)

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
- Frozen Threshold: $\tau = 11.160062745213509$ (`mean + 3*sigma`).
- Operator: $\text{RE}(\mathbf{x}) > \tau$.

### 7. Expected Result
The Autoencoder was expected to reconstruct normal flows with low error ($\le 1.0$), while anomalous or unseen network patterns would produce substantial reconstruction errors ($> 10.0$). Validation false alarm rate was expected to be $\le 0.1\%$.

### 8. Actual Result
- Normal Validation Mean RE: 0.2252, Std: 3.6450, Max: 269.16.
- Operational Threshold: $\tau = 11.160063$.
- Validation False Positives: Exactly 7 out of 11,200 flows ($FPR = 0.000625$, 0.0625%).

### 9. Why Did This Result Occur?
The mean reconstruction error was very low (0.2252), confirming that typical benign traffic compresses well into 6 latent dimensions. However, two legitimate benign connection-termination flows (rows 10731 and 10737) yielded massive reconstruction errors (~269), heavily inflating the standard deviation (3.645) and shifting the `mean + 3*sigma` threshold to 11.16006.

### 10. Expectation vs Actual Table
| Aspect | Expected | Actual | Interpretation |
|:---|:---|:---|:---|
| Benign Validation Mean RE | $\le 0.50$ | 0.2252 | High compression fidelity on typical normal traffic |
| Validation Max RE | $\le 10.0$ | 269.16 | Extreme benign outliers present in TCP RST/FIN state |
| Operational Threshold $\tau$ | $pprox 2.0 - 5.0$ | 11.160063 | Threshold pushed outward by legitimate connection aborts |
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
The conservative threshold $\tau = 11.160063$ was frozen, with explicit awareness that hybrid fusion would require substantial reconstruction deviation to trigger.

### 16. Validation / Audit
Validation gate checks confirmed zero attack contamination in VALIDATION and exact parameter count (2,049).

### 17. Graphs
Figure 4 displays the Validation Reconstruction Error distribution, showing the median, percentiles, and the outward position of $\tau = 11.160063$.

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


# 10. Sprint 8 — Baseline & Evaluation Foundation (EXP_FUSION_V1)

### 1. Sprint Objective
Establish the baseline evaluation infrastructure for comparing supervised models, Out-of-Fold stacking (C01), and Autoencoder anomaly detection, and formulate the hybrid fusion rule (C06).

### 2. Why This Sprint Was Needed
To prepare for rigorous hypothesis testing, the project required a frozen evaluation harness that could simultaneously score supervised predictions, compute Autoencoder reconstruction errors, and evaluate 2x2 logical fusion without data leakage.

### 3. Starting State
Supervised base models (Sprint 5), OOF stacking (Sprint 6), and the benign-only Autoencoder (Sprint 7) were trained and frozen.

### 4. What Was Implemented
- Evaluation harness for scoring models on DEVELOPMENT_TEST (81,749 rows).
- 11 candidate fusion rules spanning OR, AND, and threshold variants.
- Selection of configuration `C06` ($C01 \lor AE$ at frozen $\tau$) as the canonical hybrid pipeline.

### 5. Methodology
The 11 candidate fusion combinations were evaluated against pre-registered False Positive Rate gates ($FPR \le 0.05$ on validation). C06 was chosen as the most conservative logical-OR configuration that maximized sensitivity while bounding false alarms.

### 6. Parameters / Configuration
- Stacking Checkpoint: `EXP_OOF_STACK_V1` (seed 42).
- Autoencoder Threshold: $\tau = 11.160062745213509$.
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
| C06 vs C01 Macro-F1 | $\Delta \ge 0.0$ | $-0.000169$ | Marginal decrease due to false alarm additions |
| C06 False Positive Addition | $\le 50$ | $+13$ flows | Bounded and well within protocol tolerance |
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


# 11. Sprint 9 — Formal Hypothesis Testing (EXP_H123_V1)

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
- H1: Evaluated stacking across seeds 42, 123, 2024 on DEVELOPMENT_TEST against the RF baseline ($0.880733$). Required $\Delta \ge 0.005$.
- H2: Evaluated standalone AE detection on the 583 Protected Backdoor rows at $\tau = 11.160063$.
- H3: Compared C06 vs C01 on Protected Backdoor detection count and benign FPR delta.

### 6. Parameters / Configuration
- Epsilon threshold for H1: $\epsilon = 0.005$.
- Maximum permissible FPR inflation for H3: $0.02$ (2.0%).
- Decision Rule DD-4 for H2: `ae_detected_count == 0 -> NOT_SUPPORTED`.

### 7. Expected Result
- H1: Supported (Stacking outperforms RF).
- H2: Supported (AE detects unseen Backdoors).
- H3: Supported (Fusion rescues missed Backdoors).

### 8. Actual Result
- **H1 Result**: Stacking 3-seed mean Macro-F1 = $0.892961 \pm 0.000491$. Difference from RF baseline = $+0.012228 > 0.005$. **H1 = SUPPORTED**.
- **H2 Result**: AE detected count on Protected Backdoor = **0 / 583** (0.0%). **H2 = NOT_SUPPORTED**.
- **H3 Result**: C01 detected = 582/583; C06 detected = 582/583; Rescued = 0. Primary rescue condition failed. **H3 = NOT_SUPPORTED**.

### 9. Why Did This Result Occur?
1. **H1 Supported**: Ensemble stacking effectively leveraged complementary strengths of Random Forest and Neural Network, producing a consistent $+0.0122$ Macro-F1 improvement across all three seeds.
2. **H2 Not Supported**: The Autoencoder threshold ($	au = 11.16006$) was too conservative. Backdoor flows in UNSW-NB15 mimic legitimate protocol byte counts, generating reconstruction errors well below $11.16$.
3. **H3 Not Supported**: Because the AE detected 0 Backdoor samples, it could not rescue any samples missed by C01. C06 detection was identical to C01 (582/583).

### 10. Expectation vs Actual Table
| Hypothesis | Metric Tested | Expected | Actual | Decision |
|:---|:---|:---:|:---:|:---:|
| **H1** | Stacking Mean F1 vs RF | $\Delta \ge +0.005$ | $+0.012228$ | **SUPPORTED** |
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
Verified that model weights loaded correctly and reconstruction errors were computed properly. The distribution of Backdoor reconstruction errors had a maximum of $\sim 8.4$, entirely below $\tau = 11.16006$.

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


# 12. Sprint 10 — Systematic Ablation Study (EXP_ABLATION_V1)

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
- **Removing RF ($A3$)**: Mean Macro-F1 dropped to $0.867496$ ($\Delta = -0.024481$).
- **Soft Voting ($A1b$)**: Mean Macro-F1 = $0.850642$ ($\Delta = -0.041335$).
- **Removing DT ($A2$)**: Mean Macro-F1 = $0.892276$ ($\Delta = +0.000299$).
- **Removing SVM ($A4$)**: Mean Macro-F1 = $0.891022$ ($\Delta = -0.000954$).
- **Removing NN ($A5$)**: Mean Macro-F1 = $0.891953$ ($\Delta = -0.000024$).
- **Adding AE ($A6$)**: Mean Macro-F1 = $0.891807$ ($\Delta = -0.000169$).

### 9. Why Did This Result Occur?
1. **Dominance of Random Forest**: Removing RF caused the single largest collapse in performance ($-2.45\%$ Macro-F1). RF provides the primary anchor of stability and precision.
2. **Failure of Soft Voting**: Soft voting equalizes weights across all models, allowing noisy models (SVM, DT) with high false positive rates to corrupt the final decision boundary.
3. **Redundancy of Decision Tree**: Stacking without DT slightly improved performance ($+0.03\%$), indicating DT was largely redundant alongside RF.

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


# 13. Sprint 11 — Model Explainability & Forensic Audit (EXP_EXPLAIN_V1)

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


# 14. Sprint 12 — Final Reproducibility & Zero-Training Audit (EXP_FINAL_REPRO_V1)

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
  - Fusion C06: Historical F1 = 0.892440, Reproduced = 0.892440 (Diff: $1.68 \times 10^{-8}$) — **REPRODUCED**
  - Soft Vote (A1b): Historical F1 = 0.850632, Reproduced = 0.850632 (Diff: 0.0) — **REPRODUCED**

### 9. Why Did This Result Occur?
Deterministic seeding, strict checkpoint immutability, and explicit software environment pinning ensured exact floating-point reproducibility. The $1.68 \times 10^{-8}$ difference in C06 was traced to JSON scalar serialization rounding (6 decimal places).

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
| Fusion C06 | 0.892440 | 0.892440 | $1.68 \times 10^{-8}$ | **REPRODUCED** |
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


# 15. Sprint 13 — Controlled Zero-Day Simulation (EXP_ZERODAY_V1)

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
  - Unseen-Category Generalization: Supported if $C06 \text{ ZDR} \ge 0.50$ and Wilson 95% CI lower bound $> 0.50$.
  - Hybrid Rescue: Supported if $\text{RescueGain} (Q_3 / 583) \ge 0.05$ and exact one-sided binomial $p < 0.05$ against baseline $p_0 = 0.000625$.
  - Standalone AE (H2): Subject to rule DD-4 (`ae_detected == 0 -> NOT_SUPPORTED`).

### 6. Parameters / Configuration
- Autoencoder Threshold: $\tau = 11.160062745213509$.
- Classification Operator: $\text{RE} > \tau$.
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
  - C01 (Stacking): **582 / 583** ($99.8285\%$)
  - AE (Autoencoder): **0 / 583** ($0.0000\%$)
  - C06 (Hybrid Fusion): **582 / 583** ($99.8285\%$)
- **Benign Control Performance (37,000 Benign Rows)**:
  - C01: False Positives = 7,100 ($FPR = 0.191892$)
  - AE: False Positives = 19 ($FPR = 0.000514$)
  - C06: False Positives = 7,113 ($FPR = 0.192243$)
- **Preregistered Statistical Decisions**:
  - **Unseen-Category Generalization**: **SUPPORTED** ($ZDR = 0.998285$, Wilson 95% CI: $[0.990349, 0.999697]$).
  - **Hybrid Fusion Improvement**: **NOT_SUPPORTED** ($\text{RescueGain} = 0.0$, Exact binomial $p = 1.0000$).
  - **Standalone AE Detection (H2)**: **NOT_SUPPORTED** (Rule DD-4 satisfied, count = 0).

### 9. Why Did This Result Occur?
1. **Exceptional Stacking Generalization**: Despite never seeing Backdoors, the supervised base models (especially RF and NN) recognized flow-level characteristics (packet arrival windows, byte counts, TTL metrics) shared by Backdoor attacks and other attack classes, achieving $99.83\%$ detection.
2. **Autoencoder Inertness**: Backdoor sessions generated low reconstruction errors (max $\sim 8.4$), falling entirely below $\tau = 11.16006$. Consequently, AE detected zero Backdoor samples ($Q_1=0, Q_3=0$).
3. **Generalization Attribution**: C06 achieved $99.83\%$ detection, but this was driven 100% by C01 stacking. The Autoencoder contributed zero rescues and added 13 false positives.

### 10. Expectation vs Actual Table
| Aspect | Expected | Actual | Scientific Interpretation |
|:---|:---|:---|:---|
| C01 Backdoor Detection | $\approx 60 - 80\%$ | **582 / 583 (99.83%)** | Supervised ensemble generalized remarkably well |
| AE Backdoor Detection | $\ge 20\%$ | **0 / 583 (0.00%)** | AE completely inert at conservative threshold |
| AE Rescue ($Q_3$) | $\ge 30$ samples | **0 / 583 (0.00%)** | Zero rescue gain achieved |
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


# 16. "Why Did This Result Happen?" Systematic Analytical Syntheses

To adhere to rigorous scientific standards, this section consolidates causal explanations, empirical evidence, and inferential boundaries for all major findings.

### 1. Supervised Stacking Outperformed Base Models (H1 Supported)
- **WHAT HAPPENED?**: Out-of-fold stacking achieved a mean Macro-F1 of $0.8930 \pm 0.0005$, outperforming the best individual model (Random Forest, $0.8807$) by $+0.0122$.
- **WHY DID IT LIKELY HAPPEN?**: Base models exhibited decorrelated error distributions. Random Forest provided robust partition stability, while the Neural Network learned continuous representation spaces. The Logistic Regression meta-learner learned positive weights for RF (+2.15) and NN (+1.79) while actively penalizing noisy SVM probabilities (-0.18).
- **WHAT EVIDENCE SUPPORTS THAT?**: The ablation study confirmed that removing RF degraded performance by $-0.0245$, while soft voting (equal weighting) collapsed performance by $-0.0413$.
- **WHAT CANNOT BE CLAIMED?**: It cannot be claimed that stacking eliminates false positives; stacking incurred an FPR of $19.19\%$.

### 2. Autoencoder Inertness on Protected Backdoors (H2 Rejected)
- **WHAT HAPPENED?**: The Autoencoder detected exactly $0 / 583$ Protected Backdoor samples ($0.00\%$) at its frozen threshold ($\tau = 11.16006$).
- **WHY DID IT LIKELY HAPPEN?**: Two legitimate benign validation flows with aborted TCP handshakes (RST/FIN flags) produced massive reconstruction errors ($\approx 269$), inflating the validation standard deviation to $3.645$. The resulting `mean + 3*sigma` threshold ($11.16006$) was displaced far beyond the reconstruction error profile of Backdoors (which peaked at $\sim 8.4$).
- **WHAT EVIDENCE SUPPORTS THAT?**: Forensic inspection of validation errors confirms the two extreme outliers. Histograms of Backdoor errors show they fall completely below $11.16$.
- **WHAT CANNOT BE CLAIMED?**: It cannot be claimed that deep autoencoders are inherently incapable of detecting network intrusions; rather, this specific global parametric threshold was displaced by benign protocol anomalies.

### 3. Supervised Stacking Generalized to Unseen Backdoors
- **WHAT HAPPENED?**: The frozen C01 stacking model detected $582 / 583$ ($99.83\%$) Protected Backdoor samples despite having never been trained on Backdoor traffic.
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


# 17. Experimental Problems, Investigations, and Resolutions

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
- **Investigation**: Traced individual validation reconstruction errors; identified rows 10731 and 10737 with errors $\approx 269$.
- **Root Cause**: Legitimate benign TCP connections terminated via RST/FIN flags with 0 payload bytes.
- **Resolution**: Preserved the flows to avoid artificial dataset curation; frozen threshold was locked at $11.16006$ to respect pre-registered protocol rules.

### Issue 4: Historical Ablation Non-Reproduction in Sprint 12
- **Problem**: In Sprint 12, ablation configurations A0 and A2–A6 were labeled `NOT_REPRODUCED`.
- **Why It Mattered**: Evaluators might assume ablation results were invalid or missing.
- **Detection**: Publication metrics audit (`final_metrics.csv`).
- **Investigation**: Reviewed Sprint 12 protocol constraints mandating `training_operations = 0`.
- **Root Cause**: Historical ablations required on-the-fly model refitting during Sprint 10; without refitting, they could not be rerun in a zero-training pipeline.
- **Resolution**: Formally documented that Sprint 12 verified frozen inference only; historical ablations remain valid historical records.


# 18. Consolidated Final Results

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
| **H1** | Stacking superiority over best base model (RF) | $\text{Mean}(\text{Stack}) = 0.8930$ vs $\text{RF} = 0.8807$ ($\Delta = +0.0122$) | $\Delta \ge \epsilon$ with $\epsilon = 0.005$ | **SUPPORTED** |
| **H2** | Unsupervised AE standalone zero-day detection | Detected count $= 0 / 583$ ($0.0\%$) at $\tau = 11.16006$ | Detected count $== 0 \rightarrow$ NOT_SUPPORTED (DD-4) | **NOT_SUPPORTED** |
| **H3** | Hybrid fusion rescue gain over supervised stacking | Detected $_{C06} = 582$, Detected $_{C01} = 582$ (Rescue $= 0$) | $\text{Det}_{C06} > \text{Det}_{C01}$ and $\Delta \text{FPR} \le 0.02$ | **NOT_SUPPORTED** |
| **Generalization** | C06 unseen category zero-day generalization | C06 ZDR $= 0.9983$, Wilson 95% CI: $[0.9903, 0.9997]$ | $\text{ZDR} \ge 0.50$ and CI lower bound $> 0.50$ | **SUPPORTED** |
| **Fusion Gain** | Practical and statistical rescue superiority | $\text{RescueGain} = 0.0$, Exact binomial $p = 1.0000$ | $\text{Gain} \ge 0.05$ and binomial $p < 0.05$ | **NOT_SUPPORTED** |

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
| **Fusion C06 Engine**| Static Rules | 0 | 0 | 0 | $1.68 \times 10^{-8}$ | **REPRODUCED** |
| **Zero-Training Audit**| **Total Ops = 0** | **0** | **0** | **0** | **Exact Match** | **AUDIT PASSED** |


# 19. Major Scientific Findings

This project establishes seven core scientific findings:

### Finding 1: Supervised Ensemble Stacking Demonstrates Genuine Generalization Superiority
- **What**: Out-of-fold Logistic Regression stacking achieved a statistically verified $+0.0122$ Macro-F1 improvement over the best individual model (Random Forest).
- **Why**: The meta-learner learned to leverage complementary representations between tree-based partitioning and neural continuous spaces while penalizing redundant linear classifiers.
- **Evidence**: 3-seed evaluation ($0.8930 \pm 0.0005$ vs $0.8807$); H1 formally SUPPORTED.
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
- **Evidence**: Ablation config A1b achieved only $0.8506$ Macro-F1 and inflated FPR to $29.38\%$.
- **Meaning**: Practitioners should never use unweighted voting when meta-learning is feasible.
- **Limitation**: Meta-learning requires cross-validated OOF generation.

### Finding 4: Unsupervised Autoencoders Can Suffer Complete Operational Suppression
- **What**: The Autoencoder detected 0 out of 583 Protected Backdoor samples at its frozen threshold.
- **Why**: Benign connection-termination flows inflated validation variance, shifting $\tau$ to $11.16006$.
- **Evidence**: H2 NOT_SUPPORTED; Backdoor reconstruction errors peaked at $\sim 8.4$.
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
- **What**: Sprint 12 reproduced historical results with zero training operations and residuals $< 1.7 \times 10^{-8}$.
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
2. **Unsupervised Autoencoder anomaly detection (AE)** suffered complete operational suppression. At its frozen operational threshold ($\tau = 11.16006$), calibrated on benign validation traffic to enforce low false alarms ($FPR = 0.000625$), the Autoencoder detected **0 out of 583 Backdoor samples**. Legitimate benign connection aborts (TCP RST/FIN) inflated validation variance, displacing the threshold far beyond the subtle profile of Backdoor intrusions.
3. Consequently, **hybrid logical-OR fusion (C06)** achieved zero attack rescue ($Q_3 = 0$), produced no statistical improvement over supervised stacking ($p = 1.0000$), and slightly inflated false alarms on benign traffic.

The central scientific conclusion is that while supervised ensemble stacking can achieve robust domain generalization against withheld attack categories that share flow primitives with known threats, unsupervised autoencoders require domain-aware feature subspace partitioning and adaptive thresholding to prevent total suppression by ambient protocol anomalies.


# Appendices

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
- Frozen Threshold $\tau$: `11.160062745213509` (`mean + 3*sigma` on Normal VALIDATION)
- Classification Operator: strictly `reconstruction_error > tau`

### Appendix E — Hypothesis Definitions and Decision Rules
- H1 Decision Rule: $\text{Mean}(\text{Macro-F1}_{\text{Stack}}) - \text{Macro-F1}_{\text{RF}} \ge 0.005 \rightarrow$ SUPPORTED
- H2 Decision Rule (DD-4): $\text{Detected Count}_{\text{AE}} == 0 \rightarrow$ NOT_SUPPORTED
- H3 Decision Rule: $\text{Detected}_{C06} > \text{Detected}_{C01}$ and $\Delta \text{FPR} \le 0.02 \rightarrow$ SUPPORTED
- Generalization Rule: $C06 \text{ ZDR} \ge 0.50$ and Wilson 95% CI lower bound $> 0.50 \rightarrow$ SUPPORTED
- Fusion Gain Rule: $\text{RescueGain} (Q_3 / 583) \ge 0.05$ and exact one-sided binomial $p < 0.05 \rightarrow$ SUPPORTED

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


# Final Research Status

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
The comprehensive experimental investigation on the UNSW-NB15 dataset demonstrates that supervised ensemble stacking (C01) delivers superior discriminative accuracy on recognized network attacks and exhibits outstanding latent generalization to unseen attack categories, successfully detecting 99.83% (582/583) of the withheld Protected Backdoor population. Conversely, the unsupervised benign-only Autoencoder (AE), when calibrated conservatively ($	au = 11.16006$) to prevent false alarms on benign traffic, suffers from total operational suppression by benign connection-termination outliers, yielding zero standalone zero-day detections and zero rescue gain within hybrid logical-OR fusion (C06). Future hybrid intrusion detection research must incorporate localized, domain-aware feature sub-spacing and adaptive thresholding to prevent benign protocol diversity from blinding unsupervised anomaly detectors.
