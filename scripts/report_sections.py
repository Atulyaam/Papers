"""
scripts/report_sections.py
--------------------------
Generates all narrative text, analytical syntheses, sprint chapters (with 21 subsections each),
callout boxes, and appendix content for both the Markdown document and ReportLab PDF.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_data import (
    PROJECT_METADATA, DATASET_COUNTS, SPLIT_ARCHITECTURE,
    FEATURE_SELECTION_DATA, BASE_MODELS_DATA, AUTOENCODER_DATA,
    STACKING_DATA, H123_DATA, ABLATION_DATA, REPRODUCIBILITY_DATA,
    ZERODAY_DATA, TIMELINE_DATA
)

def get_title_markdown() -> str:
    return f"""# {PROJECT_METADATA['title']}
## {PROJECT_METADATA['subtitle']}

**Research Scope**: {PROJECT_METADATA['scope']}  
**Dataset**: {PROJECT_METADATA['dataset']}  
**Final Experiment Status**: {PROJECT_METADATA['final_experiment_status']}  
**Authoritative Final Commit**: `{PROJECT_METADATA['final_commit']}`  
**Authoritative Tag**: `{PROJECT_METADATA['final_tag']}`  
**Report Date**: {PROJECT_METADATA['report_date']}  
**Repository Root**: `{PROJECT_METADATA['authoritative_root']}`  

---
"""

def get_executive_summary_markdown(qa_list) -> str:
    md = [
        "# Executive Summary\n",
        "This report provides the definitive, publication-oriented documentation of the research journey conducted on the UNSW-NB15 dataset across Sprints 7 through 13. The project evaluates supervised multi-model stacking, an unsupervised benign-only Autoencoder (AE), and hybrid logical-OR fusion against known and controlled zero-day network intrusions under strict data-leakage controls.\n\n",
        "### Key Research Questions & Core Synthesis\n\n"
    ]
    for item in qa_list:
        md.append(f"**{item['q']}**  \n*{item['a']}*\n\n")
    return "".join(md)

def get_introduction_markdown() -> str:
    return """# 1. Project Introduction & Research Formulation

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

- **Hypothesis 1 (H1 — Supervised Stacking Superiority)**: Multi-seed Out-of-Fold Logistic Regression stacking achieves a mean Macro-F1 on the held-out Development-Test set that exceeds the best individual base classifier (Random Forest) by at least $\\epsilon = 0.005$:
  $$\\text{Mean}(\\text{Macro-F1}_{\\text{Stack}}) - \\text{Macro-F1}_{\\text{RF}} \\ge 0.005$$

- **Hypothesis 2 (H2 — Unsupervised Autoencoder Standalone Anomaly Detection)**: The unsupervised Autoencoder trained strictly on benign traffic can detect a non-zero, statistically meaningful count of unseen Protected Backdoor attacks at its frozen operational threshold:
  $$\\text{Detected Count}_{\\text{AE}} > 0 \\quad \\text{subject to decision rule DD-4}$$

- **Hypothesis 3 (H3 — Hybrid Fusion Rescue Efficacy)**: The hybrid logical-OR fusion system $C06 = C01 \\lor AE$ increases attack detection on the Protected Backdoor population over the supervised stacking system $C01$ without inflating false positive rate on benign traffic by more than 2 percentage points:
  $$\\text{Detected}_{C06} > \\text{Detected}_{C01} \\quad \\text{and} \\quad \\Delta \\text{FPR} \\le 0.02$$

> [!IMPORTANT]
> Hypotheses H1, H2, and H3 were formulated with strict mathematical decision rules prior to test evaluation. No thresholds, operators, or sample boundaries were altered after inspection of results.
"""

def get_dataset_markdown() -> str:
    return f"""# 2. UNSW-NB15 Dataset & Rigorous Split Architecture

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
"""

def get_features_markdown() -> str:
    return """# 3. Feature Engineering & Feature Selection

## 3.1 Preprocessing Pipeline
Raw network flows contain both numeric metrics and high-cardinality nominal variables (`proto`, `service`, `state`). Preprocessing followed strict leakage-prevention standards:
1. **One-Hot Encoding**: Nominal attributes were encoded using a frozen schema fitted strictly on the TRAIN partition. Rare categories not observed in TRAIN were mapped to an out-of-vocabulary representation.
   - Raw features: 42 attributes (excluding ID and targets).
   - Post one-hot encoding: 193 candidate features (154 discrete binary columns, 39 continuous numeric columns).
2. **Feature Normalization**: Continuous variables were standardized using z-score normalization ($(\\mu, \\sigma)$). Scaling statistics were computed strictly on the TRAIN partition and applied immutably to VALIDATION, DEVELOPMENT_TEST, and PROTECTED_BACKDOOR.

## 3.2 Mutual Information Selection Methodology (EXP_MI_V1_1)
Mutual Information (MI) was selected as the non-parametric feature scoring metric due to its ability to capture non-linear relationships without distributional assumptions. MI scoring was performed using `mutual_info_classif` with $k=3$ nearest neighbors on unscaled encoded features of the TRAIN set.

Candidate feature subset sizes $K \\in \\{10, 20, 30, 40, 50, 75, 100, 150\\}$ were evaluated using 5-fold Stratified Cross-Validation on TRAIN with a fixed, balanced Logistic Regression evaluator.

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
"""

def get_architecture_markdown() -> str:
    return """# 4. Complete System Architecture

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
2. **Supervised Base Classifiers**: Four diverse learners map input features $\\mathbf{x} \\in \\mathbb{R}^{75}$ to attack probability estimates $\\hat{p}_i \\in [0, 1]$.
3. **Out-of-Fold Stacking Engine**: Generates an unbiased cross-validated probability matrix $\\mathbf{P}_{\\text{OOF}} \\in \\mathbb{R}^{N \\times 4}$ across the 162,395 training rows without label leakage.
4. **Logistic Regression Meta-Learner (C01)**: Learns optimal combining weights $\\mathbf{w}$ and bias $b$ to produce the ensemble decision:
   $$\\hat{y}_{\\text{C01}} = \\mathbb{I}\\left(\\sigma\\left(\\sum_{i=1}^4 w_i \\hat{p}_i + b\\right) > 0.5\\right)$$
5. **Unsupervised Autoencoder (AE)**: Maps input $\\mathbf{x}$ through bottleneck latent space $\\mathbf{z} \\in \\mathbb{R}^6$ to reconstruct $\\mathbf{\\hat{x}} \\in \\mathbb{R}^{75}$. Computes Mean Squared Error:
   $$\\text{RE}(\\mathbf{x}) = \\frac{1}{75} \\sum_{j=1}^{75} (x_j - \\hat{x}_j)^2$$
   Produces binary anomaly output: $\\hat{y}_{\\text{AE}} = \\mathbb{I}(\\text{RE}(\\mathbf{x}) > \\tau)$.
6. **Logical-OR Fusion Engine (C06)**: Synthesizes supervised and unsupervised outputs:
   $$\\hat{y}_{\\text{C06}} = \\hat{y}_{\\text{C01}} \\lor \\hat{y}_{\\text{AE}}$$
   This construction guarantees that an attack detected by either subsystem is flagged, creating the theoretical potential for the Autoencoder to "rescue" samples missed by stacking.
"""

def get_base_models_markdown() -> str:
    return """# 5. Supervised Base Classifiers

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
"""

def get_autoencoder_markdown() -> str:
    return """# 6. Unsupervised Benign-Only Autoencoder (EXP_AE_V1)

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
The decision threshold $\\tau$ was calibrated strictly on the 11,200 benign flows of the VALIDATION partition to guarantee bounded false positive rates:
- Validation Reconstruction Error distribution: Mean = 0.2252, Std = 3.6450, Median = 0.0659, 95th percentile = 0.5674, 99th percentile = 1.5122, 99.9th percentile = 10.6969.
- **Operational Frozen Threshold**: Calibrated under the pre-registered `mean + 3*sigma` rule:
  $$\\tau = 0.225201 + 3 \\times 3.644954 = 11.160062745213509$$
- **Classification Operator**: Strictly $\\text{RE}(\\mathbf{x}) > \\tau$.
- **Validation Operating Behavior**: Exactly 7 out of 11,200 benign validation flows exceeded $\\tau$, yielding an empirical validation FPR of $7 / 11200 = 0.000625$ (0.0625%).

## 6.4 Observed Inertness and Root Cause Analysis
During held-out testing, the Autoencoder exhibited complete operational inertness, detecting 0 out of 583 Protected Backdoor attacks and only 19 out of 37,000 benign test flows.

**Why did this occur?**
1. **Benign Connection-Termination Outliers**: Forensic inspection of the VALIDATION set revealed two extreme normal flows: row 10731 (RE = 269.03) and row 10737 (RE = 269.09). These legitimate flows corresponded to short, aborted TCP connections terminated via RST/FIN packets with zero payload bytes.
2. **Variance Inflation**: Because standard deviation is sensitive to quadratic deviations, these two outlier flows single-handedly inflated the validation standard deviation from $\\sim 0.35$ to $3.6450$.
3. **Threshold Displacement**: The resulting `mean + 3*sigma` threshold was displaced outward to $11.16006$, an order of magnitude higher than the 99th percentile (1.5122).
4. **Subtle Attack Invisibility**: Backdoor intrusions in UNSW-NB15 mimic standard interactive sessions (SSH/Telnet), generating small, structured packet exchanges with reconstruction errors typically between $0.8$ and $3.5$. Because the frozen threshold was pushed to $11.16$, all 583 Backdoor attacks fell harmlessly below the threshold.
"""

def get_stacking_markdown() -> str:
    return """# 7. Out-of-Fold Ensemble Stacking

## 7.1 Stacking Methodology and Leakage Prevention
Traditional ensemble stacking trains base models on the training set, predicts on the same training set, and trains a meta-learner on those predictions. This causes catastrophic meta-learner overfitting because base models produce artificially optimistic predictions on data they have memorized.

To prevent meta-level target leakage, we implemented strict 5-Fold Stratified Out-of-Fold (OOF) prediction:
1. The 162,395-row TRAIN partition was divided into 5 stratified folds (seed 7).
2. For each fold $k \\in \\{1, \\dots, 5\\}$, each base model was trained on 4 folds (129,916 rows) and evaluated on the held-out fold (32,479 rows).
3. The resulting out-of-fold probability predictions were concatenated into an unbiased prediction matrix $\\mathbf{P}_{\\text{OOF}} \\in \\mathbb{R}^{162,395 \\times 4}$.
4. The meta-learner was trained strictly on $\\mathbf{P}_{\\text{OOF}}$.

## 7.2 Meta-Learner Architecture and Multi-Seed Stability
The meta-learner is a regularized Logistic Regression classifier (`lbfgs` solver, $C=1.0$, balanced class weights). To evaluate numerical stability, stacking was executed across three independent random seeds: 42, 123, and 2024.

- **Seed 42 Macro-F1**: 0.892609 (FPR = 0.191892, False Positives = 7,100 / 37,000)
- **Seed 123 Macro-F1**: 0.892619 (FPR = 0.191973, False Positives = 7,103 / 37,000)
- **Seed 2024 Macro-F1**: 0.893656 (FPR = 0.188784, False Positives = 6,985 / 37,000)
- **Three-Seed Mean Macro-F1**: $0.892961 \\pm 0.000491$

## 7.3 Learned Base Model Attributions
Inspection of the canonical seed-42 meta-learner regression coefficients reveals how the ensemble synthesizes base model outputs:
- **Random Forest ($w_{\\text{RF}} = +2.1458$)**: Heavily prioritized as the most dependable positive indicator.
- **Neural Network ($w_{\\text{NN}} = +1.7892$)**: Substantial positive weight, contributing non-linear representation capability.
- **Decision Tree ($w_{\\text{DT}} = +0.3542$)**: Modest positive weight, providing marginal rule-based refinements.
- **Support Vector Machine ($w_{\\text{SVM}} = -0.1824$)**: Negative coefficient. Because SVM outputs correlated heavily with RF while exhibiting higher false-positive noise (31.06%), the meta-learner penalizes raw SVM probabilities to suppress false alarms.
- **Bias Intercept ($b = -1.2405$)**: Conservative baseline offset.
"""

def get_fusion_markdown() -> str:
    return """# 8. Hybrid Logical Fusion Architecture

## 8.1 Fusion Formulation: C01 vs C06
The primary pipeline incorporates two distinct system configurations:
- **C01 (Canonical Supervised Stacking)**: The frozen seed-42 Out-of-Fold Logistic Regression meta-learner evaluating the four base models.
- **C06 (Hybrid Inclusive Fusion)**: The logical-OR synthesis of C01 and the Autoencoder:
  $$\\hat{y}_{\\text{C06}} = \\hat{y}_{\\text{C01}} \\lor \\hat{y}_{\\text{AE}}$$

## 8.2 The "Rescue" Concept and Mathematical Properties
The logical-OR operator has distinct mathematical properties:
1. **Monotonic Positive Prediction**: $C06(\\mathbf{x}) \\ge C01(\\mathbf{x})$ for all $\\mathbf{x}$. C06 can never classify a sample as benign if C01 classified it as an attack.
2. **False Negative Reduction (Rescue)**: If an attack sample is missed by C01 ($C01 = 0$) but flagged by the Autoencoder ($AE = 1$), C06 successfully flags the attack ($C06 = 1$). This specific transition is formally defined as a **Rescue**.
3. **False Positive Penalty**: If a benign sample is correctly classified by C01 ($C01 = 0$) but generates an Autoencoder anomaly ($AE = 1$), C06 incurs a false alarm.

## 8.3 Quadrant Decomposition on Protected Zero-Day Traffic
When evaluating C01 and AE on the 583 Protected Backdoor samples, all samples map into four mutually exclusive quadrants:
- **Quadrant 1 ($Q_1$) — Dual Detection**: $C01 = 1$ and $AE = 1$. Both systems detect the attack. AE provides redundant confirmation.
- **Quadrant 2 ($Q_2$) — Stacking Solitary Detection**: $C01 = 1$ and $AE = 0$. The supervised model detects the attack; AE fails to flag it.
- **Quadrant 3 ($Q_3$) — Autoencoder Rescue**: $C01 = 0$ and $AE = 1$. The supervised model fails, but the Autoencoder detects it. **This is the sole metric of hybrid rescue value.**
- **Quadrant 4 ($Q_4$) — Complete Evasion**: $C01 = 0$ and $AE = 0$. Both systems fail to detect the intrusion.

$$\\text{RescueGain} = \\frac{Q_3}{N_{\\text{Backdoor}}} = \\frac{Q_3}{583}$$

Under the pre-registered protocol, hybrid fusion is clinically justified if and only if $Q_3$ demonstrates statistically significant rescue beyond the validation false-alarm rate ($p_0 = 0.000625$).
"""

print("report_sections.py loaded successfully.")
