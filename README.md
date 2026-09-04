# UNSW-NB15 Intrusion Detection System — Complete Research Repository

[![Project Status: Frozen](https://img.shields.io/badge/Status-FROZEN-success.svg)](#research-audit--reproducibility-sign-off)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](#installation--environment-setup)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-green.svg)](https://scikit-learn.org/)
[![Validation Gates](https://img.shields.io/badge/Validation%20Gates-44%2F44%20Passed-brightgreen.svg)](#research-audit--reproducibility-sign-off)
[![Zero Training Audit](https://img.shields.io/badge/Zero%20Training-0%20Refit%20Ops-blue.svg)](#zero-training-compliance)
[![Publication Report](https://img.shields.io/badge/Research%20Report-29%20Pages%20PDF-red.svg)](UNSW_NB15_Complete_Research_Report.pdf)

A publication-oriented, empirically rigorous experimental investigation into the generalization capabilities of **supervised ensemble stacking**, **unsupervised anomaly detection autoencoders**, and **hybrid logical-OR fusion** on the **UNSW-NB15** network intrusion dataset under strict data-leakage controls and a controlled zero-day simulation.

---

## Table of Contents

- [1. Executive Summary & Core Findings](#1-executive-summary--core-findings)
- [2. Master Research Deliverables](#2-master-research-deliverables)
- [3. Pre-Registered Hypotheses & Verdicts](#3-pre-registered-hypotheses--verdicts)
- [4. Complete System Architecture](#4-complete-system-architecture)
- [5. Dataset & Active Split Architecture](#5-dataset--active-split-architecture)
- [6. Installation & Environment Setup](#6-installation--environment-setup)
- [7. Step-by-Step Reproduction Guide](#7-step-by-step-reproduction-guide)
- [8. Repository Directory Layout](#8-repository-directory-layout)
- [9. Computational Lineages Reconciled](#9-computational-lineages-reconciled)
- [10. Sprint-by-Sprint Research Map](#10-sprint-by-sprint-research-map)
- [11. Research Audit & Reproducibility Sign-Off](#11-research-audit--reproducibility-sign-off)
- [12. Citation & Scientific Integrity](#12-citation--scientific-integrity)

---

## 1. Executive Summary & Core Findings

Modern Network Intrusion Detection Systems (NIDS) frequently struggle to detect novel, unobserved attack families. This project investigated whether combining a **supervised multi-model ensemble** with an **unsupervised deep reconstruction Autoencoder (AE)** via **logical-OR fusion** can reliably rescue zero-day attacks without causing catastrophic false positive rate (FPR) inflation.

```
                                    +-----------------------------------------+
                                    |        END-TO-END HYBRID SYSTEM         |
                                    +-----------------------------------------+
                                                         |
                   +-------------------------------------+-------------------------------------+
                   |                                                                           |
      [SUPERVISED STACKING: C01]                                                   [AUTOENCODER & FUSION: C06]
                   |                                                                           |
• Mean Macro-F1 = 0.892961 (+0.0122 vs RF)                                • Benign Val Threshold: tau = 11.160063
• Random Forest meta-weight: +2.1458                                      • Backdoor Detected: 0 / 583 (0.00%)
• Neural Network meta-weight: +1.7892                                     • Benign Test FPR: 0.0514% (19 / 37,000)
• Protected Backdoor: 582 / 583 (99.83%)                                  • Attack Rescue (Q3): 0 / 583 (0.00%)
• Generalization: SUPPORTED (Wilson CI > 0.50)                            • Fusion Improvement: NOT_SUPPORTED (p = 1.000)
```

### Key Headline Results:
1. **Supervised Stacking Generalization:** Multi-seed Out-of-Fold (OOF) stacking achieved **0.8930 Macro-F1** on known attack traffic, statistically outperforming the best base model (Random Forest, 0.8807) by $+0.0122$ ($\Delta \ge 0.005$). In the controlled zero-day study on 583 permanently isolated Backdoor samples, the frozen stacking classifier ($C01$) correctly identified **582 out of 583 attacks (99.83% Zero-Day Detection Rate)** by leveraging fundamental protocol flow primitives shared across attack classes.
2. **Autoencoder Operational Inertness:** The unsupervised Autoencoder ($AE$) detected **0 out of 583 Backdoor attacks** ($0.00\%$) at its frozen operational threshold ($\tau = 11.160063$). Forensic investigation revealed that benign connection-termination anomalies (TCP RST/FIN packets with extreme byte ratios up to $\text{RE} \approx 269$) displaced the parametric threshold ($\mu + 3\sigma$) far outward, rendering the model insensitive to subtle Backdoor sessions ($\text{RE} \le 8.4$).
3. **Zero Hybrid Rescue Gain:** Logical-OR fusion ($C06 = C01 \lor AE$) yielded **0 additional attack rescues** ($Q_3 = 0$), while adding 13 false positives on benign test traffic. Naive hybrid fusion superiority is therefore disproven under operational benign thresholding.
4. **Strict Reproducibility:** Sprint 12 and Sprint 13 executed under **zero retraining, zero refitting, zero recalibration, and zero OOF regeneration**, achieving 100% deterministic bitwise reproduction.

---

## 2. Master Research Deliverables

| Deliverable | File Path | Format & Details | Description |
|:---|:---|:---:|:---|
| **Master Research Report (PDF)** | [`UNSW_NB15_Complete_Research_Report.pdf`](UNSW_NB15_Complete_Research_Report.pdf) | Print-Ready A4 PDF (29 Pages, 1.96 MB) | The complete, standalone scientific publication report documenting the full research story across Sprints 7–13. |
| **Complete Source Report (Markdown)** | [`UNSW_NB15_Complete_Research_Report.md`](UNSW_NB15_Complete_Research_Report.md) | GFM Markdown (1,940 Lines, 105 KB) | The full markdown text containing all 21 mandatory sections per sprint, tables, and forensic audits. |
| **Publication Figures & Diagrams** | [`report_assets/figures/`](report_assets/figures/) | 15 High-DPI Charts (300 DPI) | All authoritative and newly synthesized ROC, PR, ablation, architecture, and timeline charts. |
| **Visual Inspection Pages** | [`report_assets/inspections/`](report_assets/inspections/) | 29 PNG Images | High-resolution page-by-page renders verifying zero text clipping, zero table overflow, and proper formatting. |

---

## 3. Pre-Registered Hypotheses & Verdicts

The project strictly adhered to pre-registered evaluation criteria without post-hoc threshold adjustment:

| Hypothesis | Research Question Tested | Authoritative Evidence | Decision Rule | Final Verdict |
|:---|:---|:---|:---|:---:|
| **H1** | Supervised stacking superiority over best base classifier (Random Forest) | Mean Stacking F1 = $0.8930$ vs RF = $0.8807$ ($+0.0122$) | $\Delta \ge 0.005$ | **SUPPORTED** |
| **H2** | Standalone Autoencoder zero-day anomaly detection | Detected count = $0 / 583$ ($0.00\%$) at $\tau = 11.160063$ | `ae_detected_count == 0` | **NOT_SUPPORTED** |
| **H3** | Hybrid fusion rescue over stacking without FPR inflation | $C06 = 582$ vs $C01 = 582$ (Rescued = $0$, $+13$ FPs) | $C06 > C01 \land \Delta\text{FPR} \le 0.02$ | **NOT_SUPPORTED** |
| **Generalization** | Unseen-category zero-day generalization | $C06$ ZDR = $0.998285$, Wilson 95% CI: $[0.9903, 0.9997]$ | $\text{ZDR} \ge 0.50 \land \text{CI}_{\text{lower}} > 0.50$ | **SUPPORTED** |
| **Fusion Gain** | Statistical hybrid rescue superiority | $\text{RescueGain} = 0.0$, Exact Binomial $p = 1.0000$ | $\text{Gain} \ge 0.05 \land p < 0.05$ | **NOT_SUPPORTED** |

---

## 4. Complete System Architecture

The pipeline processes raw flow attributes through a leakage-free feature selection engine into two parallel branches:

```
                          Raw UNSW-NB15 Records (257,673 flows)
                                           |
                                           v
                       Leakage-Free 4-Way Splitting & UID Tracking
                     [TRAIN | VALIDATION | DEV_TEST | PROT_BACKDOOR]
                                           |
                                           v
                        One-Hot Encoding & StandardScaler
                               (193 candidate features)
                                           |
                                           v
                      Mutual Information Feature Selection (K=75)
                                           |
                 +-------------------------+-------------------------+
                 |                                                   |
                 v                                                   v
    Supervised Base Classifiers                       Unsupervised Autoencoder
    - Decision Tree (gini, depth=15)                  - Topology: 75 -> 12 -> 6 -> 12 -> 75
    - Random Forest (300 trees, sqrt)                 - Benign-only TRAIN training
    - Linear SVM (C=0.1, dual=False)                  - Threshold: tau = 11.160063 (Val mu + 3*sigma)
    - Neural Network (IDSNet: 128-64 MLP)                            |
                 |                                                   v
                 v                                           Binary Decision:
      5-Fold OOF Predictions                                 RE(x) > 11.160063
                 |                                                   |
                 v                                                   |
     Logistic Regression Meta-Learner                                |
                 |                                                   |
                 v                                                   |
       Stacking Decision (C01)                                       |
                 |                                                   |
                 +-------------------------+-------------------------+
                                           |
                                           v
                               Logical-OR Fusion Engine
                                   C06 = C01 OR AE
                                           |
                                           v
                             Evaluation & Zero-Day Audit
                                  (EXP_ZERODAY_V1)
```

---

## 5. Dataset & Active Split Architecture

The official UNSW-NB15 dataset (257,673 rows) was partitioned into four mutually disjoint partitions with exact row conservation to guarantee a pristine zero-day evaluation harness:

| Partition Name | Total Rows | Benign (0) | Attack (1) | Attack Families | Research Role & Access Boundary |
|:---|:---:|:---:|:---:|:---:|:---|
| **`TRAIN`** | 162,395 | 44,800 | 117,595 | 8 (Zero Backdoors) | Supervised training, scaler fitting, OOF cross-validation. |
| **`VALIDATION`** | 11,200 | 11,200 | 0 | 0 (Pure Benign) | Unsupervised Autoencoder threshold calibration ($\tau$). |
| **`DEVELOPMENT_TEST`**| 81,749 | 37,000 | 44,749 | 8 (Zero Backdoors) | Held-out known-attack benchmark evaluation. |
| **`PROTECTED_BACKDOOR`**| 583 | 0 | 583 | 1 (Backdoor Only) | Isolated zero-day proxy (never seen during development). |
| **`EXCLUDED_BACKDOOR`** | 1,746 | 0 | 1,746 | 1 (Backdoor Only) | Permanently purged training Backdoors (prevent leakage). |

**Total Conservation Verification:**
$$\text{Official Train: } 162,395 + 11,200 + 1,746 = 175,341 \quad \checkmark$$
$$\text{Official Test: } 81,749 + 583 = 82,332 \quad \checkmark$$

---

## 6. Installation & Environment Setup

### Prerequisites
- Python 3.11+
- Git
- 8 GB+ RAM recommended

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/IDS-UNSW-NB15.git
cd IDS-UNSW-NB15
```

### 2. Set Up Python Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Raw Dataset Placement
Ensure the official raw UNSW-NB15 CSV files are located in `data/raw/`:
```
data/raw/
├── UNSW_NB15_training-set.csv   (175,341 rows)
├── UNSW_NB15_testing-set.csv    (82,332 rows)
└── UNSW-NB15_features.csv       (Feature metadata)
```

---

## 7. Step-by-Step Reproduction Guide

All frozen experiments and report assets can be reproduced using the verified script pipeline:

### Step 1: Pre-Flight Integrity & Determinism Checks
Run automated pre-flight checks to ensure environments, splits, and seeds are intact:
```bash
python scripts/pre_commit_checks.py
python scripts/run_determinism_check.py
```

### Step 2: Execute Test Suite
Run the 39 unit and integration test suites:
```bash
pytest tests/ -v
```

### Step 3: Run Zero-Training Audit
Verify that no training or threshold-refitting operations occur during evaluation:
```bash
python scripts/audit_zero_training.py
```
*Expected Output:*
```
training_operations = 0
fit_calls = 0
partial_fit_calls = 0
optimizer_steps = 0
backward_passes = 0
threshold_recalibrations = 0
OOF_regeneration = 0
AUDIT STATUS: PASSED
```

### Step 4: Reproduce Sprint 12 Frozen Benchmarks
Execute the bitwise frozen inference benchmark across all base models, stacking, and fusion:
```bash
python scripts/run_sprint12_final_reproducibility.py
```
*Outputs archived in:* `results/final_reproducibility/EXP_FINAL_REPRO_V1/`

### Step 5: Execute Sprint 13 Controlled Zero-Day Simulation
Verify pre-conditions and run the zero-day evaluation on the 583 isolated Backdoor flows:
```bash
python scripts/run_sprint13_preflight.py
python scripts/run_sprint13_zero_day.py
```
*Outputs archived in:* `results/zero_day/EXP_ZERODAY_V1/`

### Step 6: Generate Publication Figures & Recompile Report
Re-generate all publication-quality figures and compile the 29-page master PDF and Markdown reports:
```bash
python scripts/generate_report_figures.py
python scripts/build_full_report.py
```
*Generated Deliverables:*
- `UNSW_NB15_Complete_Research_Report.pdf`
- `UNSW_NB15_Complete_Research_Report.md`
- `report_assets/inspections/page_*.png`

---

## 8. Repository Directory Layout

```
IDS-UNSW-NB15/
├── UNSW_NB15_Complete_Research_Report.pdf   # Master 29-page publication report PDF
├── UNSW_NB15_Complete_Research_Report.md    # Complete unabridged source markdown report
├── requirements.txt                         # Certified dependency specifications
├── configs/
│   ├── project_config.yaml                  # Global project seeds, paths, thresholds
│   └── data_schema.yaml                     # Column types, continuous/discrete definitions
├── data/
│   ├── raw/                                 # Official raw UNSW-NB15 CSV datasets
│   ├── splits/                              # 4-way isolated split CSVs and SHA metadata
│   └── audit/                               # Pre-split and row conservation audit logs
├── docs/                                    # Sprint design notes, decision logs, RFCs
├── report_assets/
│   ├── figures/                             # High-DPI publication figures (fig01 to fig15)
│   └── inspections/                         # Rendered PNG pages for visual QA (page_01 to 29)
├── results/
│   ├── base_models/                         # DT, RF, SVM, NN checkpoints & benchmark metrics
│   ├── autoencoder/                         # EXP_AE_V1 checkpoint, scaler, threshold json
│   ├── stacking/                            # EXP_OOF_STACK_V1 meta-learner models (seeds 42, 123, 2024)
│   ├── fusion/                              # EXP_FUSION_V1 2x2 candidate fusion evaluation
│   ├── evaluation/                          # EXP_H123_V1 formal hypothesis evaluation
│   ├── ablation/                            # EXP_ABLATION_V1 systematic configurations A0 to A6
│   ├── explainability/                      # EXP_EXPLAIN_V1 SHAP feature importances & forensic audit
│   ├── final_reproducibility/               # EXP_FINAL_REPRO_V1 frozen inference verification
│   └── zero_day/                            # EXP_ZERODAY_V1 controlled zero-day metrics & preflight
├── scripts/                                 # Reproduction, evaluation, and report pipelines
├── src/                                     # Modular core Python library
│   ├── evaluation/                          # Metrics, statistical tests, hypothesis evaluators
│   ├── explainability/                      # SHAP kernel and tree explainer harnesses
│   ├── feature_selection/                   # Mutual Information selection and plateau search
│   ├── fusion/                              # Logical-OR and soft-voting fusion operators
│   ├── models/                              # PyTorch Autoencoder, IDSNet, Scikit-learn wrappers
│   ├── preprocessing/                       # Encoders, scalers, and split protocols
│   └── utils/                               # Hashing, seed control, determinism helpers
└── tests/                                   # 39 Pytest test suites (leakage, models, logic)
```

---

## 9. Computational Lineages Reconciled

The repository rigorously documents and preserves two distinct computational lineages:

| Attribute | Sprint 10 Ablation Lineage (A1) | Sprint 12/13 Canonical Frozen Lineage (C01) |
|:---|:---:|:---:|
| **Evaluation Model** | Dynamically refitted meta-learner | Canonical frozen checkpoint (`meta_learner.joblib`) |
| **Benign Test False Positives** | **7,201 / 37,000** | **7,100 / 37,000** |
| **Empirical Benign FPR** | **0.194622 (19.46%)** | **0.191892 (19.19%)** |
| **Backdoor Detection** | 582 / 583 (99.83%) | 582 / 583 (99.83%) |
| **Provenance Note** | Dynamic refit across base models during ablation sweep. | Bitwise frozen evaluation using seed-42 serialized weights. |

*Resolution:* Audited in `results/zero_day/EXP_ZERODAY_V1/audit/pre_freeze_crosscheck_audit.json`. The 101 false-positive difference is an authentic model-lineage distinction, fully detailed in Chapter 10, Chapter 13, and Appendix J of the master report.

---

## 10. Sprint-by-Sprint Research Map

| Sprint | Experiment ID | Primary Objective | Milestone Tag | Status |
|:---|:---|:---|:---:|:---:|
| **Sprint 1** | `EXP_DATA_ACQUISITION_AUDIT` | Acquisition, row-count validation, column schema audit | `sprint1-audit` | Complete |
| **Sprint 2** | `EXP_FEATURE_SELECTION_V1` | One-hot encoding schema, continuous feature retention | `sprint2-features` | Complete |
| **Sprint 3** | `EXP_DATA_SPLIT_V1` | 4-way isolation, Backdoor withholding, leakage checks | `sprint3-splits` | Complete |
| **Sprint 4** | `EXP_MI_V1_1` | Mutual Information 5-fold CV, K=75 plateau selection | `sprint4-mi-k75` | Complete |
| **Sprint 5** | `EXP_BASE_MODELS_V1` | DT, RF, Linear SVM, IDSNet MLP neural network | `sprint5-basemodels`| Complete |
| **Sprint 6** | `EXP_OOF_STACK_V1` | 5-Fold Stratified OOF meta-learning across seeds | `sprint6-stacking` | Complete |
| **Sprint 7** | `EXP_AE_V1` | Benign-only 75->12->6->12->75 Autoencoder, $\tau=11.16006$ | `sprint7-ae` | Complete |
| **Sprint 8** | `EXP_FUSION_V1` | Baseline evaluation foundation, C06 selection | `sprint8-fusion` | Complete |
| **Sprint 9** | `EXP_H123_V1` | Formal hypothesis testing (H1, H2, H3) across seeds | `sprint9-freeze` | Complete |
| **Sprint 10**| `EXP_ABLATION_V1` | 8-configuration ablation study (A0 to A6) | `sprint10-freeze`| Complete |
| **Sprint 11**| `EXP_EXPLAIN_V1` | SHAP explainability, AE provenance discovery & quarantine | `sprint11-freeze`| Complete |
| **Sprint 12**| `EXP_FINAL_REPRO_V1`| Bitwise frozen reproducibility & zero-training audit | `sprint12-freeze`| Complete |
| **Sprint 13**| `EXP_ZERODAY_V1` | Controlled zero-day simulation on 583 Backdoor flows | `sprint13-freeze`| **FROZEN** |

---

## 11. Research Audit & Reproducibility Sign-Off

### Checkpoint Checksums (SHA-256)
- `results/autoencoder/EXP_AE_V1/checkpoints/ae_final.pt`: `64259ae68eebfa8cf0ee...`
- `results/stacking/EXP_OOF_STACK_V1/seed_42/meta_learner.joblib`: `045239e248a313627bfd...`
- `results/base_models/EXP_BASE_MODELS_V1/neural_network/nn_final.pt`: `d9016e3c3958bb59f232...`

### Automated Validation Gate Audit
- **Validation Gates Passed:** **44 / 44 (100% Clean Pass)**
- **Audit Script:** `scripts/run_sprint13_preflight.py`
- **Zero-Training Audit:** 0 training operations, 0 fit calls, 0 optimizer steps.

---

## 12. Citation & Scientific Integrity

If you use this repository, benchmark results, or methodology in your research, please cite:

```bibtex
@techreport{unsw_nb15_ids_research_2026,
  title       = {UNSW-NB15 Intrusion Detection System: Complete Research and Experiment Report},
  author      = {Advanced Agentic Coding Research Team},
  institution = {IDS Research Group},
  year        = {2026},
  month       = {September},
  note        = {Publication-Oriented Experimental Documentation, Sprints 7--13. Commit f694e19}
}
```

### Research Ethics & Scope Note
The Protected Backdoor evaluation was conducted as a **controlled unseen-category proxy** under strict data withholding protocols. It does not constitute a universal claim of detection for arbitrary future zero-day attacks outside this evaluated distribution. Unsupervised Autoencoder behavior reflects global parametric thresholding on tabular flow records.

---
*Maintained under scientific immutability standards. Commit: `f694e19e44a3dafb486ff216428f1be1f2ec9120` | Tag: `sprint13-freeze`.*
