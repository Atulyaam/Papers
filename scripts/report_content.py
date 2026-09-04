"""
scripts/report_content.py
-------------------------
Comprehensive narrative content generator for the Complete UNSW-NB15 Research Report.
Generates:
1. The full, unabridged source markdown document: UNSW_NB15_Complete_Research_Report.md
2. Structured section flowable specifications for ReportLab PDF compilation in build_research_report_pdf.py

Strictly adheres to all 44 prompt requirements, authoritative data, and academic standards.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.report_data import (
    PROJECT_METADATA, DATASET_COUNTS, SPLIT_ARCHITECTURE,
    FEATURE_SELECTION_DATA, BASE_MODELS_DATA, AUTOENCODER_DATA,
    STACKING_DATA, H123_DATA, ABLATION_DATA, REPRODUCIBILITY_DATA,
    ZERODAY_DATA, TIMELINE_DATA
)

def build_executive_summary_qa() -> List[Dict[str, str]]:
    """Builds the 15 mandatory questions and authoritative answers for Executive Summary."""
    return [
        {
            "q": "1. What problem does the project solve?",
            "a": "The project investigates the vulnerability of Network Intrusion Detection Systems (NIDS) to novel, zero-day network intrusions and explores whether an unsupervised anomaly detection autoencoder can rescue attacks missed by supervised ensemble classifiers under rigorous data-leakage controls."
        },
        {
            "q": "2. Why is intrusion detection important?",
            "a": "Modern enterprise networks face an expanding surface of sophisticated, polymorphic cyberattacks. Supervised classifiers trained only on known attack signatures often fail against novel intrusion variants, leading to silent network breaches and catastrophic infrastructure compromise."
        },
        {
            "q": "3. Why was UNSW-NB15 selected?",
            "a": "UNSW-NB15 was chosen because it provides modern, realistic network traffic generated via the IXIA PerfectStorm tool, containing 9 distinct contemporary attack families (e.g., Backdoors, Fuzzers, Exploits) alongside contemporary normal traffic, overcoming outdated artifacts of legacy datasets like KDD-Cup 99."
        },
        {
            "q": "4. What system was developed?",
            "a": "A hybrid multi-branch architecture combining 4 supervised base classifiers (Decision Tree, Random Forest, SVM, and Neural Network), an Out-of-Fold (OOF) Logistic Regression meta-learner (C01), and a benign-only deep reconstruction Autoencoder (AE), fused via an inclusive logical-OR decision rule (C06)."
        },
        {
            "q": "5. What models were used?",
            "a": "Four diverse supervised base models (Decision Tree, Random Forest with 300 trees, Linear Support Vector Machine, and a 2-hidden-layer MLP Neural Network named IDSNet), an Out-of-Fold Logistic Regression meta-classifier, and a symmetric 75->12->6->12->75 Tabular Autoencoder."
        },
        {
            "q": "6. Why was a hybrid architecture investigated?",
            "a": "Supervised models achieve high discriminative precision on known attack patterns but risk blind spots on zero-day attacks. An unsupervised Autoencoder trained strictly on benign traffic evaluates deviation from normality, theoretically providing a safety net to rescue novel attack variants missed by the supervised branch."
        },
        {
            "q": "7. What experiments were performed?",
            "a": "Across Sprints 7 to 13, experiments included: Autoencoder benign-only reconstruction training and threshold calibration (Sprint 7), baseline evaluation foundation (Sprint 8), formal H1/H2/H3 hypothesis testing across 3 seeds (Sprint 9), systematic 8-configuration ablation study (Sprint 10), post-hoc SHAP and AE explainability with forensic provenance auditing (Sprint 11), zero-training frozen reproducibility verification (Sprint 12), and controlled zero-day simulation using the isolated Backdoor population (Sprint 13)."
        },
        {
            "q": "8. What were the major findings?",
            "a": "Supervised OOF stacking (C01) proved exceptionally robust, achieving Macro-F1 = 0.8930 on development-test and detecting 582 out of 583 unseen Backdoor samples (99.83%). Conversely, the Autoencoder was completely inert at its frozen conservative threshold (tau = 11.16006), detecting 0/583 Backdoor samples and yielding zero rescue gain."
        },
        {
            "q": "9. What hypotheses were supported?",
            "a": "Hypothesis H1 was SUPPORTED: Learned OOF stacking outperformed the best single base model (Random Forest) by +0.0122 in Macro-F1, comfortably exceeding the pre-registered threshold (epsilon = 0.005). Furthermore, unseen-category generalization was formally SUPPORTED (C06 ZDR = 0.9983, Wilson 95% CI: [0.9903, 0.9997])."
        },
        {
            "q": "10. What hypotheses were not supported?",
            "a": "Hypothesis H2 was NOT SUPPORTED (standalone AE detected 0/583 Backdoors, rule DD-4). Hypothesis H3 was NOT SUPPORTED (C06 achieved identical detection to C01, yielding 0 additional rescues). Formal fusion improvement was NOT SUPPORTED (exact binomial p = 1.0000 against baseline p0 = 0.000625)."
        },
        {
            "q": "11. What did the zero-day experiment show?",
            "a": "The zero-day experiment on 583 isolated Backdoor samples decomposed into Quadrants: Q1=0 (both detect), Q2=582 (C01 only), Q3=0 (AE rescue), and Q4=1 (both miss). The supervised stacking model generalized remarkably well on its own to the unseen category, while the Autoencoder provided no incremental coverage."
        },
        {
            "q": "12. What did the AE contribute?",
            "a": "At its operational frozen threshold (tau = 11.16006, calibrated on Normal VALIDATION to bound FPR <= 0.001), the AE contributed zero attack rescues (Q3 = 0). It added 13 false positives on benign test traffic, slightly inflating overall FPR from 0.191892 (C01) to 0.192243 (C06)."
        },
        {
            "q": "13. What did stacking contribute?",
            "a": "Stacking delivered superior generalization over any single base classifier and drastically outperformed simple soft voting (+0.0413 Macro-F1). Logistic Regression meta-learning effectively weighted Random Forest (coef ~ +2.15) and Neural Network (coef ~ +1.79) while penalizing redundant signals."
        },
        {
            "q": "14. What are the major limitations?",
            "a": "The zero-day evaluation used Backdoor as a controlled proxy (not a guarantee for arbitrary real-world zero-days). Tabular flows lacked raw packet temporal sequencing. The AE threshold was heavily inflated by benign TCP RST/FIN connection-termination outliers, suppressing AE sensitivity to subtle payloads."
        },
        {
            "q": "15. What is the final scientific contribution?",
            "a": "The project delivers an empirically rigorous, fully auditable benchmark demonstrating that while hybrid ensemble stacking provides exceptional domain generalization to withheld network attacks, unsupervised autoencoders require domain-aware feature subspace partitioning and adaptive thresholding to avoid complete suppression by benign protocol anomalies."
        }
    ]

print("report_content.py helper loaded.")
