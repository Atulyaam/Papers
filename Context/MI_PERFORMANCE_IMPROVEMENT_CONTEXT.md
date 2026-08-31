# MI Performance Improvement Context

## Current State

Project:
Hybrid Stacked-Ensemble Intrusion Detection System

Dataset:
UNSW-NB15

Frozen:
- Sprint 1
- Sprint 2
- Sprint 3

Sprint 4 MI:
- Experiment: EXP_MI_V1_1
- Protocol: v1.1
- Current selected K: 75
- Sanity status: REVIEW_REQUIRED
- Human decision: ACCEPT K=75
- Do not add K=175/193
- Do not continue expanding MI grid

Frozen TRAIN:
- 162,395 rows
- Normal: 44,800 (27.59%)
- Attack: 117,595 (72.41%)

Encoded feature space:
- Total: 193
- OHE/discrete: 154
- Numeric/continuous: 39

MI:
- target = label
- representation = encoded + unscaled
- estimator = mutual_info_classif
- n_neighbors = 3
- random_state = 42

K candidates:
10, 20, 30, 40, 50, 75, 100, 150

Mean Macro-F1:
- K=10  -> 0.824852
- K=20  -> 0.864436
- K=30  -> 0.897442
- K=40  -> 0.916198
- K=50  -> 0.919560
- K=75  -> 0.919799
- K=100 -> 0.919775
- K=150 -> 0.919750

Final K decision:
K=75

The curve is technically non-decreasing within the configured sanity
tolerance, but performance effectively plateaus around K=75. Human review
accepted K=75 under the predefined highest-mean-Macro-F1 rule.

## Interpretation

The ~0.920 Macro-F1 is NOT the final IDS result.

It is the Macro-F1 from the fixed Logistic Regression evaluator used only
for training-only MI K selection.

The final research system still consists of:
- Decision Tree
- Random Forest
- SVM
- Neural Network
- 5-fold OOF stacking
- Logistic Regression meta-learner
- benign-trained Autoencoder
- 2x2 fusion
- SHAP

## Performance Improvement Strategy

Future F1 improvement should focus on the downstream model/ensemble
pipeline, not repeated post-hoc MI tuning.

Potential improvement areas:
- base-model hyperparameters
- class weighting
- model-specific training strategy
- probability calibration
- OOF stacking quality
- meta-learner configuration
- decision threshold where scientifically permitted
- AE/fusion contribution

Any improvement must use training-only procedures.

## Absolute Rules

Never use:
- protected Backdoor TEST
- development TEST
- final outer VALIDATION for repeated supervised tuning
- excluded training Backdoor archive

to choose performance improvements.

Do not:
- change dataset
- change withheld attack
- merge archived Backdoor rows into development
- change frozen Sprints 1–3
- keep expanding the MI K grid just to increase F1

## Current Workflow

Performance improvement must follow:

PLAN
→ DESIGN
→ DISCUSSION
→ FINAL PROMPT
→ IMPLEMENT
→ TEST
→ VALIDATE
→ FREEZE

No automatic implementation.

Current position:
Sprint 4 MI improvement/review stage.

The final Sprint 4 freeze prompt has NOT yet been approved or issued.