# Project Rules — Quick Reference

## Data
- Raw files are immutable.
- Never fit anything on TEST.
- Never fit anything on protected unseen attack.
- All learned transforms are fit on TRAIN only unless the protocol explicitly says otherwise.

## Zero/Unseen Attack
- Selection rule is fixed before results.
- Eligibility threshold is fixed.
- Protected samples are stored separately.
- Protected samples are final-evaluation-only.

## Stacking
- Exactly five OOF folds for the core method.
- Meta-learner trains on OOF predictions, never in-sample predictions.
- Final base models are retrained on full permitted TRAIN data for inference.

## Autoencoder
- Train on benign TRAIN only.
- Threshold from benign VALIDATION only.
- 95th percentile is the primary operating point.
- Threshold sweep is secondary reporting.

## Fusion
- Use the deterministic 2x2 rule.
- No learned or hidden fusion behavior.

## Explainability
- Level 1: meta-level contribution.
- Level 2: raw-feature full-pipeline explanation.
- Background = 50.
- Start with 100 explained TEST instances; cap planned range at 100-150.
- Record runtime.

## Reproducibility
- Seeds: 42, 123, 2024.
- Central seed utility.
- Save config snapshot.
- Save Git commit.
- Save dataset hashes.
- Save checkpoints.
- Save selected features and thresholds.
- Generate result files programmatically.

## Development
- PLAN -> DESIGN -> REVIEW -> IMPLEMENT -> TEST -> VALIDATE -> FREEZE.
- One major feature at a time.
- No silent methodology changes.
- Every FREEZE gets a Git commit/tag.

## Results
- No manual official metric entry.
- No result-driven protocol changes.
- Do not write interpretation before the corresponding numbers exist.
