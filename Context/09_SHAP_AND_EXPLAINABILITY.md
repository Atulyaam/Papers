# Explainability Plan

## Level 1 — Meta-Learner Explanation

Explain how the four base-model outputs contribute to the logistic-regression meta-learner.

Conceptual inputs:

```text
DT output
RF output
SVM output
NN output
```

The explanation is about the meta-level decision, not a claim that these four inputs are the original raw features.

Where appropriate, use model-matched explainers for underlying base-model analyses.

## Level 2 — Full-Pipeline Raw-Feature Explanation

Wrap the full inference pipeline:

```text
raw selected features
      -> base models
      -> meta-learner
      -> final probability
```

Use a KernelExplainer-based approach with:

- background size: 50
- explained instances: initially 100
- maximum planned range: 100-150
- fixed sampling seed

Benchmark wall-clock runtime before increasing the sample count.

## Runtime Reporting

Record:

- background size
- explained instance count
- seed
- wall-clock runtime
- software/model configuration

## AE Evidence

For flows flagged as suspected unseen attack, use per-feature reconstruction error as lightweight anomaly evidence.

## Interpretation Rule

Do not write claims about the most important features until actual SHAP results exist.
