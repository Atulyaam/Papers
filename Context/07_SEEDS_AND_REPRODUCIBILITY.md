# Seeds and Reproducibility

## 1. Central Seed Utility

Create one project utility:

```text
src/utils/reproducibility.py
```

with a central function conceptually named:

```python
set_all_seeds(seed)
```

All stochastic project components should use this policy.

## 2. Primary Experiment Seeds

```text
42
123
2024
```

These are the three seeds used for H1 mean ± std reporting.

## 3. Seed Responsibilities

The experiment seed governs, where applicable:

- train/validation split
- cross-validation fold assignment
- stochastic base-model settings
- NN initialization/training
- AE initialization/training
- sampling operations

SHAP background sampling must use an explicit deterministic seed policy.

## 4. Reproducibility Requirements

For a rerun with identical:

- dataset fingerprints
- code commit
- configuration
- protocol version
- seed

the experiment should produce the same results within expected numerical tolerance.

## 5. Do Not Scatter Randomness

Avoid ad-hoc calls such as:

```python
np.random.seed(...)
random.seed(...)
```

throughout unrelated files.

Use the central reproducibility utility and explicit `random_state` parameters where supported by libraries.

## 6. Reproducibility Metadata

Every experiment records the effective seed and protocol information in `metadata.json`.
