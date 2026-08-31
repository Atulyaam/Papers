# Experiment Registry and Naming

## Experiment IDs

Use stable IDs such as:

```text
EXP_DATA_AUDIT
EXP_BASE_DT
EXP_BASE_RF
EXP_BASE_SVM
EXP_BASE_NN
EXP_SOFT_VOTE
EXP_OOF_STACK_SEED42
EXP_OOF_STACK_SEED123
EXP_OOF_STACK_SEED2024
EXP_AE_SEED42
EXP_FUSION
EXP_FULL
EXP_SHAP_L1
EXP_SHAP_L2
```

## Required Experiment Metadata

Every experiment must define:

- experiment ID
- purpose
- protocol version
- seed
- dataset hash reference
- config file
- input artifact references
- expected outputs

## Registry File

Maintain a simple machine-readable registry:

```text
experiments/experiment_registry.json
```

Example:

```json
{
  "experiments": [
    {
      "id": "EXP_OOF_STACK_SEED42",
      "status": "frozen",
      "protocol_version": "1.0"
    }
  ]
}
```

## Status Values

Recommended:

```text
PLANNED
DESIGNED
IMPLEMENTED
TESTED
VALIDATED
FROZEN
FAILED
SUPERSEDED
```

A superseded experiment is not silently deleted; its status is preserved for auditability.
