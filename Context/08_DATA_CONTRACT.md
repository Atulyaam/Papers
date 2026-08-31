# UNSW-NB15 Data Contract

## Purpose

The data contract is a machine-readable description of the expected UNSW-NB15 schema.

It prevents silent schema drift and keeps preprocessing assumptions explicit.

## Planned Artifacts

```text
configs/data_schema.yaml
data/audit/dataset_schema.json
```

## Contract Contents

The final contract must document, based on the actual files:

- required columns
- optional columns
- target column
- attack-category column
- categorical columns
- numerical columns
- identifying/excluded columns
- allowed target values
- missing-value policy
- non-finite-value policy

## Important Principle

Do not hard-code NSL-KDD assumptions such as:

```text
41 raw features
18 selected features
```

The final UNSW-NB15 values must come from the actual dataset audit.

## Categorical Policy

Known categorical fields in UNSW-NB15 may include:

- proto
- service
- state

The actual final list must be confirmed from the acquired files.

Planned one-hot safety:

```python
OneHotEncoder(handle_unknown="ignore")
```

## Validation

The schema validator should stop the pipeline when required fields are missing or incompatible.

Unknown test categories should not crash the encoder and must be covered by an explicit test.
