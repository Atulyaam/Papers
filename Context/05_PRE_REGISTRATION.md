# Pre-Registration Protocol

## Purpose

The pre-registration artifact is the audit record showing that the withheld/unseen attack selection was determined independently of model results.

## Artifact

```text
experiments/pre_registration.json
```

## Required Fields

```json
{
  "protocol_version": "1.0",
  "dataset": "UNSW-NB15",
  "eligibility": {
    "minimum_test_instances": 50
  },
  "selection_rule": {
    "type": "predetermined_order",
    "ordering": "alphabetical"
  },
  "candidate_subclasses": [],
  "candidate_counts": {},
  "selected_subclass": null,
  "created_at": "",
  "selection_frozen_at": ""
}
```

## Procedure

1. Audit the dataset.
2. Compute TEST counts for all attack subclasses.
3. Apply the fixed eligibility threshold.
4. Produce the complete eligible-candidate list and counts.
5. Apply the predetermined selection rule.
6. Record the selected subclass.
7. Record timestamps.
8. Freeze the file.
9. Commit the pre-registration artifact to Git before model-result evaluation.

## Prohibition

The selected subclass must not be changed because of:

- expected difficulty
- model performance
- recall
- false-positive rate
- SHAP results
- convenience
- runtime

If the protocol must be changed, create a new protocol version and clearly distinguish it from the primary experiment.
