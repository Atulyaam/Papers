# Development Process

## 1. Mandatory Feature Lifecycle

Every feature, from dataset audit through SHAP, follows:

```text
1. PLAN
2. DESIGN
3. REVIEW
4. IMPLEMENT
5. TEST
6. VALIDATE
7. FREEZE
```

## 2. PLAN

Define:

- objective
- user/research need
- inputs
- outputs
- dependencies
- constraints
- success criteria
- leakage risks
- expected artifacts

No coding during PLAN.

## 3. DESIGN

Define:

- architecture
- data flow
- file/module ownership
- interfaces
- edge cases
- configuration
- logging
- checkpointing
- test strategy

## 4. REVIEW

Before implementation ask:

- Can this leak TEST information?
- Can the protected unseen-attack set enter fitting?
- Is randomness controlled?
- Can the output be reproduced?
- Are unknown categories handled?
- Are errors observable in logs?
- Can artifacts be linked to an experiment ID/config/commit?
- Does the design match the frozen methodology?

## 5. IMPLEMENT

Implementation must match the approved design.

Methodology must not be silently changed during coding.

If a design change is necessary:
1. stop;
2. document the reason;
3. revise the design;
4. review again;
5. then implement.

## 6. TEST

At minimum:

- unit tests
- integration tests
- data-contract tests
- leakage tests
- reproducibility tests where applicable

## 7. VALIDATE

Validation asks whether the implemented feature satisfies its approved design and acceptance criteria.

A feature can be code-complete but not validated.

## 8. FREEZE

A feature is frozen only when:

- implementation passes tests
- validation passes
- artifacts are saved
- configuration is captured
- provenance metadata exists
- Git commit/tag is created

## 9. One Feature at a Time

Do not implement multiple major features simultaneously.

Example:

```text
Data loader -> freeze
Schema validator -> freeze
Audit -> freeze
Preprocessing -> freeze
...
```

This makes failures easy to localize.
