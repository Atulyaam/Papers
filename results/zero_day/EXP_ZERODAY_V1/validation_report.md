# EXP_ZERODAY_V1 Validation Gates Report
**Protocol Version**: V1.4  
**Timestamp**: 2026-09-04T15:45:09.520890+00:00  
**Overall Status**: ALL GATES PASSED (PASS)  
**Gate Count**: 44 / 44 PASS

| Gate ID | Description | Status | Details |
|:---|:---|:---:|:---|
| `ZD-01` | Protected file SHA-256 matches frozen authoritative hash | **PASS** | `` |
| `ZD-02` | Protected row count == 583 | **PASS** | `` |
| `ZD-03` | Protected attack_cat is Backdoor only | **PASS** | `` |
| `ZD-04` | Protected label is 1 only | **PASS** | `` |
| `ZD-05` | TRAIN Backdoor count == 0 | **PASS** | `` |
| `ZD-06` | VALIDATION Backdoor count == 0 | **PASS** | `` |
| `ZD-07` | DEVELOPMENT_TEST Backdoor count == 0 | **PASS** | `` |
| `ZD-08` | Benign control count derived programmatically after ZD-PF-33 pass | **PASS** | `{'count': 37000}` |
| `ZD-09` | Combined population == 583 + benign_control_n | **PASS** | `{'count': 37583}` |
| `ZD-10` | Global source_row_uid uniqueness across splits | **PASS** | `` |
| `ZD-11` | Protected-vs-TRAIN leakage == 0 | **PASS** | `` |
| `ZD-12` | Protected-vs-VALIDATION leakage == 0 | **PASS** | `` |
| `ZD-13` | Protected-vs-benign-control leakage == 0 | **PASS** | `` |
| `ZD-14` | Frozen feature selection artifact hash verified | **PASS** | `` |
| `ZD-15` | Frozen DT checkpoint hash verified | **PASS** | `` |
| `ZD-16` | Frozen RF checkpoint hash verified | **PASS** | `` |
| `ZD-17` | Frozen SVM checkpoint & scaler hashes verified | **PASS** | `` |
| `ZD-18` | Frozen NN checkpoint & scaler hashes verified | **PASS** | `` |
| `ZD-19` | Frozen Stacking meta-learner hashes verified | **PASS** | `` |
| `ZD-20` | Frozen AE checkpoint & scaler hashes verified | **PASS** | `` |
| `ZD-21` | Frozen AE threshold verified (tau == 11.160062745213509) | **PASS** | `` |
| `ZD-22` | Frozen C06 OR-logic rule verified | **PASS** | `` |
| `ZD-23` | Zero training operations executed | **PASS** | `` |
| `ZD-24` | Zero threshold recalibrations executed | **PASS** | `` |
| `ZD-25` | No configuration selection using protected Backdoor | **PASS** | `` |
| `ZD-26` | Quadrant internal consistency: Q1 + Q2 + Q3 + Q4 == 583 | **PASS** | `{'q1': 0, 'q2': 582, 'q3': 0, 'q4': 1, 'sum': 583}` |
| `ZD-27` | C01 detected count identity: Q1 + Q2 == c01_detected | **PASS** | `` |
| `ZD-28` | AE detected count identity: Q1 + Q3 == ae_detected | **PASS** | `` |
| `ZD-29` | C06 detected count identity: Q1 + Q2 + Q3 == c06_detected | **PASS** | `` |
| `ZD-30` | Discordant pairs count identity: b + c == Q3 (b == 0) | **PASS** | `` |
| `ZD-31` | Headline generalization system is uniquely C06 | **PASS** | `` |
| `ZD-32` | Generalization threshold >= 0.50 and Wilson 95% CI lower bound > 0.50 evaluated | **PASS** | `{'zdr': 0.9982847341337907, 'ci_low': 0.9903487843853768, 'decision': 'SUPPORTED'}` |
| `ZD-33` | Fusion improvement practical threshold RescueGain >= 0.05 evaluated | **PASS** | `{'rescue_gain': 0.0, 'threshold': 0.05, 'met': False}` |
| `ZD-34` | Statistical significance threshold p < 0.05 against p0 = 0.000625 evaluated | **PASS** | `{'p_val': 1.0, 'p0': 0.000625, 'decision': 'FAIL_TO_REJECT_H0'}` |
| `ZD-35` | Dual criterion enforcement for fusion improvement verdict | **PASS** | `{'final_decision': 'NOT_SUPPORTED'}` |
| `ZD-36` | Single-family limitation explicitly recorded in metadata and reports | **PASS** | `` |
| `ZD-37` | Operational baseline limitation explicitly recorded in metadata and reports | **PASS** | `` |
| `ZD-38` | Primary rescue rate equals Q3 / 583 | **PASS** | `{'primary_rescue_rate': 0.0}` |
| `ZD-39` | Conditional rescue rate equals Q3 / (Q2 + Q3) when denominator > 0 | **PASS** | `` |
| `ZD-40` | If Q2 + Q3 == 0, conditional rescue rate is recorded as N/A | **PASS** | `` |
| `ZD-41` | Minimum integer Q3 satisfying RescueGain >= 0.05 equals 30 | **PASS** | `` |
| `ZD-42` | Statistical test uses p0 from frozen AE validation artifact, not zero-day-derived data | **PASS** | `{'p0': 0.000625}` |
| `ZD-43` | Statistical test implementation uses exact one-sided binomial test | **PASS** | `{'test_type': 'scipy.stats.binomtest'}` |
| `ZD-44` | Independence assumption is explicitly recorded in metadata and report | **PASS** | `` |
