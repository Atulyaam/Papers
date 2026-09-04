# Sprint 12 — Publication-Critical Evaluation Metrics
**Experiment ID**: `EXP_FINAL_REPRO_V1`  
**Execution Timestamp**: `2026-09-04T13:45:52.398959+00:00`  
**Zero-Training Compliance**: `training_operations_executed = 0`  

| Model / Pipeline | Status | Macro Precision | Macro Recall | Macro F1 | Attack Precision | Attack Recall | Attack F1 | Balanced Accuracy | FPR |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Decision Tree (Base) | **REPRODUCED** | 0.878340 | 0.844352 | 0.849852 | 0.807289 | 0.968245 | 0.880471 | 0.844352 | 0.279541 |
| Random Forest (Base) | **REPRODUCED** | 0.903932 | 0.874944 | 0.880733 | 0.837004 | 0.980916 | 0.903264 | 0.874944 | 0.231027 |
| SVM (Base) | **REPRODUCED** | 0.851945 | 0.818906 | 0.823613 | 0.786927 | 0.948379 | 0.860142 | 0.818906 | 0.310568 |
| Neural Network (Base) | **REPRODUCED** | 0.898909 | 0.891850 | 0.894293 | 0.881341 | 0.936133 | 0.907911 | 0.891850 | 0.152432 |
| OOF Stacking (Seed 42) | **REPRODUCED** | 0.906552 | 0.887931 | 0.892609 | 0.859144 | 0.967753 | 0.910220 | 0.887931 | 0.191892 |
| OOF Stacking (Seed 123) | **REPRODUCED** | 0.906591 | 0.887935 | 0.892619 | 0.859104 | 0.967843 | 0.910237 | 0.887935 | 0.191973 |
| OOF Stacking (Seed 2024) | **REPRODUCED** | 0.907007 | 0.889071 | 0.893656 | 0.861006 | 0.966927 | 0.910898 | 0.889071 | 0.188784 |
| OOF Stacking (3-Seed Mean) | **REPRODUCED** | 0.906717 | 0.888312 | 0.892961 | 0.859751 | 0.967508 | 0.910452 | 0.888312 | 0.190883 |
| Fusion C06 (Stack 42 + AE) | **REPRODUCED** | 0.906432 | 0.887755 | 0.892440 | 0.858922 | 0.967753 | 0.910096 | 0.887755 | 0.192243 |
| Ablation A1b (Soft Vote) | **REPRODUCED** | 0.886708 | 0.844649 | 0.850632 | 0.801958 | 0.982838 | 0.883232 | 0.844649 | 0.293541 |
| A0_RF (Ablation) | *NOT_REPRODUCED* | — | — | — | — | — | — | — | — |
| A1_FULL_STACK (Ablation) | *NOT_REPRODUCED* | — | — | — | — | — | — | — | — |
| A2_NO_DT (Ablation) | *NOT_REPRODUCED* | — | — | — | — | — | — | — | — |
| A3_NO_RF (Ablation) | *NOT_REPRODUCED* | — | — | — | — | — | — | — | — |
| A4_NO_SVM (Ablation) | *NOT_REPRODUCED* | — | — | — | — | — | — | — | — |
| A5_NO_NN (Ablation) | *NOT_REPRODUCED* | — | — | — | — | — | — | — | — |
| A6_STACK_PLUS_AE (Ablation) | *NOT_REPRODUCED* | — | — | — | — | — | — | — | — |