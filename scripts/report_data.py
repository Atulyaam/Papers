"""
scripts/report_data.py
----------------------
Authoritative data constants, metrics, hashes, and configurations
for the UNSW-NB15 Intrusion Detection System Research Report.
Strictly sourced from frozen experiment artifacts (Sprint 7–13).
"""

# -----------------------------------------------------------------------------
# 1. PROJECT METADATA & ARTIFACT REPOSITORY
# -----------------------------------------------------------------------------
PROJECT_METADATA = {
    "title": "UNSW-NB15 Intrusion Detection System",
    "subtitle": "Publication-Oriented Experimental Documentation (Sprint 7–Sprint 13)",
    "dataset": "UNSW-NB15 Network Intrusion Dataset",
    "scope": "Comprehensive Supervised Stacking, Benign-Only Autoencoder, Hybrid Fusion, and Controlled Zero-Day Evaluation",
    "final_experiment_status": "FROZEN (Sprint 13 — EXP_ZERODAY_V1)",
    "report_date": "September 2026",
    "authoritative_root": "C:\\Users\\Atul2\\OneDrive\\Desktop\\Papers\\IDS-UNSW-NB15",
    "final_commit": "f694e19e44a3dafb486ff216428f1be1f2ec9120",
    "final_tag": "sprint13-freeze"
}

# -----------------------------------------------------------------------------
# 2. DATASET & SPLIT MANIFEST
# -----------------------------------------------------------------------------
DATASET_COUNTS = {
    "original_train_rows": 175341,
    "original_test_rows": 82332,
    "total_rows": 257673,
    "class_mapping": {"0": "Benign (Normal)", "1": "Attack"},
    "train_source_sha256": "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
    "test_source_sha256": "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559"
}

SPLIT_ARCHITECTURE = {
    "train": {
        "name": "TRAIN",
        "rows": 162395,
        "normal_rows": 44800,
        "attack_rows": 117595,
        "backdoor_rows": 0,
        "role": "Supervised base model training, OOF generation, and meta-learner training",
        "sha256": "4a259324e604f013287a5de5fe49c46bf19418d815b550c5d1a5820b569ac41c"
    },
    "validation": {
        "name": "VALIDATION",
        "rows": 11200,
        "normal_rows": 11200,
        "attack_rows": 0,
        "backdoor_rows": 0,
        "role": "Benign-only AE reconstruction error threshold calibration and sanity checks",
        "sha256": "13caf21a076a33f50243f48f404b7e7525969f71d4b9d7c0f3768aef23589180"
    },
    "development_test": {
        "name": "DEVELOPMENT_TEST",
        "rows": 81749,
        "normal_rows": 37000,
        "attack_rows": 44749,
        "backdoor_rows": 0,
        "role": "Held-out evaluation of base models, stacking, and fusion on known attack categories",
        "sha256": "04725e85732ab2fc6d9eaaa6105418b22b083b5c651067e7b0785464f414e508"
    },
    "protected_backdoor": {
        "name": "PROTECTED_BACKDOOR",
        "rows": 583,
        "normal_rows": 0,
        "attack_rows": 583,
        "backdoor_rows": 583,
        "role": "Strictly isolated zero-day proxy population for final unseen attack generalization study",
        "sha256": "6ffd23479b575e438ad90678268f40f674a663c2b9507aaf65089623397a9d91"
    },
    "excluded_train_backdoor": {
        "name": "EXCLUDED_TRAIN_BACKDOOR",
        "rows": 1746,
        "normal_rows": 0,
        "attack_rows": 1746,
        "backdoor_rows": 1746,
        "role": "Archived and withheld Backdoor rows from training set to prevent any training leakage",
        "sha256": "b3f6e7e60c9815a53f40eb2d41df8b67d29f884b922a487c3fe83c02e0db0a02"
    }
}

# -----------------------------------------------------------------------------
# 3. FEATURE SELECTION (EXP_MI_V1_1)
# -----------------------------------------------------------------------------
FEATURE_SELECTION_DATA = {
    "experiment_id": "EXP_MI_V1_1",
    "protocol_version": "1.1",
    "method": "Mutual Information (mutual_info_classif, n_neighbors=3, seed=42)",
    "evaluator": "LogisticRegression (liblinear, C=1.0, balanced, seed=42) on 5-fold Stratified CV",
    "raw_features": 42,
    "one_hot_encoded_features": 193,
    "discrete_features": 154,
    "continuous_features": 39,
    "selected_k": 75,
    "selection_criterion": "Highest mean inner-CV Macro-F1 (0.919799) & onset of performance plateau",
    "selected_breakdown": {
        "continuous_numeric": 39,
        "proto_one_hot": 25,
        "service_one_hot": 6,
        "state_one_hot": 5
    },
    "k_curve": [
        {"k": 10, "macro_f1": 0.824852, "std": 0.003435},
        {"k": 20, "macro_f1": 0.864436, "std": 0.002428},
        {"k": 30, "macro_f1": 0.897442, "std": 0.000917},
        {"k": 40, "macro_f1": 0.916198, "std": 0.002122},
        {"k": 50, "macro_f1": 0.919560, "std": 0.002323},
        {"k": 75, "macro_f1": 0.919799, "std": 0.002393},
        {"k": 100, "macro_f1": 0.919775, "std": 0.002436},
        {"k": 150, "macro_f1": 0.919750, "std": 0.002506}
    ],
    "top_features": [
        "sbytes", "sttl", "dbytes", "ct_state_ttl", "dttl",
        "sload", "dload", "rate", "dur", "smean"
    ]
}

# -----------------------------------------------------------------------------
# 4. BASE MODELS (EXP_BASE_MODELS_V1)
# -----------------------------------------------------------------------------
BASE_MODELS_DATA = {
    "dt": {
        "name": "Decision Tree",
        "role": "High-interpretability tree partitioning",
        "config": "criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight='balanced'",
        "macro_f1": 0.849852,
        "precision": 0.878340,
        "recall": 0.844352,
        "balanced_acc": 0.844352,
        "fpr": 0.279541,
        "runtime_s": 9.42
    },
    "rf": {
        "name": "Random Forest",
        "role": "Ensemble bagging over decorrelated decision trees",
        "config": "n_estimators=300, max_features=0.3, criterion='gini', max_depth=None, min_samples_leaf=1, class_weight='balanced'",
        "macro_f1": 0.880733,
        "precision": 0.903932,
        "recall": 0.874944,
        "balanced_acc": 0.874944,
        "fpr": 0.231027,
        "runtime_s": 31.22
    },
    "svm": {
        "name": "Support Vector Machine",
        "role": "Linear maximum-margin hyperplane separator",
        "config": "LinearSVC, C=0.1, class_weight='balanced', max_iter=5000",
        "macro_f1": 0.823613,
        "precision": 0.851945,
        "recall": 0.818906,
        "balanced_acc": 0.818906,
        "fpr": 0.310568,
        "runtime_s": 79.32
    },
    "nn": {
        "name": "Neural Network (IDSNet)",
        "role": "Non-linear multi-layer representation learning",
        "config": "MLP [75 -> 128 -> 64 -> 2], ReLU, Adam, lr=0.001, weight_decay=0.0001, epoch=18",
        "macro_f1": 0.894293,
        "precision": 0.898909,
        "recall": 0.891850,
        "balanced_acc": 0.891850,
        "fpr": 0.152432,
        "runtime_s": 284.21
    }
}

# -----------------------------------------------------------------------------
# 5. AUTOENCODER (EXP_AE_V1)
# -----------------------------------------------------------------------------
AUTOENCODER_DATA = {
    "experiment_id": "EXP_AE_V1",
    "architecture": "75 -> 12 -> 6 -> 12 -> 75",
    "input_dim": 75,
    "latent_dim": 6,
    "param_count": 2049,
    "activations": "ReLU hidden, Linear output",
    "batchnorm": False,
    "dropout": False,
    "training_rows": 40320,  # Normal TRAIN (seed 42 split)
    "monitor_rows": 4480,
    "calibration_rows": 11200,  # Normal VALIDATION
    "loss": "MSELoss",
    "optimizer": "Adam (lr=0.001, weight_decay=0.0001, batch_size=256)",
    "best_epoch": 133,
    "checkpoint_sha256": "4ab66af8d4a6e61212ef5d78360f30a8caa68aa85dac3d54042218e010f9a1d6",
    "scaler_sha256": "c0128d42ed9ef5be695f261be75155e7de4ddf8e51b926e3ce516c4a88ad8211",
    "frozen_threshold": 11.160062745213509,
    "threshold_rule": "mean + 3 * sigma",
    "operator": "reconstruction_error > tau",
    "val_re_stats": {
        "mean": 0.225201,
        "std": 3.644954,
        "min": 0.009684,
        "p50": 0.065888,
        "p95": 0.567386,
        "p99": 1.512164,
        "p999": 10.696876,
        "max": 269.161896
    },
    "val_samples_above_tau": 7,
    "val_fpr": 0.000625,  # 7 / 11200
    "candidate_thresholds": [
        {"name": "p95", "val": 0.567386, "val_fp": 560, "val_fpr": 0.050000},
        {"name": "p99", "val": 1.512164, "val_fp": 112, "val_fpr": 0.010000},
        {"name": "p999", "val": 10.696876, "val_fp": 12, "val_fpr": 0.001071},
        {"name": "mean+2sigma", "val": 7.515109, "val_fp": 23, "val_fpr": 0.002054},
        {"name": "mean+3sigma (Frozen)", "val": 11.160063, "val_fp": 7, "val_fpr": 0.000625}
    ]
}

# -----------------------------------------------------------------------------
# 6. OOF STACKING & MULTI-SEED (EXP_OOF_STACK_V1)
# -----------------------------------------------------------------------------
STACKING_DATA = {
    "experiment_id": "EXP_OOF_STACK_V1",
    "folds": 5,
    "oof_seed": 7,
    "meta_learner": "LogisticRegression (lbfgs, C=1.0, balanced, max_iter=1000)",
    "training_samples": 162395,
    "seeds": [42, 123, 2024],
    "oof_results": {
        "42": {"macro_f1": 0.946958, "balanced_acc": 0.954778, "fpr": 0.049933},
        "123": {"macro_f1": 0.947290, "balanced_acc": 0.955103, "fpr": 0.049487},
        "2024": {"macro_f1": 0.947476, "balanced_acc": 0.955264, "fpr": 0.049308},
        "mean_macro_f1": 0.947242,
        "std_macro_f1": 0.000263
    },
    "dev_test_results": {
        "42": {"macro_f1": 0.892609, "precision": 0.906552, "recall": 0.887931, "fpr": 0.191892, "fp": 7100},
        "123": {"macro_f1": 0.892619, "precision": 0.906591, "recall": 0.887935, "fpr": 0.191973, "fp": 7103},
        "2024": {"macro_f1": 0.893656, "precision": 0.907007, "recall": 0.889071, "fpr": 0.188784, "fp": 6985},
        "mean_macro_f1": 0.892961,
        "std_macro_f1": 0.000491
    },
    "meta_weights_seed42": {
        "DT": 0.3542,
        "RF": 2.1458,
        "SVM": -0.1824,
        "NN": 1.7892,
        "Intercept": -1.2405
    }
}

# -----------------------------------------------------------------------------
# 7. SPRINT 9 HYPOTHESIS TESTING (EXP_H123_V1)
# -----------------------------------------------------------------------------
H123_DATA = {
    "experiment_id": "EXP_H123_V1",
    "seeds": [42, 123, 2024],
    "epsilon": 0.005,
    "h1": {
        "question": "Does multi-seed OOF stacking outperform the best single base model (RF) by at least epsilon on Development-Test?",
        "stacking_mean_macro_f1": 0.892961,
        "rf_baseline_macro_f1": 0.880733,
        "delta": 0.012228,
        "verdict": "SUPPORTED",
        "reason": "Observed delta (+0.012228) exceeds pre-registered epsilon (+0.005000)."
    },
    "h2": {
        "question": "Can the unsupervised Autoencoder alone detect a meaningful fraction of Protected Backdoor (unseen) attacks at frozen tau?",
        "ae_detected": 0,
        "ae_total": 583,
        "detection_rate": 0.000000,
        "tau": 11.160063,
        "rule": "ae_detected_count == 0 -> NOT_SUPPORTED (Rule DD-4)",
        "verdict": "NOT_SUPPORTED",
        "reason": "AE detected 0/583 Protected Backdoor rows at the frozen threshold."
    },
    "h3": {
        "question": "Does C06 (C01 OR AE) hybrid fusion improve detection over supervised stacking C01 on Protected Backdoor without exceeding 2% FPR inflation?",
        "c01_detected": 582,
        "c06_detected": 582,
        "rescued_count": 0,
        "dev_test_fpr_delta": 0.000351,
        "fpr_cap": 0.02,
        "rule": "C06 detected_count == C01 detected_count; primary rescue condition fails",
        "verdict": "NOT_SUPPORTED",
        "reason": "C06 achieved identical detection (582/583) to C01; zero additional unseen attacks were rescued."
    }
}

# -----------------------------------------------------------------------------
# 8. SPRINT 10 ABLATION STUDY (EXP_ABLATION_V1)
# -----------------------------------------------------------------------------
ABLATION_DATA = {
    "experiment_id": "EXP_ABLATION_V1",
    "reference_system": "A1_FULL_STACK",
    "configs": [
        {"id": "A0_RF", "desc": "RF baseline alone", "mean_f1": 0.881618, "std_f1": 0.000071, "delta_from_a1": -0.010359, "fpr": 0.229189, "backdoor_det": 582},
        {"id": "A1_FULL_STACK", "desc": "Full 4-model stacking (dynamically fitted)", "mean_f1": 0.891977, "std_f1": 0.000210, "delta_from_a1": 0.000000, "fpr": 0.194874, "backdoor_det": 582},
        {"id": "A1b_SOFT_VOTE", "desc": "Equal-weight soft voting", "mean_f1": 0.850642, "std_f1": 0.000234, "delta_from_a1": -0.041335, "fpr": 0.293775, "backdoor_det": 582},
        {"id": "A2_NO_DT", "desc": "Stacking without Decision Tree", "mean_f1": 0.892276, "std_f1": 0.000418, "delta_from_a1": 0.000299, "fpr": 0.194144, "backdoor_det": 582},
        {"id": "A3_NO_RF", "desc": "Stacking without Random Forest", "mean_f1": 0.867496, "std_f1": 0.001089, "delta_from_a1": -0.024481, "fpr": 0.232766, "backdoor_det": 578},
        {"id": "A4_NO_SVM", "desc": "Stacking without SVM", "mean_f1": 0.891022, "std_f1": 0.000565, "delta_from_a1": -0.000954, "fpr": 0.199748, "backdoor_det": 582},
        {"id": "A5_NO_NN", "desc": "Stacking without Neural Network", "mean_f1": 0.891953, "std_f1": 0.000320, "delta_from_a1": -0.000024, "fpr": 0.194874, "backdoor_det": 582},
        {"id": "A6_STACK_PLUS_AE", "desc": "Stacking + AE fusion (OR rule)", "mean_f1": 0.891807, "std_f1": 0.000211, "delta_from_a1": -0.000169, "fpr": 0.195225, "backdoor_det": 582}
    ],
    "lineage_note": "A1 dynamically fitted seed-42 produced FP=7,201/37,000 (FPR=0.194622). Canonical frozen C01 produced FP=7,100/37,000 (FPR=0.191892). This 101-FP difference represents dynamic refitting versus frozen checkpoint evaluation."
}

# -----------------------------------------------------------------------------
# 9. SPRINT 12 REPRODUCIBILITY AUDIT (EXP_FINAL_REPRO_V1)
# -----------------------------------------------------------------------------
REPRODUCIBILITY_DATA = {
    "experiment_id": "EXP_FINAL_REPRO_V1",
    "verification_mode": "FROZEN_INFERENCE_PIPELINE",
    "zero_training_audit": {
        "training_operations_executed": 0,
        "estimator_fit_calls": 0,
        "estimator_fit_transform_calls": 0,
        "partial_fit_calls": 0,
        "optimizer_step_calls": 0,
        "backward_passes": 0,
        "threshold_recalibrations": 0,
        "oof_fold_regeneration": 0,
        "preprocessing_pipeline_fit": 1,  # Permitted frozen categorical schema setup
        "compliance": "PASS"
    },
    "reproduced_components": [
        {"model": "Decision Tree (Base)", "hist_f1": 0.849852, "repro_f1": 0.849852, "diff": 0.0, "status": "REPRODUCED"},
        {"model": "Random Forest (Base)", "hist_f1": 0.880733, "repro_f1": 0.880733, "diff": 0.0, "status": "REPRODUCED"},
        {"model": "SVM (Base)", "hist_f1": 0.823613, "repro_f1": 0.823613, "diff": 0.0, "status": "REPRODUCED"},
        {"model": "Neural Network (Base)", "hist_f1": 0.894293, "repro_f1": 0.894293, "diff": 0.0, "status": "REPRODUCED"},
        {"model": "OOF Stacking (Seed 42)", "hist_f1": 0.892609, "repro_f1": 0.892609, "diff": 0.0, "status": "REPRODUCED"},
        {"model": "OOF Stacking (Seed 123)", "hist_f1": 0.892619, "repro_f1": 0.892619, "diff": 0.0, "status": "REPRODUCED"},
        {"model": "OOF Stacking (Seed 2024)", "hist_f1": 0.893656, "repro_f1": 0.893656, "diff": 0.0, "status": "REPRODUCED"},
        {"model": "OOF Stacking (3-Seed Mean)", "hist_f1": 0.892961, "repro_f1": 0.892961, "diff": 0.0, "status": "REPRODUCED"},
        {"model": "Fusion C06 (Stack 42 + AE)", "hist_f1": 0.892440, "repro_f1": 0.892440, "diff": 1.68e-8, "status": "REPRODUCED"},
        {"model": "Ablation A1b (Soft Vote)", "hist_f1": 0.850632, "repro_f1": 0.850632, "diff": 0.0, "status": "REPRODUCED"}
    ],
    "not_reproduced_ablations": [
        "A0_RF (Ablation)", "A1_FULL_STACK (Ablation)", "A2_NO_DT (Ablation)",
        "A3_NO_RF (Ablation)", "A4_NO_SVM (Ablation)", "A5_NO_NN (Ablation)", "A6_STACK_PLUS_AE (Ablation)"
    ],
    "not_reproduced_reason": "Sprint 12 strictly verified frozen checkpoint inference with zero model refitting. Historical ablation models A0 and A2-A6 were dynamically fit during Sprint 10 and do not have standalone frozen checkpoint files."
}

# -----------------------------------------------------------------------------
# 10. SPRINT 13 ZERO-DAY SIMULATION (EXP_ZERODAY_V1)
# -----------------------------------------------------------------------------
ZERODAY_DATA = {
    "experiment_id": "EXP_ZERODAY_V1",
    "protocol_version": "V1.4 FINAL",
    "proxy_population": "Protected Backdoor (isolated from test set)",
    "n_protected_backdoor": 583,
    "n_benign_control": 37000,
    "n_attack_control": 44749,
    "n_combined_eval": 37583,
    "quadrants": {
        "Q1": {"count": 0, "name": "Both Detect", "desc": "C01=1, AE=1"},
        "Q2": {"count": 582, "name": "C01 Only", "desc": "C01=1, AE=0"},
        "Q3": {"count": 0, "name": "AE Rescue", "desc": "C01=0, AE=1"},
        "Q4": {"count": 1, "name": "Both Miss", "desc": "C01=0, AE=0"}
    },
    "component_detections": {
        "C01": {"detected": 582, "missed": 1, "total": 583, "rate": 0.998285, "fpr": 0.191892, "benign_fp": 7100},
        "AE": {"detected": 0, "missed": 583, "total": 583, "rate": 0.000000, "fpr": 0.000514, "benign_fp": 19},
        "C06": {"detected": 582, "missed": 1, "total": 583, "rate": 0.998285, "fpr": 0.192243, "benign_fp": 7113}
    },
    "benchmark_systems": [
        {"system": "DT", "detected": 577, "missed": 6, "rate": 0.989708, "fp": 10343, "fpr": 0.279541},
        {"system": "RF", "detected": 582, "missed": 1, "rate": 0.998285, "fp": 8548, "fpr": 0.231027},
        {"system": "SVM", "detected": 577, "missed": 6, "rate": 0.989708, "fp": 11491, "fpr": 0.310568},
        {"system": "NN", "detected": 578, "missed": 5, "rate": 0.991424, "fp": 5640, "fpr": 0.152432},
        {"system": "C01 (Stack)", "detected": 582, "missed": 1, "rate": 0.998285, "fp": 7100, "fpr": 0.191892},
        {"system": "AE", "detected": 0, "missed": 583, "rate": 0.000000, "fp": 19, "fpr": 0.000514},
        {"system": "C06 (Fusion)", "detected": 582, "missed": 1, "rate": 0.998285, "fp": 7113, "fpr": 0.192243}
    ],
    "preregistered_decisions": {
        "generalization": {
            "target": "C06",
            "observed_zdr": 0.998285,
            "ci_95_wilson": [0.990349, 0.999697],
            "criterion": "C06 ZDR >= 0.50 and CI lower bound > 0.50",
            "decision": "UNSEEN_CATEGORY_GENERALIZATION_SUPPORTED",
            "attribution_caveat": "High C06 detection (582/583) is driven entirely by the supervised C01 stacking component; the AE contributed 0 detections."
        },
        "rescue_fusion": {
            "target": "C06 vs C01",
            "rescue_count_Q3": 0,
            "rescue_gain": 0.000000,
            "practical_threshold": 0.05,
            "min_integer_Q3": 30,
            "statistical_test": "Exact one-sided binomial against benign validation baseline p0=0.000625",
            "p_value": 1.0000,
            "decision": "FUSION_IMPROVEMENT_NOT_SUPPORTED"
        },
        "h2_zero_day": {
            "ae_detected": 0,
            "rule": "ae_detected == 0 -> NOT_SUPPORTED (Rule DD-4)",
            "decision": "NOT_SUPPORTED"
        }
    },
    "validation_gates": {
        "total_gates": 44,
        "passed_gates": 44,
        "failed_gates": 0,
        "verdict": "PASS"
    }
}

# -----------------------------------------------------------------------------
# 11. TIMELINE & MILESTONES
# -----------------------------------------------------------------------------
TIMELINE_DATA = [
    {"sprint": "Sprint 1", "tag": "EXP_DATA_ACQUISITION_UNSEEN_RESERVATION_V1", "date": "2026-08-31", "desc": "Data acquisition, 45-feature schema validation, and Backdoor isolation."},
    {"sprint": "Sprint 2", "tag": "EXP_PREPROCESSING_V1", "date": "2026-08-31", "desc": "Deterministic one-hot encoding, feature normalization schemas, and provenance."},
    {"sprint": "Sprint 3", "tag": "EXP_TRAIN_VAL_SPLIT_V1", "date": "2026-08-31", "desc": "Official splits: TRAIN (162,395), VALIDATION (11,200), DEV_TEST (81,749), BACKDOOR (583)."},
    {"sprint": "Sprint 4", "tag": "EXP_MI_V1_1", "date": "2026-09-01", "desc": "Mutual Information feature selection; K=75 selected at performance plateau."},
    {"sprint": "Sprint 5", "tag": "EXP_BASE_MODELS_V1", "date": "2026-09-01", "desc": "Tuning & evaluation of base classifiers: DT, RF, SVM, and NN (IDSNet)."},
    {"sprint": "Sprint 6", "tag": "EXP_OOF_STACK_V1", "date": "2026-09-02", "desc": "5-fold OOF matrix generation and Logistic Regression meta-learner training."},
    {"sprint": "Sprint 7", "tag": "EXP_AE_V1", "date": "2026-09-02", "desc": "Benign-only Autoencoder (75->12->6->12->75); threshold tau=11.16006 calibrated."},
    {"sprint": "Sprint 8", "tag": "EXP_FUSION_V1", "date": "2026-09-02", "desc": "2x2 hybrid fusion exploration; C06 (C01 OR AE) selected as primary pipeline."},
    {"sprint": "Sprint 9", "tag": "sprint9-freeze", "date": "2026-09-03", "desc": "Formal hypothesis testing: H1 SUPPORTED, H2 NOT_SUPPORTED, H3 NOT_SUPPORTED."},
    {"sprint": "Sprint 10", "tag": "sprint10-freeze", "date": "2026-09-04", "desc": "Systematic ablation study (A0-A6); RF established as core contributor; lineage identified."},
    {"sprint": "Sprint 11", "tag": "sprint11-freeze", "date": "2026-09-04", "desc": "Post-hoc SHAP explainability; forensic quarantine and resolution of AE provenance issue."},
    {"sprint": "Sprint 12", "tag": "sprint12-freeze", "date": "2026-09-04", "desc": "Final frozen reproducibility verification; zero-training compliance verified."},
    {"sprint": "Sprint 13", "tag": "sprint13-freeze", "date": "2026-09-04", "desc": "Controlled zero-day simulation (Protocol V1.4); 44/44 validation gates; project freeze."}
]
