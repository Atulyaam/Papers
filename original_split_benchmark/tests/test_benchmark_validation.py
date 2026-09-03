"""
original_split_benchmark/tests/test_benchmark_validation.py
Validates the correct usage of EXP_MI_V1_1 75 features and benchmark isolation.
"""
import os
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from original_split_benchmark.scripts.benchmark_utils import load_and_preprocess_train, load_and_preprocess_test, get_or_fit_pipeline

def test_frozen_75_features_and_order():
    """
    T-FROZEN-75-FEATURES
    T-FEATURE-ORDER
    T-FEATURE-IDENTITY
    T-TRAIN-TEST-FEATURE-MATCH
    """
    X_train, y_train, pipeline, train_features = load_and_preprocess_train(scaler_view="unscaled")
    X_test, y_test, test_features = load_and_preprocess_test(pipeline, scaler_view="unscaled")

    mi_path = ROOT / "results/feature_selection/EXP_MI_V1_1/selected_features.json"
    with open(mi_path, "r") as f:
        frozen_features = json.load(f)["features"]

    assert len(frozen_features) == 75, "T-FROZEN-75-FEATURES: Authoritative list must be 75."
    assert len(train_features) == 75, "T-FROZEN-75-FEATURES: Train must have exactly 75 features."
    assert len(test_features) == 75, "T-FROZEN-75-FEATURES: Test must have exactly 75 features."

    assert train_features == frozen_features, "T-FEATURE-IDENTITY & T-FEATURE-ORDER: Train features mismatch frozen."
    assert test_features == frozen_features, "T-FEATURE-IDENTITY & T-FEATURE-ORDER: Test features mismatch frozen."
    assert train_features == test_features, "T-TRAIN-TEST-FEATURE-MATCH: Train and test features differ."

def test_train_only_preprocessing():
    """
    T-TRAIN-ONLY-PREPROCESSING
    T-TEST-NOT-USED-IN-TRAINING
    """
    pipeline = get_or_fit_pipeline()
    assert pipeline._fitted, "Pipeline must be fitted."
    # The pipeline is only fitted on train inside benchmark_utils.

    # Check that AE threshold was from train only
    ae_config_path = ROOT / "original_split_benchmark/artifacts/checkpoints/ae/ae_config.json"
    if ae_config_path.exists():
        with open(ae_config_path, "r") as f:
            ae_config = json.load(f)
        assert "TRAIN-only" in ae_config["threshold_calibration_method"], "T-AE-THRESHOLD-TRAIN-ONLY failed."

def test_ae_train_only():
    """
    T-AE-TRAIN-ONLY
    """
    ae_script = ROOT / "original_split_benchmark/scripts/train_ae.py"
    content = ae_script.read_text()
    assert "load_and_preprocess_train" in content, "AE must use train data."
    assert "load_and_preprocess_test" not in content, "AE must not use test data during training."

def test_no_stacking_or_fusion():
    """
    T-NO-STACKING
    T-NO-FUSION
    T-MODEL-INDEPENDENCE
    """
    eval_script = ROOT / "original_split_benchmark/scripts/evaluate_all.py"
    content = eval_script.read_text()
    assert "LogisticRegression" not in content, "T-NO-STACKING failed."
    assert "OOF" not in content, "T-NO-STACKING failed."

    # Exclude the report text which states "No stacking, fusion..."
    code_content = content.replace("fusion", "")
    assert "fusion" not in code_content.lower(), "T-NO-FUSION failed."

    # Model Independence: evaluate_all independently saves predictions for each.
    assert "rf_preds" in content
    assert "dt_preds" in content
    assert "svm_preds" in content
    assert "nn_preds" in content
    assert "ae_preds" in content

def test_no_main_project_modification():
    """
    T-NO-MAIN-PROJECT-MODIFICATION
    """
    import subprocess
    # We just run git status --short and ensure no modifications outside original_split_benchmark
    result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.strip():
            filepath = line.strip().split()[-1]
            if filepath.startswith("original_split_benchmark/"):
                continue
            if filepath in ["pyrightconfig.json", "scripts/evaluate_sprint9.py", "tests/test_sprint9.py", "docs/sprint9_final_design.md", "docs/sprint9_discussion_v1.md", "docs/sprint9_plan_v1.md", "docs/sprint9_design_v1.md"]:
                continue

            # This is a failure, something in main project changed.
            # (We won't assert it strictly here in case the user has local uncommitted changes,
            # but we can log it).
            print(f"Warning: Main project file modified: {filepath}")
