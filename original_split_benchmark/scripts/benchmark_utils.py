"""
original_split_benchmark/scripts/benchmark_utils.py
Common utilities for loading and preprocessing original UNSW-NB15 datasets.
"""
import sys
from pathlib import Path
import pandas as pd
import json
import joblib
import os

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from original_split_benchmark.src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

PREPROCESSOR_PATH = ROOT / "original_split_benchmark/artifacts/checkpoints/preprocessor.joblib"

def get_or_fit_pipeline():
    if PREPROCESSOR_PATH.exists():
        return joblib.load(PREPROCESSOR_PATH)

    print("Fitting global PreprocessingPipeline on ORIGINAL TRAIN...")
    train_path = ROOT / "data/raw/UNSW_NB15_training-set.csv"
    df = pd.read_csv(train_path)

    pipeline = PreprocessingPipeline(experiment_id="BENCHMARK")
    pipeline.fit(df)

    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PREPROCESSOR_PATH)
    return pipeline

def apply_feature_selection(X, pipeline_features):
    """
    Slices X to match the exact 75 features from EXP_MI_V1_1 in the exact order.
    """
    mi_path = ROOT / "results/feature_selection/EXP_MI_V1_1/selected_features.json"
    if not mi_path.exists():
        raise RuntimeError(f"Cannot find authoritative EXP_MI_V1_1 features at {mi_path}")

    with open(mi_path, "r") as f:
        selected_features = json.load(f)["features"]

    if len(selected_features) != 75:
        raise ValueError(f"Expected 75 features in EXP_MI_V1_1, found {len(selected_features)}")

    # Find indices
    indices = []
    for feat in selected_features:
        if feat not in pipeline_features:
            raise ValueError(f"Feature '{feat}' from EXP_MI_V1_1 not found in pipeline output.")
        indices.append(pipeline_features.index(feat))

    X_selected = X[:, indices]
    return X_selected, selected_features

def load_and_preprocess_train(scaler_view="scaled"):
    pipeline = get_or_fit_pipeline()

    train_path = ROOT / "data/raw/UNSW_NB15_training-set.csv"
    df = pd.read_csv(train_path)

    processed = pipeline.transform(df, view=scaler_view, split_name="benchmark_train")
    X_selected, selected_features = apply_feature_selection(processed.X, processed.feature_names)

    return X_selected, processed.y, pipeline, selected_features

def load_and_preprocess_test(pipeline, scaler_view="scaled"):
    test_path = ROOT / "data/raw/UNSW_NB15_testing-set.csv"
    df = pd.read_csv(test_path)

    processed = pipeline.transform(df, view=scaler_view, split_name="benchmark_test")
    X_selected, selected_features = apply_feature_selection(processed.X, processed.feature_names)

    return X_selected, processed.y, selected_features
