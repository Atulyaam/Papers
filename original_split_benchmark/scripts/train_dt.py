"""
original_split_benchmark/scripts/train_dt.py
Trains Decision Tree on the Original Train Split.
"""
import sys
import json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from original_split_benchmark.scripts.benchmark_utils import load_and_preprocess_train
from sklearn.tree import DecisionTreeClassifier

def run():
    print("Loading data for DT...")
    X_train, y_train, pipeline, feature_names = load_and_preprocess_train(scaler_view="unscaled")

    print(f"Training Decision Tree on {len(X_train)} rows with {len(feature_names)} features...")
    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    dt.fit(X_train, y_train)

    out_dir = ROOT / "original_split_benchmark/artifacts/checkpoints/dt"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(dt, out_dir / "dt_final.joblib")

    config = {
        "model_type": "DecisionTree",
        "random_state": 42,
        "n_train_rows": len(X_train),
        "n_features": len(feature_names)
    }
    with open(out_dir / "dt_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Decision Tree training complete and saved.")

if __name__ == "__main__":
    run()
