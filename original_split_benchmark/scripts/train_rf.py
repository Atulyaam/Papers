"""
original_split_benchmark/scripts/train_rf.py
Trains Random Forest on the Original Train Split.
"""
import sys
import json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from original_split_benchmark.scripts.benchmark_utils import load_and_preprocess_train
from sklearn.ensemble import RandomForestClassifier

def run():
    print("Loading data for RF...")
    X_train, y_train, pipeline, feature_names = load_and_preprocess_train(scaler_view="unscaled")

    print(f"Training Random Forest on {len(X_train)} rows with {len(feature_names)} features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)

    out_dir = ROOT / "original_split_benchmark/artifacts/checkpoints/rf"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf, out_dir / "rf_final.joblib")

    config = {
        "model_type": "RandomForest",
        "random_state": 42,
        "n_train_rows": len(X_train),
        "n_features": len(feature_names)
    }
    with open(out_dir / "rf_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Random Forest training complete and saved.")

if __name__ == "__main__":
    run()
