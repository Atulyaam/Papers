"""
original_split_benchmark/scripts/train_svm.py
Trains SVM on the Original Train Split.
"""
import sys
import json
from pathlib import Path
import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from original_split_benchmark.scripts.benchmark_utils import load_and_preprocess_train

def run():
    print("Loading data for SVM...")
    X_train, y_train, pipeline, feature_names = load_and_preprocess_train(scaler_view="unscaled")

    print("Fitting scaler on TRAIN...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"Training LinearSVC on {len(X_train_scaled)} rows with {len(feature_names)} features...")
    svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
    svm.fit(X_train_scaled, y_train)

    out_dir = ROOT / "original_split_benchmark/artifacts/checkpoints/svm"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(svm, out_dir / "svm_final.joblib")
    joblib.dump(scaler, out_dir / "svm_scaler.joblib")

    config = {
        "model_type": "LinearSVC",
        "random_state": 42,
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 5000,
        "n_train_rows": len(X_train),
        "n_features": len(feature_names)
    }
    with open(out_dir / "svm_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("SVM training complete and saved.")

if __name__ == "__main__":
    run()
