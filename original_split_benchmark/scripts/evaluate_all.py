"""
original_split_benchmark/scripts/evaluate_all.py
Evaluates all five trained benchmark models independently on the ORIGINAL TEST split.
Generates metrics and comparison report.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from original_split_benchmark.scripts.benchmark_utils import load_and_preprocess_test, get_or_fit_pipeline
from original_split_benchmark.src.models.base_models.neural_network import IDSNet
from original_split_benchmark.src.models.autoencoder.ae_model import Autoencoder

def eval_supervised(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Macro-F1": macro_f1,
        "Weighted-F1": weighted_f1,
        "Balanced Accuracy": bacc,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }

def run():
    print("Loading ORIGINAL TEST data...")
    pipeline = get_or_fit_pipeline()
    X_test, y_test, feature_names = load_and_preprocess_test(pipeline, scaler_view="unscaled")

    metrics_list = []

    # Paths
    ckpt_dir = ROOT / "original_split_benchmark/artifacts/checkpoints"
    pred_dir = ROOT / "original_split_benchmark/artifacts/predictions"
    metr_dir = ROOT / "original_split_benchmark/artifacts/metrics"

    pred_dir.mkdir(parents=True, exist_ok=True)
    metr_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Decision Tree
    # ---------------------------------------------------------
    print("Evaluating Decision Tree...")
    dt = joblib.load(ckpt_dir / "dt/dt_final.joblib")
    dt_preds = dt.predict(X_test)
    pd.DataFrame({"prediction": dt_preds}).to_csv(pred_dir / "dt_predictions.csv", index=False)

    dt_metrics = eval_supervised(y_test, dt_preds, "Decision Tree")
    with open(metr_dir / "dt_metrics.json", "w") as f:
        json.dump(dt_metrics, f, indent=2)
    metrics_list.append(dt_metrics)

    # ---------------------------------------------------------
    # 2. Random Forest
    # ---------------------------------------------------------
    print("Evaluating Random Forest...")
    rf = joblib.load(ckpt_dir / "rf/rf_final.joblib")
    rf_preds = rf.predict(X_test)
    pd.DataFrame({"prediction": rf_preds}).to_csv(pred_dir / "rf_predictions.csv", index=False)

    rf_metrics = eval_supervised(y_test, rf_preds, "Random Forest")
    with open(metr_dir / "rf_metrics.json", "w") as f:
        json.dump(rf_metrics, f, indent=2)
    metrics_list.append(rf_metrics)

    # ---------------------------------------------------------
    # 3. SVM
    # ---------------------------------------------------------
    print("Evaluating SVM...")
    svm = joblib.load(ckpt_dir / "svm/svm_final.joblib")
    svm_scaler = joblib.load(ckpt_dir / "svm/svm_scaler.joblib")

    X_test_svm = svm_scaler.transform(X_test)
    svm_preds = svm.predict(X_test_svm)
    pd.DataFrame({"prediction": svm_preds}).to_csv(pred_dir / "svm_predictions.csv", index=False)

    svm_metrics = eval_supervised(y_test, svm_preds, "SVM")
    with open(metr_dir / "svm_metrics.json", "w") as f:
        json.dump(svm_metrics, f, indent=2)
    metrics_list.append(svm_metrics)

    # ---------------------------------------------------------
    # 4. Neural Network
    # ---------------------------------------------------------
    print("Evaluating Neural Network...")
    with open(ckpt_dir / "nn/nn_config.json") as f:
        nn_config = json.load(f)

    nn_scaler = joblib.load(ckpt_dir / "nn/nn_scaler.joblib")
    X_test_nn = nn_scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = IDSNet(input_dim=nn_config["n_features"], hidden_sizes=nn_config["hidden_sizes"]).to(device)
    nn_model.load_state_dict(torch.load(ckpt_dir / "nn/nn_final.pt", map_location=device, weights_only=True))
    nn_model.eval()

    with torch.no_grad():
        logits = nn_model(torch.tensor(X_test_nn, dtype=torch.float32, device=device))
        nn_probs = torch.sigmoid(logits).cpu().numpy()
        nn_preds = (nn_probs >= 0.5).astype(int)

    pd.DataFrame({"prediction": nn_preds}).to_csv(pred_dir / "nn_predictions.csv", index=False)

    nn_metrics = eval_supervised(y_test, nn_preds, "Neural Network")
    with open(metr_dir / "nn_metrics.json", "w") as f:
        json.dump(nn_metrics, f, indent=2)
    metrics_list.append(nn_metrics)

    # ---------------------------------------------------------
    # 5. Autoencoder
    # ---------------------------------------------------------
    print("Evaluating Autoencoder...")
    with open(ckpt_dir / "ae/ae_config.json") as f:
        ae_config = json.load(f)

    tau = ae_config["tau"]
    ae_scaler = joblib.load(ckpt_dir / "ae/ae_scaler.joblib")
    X_test_ae = ae_scaler.transform(X_test).astype(np.float32)

    ae_model = Autoencoder(input_dim=ae_config["input_dim"]).to(device)
    ae_model.load_state_dict(torch.load(ckpt_dir / "ae/ae_final.pt", map_location=device, weights_only=True))
    ae_model.eval()

    reconstruction_errors = []
    with torch.no_grad():
        for i in range(0, len(X_test_ae), 1024):
            batch = torch.tensor(X_test_ae[i:i+1024], device=device)
            reconstructed = ae_model(batch)
            re = ((batch - reconstructed) ** 2).mean(dim=1).cpu().numpy()
            reconstruction_errors.append(re)

    reconstruction_errors = np.concatenate(reconstruction_errors)
    ae_preds = (reconstruction_errors > tau).astype(int)

    pd.DataFrame({
        "reconstruction_error": reconstruction_errors,
        "prediction": ae_preds
    }).to_csv(pred_dir / "ae_predictions.csv", index=False)

    ae_metrics = eval_supervised(y_test, ae_preds, "Autoencoder")
    ae_metrics["threshold"] = tau
    ae_metrics["threshold_method"] = ae_config["threshold_calibration_method"]
    ae_metrics["anomaly_count"] = int(np.sum(ae_preds))
    ae_metrics["anomaly_rate"] = ae_metrics["anomaly_count"] / len(y_test)

    with open(metr_dir / "ae_metrics.json", "w") as f:
        json.dump(ae_metrics, f, indent=2)
    metrics_list.append(ae_metrics)

    # ---------------------------------------------------------
    # Generate Reports
    # ---------------------------------------------------------
    print("Generating final comparison report...")
    report_dir = ROOT / "original_split_benchmark/reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    md = "# Supplementary Original-Split Benchmark Results\n\n"
    md += "SUPPLEMENTARY ORIGINAL-SPLIT BENCHMARK. This experiment is independent of the main hybrid IDS research protocol and is intended only to provide a conventional reference using the original UNSW-NB15 train/test split.\n\n"
    md += "No stacking, fusion, C01/C06 evaluation, or main-project hypothesis decision was performed in this benchmark.\n\n"

    md += "## Final Comparison Table\n\n"
    md += "| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | Weighted-F1 | Balanced Accuracy | TP | FP | TN | FN |\n"
    md += "|-------|----------|-----------|--------|----|----------|-------------|-------------------|----|----|----|----|\n"

    for m in metrics_list:
        md += f"| {m['model']} | {m['Accuracy']:.4f} | {m['Precision']:.4f} | {m['Recall']:.4f} | {m['F1']:.4f} | {m['Macro-F1']:.4f} | {m['Weighted-F1']:.4f} | {m['Balanced Accuracy']:.4f} | {m['TP']} | {m['FP']} | {m['TN']} | {m['FN']} |\n"

    md += "\n## Autoencoder Threshold Information\n"
    md += f"- **Threshold (\u03c4)**: {ae_metrics['threshold']:.6f}\n"
    md += f"- **Calibration Method**: {ae_metrics['threshold_method']}\n"
    md += f"- **Anomaly Count**: {ae_metrics['anomaly_count']} ({ae_metrics['anomaly_rate']:.2%})\n\n"
    md += "*Note: The Autoencoder produces an anomaly decision (RE > \u03c4) rather than a directly supervised class prediction.*\n"

    with open(report_dir / "results.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("Benchmark evaluation complete.")

if __name__ == "__main__":
    run()
