"""
original_split_benchmark/scripts/train_ae.py
Trains Autoencoder on the Original Train Split using only NORMAL rows.
Splits NORMAL into AE-FIT and AE-CALIBRATION.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from original_split_benchmark.scripts.benchmark_utils import load_and_preprocess_train
from original_split_benchmark.src.models.autoencoder.ae_model import Autoencoder

def run():
    print("Loading data for AE...")
    X_train, y_train, pipeline, feature_names = load_and_preprocess_train(scaler_view="unscaled")

    # Filter NORMAL rows only
    normal_mask = (y_train == 0)
    X_normal = X_train[normal_mask]

    print(f"Total NORMAL rows available: {len(X_normal)}")

    # Split into AE-FIT (80%) and AE-CALIBRATION (20%)
    X_fit, X_cal = train_test_split(X_normal, test_size=0.2, random_state=42)

    print(f"AE-FIT subset: {len(X_fit)} rows")
    print(f"AE-CALIBRATION subset: {len(X_cal)} rows")

    print("Fitting scaler on AE-FIT only...")
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit).astype(np.float32)
    X_cal_scaled = scaler.transform(X_cal).astype(np.float32)

    # AE hyperparameters from main project
    input_dim = len(feature_names)
    epochs = 50
    batch_size = 1024
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_fit_scaled))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Training Autoencoder...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataset):.4f}")

    model.eval()

    print("Calculating reconstruction errors on AE-CALIBRATION subset...")
    with torch.no_grad():
        cal_tensor = torch.tensor(X_cal_scaled, device=device)
        cal_reconstructed = model(cal_tensor)
        # RE = mean((x - x_hat)^2) per row
        re = ((cal_tensor - cal_reconstructed) ** 2).mean(dim=1).cpu().numpy()

    tau_mean3sigma = float(np.mean(re) + 3 * np.std(re))
    tau_p99 = float(np.percentile(re, 99))

    print(f"Calibrated Threshold (mean+3sigma): {tau_mean3sigma:.6f}")

    out_dir = ROOT / "original_split_benchmark/artifacts/checkpoints/ae"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "ae_final.pt")
    joblib.dump(scaler, out_dir / "ae_scaler.joblib")

    config = {
        "model_type": "Autoencoder",
        "random_state": 42,
        "input_dim": input_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_fit_rows": len(X_fit),
        "n_cal_rows": len(X_cal),
        "threshold_calibration_method": "Internal TRAIN-only split (80% fit, 20% cal). mean+3sigma on AE-CALIBRATION subset.",
        "tau": tau_mean3sigma,
        "tau_p99": tau_p99,
        "n_features": len(feature_names)
    }
    with open(out_dir / "ae_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Autoencoder training and calibration complete and saved.")

if __name__ == "__main__":
    torch.manual_seed(42)
    run()
