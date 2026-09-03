"""
original_split_benchmark/scripts/train_nn.py
Trains Neural Network on the Original Train Split.
"""
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from original_split_benchmark.scripts.benchmark_utils import load_and_preprocess_train
from original_split_benchmark.src.models.base_models.neural_network import IDSNet

def run():
    print("Loading data for NN...")
    X_train, y_train, pipeline, feature_names = load_and_preprocess_train(scaler_view="unscaled")

    print("Fitting scaler on TRAIN...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # NN hyperparameters from main project
    hidden_sizes = [128, 64]
    lr = 0.001
    weight_decay = 0.0001
    batch_size = 256
    epochs = 18 # We will train for 18 epochs based on the main project median best epoch

    # compute class weights since the dataset size is different from the main project
    n_attack = int(np.sum(y_train))
    n_normal = int(len(y_train) - n_attack)
    pos_weight = n_normal / n_attack

    print(f"Training NN on {len(X_train_scaled)} rows with {len(feature_names)} features...")
    print(f"Pos weight: {pos_weight:.5f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IDSNet(input_dim=len(feature_names), hidden_sizes=hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataset):.4f}")

    out_dir = ROOT / "original_split_benchmark/artifacts/checkpoints/nn"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    torch.save(model.state_dict(), out_dir / "nn_final.pt")
    joblib.dump(scaler, out_dir / "nn_scaler.joblib")

    config = {
        "model_type": "IDSNet",
        "random_state": 42, # PyTorch shuffle implicitly uses torch seed, but we can just say seed is not fully fixed unless we did torch.manual_seed
        "hidden_sizes": hidden_sizes,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "pos_weight": pos_weight,
        "n_train_rows": len(X_train),
        "n_features": len(feature_names)
    }
    with open(out_dir / "nn_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("NN training complete and saved.")

if __name__ == "__main__":
    torch.manual_seed(42)
    run()
