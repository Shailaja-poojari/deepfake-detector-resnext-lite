"""
Training script for Deepfake Detector using ResNeXt Lite
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Fix for Colab path issue ---
# Ensures repo root is added to Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.models.resnext_lite import ResNeXtLite  # now should always work


def get_dummy_data(num_samples: int = 64):
    """
    Generate dummy dataset of random tensors to simulate training.
    Args:
        num_samples (int): Number of samples to generate.
    Returns:
        train_loader, val_loader (DataLoader, DataLoader)
    """
    X = torch.randn(num_samples, 3, 64, 64)  # 64x64 RGB images
    y = torch.randint(0, 2, (num_samples,))  # binary labels

    train_ds = TensorDataset(X[:48], y[:48])
    val_ds = TensorDataset(X[48:], y[48:])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    return train_loader, val_loader


def train_model():
    """
    Simple training loop on dummy dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dummy_data()

    model = ResNeXtLite(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, 3):  # keep it short for demo
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/2, Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item()
        val_loss /= len(val_loader)

        # Save if improved
        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_path = "checkpoints/model_best.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Model saved to {ckpt_path}")


if __name__ == "__main__":
    train_model()
