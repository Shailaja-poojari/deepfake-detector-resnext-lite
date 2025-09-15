# Training script placeholder"""
Training script for DeepFake Detector.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.resnext_lite import ResNeXtLite

def train_model(epochs: int = 2, batch_size: int = 4, lr: float = 1e-4):
    # Dummy data loader (replace with real dataset later)
    X = torch.randn(16, 4, 3, 224, 224)  # 16 videos, 4 frames each
    y = torch.randint(0, 2, (16,), dtype=torch.float)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNeXtLite().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "checkpoints/model_best.pth")
    print("Model saved to checkpoints/model_best.pth")

if __name__ == "__main__":
    train_model()
