"""
Training script for Deepfake Detector using ResNeXt Lite
"""

import os
import torch
from src.models.resnext_lite import ResNeXtLite
from src.utils import load_data, train_model  # Make sure you have these utils

def main():
    """
    Main function to train the deepfake detector model.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader = load_data(batch_size=16)

    # Initialize model
    model = ResNeXtLite(num_classes=2).to(device)

    # Train the model
    train_model(model, train_loader, val_loader, epochs=10, device=device)

if __name__ == "__main__":
    main()

