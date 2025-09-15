# Placeholder for ResNeXt-lite + LSTM temporal head"""
ResNeXt-lite backbone with optional temporal pooling (LSTM).
"""
import torch
import torch.nn as nn
from torchvision import models

class ResNeXtLite(nn.Module):
    def __init__(self, num_classes: int = 1, temporal: str = "mean"):
        super().__init__()
        base = models.resnext50_32x4d(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # remove fc
        self.temporal = temporal
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, C, H, W)
        Returns:
            Logits of shape (B,)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.backbone(x).view(b, t, -1)

        if self.temporal == "mean":
            feat = feat.mean(dim=1)
        elif self.temporal == "max":
            feat, _ = feat.max(dim=1)
        out = self.fc(feat)
        return out.squeeze(1)
