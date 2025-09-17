"""
ResNeXt Lite model definition for Deepfake Detection.
"""

import torch
import torch.nn as nn

class ResNeXtLite(nn.Module):
    """
    A lightweight ResNeXt-like model for deepfake detection.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Simple CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # ensures fixed-size output
        )

        # After AdaptiveAvgPool2d, feature map = [batch, 128, 1, 1] → flatten = 128
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # flatten all except batch
        x = self.classifier(x)
        return x


# Debug: run a forward pass to confirm dimensions
if __name__ == "__main__":
    model = ResNeXtLite()
    dummy = torch.randn(4, 3, 64, 64)  # 4 RGB images of 64x64
    out = model(dummy)
    print("Output shape:", out.shape)  # should be [4, 2]

