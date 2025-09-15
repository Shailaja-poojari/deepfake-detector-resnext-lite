"""
ResNeXt Lite model definition for Deepfake Detection.
"""

import torch
import torch.nn as nn

class ResNeXtLite(nn.Module):
    """
    A lightweight ResNeXt model for deepfake detection.
    """

    def __init__(self, num_classes=2):
        super(ResNeXtLite, self).__init__()

        # Example simple ResNeXt block (replace with actual architecture if needed)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
            # Add more layers as required
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
