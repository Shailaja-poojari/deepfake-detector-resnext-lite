import torch
import torch.nn as nn
import torchvision.models as models

class ResNeXtLite(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNeXtLite, self).__init__()
        self.base = models.resnext50_32x4d(weights='DEFAULT')
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)
