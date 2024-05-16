import torch
import torch.nn as nn

class DenseNetBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        x = self.model.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x