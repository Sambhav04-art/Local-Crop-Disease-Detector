import torch.nn as nn
from torchvision import models as tvm

MODEL_NAMES = ["baseline_cnn", "mobilenet_v3_large", "efficientnet_b0", "resnet50"]


class SimpleCNN(nn.Module):
    """Baseline: 5 conv blocks trained from scratch (no ImageNet knowledge)."""
    def __init__(self, n):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                                         nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 256), blk(256, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, n, pretrained=True):
    if name == "baseline_cnn":
        return SimpleCNN(n)
    if name == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
    elif name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n)
    else:
        raise ValueError(f"unknown model {name}; choose from {MODEL_NAMES}")
    return m


def n_params(m):
    return sum(p.numel() for p in m.parameters())
