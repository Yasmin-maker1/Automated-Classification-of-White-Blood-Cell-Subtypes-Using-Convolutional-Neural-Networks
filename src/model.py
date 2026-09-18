"""Models for the 4-class white blood cell classifier.

  * resnet50  - main model. ImageNet weights, final FC layer replaced by a
                4-way linear layer (softmax is applied inside the loss).
  * smallcnn  - weak from-scratch baseline, used only to show whether transfer
                learning actually helped (as promised in the proposal).

The classification head is always called `fc` (resnet50) or `classifier`
(smallcnn) so that freezing the backbone is a one-liner.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

HEAD_PREFIXES = ("fc.", "classifier.")


class SmallCNN(nn.Module):
    """Four conv blocks + global pooling. ~0.4M parameters, trained from scratch."""

    def __init__(self, num_classes: int = 4):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name: str = "resnet50", num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "smallcnn":
        return SmallCNN(num_classes)
    raise ValueError(f"Unknown model '{name}'. Choose 'resnet50' or 'smallcnn'.")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classification head."""
    for pname, p in model.named_parameters():
        is_head = pname.startswith(HEAD_PREFIXES)
        p.requires_grad = True if is_head else trainable


def freeze_batchnorm(model: nn.Module) -> None:
    """Keep BatchNorm running statistics fixed (used while the backbone is frozen)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def split_param_groups(model: nn.Module, lr: float, backbone_lr_mult: float):
    """Head gets `lr`; pretrained backbone gets a smaller lr when unfrozen."""
    head, backbone = [], []
    for pname, p in model.named_parameters():
        (head if pname.startswith(HEAD_PREFIXES) else backbone).append(p)
    groups = [{"params": head, "lr": lr}]
    if backbone:
        groups.append({"params": backbone, "lr": lr * backbone_lr_mult})
    return groups


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
