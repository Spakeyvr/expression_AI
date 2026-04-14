from __future__ import annotations

import torch.nn as nn

from common import EMOTION_LABELS

try:
    from torchvision.models import ResNet18_Weights, resnet18
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    ResNet18_Weights = None
    resnet18 = None
except ImportError:  # pragma: no cover - older torchvision without weights enum
    from torchvision.models import resnet18

    ResNet18_Weights = None


def ensure_torchvision_available() -> None:
    if resnet18 is None:
        raise ModuleNotFoundError(
            "torchvision is required for the ResNet18 model. "
            "Install the dependencies from requirements.txt before training or inference."
        )


def build_model(num_classes: int = len(EMOTION_LABELS), pretrained: bool = True) -> nn.Module:
    ensure_torchvision_available()

    if ResNet18_Weights is not None:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    else:  # pragma: no cover - compatibility fallback
        model = resnet18(pretrained=pretrained)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
