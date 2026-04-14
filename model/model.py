from __future__ import annotations

import os
import ssl
from pathlib import Path

import torch
import torch.nn as nn

from common import EMOTION_LABELS

try:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    EfficientNet_B0_Weights = None
    efficientnet_b0 = None
except ImportError:  # pragma: no cover - older torchvision without weights enum
    from torchvision.models import efficientnet_b0

    EfficientNet_B0_Weights = None


def ensure_torchvision_available() -> None:
    if efficientnet_b0 is None:
        raise ModuleNotFoundError(
            "torchvision is required for the EfficientNet-B0 model. "
            "Install the dependencies from requirements.txt before training or inference."
        )


def _resolve_ca_bundle() -> str | None:
    try:
        import certifi  # type: ignore

        return certifi.where()
    except ModuleNotFoundError:
        pass

    try:
        from pip._vendor import certifi as pip_certifi  # type: ignore

        return pip_certifi.where()
    except ModuleNotFoundError:
        return None


def configure_ssl_for_downloads() -> None:
    """Point Python HTTPS clients at a valid CA bundle when the framework path is broken."""
    default_paths = ssl.get_default_verify_paths()
    current_cafile = os.environ.get(default_paths.openssl_cafile_env) or default_paths.cafile
    if current_cafile and Path(current_cafile).is_file():
        return

    ca_bundle = _resolve_ca_bundle()
    if not ca_bundle or not Path(ca_bundle).is_file():
        return

    os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
    os.environ.setdefault("CURL_CA_BUNDLE", ca_bundle)
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=ca_bundle)


def build_model(
    num_classes: int = len(EMOTION_LABELS),
    pretrained: bool = True,
    weights_path: str | Path | None = None,
) -> nn.Module:
    ensure_torchvision_available()
    configure_ssl_for_downloads()

    checkpoint_path = Path(weights_path).expanduser() if weights_path else None
    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Pretrained weights file was not found at {checkpoint_path}.")
        if EfficientNet_B0_Weights is not None:
            model = efficientnet_b0(weights=None)
        else:  # pragma: no cover - compatibility fallback
            model = efficientnet_b0(pretrained=False)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    elif EfficientNet_B0_Weights is not None:
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
    else:  # pragma: no cover - compatibility fallback
        model = efficientnet_b0(pretrained=pretrained)

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
