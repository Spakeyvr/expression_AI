from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

EMOTION_LABELS = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)

EMOTION_ASSET_FILES = {
    "Angry": "angry.ppm",
    "Disgust": "disgust.ppm",
    "Fear": "fear.ppm",
    "Happy": "happy.ppm",
    "Sad": "sad.ppm",
    "Surprise": "surprise.ppm",
    "Neutral": "neutral.ppm",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 224

try:
    RESAMPLING_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    RESAMPLING_BILINEAR = Image.BILINEAR


class DeviceResolutionError(ValueError):
    """Raised when the requested inference or training device cannot be used."""


def resolve_device(device: str = "auto") -> torch.device:
    normalized = device.lower()

    if normalized == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized == "mps":
        if not hasattr(torch.backends, "mps"):
            raise DeviceResolutionError("This PyTorch build does not include MPS support.")
        if not torch.backends.mps.is_available():
            raise DeviceResolutionError(
                "MPS was requested but is unavailable. Use --device auto or --device cpu instead."
            )
        return torch.device("mps")

    raise DeviceResolutionError(f"Unsupported device '{device}'. Expected auto, cpu, or mps.")


def pil_to_model_tensor(image: Image.Image, size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    resized = image.convert("RGB").resize((size, size), RESAMPLING_BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def asset_path_for_label(label: str, asset_dir: Path) -> Path:
    filename = EMOTION_ASSET_FILES.get(label)
    if filename is None:
        raise KeyError(f"No asset file is defined for label '{label}'.")
    return asset_dir / filename
