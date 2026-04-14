from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import EMOTION_LABELS, pil_to_model_tensor

FER2013_IMAGE_SHAPE = (48, 48)
REQUIRED_COLUMNS = {"emotion", "pixels", "Usage"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}
USAGE_ALIASES = {
    "train": {"Training"},
    "val": {"PublicTest", "Validation", "Val"},
    "validation": {"PublicTest", "Validation", "Val"},
    "test": {"PrivateTest", "Test"},
}
SPLIT_DIR_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "validation", "valid", "test"),
    "validation": ("validation", "val", "valid", "test"),
    "test": ("test", "val", "validation", "valid"),
}
EMOTION_DIR_NAMES = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6,
}


class FER2013Dataset(Dataset[tuple[torch.Tensor, int]]):
    """Loads FER2013 CSV rows and converts grayscale faces into RGB tensors."""

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"FER2013 CSV was not found at {self.csv_path}.")

        self.split = self._canonicalize_split(split)
        self.allowed_usage_values = USAGE_ALIASES[self.split]
        self.transform = transform or pil_to_model_tensor
        self.samples = self._load_samples()

        if not self.samples:
            raise ValueError(
                f"No FER2013 rows matched split '{self.split}' in {self.csv_path}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        pixels, label = self.samples[index]
        image = Image.fromarray(pixels, mode="L")
        tensor = self.transform(image)
        return tensor, label

    @property
    def num_classes(self) -> int:
        return len(EMOTION_LABELS)

    @staticmethod
    def _canonicalize_split(split: str) -> str:
        normalized = split.lower()
        if normalized not in USAGE_ALIASES:
            raise ValueError(
                f"Unsupported split '{split}'. Expected one of: {', '.join(USAGE_ALIASES)}."
            )
        return normalized

    def _load_samples(self) -> list[tuple[np.ndarray, int]]:
        samples: list[tuple[np.ndarray, int]] = []

        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
                raise ValueError(
                    f"{self.csv_path} must contain columns: {', '.join(sorted(REQUIRED_COLUMNS))}."
                )

            for row_number, row in enumerate(reader, start=2):
                usage = (row.get("Usage") or "").strip()
                if usage not in self.allowed_usage_values:
                    continue

                label = int(row["emotion"])
                if label < 0 or label >= len(EMOTION_LABELS):
                    raise ValueError(
                        f"Row {row_number} has invalid emotion label {label}. "
                        f"Expected 0 through {len(EMOTION_LABELS) - 1}."
                    )

                pixel_values = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
                if pixel_values.size != FER2013_IMAGE_SHAPE[0] * FER2013_IMAGE_SHAPE[1]:
                    raise ValueError(
                        f"Row {row_number} has {pixel_values.size} pixel values. "
                        "Expected 2304 values for a 48x48 image."
                    )

                samples.append((pixel_values.reshape(FER2013_IMAGE_SHAPE), label))

        return samples


class EmotionImageFolderDataset(Dataset[tuple[torch.Tensor, int]]):
    """Loads split/class/image folder datasets while preserving EMOTION_LABELS ordering."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory was not found at {self.root}.")

        self.split = FER2013Dataset._canonicalize_split(split)
        self.transform = transform or pil_to_model_tensor
        self.split_dir = self._resolve_split_dir()
        self.samples = self._load_samples()

        if not self.samples:
            raise ValueError(f"No image files were found for split '{self.split}' in {self.split_dir}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            tensor = self.transform(image)
        return tensor, label

    @property
    def num_classes(self) -> int:
        return len(EMOTION_LABELS)

    def _resolve_split_dir(self) -> Path:
        for directory_name in SPLIT_DIR_ALIASES[self.split]:
            candidate = self.root / directory_name
            if candidate.is_dir():
                return candidate
        expected = ", ".join(SPLIT_DIR_ALIASES[self.split])
        raise FileNotFoundError(
            f"Could not find a split directory for '{self.split}' in {self.root}. "
            f"Expected one of: {expected}."
        )

    def _load_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []

        for class_dir in sorted(self.split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            label_key = class_dir.name.strip().lower()
            if label_key not in EMOTION_DIR_NAMES:
                raise ValueError(
                    f"Unsupported class directory '{class_dir.name}' in {self.split_dir}. "
                    f"Expected one of: {', '.join(name.lower() for name in EMOTION_LABELS)}."
                )

            label = EMOTION_DIR_NAMES[label_key]
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((image_path, label))

        return samples


def build_dataset(
    data_path: str | Path,
    split: str = "train",
    transform: Callable[[Image.Image], torch.Tensor] | None = None,
) -> Dataset[tuple[torch.Tensor, int]]:
    path = Path(data_path)
    if path.is_dir():
        return EmotionImageFolderDataset(path, split=split, transform=transform)
    return FER2013Dataset(path, split=split, transform=transform)
