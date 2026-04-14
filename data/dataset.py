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
USAGE_ALIASES = {
    "train": {"Training"},
    "val": {"PublicTest", "Validation", "Val"},
    "validation": {"PublicTest", "Validation", "Val"},
    "test": {"PrivateTest", "Test"},
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
