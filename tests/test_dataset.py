from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from torch.utils.data import DataLoader

from data.dataset import FER2013Dataset


def make_pixels(value: int) -> str:
    return " ".join([str(value)] * (48 * 48))


class FER2013DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "fer2013.csv"
        rows = [
            {"emotion": 0, "pixels": make_pixels(0), "Usage": "Training"},
            {"emotion": 3, "pixels": make_pixels(127), "Usage": "Training"},
            {"emotion": 6, "pixels": make_pixels(255), "Usage": "PublicTest"},
            {"emotion": 1, "pixels": make_pixels(90), "Usage": "PrivateTest"},
        ]
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["emotion", "pixels", "Usage"])
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_dataset_smoke_batch_shape(self) -> None:
        dataset = FER2013Dataset(self.csv_path, split="train")
        loader = DataLoader(dataset, batch_size=2)
        images, labels = next(iter(loader))

        self.assertEqual(images.shape, (2, 3, 224, 224))
        self.assertTrue(((labels >= 0) & (labels <= 6)).all().item())

    def test_usage_mapping_for_validation_and_test(self) -> None:
        val_dataset = FER2013Dataset(self.csv_path, split="val")
        test_dataset = FER2013Dataset(self.csv_path, split="test")

        self.assertEqual(len(val_dataset), 1)
        self.assertEqual(len(test_dataset), 1)
