from __future__ import annotations

import csv
import tempfile
import unittest
import sys
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import EmotionImageFolderDataset, FER2013Dataset, build_dataset, find_image_folder_roots


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


class EmotionImageFolderDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "archive"

        for split in ("train", "test"):
            for class_name, color in (("happy", 255), ("sad", 64)):
                image_dir = self.root / split / class_name
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f"{class_name}.png"
                Image.new("L", (48, 48), color=color).save(image_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_folder_dataset_smoke_batch_shape(self) -> None:
        dataset = EmotionImageFolderDataset(self.root, split="train")
        loader = DataLoader(dataset, batch_size=2)
        images, labels = next(iter(loader))

        self.assertEqual(images.shape, (2, 3, 224, 224))
        self.assertCountEqual(labels.tolist(), [3, 4])

    def test_build_dataset_uses_folder_loader(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_dataset(self.root, split="val")

    def test_parent_directory_combines_multiple_archives_for_same_split(self) -> None:
        second_root = Path(self.temp_dir.name) / "archive_2"
        for split in ("train", "val"):
            for class_name, color in (("happy", 180), ("sad", 32)):
                image_dir = second_root / split / class_name
                image_dir.mkdir(parents=True, exist_ok=True)
                Image.new("L", (48, 48), color=color).save(image_dir / f"{class_name}_2.png")

        dataset = build_dataset(Path(self.temp_dir.name), split="train")
        self.assertEqual(len(dataset), 4)

        val_dataset = build_dataset(Path(self.temp_dir.name), split="val")
        self.assertEqual(len(val_dataset), 2)

    def test_nested_parent_directory_is_discovered(self) -> None:
        nested_parent = Path(self.temp_dir.name) / "data" / "data"
        nested_root = nested_parent / "archive_2"
        for split in ("train", "val"):
            for class_name, color in (("happy", 200), ("sad", 16)):
                image_dir = nested_root / split / class_name
                image_dir.mkdir(parents=True, exist_ok=True)
                Image.new("L", (48, 48), color=color).save(image_dir / f"{class_name}_nested.png")

        roots = find_image_folder_roots(nested_parent, split="train")
        self.assertEqual(roots, [nested_root])

        dataset = build_dataset(Path(self.temp_dir.name) / "data", split="train")
        self.assertEqual(len(dataset), 2)

    def test_numeric_class_directories_are_supported(self) -> None:
        numeric_root = Path(self.temp_dir.name) / "archive(1)" / "DATASET"
        for split in ("train", "test"):
            for class_name, color in (("1", 255), ("7", 64)):
                image_dir = numeric_root / split / class_name
                image_dir.mkdir(parents=True, exist_ok=True)
                Image.new("L", (48, 48), color=color).save(image_dir / f"{split}_{class_name}.png")

        dataset = build_dataset(Path(self.temp_dir.name) / "archive(1)", split="train")
        loader = DataLoader(dataset, batch_size=2)
        _images, labels = next(iter(loader))

        self.assertCountEqual(labels.tolist(), [0, 6])


if __name__ == "__main__":
    unittest.main()
