from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import train.train as train_module
from train.train import (
    _train_transform,
    evaluate,
    maybe_subset,
    resolve_data_path,
    train_one_epoch,
)


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 224 * 224, 7),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.network(batch)


class TrainingTests(unittest.TestCase):
    def test_train_transform_returns_model_tensor(self) -> None:
        tensor = _train_transform(Image.new("L", (48, 48), color=128))

        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_train_and_evaluate_smoke(self) -> None:
        dataset = TensorDataset(
            torch.randn(4, 3, 224, 224),
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        model = TinyClassifier()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        train_loss, train_accuracy = train_one_epoch(model, loader, optimizer, torch.device("cpu"))
        val_loss, val_accuracy = evaluate(model, loader, torch.device("cpu"))

        self.assertGreaterEqual(train_loss, 0.0)
        self.assertGreaterEqual(val_loss, 0.0)
        self.assertGreaterEqual(train_accuracy, 0.0)
        self.assertGreaterEqual(val_accuracy, 0.0)

    def test_maybe_subset_is_deterministic_for_seed(self) -> None:
        dataset = TensorDataset(
            torch.arange(100).view(100, 1).float(),
            torch.arange(100, dtype=torch.long),
        )

        subset_a = maybe_subset(dataset, subset=10, seed=123)
        subset_b = maybe_subset(dataset, subset=10, seed=123)

        labels_a = [subset_a[i][1].item() for i in range(len(subset_a))]
        labels_b = [subset_b[i][1].item() for i in range(len(subset_b))]
        self.assertEqual(labels_a, labels_b)
        self.assertNotEqual(labels_a, list(range(10)))

    def test_resolve_data_path_prefers_raw_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_root = root / "raw_data"
            processed_root = root / "processed_data"
            (raw_root / "train" / "happy").mkdir(parents=True)
            (processed_root / "train" / "sad").mkdir(parents=True)

            original_dirs = train_module.AUTO_DATA_DIRS
            try:
                train_module.AUTO_DATA_DIRS = (raw_root, processed_root)
                self.assertEqual(resolve_data_path(None), raw_root)
            finally:
                train_module.AUTO_DATA_DIRS = original_dirs

    def test_resolve_data_path_falls_back_to_processed_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_root = root / "raw_data"
            processed_root = root / "processed_data"
            raw_root.mkdir(parents=True)
            (processed_root / "train" / "sad").mkdir(parents=True)

            original_dirs = train_module.AUTO_DATA_DIRS
            try:
                train_module.AUTO_DATA_DIRS = (raw_root, processed_root)
                self.assertEqual(resolve_data_path(None), processed_root)
            finally:
                train_module.AUTO_DATA_DIRS = original_dirs

    def test_resolve_data_path_expands_explicit_argument(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            explicit_path = Path(temp_dir) / "custom_dataset"
            self.assertEqual(resolve_data_path(str(explicit_path)), explicit_path)
