from __future__ import annotations

import unittest

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from train.train import _train_transform, evaluate, train_one_epoch


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
