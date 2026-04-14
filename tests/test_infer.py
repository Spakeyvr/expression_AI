from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from common import EMOTION_LABELS
from infer import LoadedCheckpoint, predict_batch_tensor, predict_pil_image, predict_tensor


class FakeModel(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((batch.size(0), len(EMOTION_LABELS)), dtype=torch.float32)
        for index in range(batch.size(0)):
            logits[index, index % len(EMOTION_LABELS)] = 5.0
        return logits


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loaded_checkpoint = LoadedCheckpoint(
            model=FakeModel(),
            device=torch.device("cpu"),
            class_names=list(EMOTION_LABELS),
            checkpoint={"input_size": 224},
            checkpoint_path=Path("fake.pt"),
        )

    def test_predict_tensor_returns_probabilities(self) -> None:
        image_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
        result = predict_tensor(image_tensor, self.loaded_checkpoint)

        self.assertEqual(result["label"], "Angry")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)

    def test_predict_pil_image_supports_static_images(self) -> None:
        image = Image.new("RGB", (96, 96), color=(120, 120, 120))
        result = predict_pil_image(image, self.loaded_checkpoint)

        self.assertEqual(result["label"], "Angry")

    def test_predict_batch_tensor_supports_multiple_images(self) -> None:
        image_tensor = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
        results = predict_batch_tensor(image_tensor, self.loaded_checkpoint)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["label"], "Angry")
        self.assertEqual(results[1]["label"], "Disgust")

    def test_predict_tensor_rejects_multi_image_batches(self) -> None:
        image_tensor = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
        with self.assertRaises(ValueError):
            predict_tensor(image_tensor, self.loaded_checkpoint)


@unittest.skipUnless(importlib.util.find_spec("torchvision"), "torchvision is not installed")
class CheckpointErrorTests(unittest.TestCase):
    def test_missing_checkpoint_raises_clear_error(self) -> None:
        from infer import load_checkpoint

        with self.assertRaises(FileNotFoundError):
            load_checkpoint("does-not-exist.pt")

    def test_invalid_image_path_raises_clear_error(self) -> None:
        from infer import predict_image

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "missing.pt"
            with self.assertRaises(FileNotFoundError):
                predict_image("missing-image.jpg", checkpoint_path)
