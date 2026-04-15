from __future__ import annotations

import importlib.util
import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import build_model


@unittest.skipUnless(importlib.util.find_spec("torchvision"), "torchvision is not installed")
class ModelTests(unittest.TestCase):
    def test_efficientnet_b0_forward_shape(self) -> None:
        model = build_model(num_classes=7, pretrained=False)
        model.eval()

        with torch.no_grad():
            logits = model(torch.randn(2, 3, 224, 224))

        self.assertEqual(tuple(logits.shape), (2, 7))


if __name__ == "__main__":
    unittest.main()
