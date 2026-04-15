from __future__ import annotations

import unittest
import sys
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import DeviceResolutionError, resolve_device


class DeviceTests(unittest.TestCase):
    def test_auto_falls_back_to_cpu_when_mps_unavailable(self) -> None:
        with mock.patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(str(resolve_device("auto")), "cpu")

    def test_mps_raises_clear_error_when_unavailable(self) -> None:
        with mock.patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaises(DeviceResolutionError):
                resolve_device("mps")
