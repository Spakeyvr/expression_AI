from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from display.app import ExpressionAIApp


class FakeCapture:
    def __init__(self, opened: bool, frame=None) -> None:
        self._opened = opened
        self._frame = frame

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if self._frame is not None:
            return True, self._frame.copy()
        return False, None

    def release(self) -> None:
        pass


@unittest.skipUnless(
    os.environ.get("EXPRESSION_AI_RUN_UI_TESTS") == "1",
    "UI tests are skipped by default in headless or sandboxed environments.",
)
class AppTests(unittest.TestCase):
    def test_camera_unavailable_updates_status(self) -> None:
        with mock.patch.object(ExpressionAIApp, "_load_assets", return_value={}):
            app = ExpressionAIApp(
                checkpoint_path=Path("missing.pt"),
                capture_factory=lambda index: FakeCapture(opened=False),
                start_loop=False,
            )
            app.withdraw()

        self.assertIn("Unable to open the webcam", app.status_label.cget("text"))
        app.shutdown()

    def test_no_face_state_updates_status(self) -> None:
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        with mock.patch.object(ExpressionAIApp, "_load_assets", return_value={}):
            app = ExpressionAIApp(
                checkpoint_path=Path("missing.pt"),
                capture_factory=lambda index: FakeCapture(opened=True, frame=frame),
                start_loop=False,
            )
            app.withdraw()
            app.face_detector = mock.Mock()
            app.face_detector.detect_faces.return_value = []

            app.process_frame()

        self.assertIn("No face detected", app.status_label.cget("text"))
        app.shutdown()
