from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

import cv2
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import asset_path_for_label
from infer import DeviceResolutionError, LoadedCheckpoint, load_checkpoint, predict_pil_image

APP_TITLE = "Expression AI"
FRAME_SIZE = (640, 480)
ASSET_SIZE = (180, 180)


class ExpressionAIApp(tk.Tk):
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        camera_index: int = 0,
        inference_stride: int = 6,
        capture_factory: Callable[[int], cv2.VideoCapture] | None = None,
        start_loop: bool = True,
    ) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x720")
        self.configure(bg="#111318")

        self.checkpoint_path = Path(checkpoint_path)
        self.device_name = device
        self.camera_index = camera_index
        self.inference_stride = max(inference_stride, 1)
        self.capture_factory = capture_factory or cv2.VideoCapture
        self.frame_counter = 0
        self.latest_video_image: ImageTk.PhotoImage | None = None
        self.latest_asset_image: ImageTk.PhotoImage | None = None
        self.loaded_checkpoint: LoadedCheckpoint | None = None
        self.capture: cv2.VideoCapture | None = None
        self.running = True

        asset_dir = Path(__file__).resolve().parent / "images"
        self.asset_images = self._load_assets(asset_dir)
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self._build_layout()
        self._set_status("Starting Expression AI…")
        self._load_model()
        self._open_camera()

        self.protocol("WM_DELETE_WINDOW", self.shutdown)
        if start_loop:
            self.after(20, self.process_frame)

    def _build_layout(self) -> None:
        title = ttk.Label(self, text="Expression AI", font=("Aptos", 26, "bold"))
        title.pack(pady=(18, 8))

        content = ttk.Frame(self, padding=16)
        content.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(content)
        self.video_label.grid(row=0, column=0, padx=(0, 16), pady=8, sticky="nsew")

        side_panel = ttk.Frame(content)
        side_panel.grid(row=0, column=1, sticky="n")

        self.asset_label = ttk.Label(side_panel)
        self.asset_label.pack(pady=(0, 12))

        self.prediction_label = ttk.Label(
            side_panel, text="Prediction: --", font=("Aptos", 18, "bold")
        )
        self.prediction_label.pack(anchor="w", pady=(0, 8))

        self.confidence_label = ttk.Label(side_panel, text="Confidence: --")
        self.confidence_label.pack(anchor="w", pady=(0, 8))

        self.status_label = ttk.Label(side_panel, text="Status: --", wraplength=240, justify="left")
        self.status_label.pack(anchor="w", pady=(0, 8))

        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

    def _load_assets(self, asset_dir: Path) -> dict[str, ImageTk.PhotoImage]:
        assets: dict[str, ImageTk.PhotoImage] = {}
        for label in (
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Sad",
            "Surprise",
            "Neutral",
        ):
            asset_path = asset_path_for_label(label, asset_dir)
            with Image.open(asset_path) as image:
                resized = image.convert("RGB").resize(ASSET_SIZE)
                assets[label] = ImageTk.PhotoImage(resized)
        return assets

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            self._set_status(
                f"Checkpoint missing: {self.checkpoint_path}. Train the model before opening the webcam app."
            )
            return

        try:
            self.loaded_checkpoint = load_checkpoint(self.checkpoint_path, device=self.device_name)
            self._set_status(f"Model ready on {self.loaded_checkpoint.device}.")
        except (DeviceResolutionError, FileNotFoundError, ModuleNotFoundError, RuntimeError) as error:
            self._set_status(f"Model failed to load: {error}")

    def _open_camera(self) -> None:
        capture = self.capture_factory(self.camera_index)
        if not capture or not capture.isOpened():
            self._set_status(
                "Unable to open the webcam. Check camera availability and macOS camera permissions."
            )
            self.capture = None
            return

        self.capture = capture
        if self.loaded_checkpoint is None:
            self._set_status(
                "Webcam connected, but the model is unavailable. Train the model or fix the checkpoint path."
            )
        else:
            self._set_status("Webcam connected. Looking for a face…")

    def _set_status(self, message: str) -> None:
        self.status_label.config(text=f"Status: {message}")

    def _select_largest_face(self, faces: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
        if not faces:
            return None
        return max(faces, key=lambda face: face[2] * face[3])

    def _update_asset(self, label: str) -> None:
        asset = self.asset_images.get(label)
        if asset is None:
            return
        self.latest_asset_image = asset
        self.asset_label.config(image=self.latest_asset_image)

    def _render_frame(self, frame_bgr) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        display_image = Image.fromarray(frame_rgb).resize(FRAME_SIZE)
        self.latest_video_image = ImageTk.PhotoImage(display_image)
        self.video_label.config(image=self.latest_video_image)

    def process_frame(self) -> None:
        if not self.running:
            return

        if self.capture is None:
            self.after(200, self.process_frame)
            return

        success, frame = self.capture.read()
        if not success:
            self._set_status(
                "Webcam frame read failed. Check camera permissions or whether another app is using the camera."
            )
            self.after(200, self.process_frame)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        largest_face = self._select_largest_face(list(faces))

        if largest_face is None:
            self.prediction_label.config(text="Prediction: --")
            self.confidence_label.config(text="Confidence: --")
            self._set_status("No face detected.")
            self._update_asset("Neutral")
        else:
            x, y, width, height = largest_face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 220, 120), 2)

            if self.loaded_checkpoint is not None and self.frame_counter % self.inference_stride == 0:
                try:
                    face_region = frame[y : y + height, x : x + width]
                    face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                    result = predict_pil_image(Image.fromarray(face_rgb), self.loaded_checkpoint)
                    self.prediction_label.config(text=f"Prediction: {result['label']}")
                    self.confidence_label.config(
                        text=f"Confidence: {result['confidence']:.2%}"
                    )
                    self._set_status(f"Tracking face on {result['device']}.")
                    self._update_asset(result["label"])
                except Exception as error:  # pragma: no cover - UI safeguard
                    self._set_status(f"Inference failed: {error}")

        self.frame_counter += 1
        self._render_frame(frame)
        self.after(20, self.process_frame)

    def shutdown(self) -> None:
        self.running = False
        if self.capture is not None:
            self.capture.release()
        self.destroy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Expression AI webcam demo.")
    parser.add_argument("--checkpoint", required=True, help="Path to model/checkpoints/best.pt")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps"),
        help="Inference device. auto prefers MPS when available.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--inference-stride",
        type=int,
        default=6,
        help="Run model inference every N frames to keep the UI responsive.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    app = ExpressionAIApp(
        checkpoint_path=args.checkpoint,
        device=args.device,
        camera_index=args.camera_index,
        inference_stride=args.inference_stride,
    )
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
