from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import asset_path_for_label
from infer import DeviceResolutionError, LoadedCheckpoint, load_checkpoint, predict_pil_image

APP_TITLE = "Expression AI"
FRAME_SIZE = (640, 480)
ASSET_SIZE = (180, 180)
FACE_MIN_SIZE = (48, 48)
ROTATED_DETECTION_ANGLES = (-20, 20)


class CascadeMultiAngleFaceDetector:
    def __init__(self) -> None:
        cascade_root = Path(cv2.data.haarcascades)
        self.frontal_detector = cv2.CascadeClassifier(
            str(cascade_root / "haarcascade_frontalface_alt2.xml")
        )
        self.profile_detector = cv2.CascadeClassifier(
            str(cascade_root / "haarcascade_profileface.xml")
        )
        if self.frontal_detector.empty() or self.profile_detector.empty():
            raise RuntimeError("OpenCV face detection cascades could not be loaded.")

    def detect_faces(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._detect_with_cascade(self.frontal_detector, gray)
        faces.extend(self._detect_profiles(gray))

        # A small rotated pass helps when the user's head is tilted relative to the camera.
        for angle in ROTATED_DETECTION_ANGLES:
            faces.extend(self._detect_rotated_frontal(gray, angle))

        return self._deduplicate_faces(faces)

    def close(self) -> None:
        return

    def _detect_with_cascade(
        self,
        cascade: cv2.CascadeClassifier,
        gray: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=FACE_MIN_SIZE,
        )
        return [tuple(int(value) for value in detection) for detection in detections]

    def _detect_profiles(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = gray.shape[:2]
        faces = self._detect_with_cascade(self.profile_detector, gray)

        flipped = cv2.flip(gray, 1)
        for x, y, box_width, box_height in self._detect_with_cascade(self.profile_detector, flipped):
            faces.append((width - (x + box_width), y, box_width, box_height))

        return [
            self._clip_box(face, width, height)
            for face in faces
            if self._clip_box(face, width, height) is not None
        ]

    def _detect_rotated_frontal(
        self,
        gray: np.ndarray,
        angle: float,
    ) -> list[tuple[int, int, int, int]]:
        height, width = gray.shape[:2]
        center = (width / 2.0, height / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        inverse_rotation = cv2.invertAffineTransform(rotation_matrix)

        faces: list[tuple[int, int, int, int]] = []
        for x, y, box_width, box_height in self._detect_with_cascade(self.frontal_detector, rotated):
            corners = np.array(
                [
                    [[x, y]],
                    [[x + box_width, y]],
                    [[x, y + box_height]],
                    [[x + box_width, y + box_height]],
                ],
                dtype=np.float32,
            )
            mapped_corners = cv2.transform(corners, inverse_rotation).reshape(-1, 2)
            min_x, min_y = mapped_corners.min(axis=0)
            max_x, max_y = mapped_corners.max(axis=0)
            mapped_box = self._clip_box(
                (
                    int(round(min_x)),
                    int(round(min_y)),
                    int(round(max_x - min_x)),
                    int(round(max_y - min_y)),
                ),
                width,
                height,
            )
            if mapped_box is not None:
                faces.append(mapped_box)
        return faces

    def _deduplicate_faces(
        self,
        faces: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        unique_faces: list[tuple[int, int, int, int]] = []
        for face in sorted(faces, key=lambda item: item[2] * item[3], reverse=True):
            if any(self._intersection_over_union(face, existing) >= 0.4 for existing in unique_faces):
                continue
            unique_faces.append(face)
        return unique_faces

    @staticmethod
    def _clip_box(
        face: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        x, y, box_width, box_height = face
        x = max(0, x)
        y = max(0, y)
        box_width = min(box_width, width - x)
        box_height = min(box_height, height - y)
        if box_width < FACE_MIN_SIZE[0] or box_height < FACE_MIN_SIZE[1]:
            return None
        return x, y, box_width, box_height

    @staticmethod
    def _intersection_over_union(
        left: tuple[int, int, int, int],
        right: tuple[int, int, int, int],
    ) -> float:
        left_x1, left_y1, left_w, left_h = left
        right_x1, right_y1, right_w, right_h = right
        left_x2 = left_x1 + left_w
        left_y2 = left_y1 + left_h
        right_x2 = right_x1 + right_w
        right_y2 = right_y1 + right_h

        intersection_x1 = max(left_x1, right_x1)
        intersection_y1 = max(left_y1, right_y1)
        intersection_x2 = min(left_x2, right_x2)
        intersection_y2 = min(left_y2, right_y2)

        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height
        if intersection_area == 0:
            return 0.0

        left_area = left_w * left_h
        right_area = right_w * right_h
        union_area = left_area + right_area - intersection_area
        return intersection_area / union_area if union_area else 0.0


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
        self.model_load_error: str | None = None
        self.capture: cv2.VideoCapture | None = None
        self.running = True

        asset_dir = Path(__file__).resolve().parent / "images"
        self.asset_images = self._load_assets(asset_dir)
        self.face_detector = CascadeMultiAngleFaceDetector()

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
            self.model_load_error = f"Checkpoint missing: {self.checkpoint_path}"
            self._set_status(f"{self.model_load_error}. Train the model before opening the webcam app.")
            return

        try:
            self.loaded_checkpoint = load_checkpoint(self.checkpoint_path, device=self.device_name)
            self._set_status(f"Model ready on {self.loaded_checkpoint.device}.")
        except Exception as error:
            self.model_load_error = f"{type(error).__name__}: {error}"
            self._set_status(f"Model failed to load: {self.model_load_error}")

    def _open_camera(self) -> None:
        capture = self.capture_factory(self.camera_index)
        if not capture or not capture.isOpened():
            self._set_status(
                "Unable to open the webcam. Check camera availability and macOS camera permissions."
            )
            self.capture = None
            return

        self.capture = capture
        if self.loaded_checkpoint is not None:
            self._set_status("Webcam connected. Looking for a face…")
        # If loaded_checkpoint is None, keep the specific error from _load_model visible.

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

    @staticmethod
    def _build_display_image(frame_rgb: np.ndarray) -> Image.Image:
        source_height, source_width = frame_rgb.shape[:2]
        target_width, target_height = FRAME_SIZE
        scale = min(target_width / source_width, target_height / source_height)
        scaled_width = max(1, int(round(source_width * scale)))
        scaled_height = max(1, int(round(source_height * scale)))

        resized = Image.fromarray(frame_rgb).resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", FRAME_SIZE, color=(17, 19, 24))
        offset_x = (target_width - scaled_width) // 2
        offset_y = (target_height - scaled_height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    def _render_frame(self, frame_bgr) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        display_image = self._build_display_image(frame_rgb)
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

        faces = self.face_detector.detect_faces(frame)
        largest_face = self._select_largest_face(faces)

        if largest_face is None:
            self.prediction_label.config(text="Prediction: --")
            self.confidence_label.config(text="Confidence: --")
            self._set_status("No face detected.")
            self._update_asset("Neutral")
        else:
            x, y, width, height = largest_face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 220, 120), 2)

            if self.loaded_checkpoint is None:
                self._set_status(
                    f"Face detected — model unavailable: {self.model_load_error}"
                    if self.model_load_error
                    else "Face detected — model unavailable."
                )
            elif self.frame_counter % self.inference_stride == 0:
                try:
                    face_region = frame[y : y + height, x : x + width]
                    face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    result = predict_pil_image(Image.fromarray(face_gray, mode="L"), self.loaded_checkpoint)
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
        self.face_detector.close()
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
