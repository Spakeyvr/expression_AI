from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from common import DEFAULT_IMAGE_SIZE, DeviceResolutionError, EMOTION_LABELS, pil_to_model_tensor, resolve_device
from model.model import build_model, ensure_torchvision_available


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    device: torch.device
    class_names: list[str]
    checkpoint: dict[str, Any]
    checkpoint_path: Path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> LoadedCheckpoint:
    ensure_torchvision_available()

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    selected_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=selected_device)
    class_names = checkpoint.get("class_names", list(EMOTION_LABELS))

    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device)
    model.eval()

    return LoadedCheckpoint(
        model=model,
        device=selected_device,
        class_names=class_names,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )


def predict_tensor(
    image_tensor: torch.Tensor,
    loaded_checkpoint: LoadedCheckpoint,
) -> dict[str, Any]:
    if image_tensor.ndim == 3:
        batch = image_tensor.unsqueeze(0)
    elif image_tensor.ndim == 4:
        batch = image_tensor
    else:
        raise ValueError("image_tensor must have shape [3, H, W] or [B, 3, H, W].")

    batch = batch.to(loaded_checkpoint.device)

    with torch.no_grad():
        logits = loaded_checkpoint.model(batch)
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    top_index = int(torch.argmax(probabilities).item())
    probability_map = {
        label: float(probabilities[index].item())
        for index, label in enumerate(loaded_checkpoint.class_names)
    }

    return {
        "label": loaded_checkpoint.class_names[top_index],
        "label_index": top_index,
        "confidence": probability_map[loaded_checkpoint.class_names[top_index]],
        "probabilities": probability_map,
        "device": str(loaded_checkpoint.device),
    }


def predict_pil_image(image: Image.Image, loaded_checkpoint: LoadedCheckpoint) -> dict[str, Any]:
    tensor = pil_to_model_tensor(
        image,
        size=loaded_checkpoint.checkpoint.get("input_size", DEFAULT_IMAGE_SIZE),
    )
    return predict_tensor(tensor, loaded_checkpoint)


def predict_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        loaded_checkpoint = load_checkpoint(checkpoint_path, device=device)
        return predict_pil_image(image, loaded_checkpoint)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-image emotion inference.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", required=True, help="Path to model/checkpoints/best.pt")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps"),
        help="Inference device. auto prefers MPS when available.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        result = predict_image(args.image, args.checkpoint, device=args.device)
    except (DeviceResolutionError, FileNotFoundError, ModuleNotFoundError, ValueError) as error:
        print(error)
        return 2

    print(f"Predicted emotion: {result['label']} ({result['confidence']:.2%})")
    for label, probability in result["probabilities"].items():
        print(f"  {label:<8} {probability:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
