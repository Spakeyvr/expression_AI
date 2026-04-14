from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    DEFAULT_IMAGE_SIZE,
    EMOTION_LABELS,
    DeviceResolutionError,
    pil_to_model_tensor,
    resolve_device,
)
from data.dataset import build_dataset
from model.model import build_model, ensure_torchvision_available

TRAIN_AUGMENTATION = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.RandomRotation(
            degrees=20,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,
        ),
    ]
)


def _train_transform(image):
    return pil_to_model_tensor(TRAIN_AUGMENTATION(image))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Expression AI emotion classifier.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a FER2013 CSV file or a dataset directory with split/class/image folders.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(ROOT / "model" / "checkpoints"),
        help="Directory where best.pt will be saved.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps"),
        help="Training device. auto prefers MPS when available.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Limit train/validation datasets to the first N samples for quick iteration.",
    )
    parser.add_argument(
        "--smoke-run",
        action="store_true",
        help="Run a tiny 1-epoch subset training job for local verification.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Default 0 is safer on macOS desktops.",
    )
    parser.add_argument(
        "--weights-path",
        default=None,
        help="Optional local path to an EfficientNet-B0 state dict (.pth). Uses this file instead of downloading weights.",
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Initialize EfficientNet-B0 with pretrained ImageNet weights via torchvision's current weights API. This may download weights if they are not already cached locally.",
    )
    pretrained_group.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Train EfficientNet-B0 from random initialization without downloading pretrained weights.",
    )
    parser.set_defaults(pretrained=True)
    return parser


def maybe_subset(
    dataset: Dataset[tuple[torch.Tensor, int]], subset: int | None
) -> Dataset[tuple[torch.Tensor, int]] | Subset:
    if subset is None or subset >= len(dataset):
        return dataset
    return Subset(dataset, range(subset))


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    average_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    average_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    epoch: int,
    val_accuracy: float,
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(EMOTION_LABELS),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "input_size": DEFAULT_IMAGE_SIZE,
            "backbone": "efficientnet_b0",
            "pretrained_backbone": args.pretrained,
        },
        checkpoint_path,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    ensure_torchvision_available()

    if args.smoke_run:
        args.epochs = 1
        args.subset = args.subset or 64
        args.pretrained = False

    try:
        device = resolve_device(args.device)
    except DeviceResolutionError as error:
        print(error, file=sys.stderr)
        return 2

    train_dataset = maybe_subset(
        build_dataset(args.data, split="train", transform=_train_transform),
        args.subset,
    )
    val_dataset = maybe_subset(build_dataset(args.data, split="val"), args.subset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        num_classes=len(EMOTION_LABELS),
        pretrained=args.pretrained,
        weights_path=args.weights_path,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    checkpoint_path = Path(args.checkpoint_dir) / "best.pt"
    best_val_accuracy = -1.0

    print(f"Training on {device} with {len(train_loader.dataset)} train samples.")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(checkpoint_path, model, epoch, val_accuracy, args)
            print(f"Saved improved checkpoint to {checkpoint_path}")

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        raise SystemExit(0)
