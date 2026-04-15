from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

CANONICAL_EMOTIONS = (
    "surprise",
    "sad",
    "neutral",
    "happy",
    "fear",
    "disgust",
    "angry",
)
FER_ZERO_BASED_EMOTIONS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)
FOLDER_ONE_BASED_EMOTIONS = {
    index + 1: emotion for index, emotion in enumerate(FER_ZERO_BASED_EMOTIONS)
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PATH_COLUMNS = ("pth", "path", "filepath", "file")
LABEL_COLUMNS = ("label", "emotion", "class")
CANONICAL_SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "validation": "val",
    "valid": "val",
    "test": "test",
}
SPLIT_DIR_NAMES = set(SPLIT_ALIASES)
SKIP_REASONS = (
    "non_image_file",
    "invalid_image",
    "unknown_label",
    "no_resolvable_label",
    "copy_error",
)
LABEL_ALIASES = {
    "angry": "angry",
    "anger": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "scared": "fear",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "joyful": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
}
KNOWN_UNSUPPORTED_LABELS = {
    "contempt",
    "contemptuous",
    "other",
    "unknown",
}
DEFAULT_OUTPUT_DIR = Path("data") / "processed_data"


@dataclass(frozen=True)
class LabelResolution:
    normalized_label: str | None
    raw_label: str
    reason: str
    normalized_split: str | None = None


@dataclass(frozen=True)
class CsvManifest:
    csv_path: Path
    base_dir: Path
    entries: dict[str, LabelResolution]


@dataclass
class ProcessingSummary:
    placed_counts: dict[str, int] = field(
        default_factory=lambda: {emotion: 0 for emotion in CANONICAL_EMOTIONS}
    )
    skipped_counts: Counter[str] = field(default_factory=Counter)
    total_processed: int = 0


def _normalize_token(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value.strip()).strip("_")


def _normalize_split(raw_split: str) -> str | None:
    return SPLIT_ALIASES.get(_normalize_token(raw_split))


def _detect_numeric_scheme(values: Iterable[str], prefer_one_based: bool) -> str:
    normalized = {_normalize_token(value) for value in values}
    if "0" in normalized:
        return "zero_based"
    if "7" in normalized:
        return "one_based"
    return "one_based" if prefer_one_based else "zero_based"


def _normalize_label(raw_label: str, numeric_scheme: str) -> str | None:
    normalized = _normalize_token(raw_label)
    if not normalized:
        return None

    alias = LABEL_ALIASES.get(normalized)
    if alias is not None:
        return alias

    if normalized.isdigit():
        value = int(normalized)
        if numeric_scheme == "zero_based" and 0 <= value < len(FER_ZERO_BASED_EMOTIONS):
            return FER_ZERO_BASED_EMOTIONS[value]
        if numeric_scheme == "one_based" and value in FOLDER_ONE_BASED_EMOTIONS:
            return FOLDER_ONE_BASED_EMOTIONS[value]
        if value == 0:
            return FER_ZERO_BASED_EMOTIONS[0]
        if value == 7:
            return FOLDER_ONE_BASED_EMOTIONS[7]

    return None


def _looks_like_label(raw_label: str) -> bool:
    normalized = _normalize_token(raw_label)
    if not normalized or normalized in SPLIT_DIR_NAMES:
        return False
    return normalized in LABEL_ALIASES or normalized in KNOWN_UNSUPPORTED_LABELS or normalized.isdigit()


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _should_exclude(path: Path, excluded_dirs: Iterable[Path]) -> bool:
    return any(_is_relative_to(path, excluded_dir) for excluded_dir in excluded_dirs)


def _build_path_variants(path_value: str | Path) -> set[str]:
    raw_parts = [part for part in str(path_value).replace("\\", "/").split("/") if part and part != "."]
    if not raw_parts:
        return set()

    directories = []
    stripped_directories = []
    for part in raw_parts[:-1]:
        normalized = _normalize_token(part)
        if not normalized:
            continue
        canonical = LABEL_ALIASES.get(normalized, normalized)
        directories.append(canonical)
        if canonical not in SPLIT_DIR_NAMES:
            stripped_directories.append(canonical)

    filename = raw_parts[-1].strip().lower()
    if not filename:
        return set()

    variants = {"/".join([*directories, filename])}
    variants.add("/".join([*stripped_directories, filename]))
    return {variant for variant in variants if variant}


def _resolve_split_from_path_value(path_value: str | Path) -> str | None:
    raw_parts = [part for part in str(path_value).replace("\\", "/").split("/") if part and part != "."]
    for part in raw_parts[:-1]:
        normalized_split = _normalize_split(part)
        if normalized_split is not None:
            return normalized_split
    return None


def _select_column(fieldnames: Iterable[str], supported_names: Iterable[str]) -> str | None:
    normalized_lookup = {_normalize_token(name): name for name in fieldnames}
    for supported_name in supported_names:
        column = normalized_lookup.get(_normalize_token(supported_name))
        if column is not None:
            return column
    return None


def _load_csv_manifest(csv_path: Path) -> CsvManifest | None:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return None

            path_column = _select_column(reader.fieldnames, PATH_COLUMNS)
            label_column = _select_column(reader.fieldnames, LABEL_COLUMNS)
            if path_column is None or label_column is None:
                return None

            rows = list(reader)
    except (OSError, csv.Error):
        return None

    numeric_values = [row.get(label_column, "") for row in rows if row.get(label_column)]
    numeric_scheme = _detect_numeric_scheme(numeric_values, prefer_one_based=False)
    entries: dict[str, LabelResolution] = {}

    for row in rows:
        raw_path = (row.get(path_column) or "").strip()
        raw_label = (row.get(label_column) or "").strip()
        if not raw_path or not raw_label:
            continue

        resolution = LabelResolution(
            normalized_label=_normalize_label(raw_label, numeric_scheme=numeric_scheme),
            raw_label=raw_label,
            reason="manifest",
            normalized_split=_resolve_split_from_path_value(raw_path),
        )
        for variant in _build_path_variants(raw_path):
            entries[variant] = resolution

    if not entries:
        return None

    return CsvManifest(csv_path=csv_path, base_dir=csv_path.parent, entries=entries)


def _discover_csv_manifests(source_dir: Path, excluded_dirs: Iterable[Path]) -> list[CsvManifest]:
    manifests: list[CsvManifest] = []
    for csv_path in sorted(source_dir.rglob("*.csv")):
        if _should_exclude(csv_path, excluded_dirs):
            continue
        manifest = _load_csv_manifest(csv_path)
        if manifest is not None:
            manifests.append(manifest)

    return sorted(manifests, key=lambda manifest: (-len(manifest.base_dir.parts), str(manifest.csv_path)))


def _resolve_manifest_label(image_path: Path, manifests: Iterable[CsvManifest]) -> LabelResolution | None:
    for manifest in manifests:
        if not _is_relative_to(image_path, manifest.base_dir):
            continue

        relative_path = image_path.relative_to(manifest.base_dir)
        for variant in _build_path_variants(relative_path):
            match = manifest.entries.get(variant)
            if match is not None:
                return match
    return None


def _resolve_parent_label(image_path: Path, source_dir: Path) -> LabelResolution | None:
    for ancestor in image_path.parents:
        if ancestor == source_dir.parent:
            break

        if ancestor == source_dir:
            continue

        name = ancestor.name
        normalized_name = _normalize_token(name)
        if not normalized_name or normalized_name in SPLIT_DIR_NAMES:
            continue
        if not _looks_like_label(name):
            continue

        sibling_names = [child.name for child in ancestor.parent.iterdir() if child.is_dir()]
        numeric_scheme = _detect_numeric_scheme(sibling_names, prefer_one_based=True)
        return LabelResolution(
            normalized_label=_normalize_label(name, numeric_scheme=numeric_scheme),
            raw_label=name,
            reason="parent_folder",
            normalized_split=_resolve_parent_split(image_path, source_dir),
        )

    return None


def _resolve_parent_split(image_path: Path, source_dir: Path) -> str | None:
    for ancestor in image_path.parents:
        if ancestor == source_dir.parent:
            break

        if ancestor == source_dir:
            continue

        normalized_split = _normalize_split(ancestor.name)
        if normalized_split is not None:
            return normalized_split

    return None


def _verify_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in CANONICAL_SPLITS:
        split_dir = output_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for emotion in CANONICAL_EMOTIONS:
            (split_dir / emotion).mkdir(parents=True, exist_ok=True)


def _next_available_destination(destination_dir: Path, filename: str) -> Path:
    candidate = destination_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        candidate = destination_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def normalize_dataset(
    source_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_DIR,
) -> ProcessingSummary:
    source_dir = Path(source_path).expanduser().resolve()
    output_dir = Path(output_path).expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset was not found: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source dataset must be a directory: {source_dir}")
    if source_dir == output_dir:
        raise ValueError("Source and output paths must be different.")

    excluded_dirs = [output_dir]
    manifests = _discover_csv_manifests(source_dir, excluded_dirs)
    manifest_paths = {manifest.csv_path for manifest in manifests}

    source_files = [
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and not _should_exclude(path, excluded_dirs)
    ]

    _prepare_output_dir(output_dir)
    summary = ProcessingSummary()

    for file_path in source_files:
        if file_path in manifest_paths:
            continue

        suffix = file_path.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            summary.skipped_counts["non_image_file"] += 1
            _warn(f"Skipping non-image file: {file_path}")
            continue

        resolution = _resolve_manifest_label(file_path, manifests)
        if resolution is None:
            resolution = _resolve_parent_label(file_path, source_dir)

        if resolution is None:
            summary.skipped_counts["no_resolvable_label"] += 1
            _warn(f"No resolvable label for: {file_path}")
            continue

        if resolution.normalized_label is None:
            summary.skipped_counts["unknown_label"] += 1
            _warn(
                f"Unsupported label '{resolution.raw_label}' for {file_path} "
                f"(source: {resolution.reason})"
            )
            continue

        if not _verify_image(file_path):
            summary.skipped_counts["invalid_image"] += 1
            _warn(f"Skipping invalid image file: {file_path}")
            continue

        split_name = resolution.normalized_split or _resolve_parent_split(file_path, source_dir) or "train"
        destination_dir = output_dir / split_name / resolution.normalized_label
        destination_path = _next_available_destination(destination_dir, file_path.name)
        try:
            shutil.copy2(file_path, destination_path)
        except OSError as error:
            summary.skipped_counts["copy_error"] += 1
            _warn(f"Failed to copy {file_path} to {destination_path}: {error}")
            continue

        summary.placed_counts[resolution.normalized_label] += 1
        summary.total_processed += 1

    return summary


def print_summary(summary: ProcessingSummary) -> None:
    print("Dataset normalization complete.")
    print(f"Total images processed: {summary.total_processed}")
    print("Images placed by emotion:")
    for emotion in CANONICAL_EMOTIONS:
        print(f"  {emotion}: {summary.placed_counts[emotion]}")

    print("Skipped files:")
    for reason in SKIP_REASONS:
        print(f"  {reason}: {summary.skipped_counts.get(reason, 0)}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize a raw image dataset into data/processed_data/<split>/<emotion> folders."
    )
    parser.add_argument("source", help="Path to the source dataset directory.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the normalized dataset. Defaults to data/processed_data.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        summary = normalize_dataset(args.source, args.output)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
