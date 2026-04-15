from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from PIL import Image

import process_data
from data.dataset import build_dataset


class ProcessDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source"
        self.output = self.root / "normalized"
        self.default_output = self.root / process_data.DEFAULT_OUTPUT_DIR
        self.source.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_image(self, path: Path, color: int = 128) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 24), color=(color, color, color)).save(path)

    def _write_text(self, path: Path, content: str = "not an image") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_normalizes_folder_aliases_and_numeric_labels(self) -> None:
        self._write_image(self.source / "Train" / "anger" / "alias.jpg")
        self._write_image(self.source / "Test" / "sadness" / "sad_case.png")
        self._write_image(self.source / "zero_based" / "Train" / "0" / "zero_angry.jpg")
        self._write_image(self.source / "zero_based" / "Train" / "1" / "zero_disgust.jpg")
        self._write_image(self.source / "one_based" / "Test" / "1" / "one_angry.jpg")
        self._write_image(self.source / "one_based" / "Test" / "7" / "one_neutral.png")

        summary = process_data.normalize_dataset(self.source, self.output)

        self.assertEqual(summary.total_processed, 6)
        self.assertEqual(summary.placed_counts["angry"], 3)
        self.assertEqual(summary.placed_counts["sad"], 1)
        self.assertEqual(summary.placed_counts["disgust"], 1)
        self.assertEqual(summary.placed_counts["neutral"], 1)

        self.assertTrue((self.output / "train" / "angry" / "alias.jpg").exists())
        self.assertTrue((self.output / "test" / "sad" / "sad_case.png").exists())
        self.assertTrue((self.output / "train" / "disgust" / "zero_disgust.jpg").exists())
        self.assertTrue((self.output / "test" / "neutral" / "one_neutral.png").exists())

    def test_csv_manifest_takes_precedence_over_folder_name(self) -> None:
        self._write_image(self.source / "Train" / "anger" / "image0000006.jpg")
        (self.source / "labels.csv").write_text(
            "pth,label\nanger/image0000006.jpg,surprise\n",
            encoding="utf-8",
        )

        summary = process_data.normalize_dataset(self.source, self.output)

        self.assertEqual(summary.total_processed, 1)
        self.assertEqual(summary.placed_counts["surprise"], 1)
        self.assertTrue((self.output / "train" / "surprise" / "image0000006.jpg").exists())
        self.assertFalse((self.output / "train" / "angry" / "image0000006.jpg").exists())
        self.assertEqual(summary.skipped_counts["non_image_file"], 0)

    def test_skips_invalid_non_image_unknown_and_unlabeled_files(self) -> None:
        self._write_image(self.source / "Train" / "happy" / "valid.jpg")
        self._write_text(self.source / "Train" / "happy" / "broken.jpg")
        self._write_text(self.source / "notes.txt")
        self._write_image(self.source / "Train" / "contempt" / "unsupported.png")
        self._write_image(self.source / "misc" / "unlabeled.png")

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exit_code = process_data.main([str(self.source), "--output", str(self.output)])

        self.assertEqual(exit_code, 0)
        summary_output = stdout_buffer.getvalue()
        warning_output = stderr_buffer.getvalue()

        self.assertIn("Total images processed: 1", summary_output)
        self.assertIn("happy: 1", summary_output)
        self.assertIn("non_image_file: 1", summary_output)
        self.assertIn("invalid_image: 1", summary_output)
        self.assertIn("unknown_label: 1", summary_output)
        self.assertIn("no_resolvable_label: 1", summary_output)
        self.assertIn("Skipping non-image file", warning_output)
        self.assertIn("Skipping invalid image file", warning_output)
        self.assertIn("Unsupported label 'contempt'", warning_output)
        self.assertIn("No resolvable label", warning_output)

    def test_handles_filename_collisions(self) -> None:
        self._write_image(self.source / "Train" / "happy" / "duplicate.jpg", color=20)
        self._write_image(self.source / "archive" / "Train" / "happy" / "duplicate.jpg", color=200)

        summary = process_data.normalize_dataset(self.source, self.output)

        self.assertEqual(summary.placed_counts["happy"], 2)
        self.assertTrue((self.output / "train" / "happy" / "duplicate.jpg").exists())
        self.assertTrue((self.output / "train" / "happy" / "duplicate_1.jpg").exists())

    def test_output_contains_exact_split_and_emotion_directories(self) -> None:
        self._write_image(self.source / "Train" / "happy" / "sample.jpg")

        process_data.normalize_dataset(self.source, self.output)

        self.assertEqual(
            sorted(path.name for path in self.output.iterdir()),
            sorted(process_data.CANONICAL_SPLITS),
        )
        self.assertTrue(all(path.is_dir() for path in self.output.iterdir()))
        for split in process_data.CANONICAL_SPLITS:
            self.assertEqual(
                sorted(path.name for path in (self.output / split).iterdir()),
                sorted(process_data.CANONICAL_EMOTIONS),
            )

    def test_main_uses_processed_data_default_output_path(self) -> None:
        self._write_image(self.source / "Train" / "happy" / "sample.jpg")

        original_cwd = Path.cwd()
        os.chdir(self.root)
        try:
            exit_code = process_data.main([str(self.source)])
        finally:
            os.chdir(original_cwd)

        self.assertEqual(exit_code, 0)
        self.assertTrue((self.default_output / "train" / "happy" / "sample.jpg").exists())

    def test_normalized_output_can_be_loaded_by_training_dataset_builder(self) -> None:
        self._write_image(self.source / "Train" / "happy" / "train_sample.jpg")
        self._write_image(self.source / "Test" / "sad" / "test_sample.jpg")

        process_data.normalize_dataset(self.source, self.output)

        train_dataset = build_dataset(self.output, split="train")
        test_dataset = build_dataset(self.output, split="test")

        self.assertEqual(len(train_dataset), 1)
        self.assertEqual(len(test_dataset), 1)


if __name__ == "__main__":
    unittest.main()
