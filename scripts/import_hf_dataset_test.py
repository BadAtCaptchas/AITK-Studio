import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from scripts.import_hf_dataset import (
    normalize_caption,
    rank_caption_columns,
    write_caption_file,
    write_image_file,
)


class ImportHfDatasetTests(unittest.TestCase):
    def test_rank_caption_columns_prefers_common_names(self):
        self.assertEqual(
            rank_caption_columns(["notes", "Text", "caption", "title", "prompt"]),
            ["caption", "prompt", "Text", "title", "notes"],
        )

    def test_write_image_file_uses_deterministic_png_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            image = Image.new("RGB", (4, 3), "red")
            image_path = write_image_file({"image": image}, "image", output_dir, 7)

            self.assertEqual(image_path.name, "row_000007.png")
            self.assertTrue(image_path.exists())
            with Image.open(image_path) as imported:
                self.assertEqual(imported.size, (4, 3))

    def test_write_image_file_infers_jpeg_from_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            buffer = BytesIO()
            Image.new("RGB", (5, 2), "blue").save(buffer, format="JPEG")

            image_path = write_image_file({"image": {"bytes": buffer.getvalue()}}, "image", output_dir, 2)

            self.assertEqual(image_path.name, "row_000002.jpg")
            self.assertTrue(image_path.exists())

    def test_caption_sidecars_skip_blank_captions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "row_000000.png"
            image_path.write_bytes(b"image")

            self.assertFalse(write_caption_file(image_path, ""))
            self.assertFalse(image_path.with_suffix(".txt").exists())
            self.assertTrue(write_caption_file(image_path, "a caption"))
            self.assertEqual(image_path.with_suffix(".txt").read_text(encoding="utf-8"), "a caption")

    def test_invalid_image_rows_are_skippable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(write_image_file({"image": None}, "image", Path(temp_dir), 0))

    def test_normalize_caption_handles_missing_and_structured_values(self):
        self.assertEqual(normalize_caption(None), "")
        self.assertEqual(normalize_caption("  hello  "), "hello")
        self.assertEqual(normalize_caption({"text": "hello"}), '{"text": "hello"}')


if __name__ == "__main__":
    unittest.main()
