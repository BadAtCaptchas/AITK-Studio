import ast
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_MODULE = ROOT / "toolkit" / "config_modules.py"


def load_generate_image_methods(*names):
    tree = ast.parse(CONFIG_MODULE.read_text(encoding="utf-8"))
    generate_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GenerateImageConfig"
    )
    methods = [
        node
        for node in generate_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    namespace = {"os": os, "shutil": shutil, "tempfile": tempfile}
    exec(compile(module, str(CONFIG_MODULE), "exec"), namespace)
    return {name: namespace[name] for name in names}


METHODS = load_generate_image_methods("save_image_atomic", "_generate_thumbnail")


class AtomicSaveHarness:
    save_image_atomic = METHODS["save_image_atomic"]

    def __init__(self, output_folder, save_image):
        self.output_folder = output_folder
        self._sample_write_lock = threading.Lock()
        self.save_image = types.MethodType(save_image, self)

    def get_image_path(self, count=0, max_count=0):
        return os.path.join(self.output_folder, f"sample-{count}.png")

    def _generate_thumbnail(self, media_path, thumb_path):
        with open(thumb_path, "wb") as handle:
            handle.write(b"jpeg")
        return True


class SampleAtomicWriteTests(unittest.TestCase):
    def test_partial_media_is_hidden_and_thumbnail_is_published(self):
        with tempfile.TemporaryDirectory() as root:
            wrote_partial = threading.Event()
            finish_write = threading.Event()

            def save_image(harness, image, count, max_count):
                with open(harness.get_image_path(count, max_count), "wb") as handle:
                    handle.write(b"partial")
                    handle.flush()
                    wrote_partial.set()
                    self.assertTrue(finish_write.wait(5))
                    handle.write(b"-complete")

            harness = AtomicSaveHarness(root, save_image)
            worker = threading.Thread(target=harness.save_image_atomic, args=(object(), 1))
            worker.start()
            self.assertTrue(wrote_partial.wait(5))
            self.assertFalse(os.path.exists(os.path.join(root, "sample-1.png")))
            finish_write.set()
            worker.join(5)

            self.assertFalse(worker.is_alive())
            self.assertEqual(Path(root, "sample-1.png").read_bytes(), b"partial-complete")
            self.assertEqual(Path(root, ".thumbs", "sample-1.png.jpg").read_bytes(), b"jpeg")

    def test_concurrent_saves_use_isolated_staging_directories(self):
        with tempfile.TemporaryDirectory() as root:
            barrier = threading.Barrier(2)

            def make_save_image():
                def save_image(harness, image, count, max_count):
                    barrier.wait(5)
                    Path(harness.get_image_path(count, max_count)).write_bytes(str(count).encode())

                return save_image

            first = AtomicSaveHarness(root, make_save_image())
            second = AtomicSaveHarness(root, make_save_image())
            workers = [
                threading.Thread(target=first.save_image_atomic, args=(object(), 1)),
                threading.Thread(target=second.save_image_atomic, args=(object(), 2)),
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(5)

            self.assertTrue(all(not worker.is_alive() for worker in workers))
            self.assertEqual(Path(root, "sample-1.png").read_bytes(), b"1")
            self.assertEqual(Path(root, "sample-2.png").read_bytes(), b"2")
            self.assertFalse(os.path.exists(os.path.join(root, ".tmp")))

    def test_failed_save_restores_output_folder_and_removes_staging(self):
        with tempfile.TemporaryDirectory() as root:
            def save_image(harness, image, count, max_count):
                Path(harness.get_image_path(count, max_count)).write_bytes(b"partial")
                raise RuntimeError("save failed")

            harness = AtomicSaveHarness(root, save_image)
            with self.assertRaisesRegex(RuntimeError, "save failed"):
                harness.save_image_atomic(object())

            self.assertEqual(harness.output_folder, root)
            self.assertFalse(os.path.exists(os.path.join(root, ".tmp")))


class ThumbnailGenerationTests(unittest.TestCase):
    def setUp(self):
        self.harness = types.SimpleNamespace()
        self.generate = types.MethodType(METHODS["_generate_thumbnail"], self.harness)

    def test_static_and_animated_images_generate_valid_jpegs(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as root:
            static_path = os.path.join(root, "wide.png")
            animated_path = os.path.join(root, "animated.webp")
            static_thumb = os.path.join(root, "wide.jpg")
            animated_thumb = os.path.join(root, "animated.jpg")
            Image.new("RGB", (640, 320), "red").save(static_path)
            Image.new("RGB", (320, 640), "blue").save(
                animated_path,
                format="WEBP",
                save_all=True,
                append_images=[Image.new("RGB", (320, 640), "green")],
            )

            self.assertTrue(self.generate(static_path, static_thumb))
            self.assertTrue(self.generate(animated_path, animated_thumb))
            for thumb_path in (static_thumb, animated_thumb):
                with Image.open(thumb_path) as thumb:
                    self.assertEqual(thumb.format, "JPEG")
                    self.assertEqual(thumb.size, (300, 300))

    def test_mp4_uses_first_frame_and_releases_capture(self):
        from PIL import Image

        released = []

        class FakeCapture:
            def __init__(self, media_path):
                self.media_path = media_path

            def read(self):
                return True, object()

            def release(self):
                released.append(self.media_path)

        fake_cv2 = types.SimpleNamespace(
            COLOR_BGR2RGB=1,
            VideoCapture=FakeCapture,
            cvtColor=lambda frame, mode: frame,
        )
        original_cv2 = sys.modules.get("cv2")
        original_fromarray = Image.fromarray
        sys.modules["cv2"] = fake_cv2
        Image.fromarray = lambda frame: Image.new("RGB", (400, 200), "purple")
        try:
            with tempfile.TemporaryDirectory() as root:
                media_path = os.path.join(root, "sample.mp4")
                thumb_path = os.path.join(root, "sample.jpg")
                Path(media_path).write_bytes(b"video")
                self.assertTrue(self.generate(media_path, thumb_path))
                self.assertEqual(released, [media_path])
                with Image.open(thumb_path) as thumb:
                    self.assertEqual(thumb.size, (300, 300))
        finally:
            Image.fromarray = original_fromarray
            if original_cv2 is None:
                sys.modules.pop("cv2", None)
            else:
                sys.modules["cv2"] = original_cv2


if __name__ == "__main__":
    unittest.main()
