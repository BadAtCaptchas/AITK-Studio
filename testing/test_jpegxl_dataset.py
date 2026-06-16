import base64
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    if importlib.util.find_spec("pillow_jxl") is None:
        raise ImportError("pillow_jxl is not importable")
    import pillow_jxl  # noqa: F401
    JXL_AVAILABLE = True
except Exception:
    JXL_AVAILABLE = False

from toolkit import image_io, image_utils
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset
from toolkit.encrypted_dataset import EncryptedDatasetReader


class FakeSD:
    def __init__(self):
        self.adapter = None
        self.device = "cpu"
        self.device_torch = torch.device("cpu")
        self.encode_control_in_text_embeddings = False
        self.is_audio_model = False
        self.is_auraflow = False
        self.is_flux = False
        self.is_v3 = False
        self.is_xl = False
        self.latent_space_version = None
        self.model_config = SimpleNamespace(
            arch="sd1",
            is_pixart_sigma=False,
            latent_space_version=None,
        )
        self.sample_rate = 48000
        self.te_padding_side = "right"
        self.text_embedding_space_version = "sd1"
        self.torch_dtype = torch.float32
        self.unet = SimpleNamespace()
        self.use_raw_control_images = False
        self.vae = SimpleNamespace()

    def get_bucket_divisibility(self):
        return 32


def _write_rgb_jxl(path, size=(96, 80), color=(32, 64, 128)):
    Image.new("RGB", size, color).save(path, lossless=True)


def _write_rgba_jxl(path, size=(64, 64)):
    image = Image.new("RGBA", size, (20, 40, 60, 255))
    alpha = Image.new("L", size, 255)
    draw = ImageDraw.Draw(alpha)
    draw.rectangle((0, 0, size[0] // 2 - 1, size[1] - 1), fill=0)
    image.putalpha(alpha)
    image.save(path, lossless=True)


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


class JPEGXLMissingPluginMessageTest(unittest.TestCase):
    def test_jxl_bytes_without_plugin_raise_clear_error(self):
        previous = image_io._jpegxl_plugin_available
        image_io._jpegxl_plugin_available = False
        try:
            with self.assertRaisesRegex(image_io.JpegXLSupportError, "pillow-jxl-plugin"):
                image_io.open_static_image_from_bytes(b"\xff\x0a" + (b"\x00" * 32), source="sample.jxl")
            with self.assertRaisesRegex(image_io.JpegXLSupportError, "pillow-jxl-plugin"):
                image_utils.get_image_metadata_from_bytesio(io.BytesIO(b"\xff\x0a" + (b"\x00" * 32)), 34)
        finally:
            image_io._jpegxl_plugin_available = previous


@unittest.skipUnless(JXL_AVAILABLE, "pillow_jxl is required for JPEG XL encode/decode tests")
class JPEGXLDatasetTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="aitk-jxl-")
        self.dataset_dir = os.path.join(self.root, "dataset")
        self.inpaint_dir = os.path.join(self.root, "inpaint")
        os.makedirs(self.dataset_dir)
        os.makedirs(self.inpaint_dir)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _dataset_config(self, **overrides):
        kwargs = {
            "dataset_path": self.dataset_dir,
            "resolution": 32,
            "buckets": False,
        }
        kwargs.update(overrides)
        return DatasetConfig(**kwargs)

    def _skip_if_alpha_unavailable(self, path):
        try:
            image_io.open_static_image(path, mode="RGBA", require_alpha=True)
        except Exception as exc:
            self.skipTest(f"pillow_jxl alpha support is unavailable: {exc}")

    def test_static_jxl_is_discovered_fast_sized_and_loaded_as_rgb_tensor(self):
        _write_rgb_jxl(os.path.join(self.dataset_dir, "sample.jxl"))

        dataset = AiToolkitDataset(
            self._dataset_config(fast_image_size=True),
            batch_size=1,
            sd=FakeSD(),
        )

        self.assertEqual(len(dataset.file_list), 1)
        self.assertEqual(os.path.basename(dataset.file_list[0].path), "sample.jxl")
        self.assertEqual((dataset.file_list[0].width, dataset.file_list[0].height), (96, 80))

        item = dataset._get_single_item(0)
        self.assertEqual(tuple(item.tensor.shape), (3, 32, 32))
        self.assertTrue(torch.isfinite(item.tensor).all())

    def test_fast_metadata_reads_jxl_dimensions(self):
        jxl_path = os.path.join(self.dataset_dir, "metadata.jxl")
        _write_rgb_jxl(jxl_path, size=(123, 77))

        metadata = image_utils.get_image_metadata(jxl_path)

        self.assertEqual((metadata.width, metadata.height), (123, 77))
        self.assertEqual(metadata.type, "JPEGXL")

    def test_alpha_mask_extracts_jxl_alpha_channel(self):
        jxl_path = os.path.join(self.dataset_dir, "alpha.jxl")
        _write_rgba_jxl(jxl_path)
        self._skip_if_alpha_unavailable(jxl_path)

        dataset = AiToolkitDataset(
            self._dataset_config(buckets=True, alpha_mask=True),
            batch_size=1,
            sd=FakeSD(),
        )

        item = dataset._get_single_item(0)

        self.assertIsNotNone(item.mask_tensor)
        self.assertEqual(tuple(item.mask_tensor.shape), (1, 32, 32))
        left_alpha = item.mask_tensor[:, :, :16].mean().item()
        right_alpha = item.mask_tensor[:, :, 16:].mean().item()
        self.assertLess(left_alpha, 0.1)
        self.assertGreater(right_alpha, 0.9)

    def test_rgba_jxl_can_feed_inpaint_path(self):
        Image.new("RGB", (64, 64), (10, 20, 30)).save(os.path.join(self.dataset_dir, "plain.png"))
        inpaint_path = os.path.join(self.inpaint_dir, "plain.jxl")
        _write_rgba_jxl(inpaint_path, size=(64, 64))
        self._skip_if_alpha_unavailable(inpaint_path)

        dataset = AiToolkitDataset(
            self._dataset_config(buckets=True, inpaint_path=self.inpaint_dir),
            batch_size=1,
            sd=FakeSD(),
        )

        item = dataset._get_single_item(0)

        self.assertIsNotNone(item.inpaint_tensor)
        self.assertEqual(tuple(item.inpaint_tensor.shape), (4, 32, 32))

    def test_encrypted_jxl_opens_from_detached_bytes(self):
        encrypted_root = os.path.join(self.root, "encrypted")
        objects_root = os.path.join(encrypted_root, "objects")
        os.makedirs(objects_root)
        key = os.urandom(32)
        aes = AESGCM(key)

        image_path = os.path.join(self.root, "private.jxl")
        _write_rgb_jxl(image_path, size=(11, 7), color=(12, 34, 56))
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        def encrypt(data: bytes, aad: bytes):
            nonce = os.urandom(12)
            return {"nonce": _b64(nonce), "data": _b64(aes.encrypt(nonce, data, aad))}

        object_path = "objects/media.bin"
        with open(os.path.join(encrypted_root, object_path), "w", encoding="utf-8") as f:
            json.dump(encrypt(image_bytes, b"aitk-encrypted-object:objects/media.bin"), f)

        catalog = {
            "version": 1,
            "items": [
                {
                    "id": "jxl1",
                    "name": "private.jxl",
                    "extension": ".jxl",
                    "mimeType": "image/jxl",
                    "mediaKind": "image",
                    "objectPath": object_path,
                    "size": len(image_bytes),
                    "width": 11,
                    "height": 7,
                    "createdAt": "2026-06-16T00:00:00Z",
                    "updatedAt": "2026-06-16T00:00:00Z",
                }
            ],
        }
        manifest = {
            "format": "aitk-encrypted-dataset",
            "version": 1,
            "crypto": {
                "algorithm": "AES-256-GCM",
                "kdf": {"type": "KEYFILE-SHA256", "keyLength": 32},
            },
            "catalog": encrypt(json.dumps(catalog).encode("utf-8"), b"aitk-encrypted-catalog:v1"),
        }
        with open(os.path.join(encrypted_root, ".aitk_encrypted_dataset.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        reader = EncryptedDatasetReader(encrypted_root, key=key)
        loaded = reader.open_image(reader.items[0], mode="RGB")

        self.assertEqual(loaded.size, (11, 7))
        self.assertEqual(loaded.getpixel((0, 0)), (12, 34, 56))


if __name__ == "__main__":
    unittest.main()
