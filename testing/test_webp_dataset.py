import base64
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
from PIL import Image, ImageDraw, features

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset
from toolkit.encrypted_dataset import EncryptedDatasetReader
from toolkit import image_utils


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
        self.torch_dtype = torch.float32
        self.unet = SimpleNamespace()
        self.use_raw_control_images = False
        self.vae = SimpleNamespace()

    def get_bucket_divisibility(self):
        return 32


def _write_rgb_webp(path, size=(96, 80), color=(32, 64, 128)):
    Image.new("RGB", size, color).save(path, format="WEBP", lossless=True)


def _write_rgba_webp(path, size=(64, 64)):
    image = Image.new("RGBA", size, (20, 40, 60, 255))
    alpha = Image.new("L", size, 255)
    draw = ImageDraw.Draw(alpha)
    draw.rectangle((0, 0, size[0] // 2 - 1, size[1] - 1), fill=0)
    image.putalpha(alpha)
    image.save(path, format="WEBP", lossless=True)


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


@unittest.skipUnless(features.check("webp"), "Pillow WebP support is required")
class WebPDatasetTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="aitk-webp-")
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

    def test_static_webp_is_discovered_fast_sized_and_loaded_as_rgb_tensor(self):
        _write_rgb_webp(os.path.join(self.dataset_dir, "sample.webp"))

        dataset = AiToolkitDataset(
            self._dataset_config(fast_image_size=True),
            batch_size=1,
            sd=FakeSD(),
        )

        self.assertEqual(len(dataset.file_list), 1)
        self.assertEqual(os.path.basename(dataset.file_list[0].path), "sample.webp")
        self.assertEqual((dataset.file_list[0].width, dataset.file_list[0].height), (96, 80))

        item = dataset._get_single_item(0)
        self.assertEqual(tuple(item.tensor.shape), (3, 32, 32))
        self.assertTrue(torch.isfinite(item.tensor).all())

    def test_fast_metadata_reads_webp_dimensions(self):
        webp_path = os.path.join(self.dataset_dir, "metadata.webp")
        _write_rgb_webp(webp_path, size=(123, 77))

        self.assertEqual(image_utils.get_image_size(webp_path), (123, 77))

    def test_alpha_mask_extracts_webp_alpha_channel(self):
        _write_rgba_webp(os.path.join(self.dataset_dir, "alpha.webp"))
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

    def test_inpaint_webp_without_alpha_raises_clear_error(self):
        Image.new("RGB", (64, 64), (10, 20, 30)).save(os.path.join(self.dataset_dir, "plain.png"))
        _write_rgb_webp(os.path.join(self.inpaint_dir, "plain.webp"), size=(64, 64))
        dataset = AiToolkitDataset(
            self._dataset_config(buckets=True, inpaint_path=self.inpaint_dir),
            batch_size=1,
            sd=FakeSD(),
        )

        with self.assertRaisesRegex(Exception, "alpha channel"):
            dataset._get_single_item(0)

    def test_encrypted_webp_opens_from_detached_bytes(self):
        encrypted_root = os.path.join(self.root, "encrypted")
        objects_root = os.path.join(encrypted_root, "objects")
        os.makedirs(objects_root)
        key = os.urandom(32)
        aes = AESGCM(key)

        image = Image.new("RGB", (11, 7), (12, 34, 56))
        image_io = io.BytesIO()
        image.save(image_io, format="WEBP", lossless=True)

        def encrypt(data: bytes, aad: bytes):
            nonce = os.urandom(12)
            return {"nonce": _b64(nonce), "data": _b64(aes.encrypt(nonce, data, aad))}

        object_path = "objects/media.bin"
        with open(os.path.join(encrypted_root, object_path), "w", encoding="utf-8") as f:
            json.dump(encrypt(image_io.getvalue(), b"aitk-encrypted-object:objects/media.bin"), f)

        catalog = {
            "version": 1,
            "items": [
                {
                    "id": "webp1",
                    "name": "private.webp",
                    "extension": ".webp",
                    "mimeType": "image/webp",
                    "mediaKind": "image",
                    "objectPath": object_path,
                    "size": len(image_io.getvalue()),
                    "width": 11,
                    "height": 7,
                    "createdAt": "2026-05-21T00:00:00Z",
                    "updatedAt": "2026-05-21T00:00:00Z",
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
        shutil.rmtree(encrypted_root)

        self.assertEqual(loaded.size, (11, 7))
        self.assertEqual(loaded.getpixel((0, 0)), (12, 34, 56))


if __name__ == "__main__":
    unittest.main()
