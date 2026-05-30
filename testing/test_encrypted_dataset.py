import base64
import io
import json
import os
import tempfile
import unittest

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image

from toolkit.encrypted_dataset import EncryptedDatasetError, EncryptedDatasetReader


def b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


class EncryptedDatasetReaderTest(unittest.TestCase):
    def make_dataset(self, kdf=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = tmp.name
        os.makedirs(os.path.join(root, "objects"))
        key = os.urandom(32)
        aes = AESGCM(key)

        image = Image.new("RGB", (4, 3), (12, 34, 56))
        image_io = io.BytesIO()
        image.save(image_io, format="PNG")

        def encrypt(data: bytes, aad: bytes):
            nonce = os.urandom(12)
            return {"nonce": b64(nonce), "data": b64(aes.encrypt(nonce, data, aad))}

        media_payload = encrypt(image_io.getvalue(), b"aitk-encrypted-object:objects/media.bin")
        with open(os.path.join(root, "objects", "media.bin"), "w", encoding="utf-8") as f:
            json.dump(media_payload, f)

        catalog = {
            "version": 1,
            "items": [
                {
                    "id": "item1",
                    "name": "private_name.png",
                    "extension": ".png",
                    "mimeType": "image/png",
                    "mediaKind": "image",
                    "objectPath": "objects/media.bin",
                    "size": len(image_io.getvalue()),
                    "width": 4,
                    "height": 3,
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
                "kdf": kdf or {"type": "KEYFILE-SHA256", "keyLength": 32},
            },
            "catalog": encrypt(json.dumps(catalog).encode("utf-8"), b"aitk-encrypted-catalog:v1"),
        }
        with open(os.path.join(root, ".aitk_encrypted_dataset.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        return root, key

    def test_decrypts_catalog_and_media_in_memory(self):
        root, key = self.make_dataset()

        reader = EncryptedDatasetReader(root, key=key)
        self.assertEqual(len(reader.items), 1)
        item = reader.items[0]
        image = reader.open_image(item)

        self.assertEqual(image.size, (4, 3))
        self.assertFalse(os.path.exists(os.path.join(root, "private_name.png")))
        with open(os.path.join(root, ".aitk_encrypted_dataset.json"), "r", encoding="utf-8") as f:
            self.assertNotIn("private_name.png", f.read())

    def test_wrong_key_fails_authentication(self):
        root, _key = self.make_dataset()

        with self.assertRaises(EncryptedDatasetError):
            EncryptedDatasetReader(root, key=os.urandom(32))

    def test_manifest_tamper_fails_authentication(self):
        root, key = self.make_dataset()
        manifest_path = os.path.join(root, ".aitk_encrypted_dataset.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["catalog"]["data"] = manifest["catalog"]["data"][:-2] + "AA"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        with self.assertRaises(EncryptedDatasetError):
            EncryptedDatasetReader(root, key=key)

    def test_caption_write_stays_encrypted(self):
        root, key = self.make_dataset()
        reader = EncryptedDatasetReader(root, key=key)
        item = reader.items[0]

        reader.save_caption(item, "secret caption")

        caption_path = os.path.join(root, "objects", "item1.caption.bin")
        with open(caption_path, "r", encoding="utf-8") as f:
            stored = f.read()
        self.assertNotIn("secret caption", stored)
        refreshed = EncryptedDatasetReader(root, key=key)
        self.assertEqual(refreshed.get_caption(refreshed.items[0]), "secret caption")

    def test_webauthn_prf_manifest_reads_with_supplied_raw_key(self):
        root, key = self.make_dataset(
            {
                "type": "WEBAUTHN-PRF",
                "keyLength": 32,
                "rpId": "localhost",
                "credentials": [
                    {
                        "id": "mockCredentialId",
                        "label": "Mock YubiKey",
                        "transports": ["usb"],
                        "saltB64": b64(os.urandom(32)),
                        "createdAt": "2026-05-30T00:00:00.000Z",
                        "wrappedKey": {
                            "algorithm": "AES-256-GCM",
                            "nonce": b64(os.urandom(12)),
                            "data": b64(os.urandom(48)),
                        },
                    }
                ],
                "nativeUsb": {
                    "provider": "ctap2-hmac-secret",
                    "status": "planned",
                },
            }
        )

        reader = EncryptedDatasetReader(root, key=key)

        self.assertEqual(len(reader.items), 1)
        self.assertEqual(reader.open_image(reader.items[0]).size, (4, 3))


if __name__ == "__main__":
    unittest.main()
