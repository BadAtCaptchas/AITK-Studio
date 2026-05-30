import base64
import io
import json
import os
import random
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image
from toolkit import image_io

ENCRYPTED_DATASET_MANIFEST = ".aitk_encrypted_dataset.json"
CATALOG_AAD = b"aitk-encrypted-catalog:v1"

_KEY_MAP: Optional[Dict[str, bytes]] = None
_ENV_KEY_PAYLOAD = os.environ.pop("AITK_ENCRYPTED_DATASET_KEYS_B64", "")


class EncryptedDatasetError(RuntimeError):
    pass


@dataclass
class EncryptedDatasetItem:
    id: str
    name: str
    extension: str
    mimeType: str
    mediaKind: str
    objectPath: str
    size: int
    width: Optional[int] = None
    height: Optional[int] = None
    durationMs: Optional[int] = None
    captionObjectPath: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def is_encrypted_dataset_path(dataset_path: str) -> bool:
    return os.path.exists(os.path.join(dataset_path, ENCRYPTED_DATASET_MANIFEST))


def _normalize_key(value: str) -> str:
    return os.path.abspath(value).rstrip("\\/").lower()


def _load_env_key_map() -> Dict[str, bytes]:
    global _KEY_MAP
    if _KEY_MAP is not None:
        return _KEY_MAP

    _KEY_MAP = {}
    payload = _ENV_KEY_PAYLOAD
    if not payload:
        return _KEY_MAP

    try:
        raw = base64.b64decode(payload.encode("ascii")).decode("utf-8")
        for item in json.loads(raw):
            dataset_path = item.get("datasetPath")
            key_b64 = item.get("keyB64")
            if not dataset_path or not key_b64:
                continue
            key = _b64decode(key_b64)
            _KEY_MAP[_normalize_key(dataset_path)] = key
            _KEY_MAP[os.path.basename(dataset_path).lower()] = key
    except Exception as exc:
        raise EncryptedDatasetError("Invalid encrypted dataset key payload") from exc

    return _KEY_MAP


def _object_aad(object_path: str) -> bytes:
    return f"aitk-encrypted-object:{object_path.replace(os.sep, '/')}".encode("utf-8")


def _validate_manifest(manifest: dict) -> None:
    if manifest.get("format") != "aitk-encrypted-dataset" or manifest.get("version") != 1:
        raise EncryptedDatasetError("Unsupported encrypted dataset format")
    crypto = manifest.get("crypto") or {}
    if crypto.get("algorithm") != "AES-256-GCM":
        raise EncryptedDatasetError("Unsupported encrypted dataset algorithm")
    kdf_type = (crypto.get("kdf") or {}).get("type")
    if kdf_type not in ("PBKDF2-SHA256", "KEYFILE-SHA256", "WEBAUTHN-PRF"):
        raise EncryptedDatasetError("Unsupported encrypted dataset KDF")
    catalog = manifest.get("catalog") or {}
    if not catalog.get("nonce") or not catalog.get("data"):
        raise EncryptedDatasetError("Encrypted dataset catalog is missing")


class EncryptedDatasetReader:
    def __init__(self, dataset_path: str, key: Optional[bytes] = None):
        self.dataset_path = os.path.abspath(dataset_path)
        self.manifest_path = os.path.join(self.dataset_path, ENCRYPTED_DATASET_MANIFEST)
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        _validate_manifest(self.manifest)

        self.key = key or self._lookup_key()
        if len(self.key) != 32:
            raise EncryptedDatasetError("Encrypted dataset key must be 32 bytes")
        self.aesgcm = AESGCM(self.key)
        self.catalog = self._decrypt_catalog()
        self.items: List[EncryptedDatasetItem] = [
            EncryptedDatasetItem(**item) for item in self.catalog.get("items", [])
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["aesgcm"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.aesgcm = AESGCM(self.key)

    def _lookup_key(self) -> bytes:
        key_map = _load_env_key_map()
        key = key_map.get(_normalize_key(self.dataset_path)) or key_map.get(os.path.basename(self.dataset_path).lower())
        if not key:
            raise EncryptedDatasetError(f"decryption key required for encrypted dataset: {self.dataset_path}")
        return key

    def _decrypt_payload(self, payload: dict, aad: bytes) -> bytes:
        try:
            return self.aesgcm.decrypt(_b64decode(payload["nonce"]), _b64decode(payload["data"]), aad)
        except Exception as exc:
            raise EncryptedDatasetError("Encrypted dataset authentication failed") from exc

    def _encrypt_payload(self, plaintext: bytes, aad: bytes) -> dict:
        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, plaintext, aad)
        return {"nonce": _b64encode(nonce), "data": _b64encode(ciphertext)}

    def _decrypt_catalog(self) -> dict:
        catalog_bytes = self._decrypt_payload(self.manifest["catalog"], CATALOG_AAD)
        catalog = json.loads(catalog_bytes.decode("utf-8"))
        if catalog.get("version") != 1 or not isinstance(catalog.get("items"), list):
            raise EncryptedDatasetError("Invalid encrypted dataset catalog")
        return catalog

    def _write_manifest(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def _encrypt_catalog(self) -> None:
        self.catalog["items"] = [item.__dict__ for item in self.items]
        self.manifest["catalog"] = self._encrypt_payload(
            json.dumps(self.catalog, separators=(",", ":")).encode("utf-8"),
            CATALOG_AAD,
        )
        self._write_manifest()

    def list_items(self, media_kinds: Optional[List[str]] = None, extensions: Optional[List[str]] = None) -> List[EncryptedDatasetItem]:
        result = self.items
        if media_kinds is not None:
            result = [item for item in result if item.mediaKind in media_kinds]
        if extensions is not None:
            normalized = {ext.lower().lstrip(".") for ext in extensions}
            result = [item for item in result if item.extension.lower().lstrip(".") in normalized]
        return result

    def virtual_path(self, item: EncryptedDatasetItem) -> str:
        return f"aitk-encrypted://{item.id}/{item.name}"

    def item_signature(self, item: EncryptedDatasetItem) -> str:
        return f"{item.size}:{item.updatedAt or item.createdAt or item.id}"

    def _resolve_object_path(self, object_path: str) -> str:
        normalized = object_path.replace("\\", "/")
        if not normalized.startswith("objects/") or ".." in normalized.split("/"):
            raise EncryptedDatasetError("Invalid encrypted object path")
        resolved = os.path.abspath(os.path.join(self.dataset_path, *normalized.split("/")))
        objects_root = os.path.abspath(os.path.join(self.dataset_path, "objects"))
        if not resolved.startswith(objects_root + os.sep):
            raise EncryptedDatasetError("Invalid encrypted object path")
        return resolved

    def decrypt_object_bytes(self, object_path: str) -> bytes:
        with open(self._resolve_object_path(object_path), "r", encoding="utf-8") as f:
            payload = json.load(f)
        return self._decrypt_payload(payload, _object_aad(object_path))

    def encrypt_object_bytes(self, object_path: str, data: bytes) -> None:
        os.makedirs(os.path.join(self.dataset_path, "objects"), exist_ok=True)
        payload = self._encrypt_payload(data, _object_aad(object_path))
        with open(self._resolve_object_path(object_path), "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

    def open_image(
        self,
        item: EncryptedDatasetItem,
        mode: Optional[str] = None,
        require_alpha: bool = False,
    ) -> Image.Image:
        data = self.decrypt_object_bytes(item.objectPath)
        return image_io.open_static_image_from_bytes(
            data,
            source=self.virtual_path(item),
            mode=mode,
            require_alpha=require_alpha,
        )

    def get_caption(self, item: EncryptedDatasetItem) -> Optional[str]:
        if not item.captionObjectPath:
            return None
        return self.decrypt_object_bytes(item.captionObjectPath).decode("utf-8")

    def save_caption(self, item: EncryptedDatasetItem, caption: str) -> None:
        if not item.captionObjectPath:
            item.captionObjectPath = f"objects/{item.id}.caption.bin"
        self.encrypt_object_bytes(item.captionObjectPath, caption.encode("utf-8"))
        item.updatedAt = datetime.now(timezone.utc).isoformat()
        self._encrypt_catalog()

    def load_audio_waveform(self, item: EncryptedDatasetItem) -> Tuple[torch.Tensor, int]:
        import torchaudio

        data = self.decrypt_object_bytes(item.objectPath)
        return torchaudio.load(io.BytesIO(data))

    def load_audio_numpy(self, item: EncryptedDatasetItem, target_sample_rate: int):
        import torchaudio

        waveform, sample_rate = self.load_audio_waveform(item)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        return waveform.squeeze(0).cpu().numpy(), target_sample_rate

    def load_video_frames(
        self,
        item: EncryptedDatasetItem,
        num_frames: int,
        dataset_fps: float,
        shrink_video_to_frames: bool,
        auto_frame_count: bool,
        temporal_compression: int,
    ):
        data = self.decrypt_object_bytes(item.objectPath)
        import av

        container = av.open(io.BytesIO(data))
        video_stream = container.streams.video[0]
        video_fps = float(video_stream.average_rate or video_stream.base_rate or dataset_fps or 24)
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
        container.close()
        total_frames = len(frames)
        if total_frames <= 0:
            raise EncryptedDatasetError(f"Encrypted video has no decodable frames: {item.name}")
        max_frame_index = total_frames - 1

        selected_num_frames = num_frames
        if auto_frame_count:
            duration_seconds = total_frames / video_fps
            selected_num_frames = int(duration_seconds * dataset_fps)
            selected_num_frames = selected_num_frames // temporal_compression * temporal_compression
            selected_num_frames += 1
            if selected_num_frames <= 1:
                raise EncryptedDatasetError("Computed encrypted video frame count is invalid")

        if shrink_video_to_frames or total_frames < selected_num_frames:
            interval = max_frame_index / (selected_num_frames - 1) if selected_num_frames > 1 else 0
            frame_indices = [min(int(round(i * interval)), max_frame_index) for i in range(selected_num_frames)]
        else:
            fps_ratio = video_fps / dataset_fps
            frame_interval = max(1, int(round(fps_ratio)))
            max_consecutive_frames = total_frames // frame_interval
            if max_consecutive_frames < selected_num_frames:
                interval = max_frame_index / (selected_num_frames - 1) if selected_num_frames > 1 else 0
                frame_indices = [min(int(round(i * interval)), max_frame_index) for i in range(selected_num_frames)]
            else:
                max_start_frame = max_frame_index - ((selected_num_frames - 1) * frame_interval)
                start_frame = random.randint(0, max(0, max_start_frame))
                frame_indices = [start_frame + (i * frame_interval) for i in range(selected_num_frames)]

        return [frames[idx] for idx in frame_indices], video_fps, frame_indices, selected_num_frames
