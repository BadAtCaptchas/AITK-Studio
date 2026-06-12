import base64
import json
import os
from typing import Any, Dict

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


VERSION = 1
NONCE_BYTES = 12
SALT_BYTES = 32


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def _aad(direction: str, job_id: str, item_id: str) -> bytes:
    return f"aitk-secure-caption:v{VERSION}:{direction}:{job_id}:{item_id}".encode("utf-8")


def _derive_key(token: str, salt: bytes, direction: str) -> bytes:
    if not token:
        raise ValueError("Secure caption token is required")
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=f"aitk-secure-caption:{direction}:v{VERSION}".encode("utf-8"),
    )
    return hkdf.derive(token.encode("utf-8"))


def encrypt_secure_caption_json(
    token: str,
    direction: str,
    job_id: str,
    item_id: str,
    value: Dict[str, Any],
) -> Dict[str, Any]:
    salt = os.urandom(SALT_BYTES)
    nonce = os.urandom(NONCE_BYTES)
    key = _derive_key(token, salt, direction)
    aesgcm = AESGCM(key)
    plaintext = json.dumps(value, separators=(",", ":")).encode("utf-8")
    ciphertext = aesgcm.encrypt(nonce, plaintext, _aad(direction, job_id, item_id))
    return {
        "version": VERSION,
        "jobId": job_id,
        "itemId": item_id,
        "salt": _b64encode(salt),
        "nonce": _b64encode(nonce),
        "data": _b64encode(ciphertext),
    }


def decrypt_secure_caption_json(
    token: str,
    direction: str,
    envelope: Dict[str, Any],
) -> Dict[str, Any]:
    if envelope.get("version") != VERSION:
        raise ValueError("Unsupported secure caption envelope")
    job_id = envelope.get("jobId")
    item_id = envelope.get("itemId")
    if not job_id or not item_id:
        raise ValueError("Secure caption envelope is missing context")
    salt = _b64decode(envelope["salt"])
    nonce = _b64decode(envelope["nonce"])
    ciphertext = _b64decode(envelope["data"])
    key = _derive_key(token, salt, direction)
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, _aad(direction, job_id, item_id))
    decoded = plaintext.decode("utf-8", errors="replace")
    try:
        return json.loads(decoded)
    except json.JSONDecodeError as exc:
        preview = decoded[:2000]
        raise ValueError(f"Secure caption payload is not valid JSON: {preview}") from exc
