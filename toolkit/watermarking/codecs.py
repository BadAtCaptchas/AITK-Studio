import os
from typing import Dict, List


CODECS_DIR = os.path.join(os.path.dirname(__file__), "codecs")

BUILTIN_AUTHENLORA_CODECS: Dict[str, dict] = {
    "builtin:authenlora_48bits": {
        "filename": "authenlora_codec_48bits.pth",
        "msg_bits": 48,
        "label": "AuthenLoRA 48-bit official codec",
    },
    "builtin:authenlora_80bits": {
        "filename": "authenlora_codec_80bits.pth",
        "msg_bits": 80,
        "label": "AuthenLoRA 80-bit official codec",
    },
    "builtin:authenlora_100bits": {
        "filename": "authenlora_codec_100bits.pth",
        "msg_bits": 100,
        "label": "AuthenLoRA 100-bit official codec",
    },
}


def get_builtin_codec_options() -> List[dict]:
    return [
        {
            "id": codec_id,
            "path": resolve_builtin_codec_path(codec_id),
            "msg_bits": spec["msg_bits"],
            "label": spec["label"],
        }
        for codec_id, spec in BUILTIN_AUTHENLORA_CODECS.items()
    ]


def is_builtin_codec_reference(codec_path: str) -> bool:
    return str(codec_path).strip() in BUILTIN_AUTHENLORA_CODECS


def resolve_builtin_codec_path(codec_id: str) -> str:
    spec = BUILTIN_AUTHENLORA_CODECS.get(str(codec_id).strip())
    if spec is None:
        known = ", ".join(BUILTIN_AUTHENLORA_CODECS.keys())
        raise ValueError(f"Unknown built-in AuthenLoRA codec {codec_id!r}. Available codecs: {known}")
    return os.path.join(CODECS_DIR, spec["filename"])


def resolve_codec_path(codec_path: str) -> str:
    value = str(codec_path).strip()
    if is_builtin_codec_reference(value):
        return resolve_builtin_codec_path(value)
    return value


def get_builtin_codec_msg_bits(codec_path: str):
    spec = BUILTIN_AUTHENLORA_CODECS.get(str(codec_path).strip())
    if spec is None:
        return None
    return spec["msg_bits"]
