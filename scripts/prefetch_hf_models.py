import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolkit.network_policy import install_offline_network_guard
from toolkit.paths import MODELS_PATH, TOOLKIT_ROOT

install_offline_network_guard()

from huggingface_hub import hf_hub_download, snapshot_download


MODEL_FILE_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")
LOCAL_PATH_PREFIXES = ("./", "../", "~/", "/")
LOCAL_TOP_LEVEL_NAMES = {
    "config",
    "datasets",
    "extensions",
    "extensions_built_in",
    "jobs",
    "models",
    "output",
    "toolkit",
}
URI_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def normalize_reference(value):
    if not isinstance(value, str):
        return ""
    return value.strip()


def normalized_path(value):
    return value.replace("\\", "/")


def get_token():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip() or None

    token_path = os.getenv("HF_TOKEN_PATH")
    if token_path:
        try:
            return Path(token_path).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    return None


def resolve_config_path(value):
    if os.path.isabs(value) or WINDOWS_DRIVE_RE.match(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(TOOLKIT_ROOT, value))


def is_existing_local_path(value):
    return os.path.exists(resolve_config_path(value))


def is_local_like_reference(value):
    if URI_RE.match(value):
        return False
    if WINDOWS_DRIVE_RE.match(value):
        return True
    normalized = normalized_path(value)
    if normalized.startswith(LOCAL_PATH_PREFIXES):
        return True
    parts = normalized.split("/")
    return bool(parts and parts[0] in LOCAL_TOP_LEVEL_NAMES)


def classify_hf_reference(value):
    value = normalize_reference(value)
    if not value or is_existing_local_path(value) or is_local_like_reference(value):
        return None
    if URI_RE.match(value):
        return None

    normalized = normalized_path(value)
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 2:
        return None
    if any(part in (".", "..", "~") for part in parts[:2]):
        return None

    repo_id = "/".join(parts[:2])
    if len(parts) >= 3 and parts[-1].lower().endswith(MODEL_FILE_EXTENSIONS):
        return {
            "kind": "file",
            "repo_id": repo_id,
            "filename": "/".join(parts[2:]),
        }
    if len(parts) == 2 and not Path(parts[-1]).suffix:
        return {
            "kind": "snapshot",
            "repo_id": repo_id,
        }
    return None


def prefetch_reference(value, token):
    classification = classify_hf_reference(value)
    if classification is None:
        return None

    if classification["kind"] == "file":
        filename = classification["filename"]
        target_dir = os.path.join(MODELS_PATH, "diffusion_models")
        target_path = os.path.join(target_dir, *filename.split("/"))
        if os.path.isfile(target_path):
            return {
                "kind": "file",
                "value": value,
                "path": target_path,
                "cached": True,
            }

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        print(
            f"Downloading model file {value} to {target_dir}",
            file=sys.stderr,
            flush=True,
        )
        local_path = hf_hub_download(
            repo_id=classification["repo_id"],
            filename=filename,
            local_dir=target_dir,
            token=token,
        )
        return {
            "kind": "file",
            "value": value,
            "path": local_path,
            "cached": False,
        }

    print(
        f"Downloading model repository {classification['repo_id']} to Hugging Face cache",
        file=sys.stderr,
        flush=True,
    )
    local_path = snapshot_download(
        repo_id=classification["repo_id"],
        token=token,
    )
    return {
        "kind": "snapshot",
        "value": value,
        "path": local_path,
        "cached": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    references = payload.get("references", [])
    values = []
    seen = set()
    for reference in references:
        value = normalize_reference(reference.get("value") if isinstance(reference, dict) else reference)
        if value and value not in seen:
            seen.add(value)
            values.append(value)

    token = get_token()
    result = {
        "handledValues": [],
        "downloads": [],
        "warnings": [],
    }

    for value in values:
        download = prefetch_reference(value, token)
        if download is None:
            continue
        result["handledValues"].append(value)
        result["downloads"].append(download)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
