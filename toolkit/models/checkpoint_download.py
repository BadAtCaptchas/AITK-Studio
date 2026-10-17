"""Atomic, coordinated publication of downloaded model weights."""
import hashlib
import os
import tempfile
from contextlib import contextmanager

from filelock import FileLock
from safetensors import safe_open

from toolkit.hf_offline import is_hf_offline_mode


def repository_filename(repo_id: str, filename: str) -> str:
    return f"{hashlib.sha256(repo_id.encode()).hexdigest()[:20]}_{os.path.basename(filename)}"


def validate_safetensors(filename: str) -> None:
    # Opening validates the header, tensor offsets, dtypes, and complete data length.
    with safe_open(filename, framework="pt") as handle:
        if not list(handle.keys()):
            raise ValueError("Checkpoint contains no tensors")


@contextmanager
def checkpoint_staging(target: str):
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    with FileLock(target + ".lock"):
        with tempfile.TemporaryDirectory(prefix=".aitk-download-", dir=parent) as staging:
            yield staging


def download_checkpoint(repo_id: str, filename: str, target: str, token=None) -> str:
    from huggingface_hub import hf_hub_download

    if not filename or any(part in ("", ".", "..") for part in filename.split("/")) or "\\" in filename or ":" in filename:
        raise ValueError("Invalid checkpoint filename")

    with checkpoint_staging(target) as staging:
        if os.path.isfile(target):
            validate_safetensors(target)
            return target
        source = hf_hub_download(repo_id, filename, token=token, local_dir=staging,
                                 local_files_only=is_hf_offline_mode())
        validate_safetensors(source)
        os.replace(source, target)
    return target
