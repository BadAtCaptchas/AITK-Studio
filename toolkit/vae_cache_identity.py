"""Offline cache identity for explicitly selected VAE weights."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from huggingface_hub.errors import HFValidationError


def _local_source(source: str):
    path = Path(source).expanduser()
    if path.exists():
        # A pipeline repository can be selected as the VAE source.
        return path / "vae" if (path / "vae").is_dir() else path
    parts = source.split('/')
    if len(parts) > 2 and source.endswith(('.safetensors', '.bin', '.pt', '.pth')):
        try:
            cached = try_to_load_from_cache('/'.join(parts[:2]), '/'.join(parts[2:]))
        except HFValidationError:
            return None
        return Path(cached) if isinstance(cached, str) else None
    # Inspect the existing Hub cache only; never resolve this by downloading.
    for filename in ("config.json", "vae/config.json"):
        try:
            cached = try_to_load_from_cache(source, filename)
        except HFValidationError:
            return None
        if isinstance(cached, str):
            return Path(cached).parent
    return None


def with_vae_cache_identity(model, version: str) -> str:
    """Namespace custom VAEs, while respecting an explicit user cache namespace.

    Hash once per loaded VAE. Content hashing also detects local replacements
    which preserve size and timestamps, and works for sharded Diffusers VAEs.
    """
    config = model.model_config
    source = getattr(config, "vae_path", None)
    if not source or getattr(config, "latent_space_version", None) is not None:
        return version
    vae = getattr(model, "vae", None)
    precision = str(getattr(model, "vae_torch_dtype", getattr(config, "vae_dtype", None)))
    resolved_source = getattr(vae, '_aitk_vae_source', None)
    key = (str(source), precision, resolved_source)
    cached = getattr(vae, "_aitk_latent_cache_identity", None)
    if cached is not None and cached[0] == key:
        return f"{version}_vae_{cached[1]}"

    digest = hashlib.sha256()
    vae_config = dict(getattr(vae, "config", {}) or {})
    digest.update(json.dumps({"source": str(source), "dtype": precision,
                              "config": vae_config}, sort_keys=True, default=str).encode())
    local = _local_source(str(resolved_source or source))
    if local is not None:
        if local.is_file():
            files = [local]
        else:
            files = sorted(path for path in local.rglob("*") if path.is_file()
                           and path.suffix in (".json", ".safetensors", ".bin", ".pt", ".pth"))
        for path in files:
            digest.update((path.name if local.is_file() else path.relative_to(local).as_posix()).encode())
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    identity = digest.hexdigest()[:32]
    if vae is not None:
        vae._aitk_latent_cache_identity = (key, identity)
    return f"{version}_vae_{identity}"
