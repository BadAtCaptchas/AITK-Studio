import hashlib
import json
import os
import shutil
from importlib import metadata
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from optimum.quanto.quantize import quantization_map, requantize
from optimum.quanto.tensor import qtypes
from safetensors.torch import load_file, save_file

from toolkit.paths import MODELS_PATH


CACHE_SCHEMA_VERSION = 1
CACHE_WEIGHTS_NAME = "model.safetensors"
CACHE_QMAP_NAME = "quantization_map.json"
CACHE_METADATA_NAME = "metadata.json"


def is_quanto_qtype(qtype_name: Any) -> bool:
    return isinstance(qtype_name, str) and qtype_name in qtypes


def get_package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def get_raw_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    raw_state_dict = getattr(model, "_aitk_orig_state_dict", None)
    if raw_state_dict is None:
        raw_state_dict = getattr(model, "orig_state_dict", None)
    if raw_state_dict is None:
        raw_state_dict = model.state_dict
    return raw_state_dict()


def _normalize(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    return str(value)


def _file_fingerprint(path: str) -> Dict[str, Any]:
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _directory_fingerprint(path: str) -> Dict[str, Any]:
    entries = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if not filename.endswith((".json", ".safetensors")):
                continue
            full_path = os.path.join(root, filename)
            stat = os.stat(full_path)
            entries.append(
                {
                    "path": os.path.relpath(full_path, path).replace("\\", "/"),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    entries.sort(key=lambda item: item["path"])
    return {"path": os.path.abspath(path), "entries": entries}


def source_fingerprint(source: Optional[str]) -> Dict[str, Any]:
    if source is None:
        return {"source": None}
    if os.path.isfile(source):
        return _file_fingerprint(source)
    if os.path.isdir(source):
        return _directory_fingerprint(source)
    return {"source": source}


def quantized_cache_key(
    component: str,
    values: Dict[str, Any],
    sources: Optional[Iterable[Optional[str]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "component": component,
        "values": _normalize(values),
        "sources": [source_fingerprint(source) for source in sources or []],
        "versions": {
            "torch": torch.__version__,
            "torch_cuda": getattr(torch.version, "cuda", None),
            "optimum_quanto": get_package_version("optimum-quanto"),
            "torchao": get_package_version("torchao"),
            "transformers": get_package_version("transformers"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


class QuantizedModelCache:
    def __init__(self, cache_root: Optional[str] = None):
        self.cache_root = cache_root or os.path.join(MODELS_PATH, ".aitk_quantized_cache")

    def get_cache_dir(self, component: str, cache_key: str) -> str:
        return os.path.join(self.cache_root, component, cache_key)

    def has_entry(self, component: str, cache_key: str) -> bool:
        cache_dir = self.get_cache_dir(component, cache_key)
        return os.path.exists(os.path.join(cache_dir, CACHE_WEIGHTS_NAME)) and os.path.exists(
            os.path.join(cache_dir, CACHE_QMAP_NAME)
        )

    def load(
        self,
        model: torch.nn.Module,
        component: str,
        cache_key: str,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        cache_dir = self.get_cache_dir(component, cache_key)
        weights_path = os.path.join(cache_dir, CACHE_WEIGHTS_NAME)
        qmap_path = os.path.join(cache_dir, CACHE_QMAP_NAME)
        metadata_path = os.path.join(cache_dir, CACHE_METADATA_NAME)

        with open(qmap_path, "r", encoding="utf-8") as qmap_file:
            qmap = json.load(qmap_file)
        state_dict = load_file(weights_path, device="cpu")
        requantize(model, state_dict=state_dict, quantization_map=qmap, device=device)
        model.eval()

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                return json.load(metadata_file)
        return {}

    def save(
        self,
        model: torch.nn.Module,
        component: str,
        cache_key: str,
        key_payload: Dict[str, Any],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        qmap = quantization_map(model)
        if not qmap:
            raise ValueError("Model has no optimum.quanto quantization map to cache")

        final_dir = self.get_cache_dir(component, cache_key)
        parent_dir = os.path.dirname(final_dir)
        tmp_dir = f"{final_dir}.tmp-{os.getpid()}"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            save_file(get_raw_state_dict(model), os.path.join(tmp_dir, CACHE_WEIGHTS_NAME))
            with open(os.path.join(tmp_dir, CACHE_QMAP_NAME), "w", encoding="utf-8") as qmap_file:
                json.dump(qmap, qmap_file, indent=2, sort_keys=True)
            metadata_payload = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "key": cache_key,
                "key_payload": key_payload,
            }
            if extra_metadata:
                metadata_payload.update(_normalize(extra_metadata))
            with open(os.path.join(tmp_dir, CACHE_METADATA_NAME), "w", encoding="utf-8") as metadata_file:
                json.dump(metadata_payload, metadata_file, indent=2, sort_keys=True)

            os.makedirs(parent_dir, exist_ok=True)
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            os.replace(tmp_dir, final_dir)
            return final_dir
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

