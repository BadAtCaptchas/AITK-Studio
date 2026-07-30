import hashlib
import json
import os
import shutil
import time
from importlib import metadata
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from optimum.quanto.quantize import quantization_map, requantize
from optimum.quanto.tensor import qtypes
from safetensors.torch import load_file, save_file

from toolkit.paths import MODELS_PATH
from toolkit.util.ostris_quant import (
    OstrisLinear,
    get_ostris_backend_metadata,
    get_ostris_quantizer,
    is_ostris_qtype,
    prepare_linear_for_ostris_cache,
)


CACHE_SCHEMA_VERSION = 1
ORBIT_MANIFEST_SCHEMA_VERSION = 1
CACHE_WEIGHTS_NAME = "model.safetensors"
CACHE_QMAP_NAME = "quantization_map.json"
CACHE_MANIFEST_NAME = "manifest.json"
CACHE_METADATA_NAME = "metadata.json"


def is_quanto_qtype(qtype_name: Any) -> bool:
    return isinstance(qtype_name, str) and qtype_name in qtypes


def is_quantized_cache_qtype(qtype_name: Any) -> bool:
    """Return whether a qtype has a lossless packed cache representation."""
    return is_quanto_qtype(qtype_name) or is_ostris_qtype(qtype_name)


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
            "triton": get_package_version("triton"),
            "triton_windows": get_package_version("triton-windows"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype)


def _parse_dtype(name: str) -> torch.dtype:
    known = {
        str(torch.float64): torch.float64,
        str(torch.float32): torch.float32,
        str(torch.float16): torch.float16,
        str(torch.bfloat16): torch.bfloat16,
    }
    if name not in known:
        raise ValueError(f"unsupported cached original dtype {name!r}")
    return known[name]


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _module_backend_name(module: OstrisLinear) -> str:
    name = getattr(module, "ostris_backend_name", None)
    if isinstance(name, str) and is_ostris_qtype(name):
        return name
    quantizer = module.ostris_quantizer
    name = getattr(quantizer, "backend_name", None)
    if isinstance(name, str) and is_ostris_qtype(name):
        return name
    if hasattr(module, "cr_qdata"):
        return "convrot4"
    if hasattr(module, "cr8_qdata"):
        return "convrot8"
    if hasattr(module, "crn_qdata"):
        layout = getattr(module, "convrot_packed_layout", "")
        if layout == "ternary_base3_v1":
            return "convrotbitnet"
        if layout == "comfy_w4a4_int4_v1":
            return "convrotcomfyw4a4"
        return f"convrotint{int(getattr(module, 'crn_bits', 0))}"
    if hasattr(module, "uintx_packed"):
        return f"uint{int(getattr(module, 'uintx_bits', 0))}"
    bits = int(getattr(quantizer, "bits", 0))
    inferred = f"orbitvq{bits}" if hasattr(module, "ovq_packed") else f"orbit{bits}"
    if not is_ostris_qtype(inferred):
        raise ValueError(f"cannot identify packed backend for {type(quantizer).__name__}")
    return inferred


def _expected_packed_layout(qtype_name: str) -> str:
    if qtype_name == "convrot4":
        return "nvfp4_e2m1_e4m3_v1"
    if qtype_name == "convrot8":
        return "int8_per_row_v1"
    if qtype_name == "convrotbitnet":
        return "ternary_base3_v1"
    if qtype_name == "convrotcomfyw4a4":
        return "comfy_w4a4_int4_v1"
    if qtype_name.startswith("convrotint"):
        return f"int{int(qtype_name.removeprefix('convrotint'))}_bitpacked_v1"
    if qtype_name.startswith("orbitvq"):
        return "vq_bitstream_msb_v1"
    if qtype_name.startswith("uint"):
        return "power2_shards_v1"
    if qtype_name == "orbit4":
        return "nibbles_v1"
    if qtype_name in ("orbit2", "orbit3"):
        return "bitstream_msb_v1"
    raise ValueError(f"no packed layout is defined for backend {qtype_name!r}")


def _expected_buffer_names(qtype_name: str) -> set[str]:
    if qtype_name == "convrot4":
        return {"cr_qdata", "cr_scales", "cr_scales_blocked", "cr_pts"}
    if qtype_name == "convrot8":
        return {"cr8_qdata", "cr8_scales"}
    if qtype_name in {"convrotbitnet"} or (
        qtype_name.startswith("convrotint")
        and int(qtype_name.removeprefix("convrotint")) <= 7
    ):
        return {"crn_qdata", "crn_scales", "crn_gratio"}
    if qtype_name == "convrotcomfyw4a4" or qtype_name == "convrotint8":
        return {"crn_qdata", "crn_scales"}
    if qtype_name.startswith("orbitvq"):
        return {
            "ovq_packed",
            "ovq_scales",
            "ovq_perm",
            "ovq_inv_perm",
            "ovq_signs",
        }
    if qtype_name.startswith("uint"):
        return {
            "uintx_packed",
            "uintx_scale",
            "uintx_zero_point",
        }
    if qtype_name in ("orbit2", "orbit3", "orbit4"):
        return {
            "orbit_packed",
            "orbit_row_norms",
            "orbit_codebook",
            "orbit_perm",
            "orbit_inv_perm",
            "orbit_signs",
        }
    raise ValueError(f"no packed buffers are defined for backend {qtype_name!r}")


def _module_manifest(name: str, module: OstrisLinear) -> Dict[str, Any]:
    qtype_name = _module_backend_name(module)
    backend = get_ostris_backend_metadata(qtype_name)
    if backend is None:
        raise ValueError(f"missing registry metadata for {qtype_name}")
    buffer_names = sorted(
        key for key, value in module._buffers.items() if isinstance(value, torch.Tensor)
    )
    buffer_shapes = {
        key: {
            "shape": list(module._buffers[key].shape),
            "dtype": str(module._buffers[key].dtype),
            "digest_algorithm": "sha256",
            "digest": _tensor_digest(module._buffers[key]),
        }
        for key in buffer_names
    }
    options = getattr(module.ostris_quantizer, "backend_options", None)
    if options is not None:
        options_payload = options.to_dict()
    else:
        options_payload = {
            "kernel": getattr(
                module, "convrot_kernel", getattr(module, "orbit_kernel", "auto")
            ),
            "max_workspace_mb": int(
                getattr(
                    module,
                    "convrot_max_workspace_mb",
                    getattr(module, "orbit_max_workspace_mb", 64),
                )
            ),
        }
    attributes: Dict[str, Any] = {}
    for attribute in (
        "orbit_bits",
        "orbit_block",
        "orbit_kernel",
        "orbit_max_workspace_mb",
        "orbit_packed_layout",
        "ovq_block",
        "ovq_group",
        "cr_rot_size",
        "cr8_rot_size",
        "crn_bits",
        "crn_rot_size",
        "convrot_kernel",
        "convrot_max_workspace_mb",
        "convrot_packed_layout",
        "uintx_bits",
        "uintx_group_size",
        "uintx_max_workspace_mb",
        "uintx_packed_layout",
    ):
        if hasattr(module, attribute):
            attributes[attribute] = _normalize(getattr(module, attribute))
    codebook_identity = None
    if isinstance(module._buffers.get("orbit_codebook"), torch.Tensor):
        codebook_identity = {
            "algorithm": "sha256",
            "digest": _tensor_digest(module.orbit_codebook),
        }
    return {
        "path": name,
        "qtype": qtype_name,
        "backend_format_version": backend.format_version,
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "original_dtype": _dtype_name(module.ostris_orig_dtype),
        "has_bias": module.bias is not None,
        "buffers": buffer_shapes,
        "attributes": attributes,
        "options": options_payload,
        "packed_layout": attributes.get(
            "convrot_packed_layout",
            attributes.get(
                "orbit_packed_layout",
                attributes.get(
                    "uintx_packed_layout",
                    _expected_packed_layout(qtype_name),
                ),
            ),
        ),
        "rotation": {
            "scheme": (
                "regular_hadamard_v1"
                if qtype_name.startswith("convrot")
                else "none"
                if qtype_name.startswith("uint")
                else "deterministic_rpbh_v1"
            ),
            "dimension": int(module.in_features),
            "block": int(
                attributes.get(
                    "cr_rot_size",
                    attributes.get(
                        "cr8_rot_size",
                        attributes.get(
                            "crn_rot_size",
                            attributes.get(
                                "orbit_block",
                                attributes.get(
                                    "ovq_block",
                                    attributes.get("uintx_group_size", 0),
                                ),
                            ),
                        ),
                    ),
                )
            ),
        },
        "codebook_identity": codebook_identity,
    }


def _named_modules_with_duplicates(model: torch.nn.Module):
    try:
        return model.named_modules(remove_duplicate=False)
    except TypeError:
        return model.named_modules()


def _cpu_tensor_for_save(
    tensor: torch.Tensor,
    seen_storages: set[int],
) -> torch.Tensor:
    if tensor.is_meta:
        raise ValueError("cannot cache a meta tensor")
    value = tensor.detach().to("cpu").contiguous()
    try:
        storage_id = value.untyped_storage().data_ptr()
    except Exception:
        storage_id = id(value)
    if storage_id in seen_storages:
        value = value.clone()
    else:
        seen_storages.add(storage_id)
    return value


def _collect_ostris_state_dict(
    model: torch.nn.Module,
    packed_modules: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """Collect state without invoking OstrisLinear's dense save compatibility path."""
    destination: Dict[str, torch.Tensor] = {}
    packed_paths = {entry["path"] for entry in packed_modules}
    for module_name, module in _named_modules_with_duplicates(model):
        prefix = f"{module_name}." if module_name else ""
        if module_name in packed_paths and isinstance(module, OstrisLinear):
            for parameter_name, parameter in module._parameters.items():
                if isinstance(parameter, torch.Tensor):
                    destination[prefix + parameter_name] = parameter
            for buffer_name, buffer in module._buffers.items():
                if isinstance(buffer, torch.Tensor):
                    destination[prefix + buffer_name] = buffer
            continue
        module._save_to_state_dict(destination, prefix, keep_vars=False)

    seen_storages: set[int] = set()
    return {
        name: _cpu_tensor_for_save(value, seen_storages)
        for name, value in destination.items()
    }


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)


def _replace_with_retry(source: str, destination: str) -> None:
    """Use atomic rename, tolerating brief Windows file-scanner locks."""
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.02 * (attempt + 1))


class QuantizedModelCache:
    def __init__(self, cache_root: Optional[str] = None):
        self.cache_root = cache_root or os.path.join(MODELS_PATH, ".aitk_quantized_cache")

    def get_cache_dir(self, component: str, cache_key: str) -> str:
        return os.path.join(self.cache_root, component, cache_key)

    def get_metadata_path(self, component: str, cache_key: str) -> str:
        return os.path.join(self.get_cache_dir(component, cache_key), CACHE_METADATA_NAME)

    def has_entry(self, component: str, cache_key: str) -> bool:
        cache_dir = self.get_cache_dir(component, cache_key)
        weights_path = os.path.join(cache_dir, CACHE_WEIGHTS_NAME)
        if not os.path.isfile(weights_path):
            return False
        manifest_path = os.path.join(cache_dir, CACHE_MANIFEST_NAME)
        qmap_path = os.path.join(cache_dir, CACHE_QMAP_NAME)
        try:
            if os.path.isfile(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as source:
                    manifest = json.load(source)
                return (
                    manifest.get("schema_version") == ORBIT_MANIFEST_SCHEMA_VERSION
                    and manifest.get("backend") == "ostris"
                    and manifest.get("key") == cache_key
                    and bool(manifest.get("modules"))
                )
            if os.path.isfile(qmap_path):
                with open(qmap_path, "r", encoding="utf-8") as source:
                    return bool(json.load(source))
        except (OSError, ValueError, TypeError):
            return False
        return False

    def invalidate(self, component: str, cache_key: str) -> None:
        shutil.rmtree(self.get_cache_dir(component, cache_key), ignore_errors=True)

    def load_metadata(self, component: str, cache_key: str) -> Dict[str, Any]:
        metadata_path = self.get_metadata_path(component, cache_key)
        if not os.path.exists(metadata_path):
            return {}
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    def update_metadata(
        self,
        component: str,
        cache_key: str,
        extra_metadata: Dict[str, Any],
    ) -> None:
        metadata_path = self.get_metadata_path(component, cache_key)
        if not os.path.exists(metadata_path):
            return
        metadata_payload = self.load_metadata(component, cache_key)
        metadata_payload.update(_normalize(extra_metadata))
        tmp_path = f"{metadata_path}.tmp-{os.getpid()}"
        _write_json(tmp_path, metadata_payload)
        os.replace(tmp_path, metadata_path)

    def _load_ostris(
        self,
        model: torch.nn.Module,
        cache_dir: str,
        manifest: Dict[str, Any],
        device: Optional[torch.device],
    ) -> None:
        state_dict = load_file(os.path.join(cache_dir, CACHE_WEIGHTS_NAME), device="cpu")
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        packed_keys: set[str] = set()
        validated_entries = []
        seen_paths: set[str] = set()

        # Validate the complete cache before changing even one module. This
        # keeps a stale format or same-shape tensor corruption from leaving the
        # caller with a partially converted model.
        for entry in manifest["modules"]:
            path = entry["path"]
            if path in seen_paths:
                raise ValueError(f"packed cache contains duplicate module path {path!r}")
            seen_paths.add(path)
            qtype_name = entry["qtype"]
            backend = get_ostris_backend_metadata(qtype_name)
            if backend is None:
                raise ValueError(f"unknown cached qtype {qtype_name!r}")
            if entry.get("backend_format_version") != backend.format_version:
                raise ValueError(
                    f"cached backend format for {path!r} is "
                    f"{entry.get('backend_format_version')!r}, expected "
                    f"{backend.format_version}"
                )
            expected_layout = _expected_packed_layout(qtype_name)
            if entry.get("packed_layout") != expected_layout:
                raise ValueError(
                    f"cached packed layout for {path!r} is "
                    f"{entry.get('packed_layout')!r}, expected {expected_layout!r}"
                )
            buffers = entry.get("buffers")
            if not isinstance(buffers, dict):
                raise ValueError(f"packed cache buffers are missing for {path!r}")
            if set(buffers) != _expected_buffer_names(qtype_name):
                raise ValueError(f"packed cache buffer set is incompatible for {path!r}")

            module = model if not path else model.get_submodule(path)
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"cached module {path!r} is not an nn.Linear")
            if (
                int(module.in_features) != int(entry["in_features"])
                or int(module.out_features) != int(entry["out_features"])
            ):
                raise ValueError(f"cached shape does not match module {path!r}")
            if bool(module.bias is not None) != bool(entry.get("has_bias")):
                raise ValueError(f"cached bias configuration does not match module {path!r}")
            options = entry.get("options") or {}
            quantizer = get_ostris_quantizer(
                qtype_name,
                kernel=options.get("kernel", "auto"),
                max_workspace_mb=int(options.get("max_workspace_mb", 64)),
            )
            if quantizer is None:
                raise ValueError(f"unknown cached qtype {qtype_name!r}")
            original_dtype = _parse_dtype(entry["original_dtype"])
            prefix = f"{path}." if path else ""
            for buffer_name, expected in buffers.items():
                key = prefix + buffer_name
                if key not in state_dict:
                    raise ValueError(f"packed cache is missing tensor {key!r}")
                value = state_dict[key]
                if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
                    raise ValueError(f"packed cache tensor {key!r} does not match manifest")
                if expected.get("digest_algorithm") != "sha256" or not isinstance(
                    expected.get("digest"), str
                ):
                    raise ValueError(f"packed cache tensor {key!r} has no supported digest")
                if _tensor_digest(value) != expected["digest"]:
                    raise ValueError(f"packed cache tensor checksum failed for {key!r}")
                packed_keys.add(key)
            codebook_identity = entry.get("codebook_identity")
            if codebook_identity is not None:
                codebook = buffers.get("orbit_codebook")
                if (
                    codebook_identity.get("algorithm") != "sha256"
                    or codebook is None
                    or codebook_identity.get("digest") != codebook.get("digest")
                ):
                    raise ValueError(f"cached codebook identity is invalid for {path!r}")
            validated_entries.append(
                (entry, module, quantizer, original_dtype, prefix)
            )

        for entry, module, quantizer, original_dtype, prefix in validated_entries:
            prepare_linear_for_ostris_cache(module, quantizer, original_dtype)
            for buffer_name in entry["buffers"]:
                key = prefix + buffer_name
                value = state_dict[key]
                module.register_buffer(
                    buffer_name,
                    value.to(target_device),
                    persistent=False,
                )
            for attribute, value in entry.get("attributes", {}).items():
                setattr(module, attribute, value)
            module.ostris_backend_name = entry["qtype"]

        remaining = {
            key: value.to(target_device) if target_device.type != "cpu" else value
            for key, value in state_dict.items()
            if key not in packed_keys
        }
        result = model.load_state_dict(remaining, strict=False, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                "packed cache model state mismatch: "
                f"missing={result.missing_keys[:4]}, unexpected={result.unexpected_keys[:4]}"
            )
        model.eval()

    def load(
        self,
        model: torch.nn.Module,
        component: str,
        cache_key: str,
        device: Optional[torch.device] = None,
        *,
        invalidate_on_error: bool = True,
    ) -> Dict[str, Any]:
        cache_dir = self.get_cache_dir(component, cache_key)
        manifest_path = os.path.join(cache_dir, CACHE_MANIFEST_NAME)
        try:
            if os.path.isfile(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as source:
                    manifest = json.load(source)
                if (
                    manifest.get("schema_version") != ORBIT_MANIFEST_SCHEMA_VERSION
                    or manifest.get("backend") != "ostris"
                    or manifest.get("key") != cache_key
                ):
                    raise ValueError("packed cache manifest is incompatible")
                self._load_ostris(model, cache_dir, manifest, device)
            else:
                qmap_path = os.path.join(cache_dir, CACHE_QMAP_NAME)
                with open(qmap_path, "r", encoding="utf-8") as qmap_file:
                    qmap = json.load(qmap_file)
                state_dict = load_file(
                    os.path.join(cache_dir, CACHE_WEIGHTS_NAME),
                    device="cpu",
                )
                requantize(model, state_dict=state_dict, quantization_map=qmap, device=device)
                model.eval()
            return self.load_metadata(component, cache_key)
        except Exception:
            if invalidate_on_error:
                self.invalidate(component, cache_key)
            raise

    def _atomic_publish(self, tmp_dir: str, final_dir: str) -> None:
        parent_dir = os.path.dirname(final_dir)
        os.makedirs(parent_dir, exist_ok=True)
        backup_dir = f"{final_dir}.old-{os.getpid()}"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        had_previous = os.path.exists(final_dir)
        if had_previous:
            _replace_with_retry(final_dir, backup_dir)
        try:
            _replace_with_retry(tmp_dir, final_dir)
        except Exception:
            if had_previous and os.path.exists(backup_dir):
                _replace_with_retry(backup_dir, final_dir)
            raise
        finally:
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir, ignore_errors=True)

    def _save_ostris(
        self,
        model: torch.nn.Module,
        tmp_dir: str,
        cache_key: str,
        key_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        modules = [
            _module_manifest(name, module)
            for name, module in model.named_modules()
            if isinstance(module, OstrisLinear)
        ]
        if not modules:
            raise ValueError("Model has no packed Ostris quantization to cache")
        state_dict = _collect_ostris_state_dict(model, modules)
        save_file(state_dict, os.path.join(tmp_dir, CACHE_WEIGHTS_NAME))
        qtypes_in_cache = sorted({entry["qtype"] for entry in modules})
        values = key_payload.get("values", {}) if isinstance(key_payload, dict) else {}
        quantize_kwargs = values.get("quantize_kwargs", {}) if isinstance(values, dict) else {}
        manifest = {
            "schema_version": ORBIT_MANIFEST_SCHEMA_VERSION,
            "backend": "ostris",
            "key": cache_key,
            "qtypes": qtypes_in_cache,
            "modules": modules,
            "exclusions": _normalize(
                quantize_kwargs.get("exclude", [])
                if isinstance(quantize_kwargs, dict)
                else []
            ),
            "source_fingerprints": key_payload.get("sources", []),
            "dependency_versions": key_payload.get("versions", {}),
        }
        _write_json(os.path.join(tmp_dir, CACHE_MANIFEST_NAME), manifest)
        return {"cache_backend": "ostris", "qtypes": qtypes_in_cache}

    def save(
        self,
        model: torch.nn.Module,
        component: str,
        cache_key: str,
        key_payload: Dict[str, Any],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        final_dir = self.get_cache_dir(component, cache_key)
        tmp_dir = f"{final_dir}.tmp-{os.getpid()}"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            has_ostris = any(isinstance(module, OstrisLinear) for module in model.modules())
            backend_metadata: Dict[str, Any]
            if has_ostris:
                backend_metadata = self._save_ostris(
                    model,
                    tmp_dir,
                    cache_key,
                    key_payload,
                )
            else:
                qmap = quantization_map(model)
                if not qmap:
                    raise ValueError("Model has no supported quantization map to cache")
                save_file(get_raw_state_dict(model), os.path.join(tmp_dir, CACHE_WEIGHTS_NAME))
                _write_json(os.path.join(tmp_dir, CACHE_QMAP_NAME), qmap)
                backend_metadata = {"cache_backend": "quanto"}

            metadata_payload = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "key": cache_key,
                "key_payload": key_payload,
                **backend_metadata,
            }
            if extra_metadata:
                metadata_payload.update(_normalize(extra_metadata))
            _write_json(os.path.join(tmp_dir, CACHE_METADATA_NAME), metadata_payload)
            self._atomic_publish(tmp_dir, final_dir)
            return final_dir
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
