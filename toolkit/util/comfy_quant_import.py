"""Import ComfyUI pre-quantized checkpoints onto toolkit modules.

ComfyUI quantized checkpoints mark each quantized submodule with a
``<prefix>.comfy_quant`` uint8 tensor holding a JSON config, alongside the
quantized ``weight`` and its scale tensors. This module walks those markers
and converts the matching submodules in place:

  - ``{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": G}``
    per-output-row symmetric int8 on regular-Hadamard-rotated weights — the
    exact storage of the toolkit's convrot8 backend
    (toolkit/util/convrot_quant.py:ConvRotInt8Quantizer), so the tensors are
    attached to its buffers directly (no requantization). Without the
    ``convrot`` flag the rotation block is 1, i.e. plain per-row int8, which
    the same backend also decodes (rotate is the identity at rot_size 1).
  - ``{"format": "nvfp4"}`` block-16 fp4 with e4m3 block scales, an fp32
    per-tensor scale and an optional AWQ ``pre_quant_scale`` — attached to
    the nvfp4 backend (toolkit/util/nvfp4_quant.py).
  - an int8 marker on an ``nn.Embedding`` swaps in :class:`Int8Embedding`
    (per-row scales, dequantized per lookup).

Linears become OstrisLinear (class swap in place, like
convert_linear_to_ostris), so LoRA attachment, memory management and the
quantized save paths all work unchanged.
"""

import json
from typing import Callable, Dict, Mapping, Optional, Tuple

import torch

from toolkit.util.nvfp4_quant import (
    Nvfp4Quantizer,
    swap_nvfp4_nibbles,
    unswizzle_nvfp4_scales,
)
from toolkit.util.ostris_quant import (
    OstrisLinear,
    get_ostris_quantizer,
    prepare_linear_for_ostris_cache,
)


INT8_EMBEDDING_QTYPE = "int8_embedding"
INT8_EMBEDDING_FORMAT_VERSION = 1
INT8_EMBEDDING_PACKED_LAYOUT = "int8_per_row_embedding_v1"


def parse_comfy_quant_blob(blob: torch.Tensor) -> Dict[str, object]:
    if blob.dtype != torch.uint8 or blob.ndim != 1:
        raise ValueError("comfy_quant metadata must be a one-dimensional uint8 tensor")
    try:
        value = json.loads(bytes(blob.detach().cpu().tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("comfy_quant metadata is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError("comfy_quant metadata must decode to a JSON object")
    return value


class Int8Embedding(torch.nn.Module):
    """An embedding table stored as per-row symmetric int8. Rows are
    dequantized per lookup, so the full-precision table never materializes."""

    packed_backend_name = INT8_EMBEDDING_QTYPE
    packed_format_version = INT8_EMBEDDING_FORMAT_VERSION
    packed_layout = INT8_EMBEDDING_PACKED_LAYOUT

    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        dtype: torch.dtype,
        *,
        packed_scales: bool = False,
    ):
        super().__init__()
        if qweight.ndim != 2 or qweight.dtype != torch.int8:
            raise ValueError("int8 embedding weights must be a two-dimensional int8 tensor")
        if not dtype.is_floating_point:
            raise ValueError("int8 embedding output dtype must be floating point")
        self.num_embeddings, self.embedding_dim = qweight.shape
        self.output_dtype = dtype
        self.register_buffer("qweight", qweight.contiguous(), persistent=False)
        if packed_scales:
            if scales.dtype != torch.uint8 or scales.numel() != self.num_embeddings * 4:
                raise ValueError("packed int8 embedding scales are incompatible")
            scale_buffer = scales.detach().contiguous()
        else:
            if scales.numel() != self.num_embeddings:
                raise ValueError("int8 embedding requires one scale per row")
            scale_buffer = (
                scales.detach().float().reshape(-1).contiguous().view(torch.uint8)
            )
        self.register_buffer(
            "scales",
            scale_buffer,
            persistent=False,
        )

    @property
    def weight(self):
        # full dequantized table, for code that inspects it
        scales = self.scales.view(torch.float32)
        return (self.qweight.float() * scales.unsqueeze(1)).to(self.output_dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # the table may stay CPU-resident under text-encoder offloading: run
        # the (tiny) lookup on the table's device, return on the caller's
        flat = input_ids.reshape(-1).to(self.qweight.device)
        rows = self.qweight.index_select(0, flat).float()
        scales = self.scales.view(torch.float32).index_select(0, flat)
        out = (rows * scales.unsqueeze(1)).to(self.output_dtype)
        return out.to(input_ids.device).reshape(*input_ids.shape, self.embedding_dim)


def _to_ostris(
    module: torch.nn.Linear,
    quantizer,
    orig_dtype: torch.dtype,
) -> OstrisLinear:
    prepare_linear_for_ostris_cache(module, quantizer, orig_dtype)
    if module.bias is not None:
        module.bias.requires_grad_(False)
    adapter_ref = getattr(module, "ara_lora_ref", None)
    adapter = adapter_ref() if callable(adapter_ref) else adapter_ref
    if adapter is not None and hasattr(adapter, "can_merge_in"):
        adapter.can_merge_in = False
    return module


def _required_tensor(
    state_dict: Dict[str, torch.Tensor],
    key: str,
) -> torch.Tensor:
    value = state_dict.pop(key, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"comfy quantized checkpoint is missing tensor {key!r}")
    return value


def _resolve_module(
    root: torch.nn.Module,
    prefix: str,
    key_map: Optional[Callable[[str], str]],
) -> Tuple[str, torch.nn.Module]:
    module_path = key_map(prefix) if key_map is not None else prefix
    if not isinstance(module_path, str) or not module_path:
        raise ValueError(f"invalid module path mapped from {prefix!r}")
    try:
        return module_path, root.get_submodule(module_path)
    except AttributeError as error:
        raise ValueError(
            f"comfy_quant marker {prefix!r} maps to missing module {module_path!r}"
        ) from error


@torch.no_grad()
def import_comfy_quantized_layers(
    root: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    orig_dtype: torch.dtype = torch.bfloat16,
    key_map: Optional[Callable[[str], str]] = None,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Convert every module a ``comfy_quant`` marker points at and attach its
    quantized tensors. Consumes the quantized entries from ``state_dict`` and
    returns ``(remaining_state_dict, num_converted)`` — load the remainder
    with the regular load_state_dict.

    ``key_map`` optionally maps a checkpoint prefix to the module path in
    ``root`` (e.g. comfy text encoder keys onto transformers module paths).
    """
    state_dict = dict(state_dict)
    converted = 0

    marker_keys = [k for k in state_dict.keys() if k.endswith(".comfy_quant")]
    # Validate every marker before mutating the first module. A malformed later
    # marker must not leave a large meta model half converted and unusable.
    mapped_paths = set()
    for marker_key in marker_keys:
        prefix = marker_key[: -len(".comfy_quant")]
        conf = parse_comfy_quant_blob(state_dict[marker_key])
        fmt = conf.get("format")
        module_path, module = _resolve_module(root, prefix, key_map)
        if module_path in mapped_paths:
            raise ValueError(f"multiple comfy_quant markers map to {module_path!r}")
        mapped_paths.add(module_path)
        weight_key = f"{prefix}.weight"
        scale_key = f"{prefix}.weight_scale"
        if weight_key not in state_dict:
            raise ValueError(
                f"comfy quantized checkpoint is missing tensor {weight_key!r}"
            )
        if scale_key not in state_dict:
            raise ValueError(f"quantized module {prefix!r} is missing weight_scale")
        weight = state_dict[weight_key]
        weight_scale = state_dict[scale_key]
        if not isinstance(weight, torch.Tensor) or not isinstance(
            weight_scale, torch.Tensor
        ):
            raise ValueError(f"quantized module {prefix!r} has non-tensor weights")
        if isinstance(module, OstrisLinear):
            raise ValueError(
                f"comfy_quant marker {prefix!r} targets an already packed linear"
            )
        if isinstance(module, torch.nn.Embedding):
            if fmt != "int8_tensorwise":
                raise ValueError(
                    f"Unsupported comfy quant format {fmt!r} on embedding {prefix}"
                )
            expected_shape = (module.num_embeddings, module.embedding_dim)
            if (
                weight.dtype != torch.int8
                or tuple(weight.shape) != expected_shape
                or weight_scale.numel() != expected_shape[0]
            ):
                raise ValueError(f"int8 embedding {prefix!r} has incompatible tensor shapes")
        elif isinstance(module, torch.nn.Linear):
            if fmt not in {"int8_tensorwise", "nvfp4"}:
                raise ValueError(
                    f"Unsupported comfy quant format {fmt!r} on {prefix} "
                    "(supported: int8_tensorwise, nvfp4)"
                )
            if fmt == "int8_tensorwise":
                rot = (
                    int(conf.get("convrot_groupsize", 256))
                    if conf.get("convrot")
                    else 1
                )
                if (
                    weight.dtype != torch.int8
                    or tuple(weight.shape)
                    != (module.out_features, module.in_features)
                    or weight_scale.numel() != module.out_features
                ):
                    raise ValueError(f"int8 linear {prefix!r} has incompatible tensor shapes")
                if rot <= 0 or rot & (rot - 1) or module.in_features % rot:
                    raise ValueError(
                        f"int8 linear {prefix!r} has invalid convrot_groupsize {rot}"
                    )
            else:
                pts_key = f"{prefix}.weight_scale_2"
                if pts_key not in state_dict:
                    raise ValueError(
                        f"comfy quantized checkpoint is missing tensor {pts_key!r}"
                    )
                pts = state_dict[pts_key]
                expected_scale_values = (
                    ((module.out_features + 127) // 128) * 128
                    * ((module.in_features // 16 + 3) // 4) * 4
                )
                if (
                    module.in_features % 16
                    or weight.dtype != torch.uint8
                    or tuple(weight.shape)
                    != (module.out_features, module.in_features // 2)
                    or weight_scale.numel() != expected_scale_values
                    or not isinstance(pts, torch.Tensor)
                    or pts.numel() != 1
                ):
                    raise ValueError(f"nvfp4 linear {prefix!r} has incompatible tensor shapes")
                pre_scale = state_dict.get(f"{prefix}.pre_quant_scale")
                if pre_scale is not None and (
                    not isinstance(pre_scale, torch.Tensor)
                    or pre_scale.numel() != module.in_features
                ):
                    raise ValueError(f"nvfp4 linear {prefix!r} has invalid pre_quant_scale")
        else:
            raise ValueError(
                f"comfy_quant marker {prefix} points at {type(module).__name__}, "
                "expected nn.Linear or nn.Embedding"
            )
        bias = state_dict.get(f"{prefix}.bias")
        if bias is not None and (
            not isinstance(bias, torch.Tensor)
            or not hasattr(module, "out_features")
            or bias.numel() != module.out_features
        ):
            raise ValueError(f"quantized module {prefix!r} has incompatible bias")

    for marker_key in marker_keys:
        prefix = marker_key[: -len(".comfy_quant")]
        conf = parse_comfy_quant_blob(state_dict.pop(marker_key))
        fmt = conf.get("format")
        module_path, module = _resolve_module(root, prefix, key_map)

        weight = _required_tensor(state_dict, f"{prefix}.weight")
        weight_scale = state_dict.pop(f"{prefix}.weight_scale", None)

        if isinstance(module, torch.nn.Embedding):
            if fmt != "int8_tensorwise":
                raise ValueError(
                    f"Unsupported comfy quant format {fmt!r} on embedding {prefix}"
                )
            if not isinstance(weight_scale, torch.Tensor):
                raise ValueError(f"int8 embedding {prefix!r} is missing weight_scale")
            parent_path, _, attr = module_path.rpartition(".")
            parent = root.get_submodule(parent_path) if parent_path else root
            setattr(parent, attr, Int8Embedding(weight, weight_scale, orig_dtype))
            converted += 1
            continue

        if not isinstance(module, torch.nn.Linear):
            raise ValueError(
                f"comfy_quant marker {prefix} points at {type(module).__name__}, "
                "expected nn.Linear or nn.Embedding"
            )

        if fmt == "int8_tensorwise":
            if not isinstance(weight_scale, torch.Tensor):
                raise ValueError(f"int8 linear {prefix!r} is missing weight_scale")
            rot = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1
            quantizer = get_ostris_quantizer("convrot8")
            if quantizer is None:
                raise RuntimeError("convrot8 quantization backend is unavailable")
            _to_ostris(module, quantizer, orig_dtype)
            module.register_buffer("cr8_qdata", weight.contiguous(), persistent=False)
            module.register_buffer(
                "cr8_scales",
                weight_scale.detach().float().reshape(-1).contiguous().view(torch.uint8),
                persistent=False,
            )
            module.cr8_rot_size = rot
        elif fmt == "nvfp4":
            if not isinstance(weight_scale, torch.Tensor):
                raise ValueError(f"nvfp4 linear {prefix!r} is missing weight_scale")
            quantizer = get_ostris_quantizer("nvfp4")
            if quantizer is None:
                raise RuntimeError("nvfp4 quantization backend is unavailable")
            pts = _required_tensor(state_dict, f"{prefix}.weight_scale_2")
            pre_scale = state_dict.pop(f"{prefix}.pre_quant_scale", None)
            if pre_scale is not None and not isinstance(pre_scale, torch.Tensor):
                raise ValueError(f"nvfp4 linear {prefix!r} has invalid pre_quant_scale")
            # normalize comfy_kitchen's storage to the toolkit's conventions:
            # fp4 pairs are packed high-nibble-first and the e4m3 block scales
            # are stored in the swizzled cuBLAS 128x4 tile layout
            scales = unswizzle_nvfp4_scales(
                weight_scale.view(torch.float8_e4m3fn),
                module.out_features,
                module.in_features // 16,
            )
            packed = swap_nvfp4_nibbles(weight)
            _to_ostris(module, quantizer, orig_dtype)
            Nvfp4Quantizer.attach_(
                module,
                packed=packed,
                scales=scales,
                pts=pts,
                pre_scale=pre_scale,
            )
        else:
            raise ValueError(
                f"Unsupported comfy quant format {fmt!r} on {prefix} "
                "(supported: int8_tensorwise, nvfp4)"
            )

        # drop unused calibration extras if present
        state_dict.pop(f"{prefix}.input_scale", None)

        bias = state_dict.pop(f"{prefix}.bias", None)
        if bias is not None and module.bias is not None:
            # bias may still be a meta parameter when the model was built under
            # a meta device context
            module._parameters["bias"] = torch.nn.Parameter(
                bias.detach().clone(), requires_grad=False
            )
        converted += 1

    return state_dict, converted
